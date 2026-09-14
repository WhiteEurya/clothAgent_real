"""RGB-only company Claude adapter for the real fold loop.

The existing reference selection gates, grasp-height policy and execution gates
stay on the host. Remote movement proposals use upright pixels and relative
heights, never measured XYZ. No local Claude process or remote MCP is needed.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .auto_exploration import (
    AUTO_EVALUATION_JSON_SCHEMA, VISUAL_PLAN_JSON_SCHEMA,
    ClaudeAutoClient, ClaudeEvaluationResult, ClaudeVisualPlanResult,
    _now, prepare_rollout_video_evidence, validate_evaluation_payload,
    validate_visual_plan_payload,
)
from .free_exploration import (
    ClaudeExplorationResult, ExplorationPlanningError, MAX_EXPLORATION_ACTIONS,
    validate_exploration_payload,
)
from .garment_grounding_mcp import GarmentGrounding
from .grasp_height import resolve_grasp_height
from .planner_backend import RemoteClaudeBackend, parse_claude_json


# Explicit allow-list of RGB artifacts produced by the fold pipeline. Never
# select heatmaps, reports, masks or geometry merely because they are PNGs.
RGB_NAMES = frozenset({
    "camera_0_a.png", "camera_1_b.png", "camera_a_rgb_upright.png",
    "camera_a_rxxx_overlay_upright.png", "camera_a_flat_reference.png",
    "camera_a_flat_reference_anchors.png",
    "camera_a_rgb_contact_sheet.png", "camera_b_rgb_contact_sheet.png",
    "camera_c_rgb_contact_sheet.png", "camera_c.png",
    "camera_c_observer_rgb.png", "camera_c_observer_rgb_hold_check.png",
})


def rgb_evidence(paths: Sequence[Path], root: Path) -> list[Path]:
    result: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if path.name.lower() not in RGB_NAMES:
            continue
        if root.resolve() not in path.parents or not path.is_file():
            raise ExplorationPlanningError("RGB evidence is missing or outside the run")
        if path not in result:
            result.append(path)
    return result


def image_manifest(images: Sequence[Path], label: str) -> list[dict[str, Any]]:
    # Indices correspond to the backend's image_N.png names, never local paths.
    return [{"image_index": i, "role": f"{label}: {p.name}"}
            for i, p in enumerate(images)]


_SEMANTIC_KEYS = frozenset({
    "step", "current_step", "completed_steps", "action_mode", "mode", "status",
    "evaluation", "supervisor", "supervisor_before", "supervisor_after",
    "target_selection", "grasp_acquisition", "target_structure_acquired", "transport",
    "laydown", "task_progress", "earliest_failure_stage", "next_experiment",
    "confidence", "evidence", "keep", "change", "reason", "garment_condition",
    "trajectory_decision", "visibility", "condition", "metrics",
    "planned_step", "garment_visibility", "garment_condition_before", "garment_condition_after",
    "fallback", "completion_ledger_source", "local_deterministic",
    "visible_area_delta", "overlap_delta", "relief_delta", "boundary_change",
})


def semantic_history(value: Any) -> Any:
    """Project known semantic fields; do not relay full run/robot records."""
    if isinstance(value, Mapping):
        return {k: semantic_history(v) for k, v in value.items() if k in _SEMANTIC_KEYS}
    if isinstance(value, (list, tuple)):
        return [semantic_history(v) for v in value][-8:]
    if isinstance(value, str):
        # Host failure strings sometimes embed measured coordinates or paths.
        if re.search(r"(?:/home/|/tmp/|[XYZxyz]\s*[=:]|\b(?:XYZ|depth|calibration|extrinsics|intrinsics)\b|\bmm\b)", value):
            return "Host-only detail omitted."
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


def semantic_task(objective: str) -> dict[str, str]:
    # Fold's first paragraph contains only the user task; appended context
    # includes local measurements, paths and Molmo records and is not relayed.
    task = objective.split("\n", 1)[0]
    mode = "ACQUISITION_PROBE" if "EXECUTION CONTRACT — ACQUISITION PROBE" in objective else (
        "REPAIR_SLEEVE" if "ACTION MODE — REPAIR_SLEEVE" in objective else "FOLD"
    )
    return {"objective": semantic_history(task), "mode": mode}


def rejection_category(feedback: str | None) -> str | None:
    if not feedback:
        return None
    lower = feedback.lower()
    for term, category in (("workspace", "waypoint_outside_workspace"),
            ("bound", "waypoint_outside_workspace"), ("ik", "pose_unreachable"),
            ("acquisition", "acquisition_contract_or_strategy_rejected"),
            ("grasp", "contact_strategy_rejected"), ("sweep", "unsafe_lateral_entry")):
        if term in lower:
            return category
    return "host_rejected_previous_candidate"


MOVE_ARGS = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "target": {"enum": ["grasp", "pixel"]},
        "pixel_xy": {"anyOf": [{"type": "null"}, {"type": "array", "minItems": 2,
            "maxItems": 2, "items": {"type": "integer", "minimum": 0}}]},
        "height_above_grasp_mm": {"type": "number", "minimum": 0},
        "yaw_deg": {"type": "number"},
    },
    "required": ["target", "pixel_xy", "height_above_grasp_mm", "yaw_deg"],
}
MOTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "actions": {"type": "array", "minItems": 1, "maxItems": MAX_EXPLORATION_ACTIONS,
            "items": {"oneOf": [
                {"type": "object", "additionalProperties": False,
                 "properties": {"name": {"const": "move"}, "args": MOVE_ARGS},
                 "required": ["name", "args"]},
                {"type": "object", "additionalProperties": False,
                 "properties": {"name": {"enum": ["open_gripper", "close_gripper", "home"]},
                    "args": {"type": "object", "additionalProperties": False}},
                 "required": ["name", "args"]},
            ]}},
        "requires_lift_checkpoint": {"type": "boolean"},
    },
    "required": ["actions", "requires_lift_checkpoint"],
}


def compile_pixel_motion(payload, visual, grounding, robot_config, upright_size):
    """Convert explicitly proposed waypoints, without adding/defaulting actions."""
    if not isinstance(payload, dict) or set(payload) != {"actions", "requires_lift_checkpoint"}:
        raise ExplorationPlanningError("invalid remote motion fields")
    raw_actions = payload["actions"]
    if not isinstance(raw_actions, list) or not 1 <= len(raw_actions) <= MAX_EXPLORATION_ACTIONS:
        raise ExplorationPlanningError("invalid remote action count")
    selected = visual.selected_reference
    if selected["camera"] != "A":
        raise ExplorationPlanningError("remote fold grounding requires wrist Camera A")
    measurement = grounding.lookup_reference("A", selected["reference_id"])
    surface = grounding.sample_local_surface("A", *measurement["pixel_xy"],
        radius_px=3, include_nearest_reference=False)
    if surface.get("valid") is not True:
        raise ExplorationPlanningError("selected pixel has no valid local surface")
    height = resolve_grasp_height(measurement=surface, table_plane_abc=None,
                                 robot_config=robot_config)
    grasp_xyz = [*measurement["base_xyz_mm"][:2], float(height.target_xyz_mm[2])]
    if not all(math.isfinite(float(v)) for v in grasp_xyz):
        raise ExplorationPlanningError("non-finite local grasp geometry")
    actions = []
    for action in raw_actions:
        if not isinstance(action, dict) or set(action) != {"name", "args"}:
            raise ExplorationPlanningError("invalid remote action object")
        name, args = action["name"], action["args"]
        if name != "move":
            if name not in {"open_gripper", "close_gripper", "home"} or args != {}:
                raise ExplorationPlanningError("invalid remote gripper/home action")
            actions.append({"name": name, "args": {}})
            continue
        if not isinstance(args, dict) or set(args) != set(MOVE_ARGS["required"]):
            raise ExplorationPlanningError("invalid remote move fields")
        offset, yaw = args["height_above_grasp_mm"], args["yaw_deg"]
        for number in (offset, yaw):
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ExplorationPlanningError("remote move values must be finite numbers")
        if offset < 0:
            raise ExplorationPlanningError("remote motion cannot descend below the local grasp height")
        if args["target"] == "grasp" and args["pixel_xy"] is None:
            x, y = grasp_xyz[:2]
        elif args["target"] == "pixel":
            pixel = args["pixel_xy"]
            if (not isinstance(pixel, list) or len(pixel) != 2 or
                    any(type(v) is not int for v in pixel) or
                    not (0 <= pixel[0] < upright_size[0] and 0 <= pixel[1] < upright_size[1])):
                raise ExplorationPlanningError("remote target is not a valid upright pixel")
            # raw -> clockwise90: (x,y) -> (raw_height-1-y,x).
            sample = grounding.sample_pixel_xyz("A", pixel[1], upright_size[0] - 1 - pixel[0])
            if sample.get("valid") is not True:
                raise ExplorationPlanningError("remote target pixel has no measured depth")
            xyz = sample.get("base_xyz_mm", [])
            if len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz):
                raise ExplorationPlanningError("remote target pixel has invalid XYZ")
            x, y = xyz[:2]
        else:
            raise ExplorationPlanningError("remote target must be grasp/null or pixel/[u,v]")
        z = grasp_xyz[2] + offset
        robot_config.validate_workspace_pose(x, y, z, yaw)
        actions.append({"name": "move", "args": {"x": x, "y": y, "z": z, "yaw": yaw}})
    proposal = validate_exploration_payload({
        "garment_observation": visual.garment_observation,
        "reveal_strategy": visual.opening_strategy, "confidence": visual.confidence,
        "actions": actions, "expected_observation": visual.expected_observation,
        "safety_notes": list(visual.safety_notes),
        "requires_lift_checkpoint": payload["requires_lift_checkpoint"],
    })
    close = next(i for i, a in enumerate(actions) if a["name"] == "close_gripper")
    contact = next((a for a in reversed(raw_actions[:close]) if a["name"] == "move"), None)
    if (contact is None or contact["args"]["target"] != "grasp" or
            contact["args"]["height_above_grasp_mm"] != 0):
        raise ExplorationPlanningError("closure must use the selected reference at locally resolved grasp height")
    proposal = replace(proposal, skill_invocations=visual.skill_invocations)
    return proposal, {"measurement": measurement, "grasp_xy_error_mm": 0.0,
                      "height_resolution": height.as_dict(), "authority": "local_pixel_compiler"}


class RemoteFoldClient(ClaudeAutoClient):
    """Use the parent's local selection/reselection gates with remote model stages."""

    def __init__(self, *, backend: RemoteClaudeBackend, **kwargs):
        kwargs["persistent_session"] = None  # Company CLI calls are stateless.
        super().__init__(**kwargs)
        self.backend = backend
        self._remote_context: dict[str, Any] | None = None
        self._remote_images: list[Path] = []

    def plan(self, image_paths, session, objective, feedback=None, history=None,
             phase_callback=None, reference_policy="uniform", workspace_recovery=None):
        self.last_plan_result = None
        self.last_visual_plan_result = None
        self._remote_context = None
        self._remote_images = []
        if workspace_recovery is not None and workspace_recovery.required:
            raise ExplorationPlanningError("remote fold does not support metric workspace-recovery requests")
        images = rgb_evidence(image_paths, session.run_dir)
        by_name = {p.name.lower(): p for p in images}
        required = {"camera_a_rgb_upright.png", "camera_a_rxxx_overlay_upright.png"}
        if not required <= set(by_name):
            raise ExplorationPlanningError("remote fold requires the current upright RGB and Rxxx overlay")
        raw = session.run_dir / "workspace" / "perception_views" / "camera_0_A.png"
        with Image.open(raw) as source, Image.open(by_name["camera_a_rgb_upright.png"]) as current:
            expected = source.convert("RGB").rotate(-90, expand=True)
            if current.size != expected.size or current.convert("RGB").tobytes() != expected.tobytes():
                raise ExplorationPlanningError("upright RGB does not match the current local grounding capture")
        self._remote_images = images
        self._remote_context = {**semantic_task(objective), "recent_outcomes": semantic_history(history or []),
                                "previous_candidate_rejected": rejection_category(feedback)}
        return super().plan(image_paths, session, objective, feedback, history,
                            phase_callback, reference_policy, workspace_recovery)

    def _ask(self, stage, context, schema, images, root, instructions):
        prompt = instructions + "\n" + json.dumps(context, ensure_ascii=False)
        started = time.monotonic()
        try:
            result = self.backend.invoke(prompt=prompt, image_paths=images, schema=schema,
                timeout_s=self.grounding_timeout_s if stage == "pixel_motion" else self.timeout_s,
                system_prompt="You are a read-only garment reasoning assistant. Read the supplied RGB files. Return only the requested JSON. No tools except Read; no robot access.")
            payload = parse_claude_json(result.stdout)
        except Exception as exc:
            self._save_visual_log(root, {"stage": stage, "backend": "remote",
                "error": f"{type(exc).__name__}: {exc}", "created_at": _now()}, failed=True)
            raise
        return payload, result, prompt, time.monotonic() - started

    def _visual_plan(self, image_paths, base_prompt, run_dir):
        if self._remote_context is None:
            raise ExplorationPlanningError("remote visual stage has no current request")
        context = {**self._remote_context, "images": image_manifest(self._remote_images, "current/reference RGB"),
            "approved_skill_names": list(self.skill_names),
            "locally_executable_reference_ids": (self.last_reference_candidate_report or {}).get("executable_reference_ids"),
            "rejected_references": [{"camera": r["camera"], "reference_id": r["reference_id"]}
                                    for r in self.last_rejected_visual_references]}
        payload, result, prompt, duration = self._ask("visual_planning", context,
            VISUAL_PLAN_JSON_SCHEMA, self._remote_images, run_dir,
            "Select one visible Camera-A Rxxx marker for the exact current task. The current RGB and marker overlay are rotated clockwise90 upright; left/right refer to that displayed image, not anatomy. Flat reference images are topology references only. Do not choose an already rejected marker. Describe your motion strategy and expected physical evidence. Do not output XYZ or actions.")
        decision = validate_visual_plan_payload(payload, allowed_skill_names=self.skill_names)
        record = ClaudeVisualPlanResult(prompt, result.command, result.returncode,
            result.stdout, result.stderr, _now(), duration, decision)
        self._save_visual_log(run_dir, record.as_dict())
        return record

    def _ground_final_plan(self, visual, session, objective, history=None, workspace_recovery=None):
        self.last_plan_result = None
        self.last_grounding_verification = None
        if self._remote_context is None or not self._remote_images:
            raise ExplorationPlanningError("remote motion stage has no current observation")
        context = {**self._remote_context, "visual_plan": visual.as_dict(),
            "images": image_manifest(self._remote_images, "current/reference RGB"),
            "repair_requested": "HOST VALIDATION CORRECTION" in objective,
            "repair_category": rejection_category(objective.split("HOST VALIDATION CORRECTION", 1)[1])
                               if "HOST VALIDATION CORRECTION" in objective else None}
        payload, result, prompt, duration = self._ask("pixel_motion", context, MOTION_SCHEMA,
            self._remote_images, session.run_dir,
            "Return the complete proposed move/open_gripper/close_gripper/home sequence. Each move uses target=grasp with pixel_xy=null for the fixed selected marker, or target=pixel with [u,v] in the CURRENT upright RGB for transport destinations. height_above_grasp_mm is a proposed NONNEGATIVE relative lift above the host-resolved closure height; it is not a measured coordinate. yaw_deg is relative to calibrated Home. All conversions, depth checks and execution checks are local. Approach with clearance, open, descend to target=grasp and height=0, close, lift before lateral transport, lay down and release, retreat and home. Explicitly include every action; the host does not insert missing actions. In ACQUISITION_PROBE mode use only target=grasp: lift, reverse to the same contact, release and home; set requires_lift_checkpoint=true. In FOLD mode actually transport inward; in REPAIR_SLEEVE mode transport outward to unbunch, then release. Do not send measured XYZ or code.")
        rgb = next(p for p in self._remote_images if p.name.lower() == "camera_a_rgb_upright.png")
        with Image.open(rgb) as image:
            size = image.size
        try:
            proposal, verification = compile_pixel_motion(payload, visual,
                GarmentGrounding(session.run_dir / "workspace" / "perception_views"),
                session.robot_config, size)
        except Exception as exc:
            self.planner._save_invocation_log(session.run_dir, {
                "stage": "local_pixel_grounding", "remote_motion": payload,
                "error": f"{type(exc).__name__}: {exc}"}, failed=True)
            raise
        # The same downstream mode, grounding, IK and execution checks still run.
        record = ClaudeExplorationResult(prompt, result.command, result.returncode,
            result.stdout, result.stderr, _now(), proposal)
        self.last_grounding_verification = verification
        self.planner._save_invocation_log(session.run_dir, {
            "stage": "local_pixel_grounding", "duration_s": duration,
            "remote_motion": payload, "grounding_verification": verification,
            "proposal": proposal.as_dict(), "command": list(result.command)})
        return record

    def evaluate(self, before_images, after_images, *, proposal, run_dir, objective=None,
                 rollout_recording_dir=None, observer_images=(), **kwargs):
        return self._evaluate_remote(before_images, after_images, proposal=proposal,
            run_dir=run_dir, objective=objective, rollout_recording_dir=rollout_recording_dir,
            observer_images=observer_images, acquisition=False)

    def evaluate_acquisition_probe(self, before_images, after_images, *, proposal, run_dir,
            rollout_recording_dir=None, rollout_evidence_images=(), observer_images=(), **kwargs):
        return self._evaluate_remote(before_images, after_images, proposal=proposal,
            run_dir=run_dir, rollout_recording_dir=rollout_recording_dir,
            rollout_evidence_images=rollout_evidence_images, observer_images=observer_images,
            acquisition=True)

    def _evaluate_remote(self, before_images, after_images, *, proposal, run_dir,
            objective=None, rollout_recording_dir=None, rollout_evidence_images=(),
            observer_images=(), acquisition=False):
        self.last_evaluation_result = None
        before, after = rgb_evidence(before_images, run_dir), rgb_evidence(after_images, run_dir)
        if not before or not after:
            raise ExplorationPlanningError("remote evaluation requires before and after RGB")
        video, refs, errors = list(rollout_evidence_images), [], []
        if not video and rollout_recording_dir is not None:
            video, refs, errors = prepare_rollout_video_evidence(rollout_recording_dir)
        video = rgb_evidence(video, run_dir)
        observers = rgb_evidence(observer_images, run_dir)
        images = [*before, *after, *video, *observers]
        roles = (["before"] * len(before) + ["after"] * len(after) +
                 ["rollout RGB contact sheet"] * len(video) + ["observer after"] * len(observers))
        context = {"task": semantic_task(objective or "Evaluate the current garment task."),
            "acquisition_only": acquisition,
            "proposed_strategy": semantic_history(proposal.reveal_strategy),
            "expected_observation": semantic_history(proposal.expected_observation),
            "images": [{"image_index": i, "role": role, "name": path.name}
                       for i, (path, role) in enumerate(zip(images, roles))]}
        payload, result, prompt, _ = self._ask("acquisition_evaluation" if acquisition else "evaluation",
            context, AUTO_EVALUATION_JSON_SCHEMA, images, run_dir,
            "Evaluate actual visible before/after and chronological rollout evidence, never infer success from the proposed strategy. No telemetry or depth is supplied. Mark acquisition/target UNKNOWN when images do not establish them. For acquisition-only probes, transport status must be UNKNOWN, laydown NOT_REACHED, task_progress NEUTRAL; do not claim that returning to the initial scene proves successful acquisition. Provide the full requested evaluation schema, with causal next_experiment suggestions.")
        evaluation = validate_evaluation_payload(payload)
        if acquisition and (evaluation.transport.status != "UNKNOWN" or
                evaluation.laydown.status != "NOT_REACHED" or evaluation.task_progress.status != "NEUTRAL"):
            raise ExplorationPlanningError("acquisition evaluation incorrectly claims transport/fold progress")
        record = ClaudeEvaluationResult(prompt, result.command, result.returncode, result.stdout,
            result.stderr, _now(), evaluation, tuple(map(str, images)), tuple(map(str, refs)), tuple(errors))
        self.last_evaluation_result = record
        self._save_evaluation_log(run_dir, record.as_dict())
        return evaluation
