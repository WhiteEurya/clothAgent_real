"""RGB-only company Claude adapter for the real fold loop.

The existing reference selection gates, grasp-height policy and execution gates
stay on the host. Remote movement proposals use upright pixels and relative
heights, never measured XYZ. Remote MCP exposes only RGB inspection tools.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
import time
import uuid
import copy
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
from .grasp_checkpoint import ACQUISITION_PROBE_LIFT_CONTRACT
from .perception_comparison import COMPARISON_INSTRUCTION, comparison_schema
from .planner_backend import RemoteClaudeBackend, parse_claude_json
from .config import SafetyError
from .workspace_debug import WorkspaceTargetError, lateral_clearance, save_workspace_debug
from .fold_frame import CLAUDE_FOLD_RULE, load_frame
from .claude_image_debug import debug_directory
from .motion_image_sources import resolve_motion_sources
from .claude_molmo_view import prepare_molmo_view
from .trajectory_memory import HISTORY_RGB_NAMES


# Explicit allow-list of RGB artifacts produced by the fold pipeline. Never
# select heatmaps, reports, masks or geometry merely because they are PNGs.
RGB_NAMES = frozenset({
    "camera_0_a.png", "camera_1_b.png", "camera_a_rgb_upright.png",
    "camera_a_rxxx_overlay_upright.png", "camera_a_flat_reference.png",
    "camera_a_flat_reference_anchors.png",
    "camera_a_rgb_contact_sheet.png", "camera_b_rgb_contact_sheet.png",
    "camera_c_rgb_contact_sheet.png", "camera_c.png",
    "camera_c_observer_rgb.png", "camera_c_observer_rgb_hold_check.png",
    "camera_a_grasp_after_close.png", "camera_a_grasp_after_lift.png",
    "camera_a_grasp_before_lift.png",
    "fold_reference_source.png", "fold_reference_target.png",
    "camera_a_molmo_frame_hint.png", "camera_a_molmo_hint_upright.png",
    "camera_a_molmo_hint_collar_up.png",
})


def rgb_evidence(paths: Sequence[Path], root: Path) -> list[Path]:
    result: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path.name.lower() not in RGB_NAMES | HISTORY_RGB_NAMES and not re.fullmatch(r"camera_a_lift_checkpoint_\d+\.png", path.name.lower()):
            continue
        if root.resolve() not in path.parents or not path.is_file():
            raise ExplorationPlanningError("RGB evidence is missing or outside the run")
        # The pipeline stages copies of the same named RGB in multiple folders.
        # Keep distinct roles/names; before and after are filtered separately.
        identity = (path.name.lower(), hashlib.sha256(path.read_bytes()).hexdigest())
        if identity not in seen:
            seen.add(identity)
            result.append(path)
    return result


def image_manifest(images: Sequence[Path], label: str) -> list[dict[str, Any]]:
    # Indices correspond to the backend's image_N.png names, never local paths.
    return [{"image_index": i, "role": f"{'HISTORICAL ONLY; not executable' if p.name.lower() in HISTORY_RGB_NAMES else label}: {p.name}"}
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
    "failure_detection", "category", "safe_return_confirmed", "failed_stage", "inherited_lesson",
    "perception_comparison",
    "acquisition_learning", "phase", "instruction", "consecutive_acquisition_failures",
    "require_non_height_change", "height_only_retry_pattern", "uncertain_since_last_evidence",
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
REMOTE_MOVE_ARGS = copy.deepcopy(MOVE_ARGS)
REMOTE_MOVE_ARGS['properties']['image_id'] = {'type': ['string', 'null']}
REMOTE_MOVE_ARGS['properties']['pixel_xy']['anyOf'][1]['items']['type'] = 'number'
REMOTE_MOVE_ARGS['required'].append('image_id')
MOTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "actions": {"type": "array", "minItems": 1, "maxItems": MAX_EXPLORATION_ACTIONS,
            "items": {"oneOf": [
                {"type": "object", "additionalProperties": False,
                 "properties": {"name": {"const": "move"}, "args": REMOTE_MOVE_ARGS},
                 "required": ["name", "args"]},
                {"type": "object", "additionalProperties": False,
                 "properties": {"name": {"enum": ["open_gripper", "close_gripper", "home"]},
                    "args": {"type": "object", "additionalProperties": False}},
                 "required": ["name", "args"]},
            ]}},
        "contact_descent_mm": {"type": "number", "description": "Claude-chosen descent below estimated surface; NOT observed compression."},
        "requires_lift_checkpoint": {"type": "boolean"},
    },
    "required": ["actions", "requires_lift_checkpoint", "contact_descent_mm"],
}


def compile_pixel_motion(payload, visual, grounding, robot_config, upright_size):
    """Convert explicitly proposed waypoints, without adding/defaulting actions."""
    if not isinstance(payload, dict) or set(payload) != {"actions", "requires_lift_checkpoint", "contact_descent_mm"}:
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
                                 robot_config=robot_config, proposed_descent_mm=payload["contact_descent_mm"])
    grasp_xyz = [*measurement["base_xyz_mm"][:2], float(height.target_xyz_mm[2])]
    if not all(math.isfinite(float(v)) for v in grasp_xyz):
        raise ExplorationPlanningError("non-finite local grasp geometry")
    actions = []
    trace = {"selected_reference": dict(selected), "moves": [], "status": "COMPILING",
             "remote_motion": payload, "visual_plan": visual.as_dict()}
    for action_index, action in enumerate(raw_actions, 1):
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
            raw_pixel = list(measurement["pixel_xy"])
            upright_pixel = [upright_size[0] - 1 - raw_pixel[1], raw_pixel[0]]
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
            upright_pixel = list(pixel)
            raw_pixel = [pixel[1], upright_size[0] - 1 - pixel[0]]
        else:
            raise ExplorationPlanningError("remote target must be grasp/null or pixel/[u,v]")
        z = grasp_xyz[2] + offset
        point = {"action_index": action_index, "target": args["target"],
                 "raw_pixel_xy": raw_pixel, "upright_pixel_xy": upright_pixel,
                 "base_xyz_mm": [float(x), float(y), float(z)], "yaw_deg": yaw,
                 "effective_y_limits_mm": list(robot_config.y_workspace_bounds_mm(yaw)),
                 "lateral": lateral_clearance(robot_config.boundaries, x, y, robot_config.workspace_margin_mm)}
        trace["moves"].append(point)
        try:
            robot_config.validate_workspace_pose(x, y, z, yaw)
        except SafetyError as exc:
            point["error"] = str(exc)
            trace["status"] = "REJECTED"
        actions.append({"name": "move", "args": {"x": x, "y": y, "z": z, "yaw": yaw}})
    if trace["status"] == "REJECTED":
        raise WorkspaceTargetError(trace)
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
    trace["status"] = "WORKSPACE_VALIDATED_NOT_EXECUTED"
    return proposal, {"measurement": measurement, "grasp_xy_error_mm": 0.0, "workspace_trace": trace,
                      "height_resolution": height.as_dict(), "authority": "local_pixel_compiler"}


class RemoteFoldClient(ClaudeAutoClient):
    """Use the parent's local selection/reselection gates with remote model stages."""

    def __init__(self, *, backend: RemoteClaudeBackend, **kwargs):
        kwargs["persistent_session"] = None  # Company CLI calls are stateless.
        super().__init__(**kwargs)
        self.backend = backend
        self._remote_context: dict[str, Any] | None = None
        self._remote_images: list[Path] = []
        self.diagnostics_dir: Path | None = None

    def prepare_molmo_view(self, canonical_image: Path, output: Path):
        return prepare_molmo_view(self.backend, canonical_image, output, timeout_s=self.timeout_s)

    def plan(self, image_paths, session, objective, feedback=None, history=None,
             phase_callback=None, reference_policy="uniform", workspace_recovery=None):
        objective += '\n' + CLAUDE_FOLD_RULE
        self.last_plan_result = None
        self.last_grounding_verification = None
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
        self._phase_callback = phase_callback
        self._call_diagnostics = (self.diagnostics_dir or
            by_name["camera_a_rgb_upright.png"].parent / "remote_diagnostics") / uuid.uuid4().hex[:12]
        views = session.run_dir / "workspace" / "perception_views"
        precheck_started = time.monotonic()
        precheck = save_workspace_debug(self._call_diagnostics, views, session.robot_config)
        if phase_callback is not None:
            phase_callback("workspace_precheck_and_images", "completed", time.monotonic() - precheck_started)
        # This context contains only image-plane coordinates. The metric report
        # and all diagnostic images remain on the host.
        transport_pixels = [[int(expected.width - 1 - r["pixel_xy"][1]), int(r["pixel_xy"][0])]
                            for r in precheck.get("references", []) if r["xy_eligible"]]
        fold_reference_context: dict[str, Any] | None = None
        for candidate in images:
            if candidate.name.lower() != "fold_reference_source.png":
                continue
            manifest_path = candidate.parent / "reference_manifest.json"
            if manifest_path.is_file():
                try:
                    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ExplorationPlanningError("fold-state reference manifest is unreadable") from exc
                if not isinstance(loaded, dict) or loaded.get("reference_type") != "static_cross_garment_fold_states":
                    raise ExplorationPlanningError("fold-state reference manifest has an invalid type")
                fold_reference_context = {
                    key: loaded[key]
                    for key in ("reference_type", "current_step", "source_state", "target_state",
                                "source_image", "target_image", "role", "coordinate_policy")
                    if key in loaded
                }
            break
        self._remote_context = {**semantic_task(objective), "recent_outcomes": semantic_history(history or []),
                                "acquisition_learning": semantic_history(getattr(self, "acquisition_learning", {})),
                                "previous_candidate_rejected": rejection_category(feedback),
                                "xy_eligible_transport_pixels_upright": transport_pixels,
                                "fold_state_reference": fold_reference_context}
        # This host-built context has its own explicit field selection. Applying
        # semantic_history would erase the action sequence and relative motion.
        memory = copy.deepcopy(getattr(self, "trajectory_memory", None))
        if memory is not None:
            historical_ids = {p.name: f"image_{i}" for i, p in enumerate(images)
                              if p.name.lower() in HISTORY_RGB_NAMES}
            previous = memory.get("previous_physical_attempt") or {}
            for evidence in previous.get("images", []):
                evidence["image_id"] = historical_ids.get(evidence.get("name"))
            if previous.get("grasp_in_before_image"):
                grasp = previous["grasp_in_before_image"]
                grasp["image_id"] = historical_ids.get(grasp.get("name"))
            self._remote_context["trajectory_memory"] = memory
        if getattr(self, "experience_context", None) is not None:
            self._remote_context["experience_context"] = copy.deepcopy(self.experience_context)
        if (views / 'garment_frame.json').exists():
            try:
                self._remote_context['molmo_frame_hint'] = load_frame(views, expected)
            except (OSError, ValueError, KeyError):
                self._remote_context['molmo_frame_hint'] = None
        self._remote_context['semantic_authority'] = CLAUDE_FOLD_RULE
        try:
            return super().plan(image_paths, session, objective, feedback, history,
                                phase_callback, reference_policy, workspace_recovery)
        except Exception as exc:
            path = self._call_diagnostics / "workspace_diagnostics.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["plan_error"] = f"{type(exc).__name__}: {exc}"
            if report.get("status") != "REJECTED":
                report["status"] = "PLANNING_FAILED_NO_ACTION"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise
        finally:
            (self._call_diagnostics / "reference_prevalidation.json").write_text(
                json.dumps(self.last_reference_candidate_report, indent=2), encoding="utf-8")

    def _ask(self, stage, context, schema, images, root, instructions):
        evaluation_stage = stage in {'evaluation', 'acquisition_evaluation', 'experience_update'}
        instructions = instructions.replace('with [u,v] in the CURRENT upright RGB for transport destinations.',
            'with image_id naming the exact source RGB/view and pixel_xy in that view; the host maps transport destinations.')
        instructions = instructions.replace(
            'left/right refer to that displayed image, not anatomy.',
            'left/right are garment-relative; Claude determines them from the current RGB.')
        instructions += ' ' + CLAUDE_FOLD_RULE
        prompt = instructions + "\n" + json.dumps(context, ensure_ascii=False)
        started = time.monotonic()
        diagnostics = getattr(self, "_call_diagnostics", None)
        invocation = {"stage": stage, "evidence_images": [str(p) for p in images], "status": "RUNNING"}
        image_debug = debug_directory(images, diagnostics or root, stage)
        invocation["image_debug_directory"] = str(image_debug)
        manifest = diagnostics / f"{stage}_invocation.json" if diagnostics is not None else None
        if manifest is not None:
            manifest.write_text(json.dumps(invocation, indent=2), encoding="utf-8")
        try:
            result = self.backend.invoke(prompt=prompt, image_paths=images, schema=schema,
                debug_dir=image_debug,
                image_edit_limit=2 if evaluation_stage else 6,
                max_turns=8 if evaluation_stage else None,
                timeout_s=self.grounding_timeout_s if stage == "pixel_motion" else self.timeout_s,
                system_prompt="You are a garment reasoning assistant. Inspect RGB using view_image and the images returned directly by editing tools. Claude decides semantic targets; Molmo annotations are optional hints. Follow the response schema's image_id/pixel source contract exactly; the host performs coordinate transforms and safety checks. Return only the requested JSON. No robot access.")
            payload = parse_claude_json(result.stdout)
        except Exception as exc:
            invocation.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                              timings=getattr(exc, "timings", {}), duration_s=time.monotonic() - started,
                              image_tool_events=getattr(exc, "image_tool_events", ()))
            if manifest is not None:
                manifest.write_text(json.dumps(invocation, indent=2), encoding="utf-8")
            self._save_visual_log(root, {"stage": stage, "backend": "remote",
                **invocation, "created_at": _now()}, failed=True)
            raise
        self._save_visual_log(root, {"stage": stage, "backend": "remote", "created_at": _now(),
            "evidence_images": [str(p) for p in images], "timings": getattr(result, "timings", {}),
            "image_tool_events": getattr(result, "image_tool_events", ()),
            "duration_s": time.monotonic() - started})
        if manifest is not None:
            manifest.write_text(json.dumps({
                "stage": stage, "evidence_images": [str(p) for p in images],
                "image_debug_directory": str(image_debug),
                "timings": getattr(result, "timings", {}), "duration_s": time.monotonic() - started,
                "image_tool_events": getattr(result, "image_tool_events", ()),
                "status": "COMPLETED", "response": payload}, indent=2), encoding="utf-8")
        return payload, result, prompt, time.monotonic() - started

    def _visual_plan(self, image_paths, base_prompt, run_dir):
        if self._remote_context is None:
            raise ExplorationPlanningError("remote visual stage has no current request")
        context = {**self._remote_context, "images": image_manifest(self._remote_images, "current/reference RGB"),
            "approved_skill_names": list(self.skill_names),
            "skill_guidance": self.skill_guidance,
            "skill_scope": "The task is ordered folding, not opening. Use transferable failure detectors and contact lessons only; discard guidance that contradicts the current fold step. Provisional lessons are hypotheses, not measured facts.",
            "locally_executable_reference_ids": (self.last_reference_candidate_report or {}).get("executable_reference_ids"),
            "rejected_references": [{"camera": r["camera"], "reference_id": r["reference_id"]}
                                    for r in self.last_rejected_visual_references]}
        payload, result, prompt, duration = self._ask("visual_planning", context,
            VISUAL_PLAN_JSON_SCHEMA, self._remote_images, run_dir,
            "Select one visible Camera-A Rxxx marker for the exact current task. The current RGB and marker overlay are rotated clockwise90 upright; left/right refer to that displayed image, not anatomy. Flat reference images are topology references only. If fold_state_reference is present, its source/target images are static cross-garment visual examples of the requested state transition. Use them only for semantic fold geometry and the desired target state; never copy their pixels, scale, grasp points, depth, XYZ, or robot coordinates. All executable points must come from the current Camera-A RGB and current Rxxx overlay. Do not choose an already rejected marker. Describe your motion strategy and expected physical evidence. Do not output XYZ or actions.")
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
        context["transport_guidance"] = (
            "Prefer xy_eligible_transport_pixels_upright for transport destinations. These are image-plane "
            "samples which passed local XY limits only, not final height, yaw or IK approval. "
            "Do not assume the entire visible garment is reachable. Choose a semantically suitable inward "
            "destination for FOLD, or outward destination for REPAIR_SLEEVE; do not change the task to fit a point.")
        current_index = next(i for i, p in enumerate(self._remote_images) if p.name.lower() == 'camera_a_rgb_upright.png')
        context['pixel_source_contract'] = (
            f'Current executable RGB is image_{current_index}. For target=pixel, return image_id naming '
            'the exact image/view in which pixel_xy was selected, with its original floating-point coordinates. '
            'The host maps and rounds it; do not mix a view ID with already mapped coordinates. '
            'Only current RGB or verified tool views derived from it can supply transport pixels. '
            'For target=grasp, image_id=null and pixel_xy=null; the selected current Rxxx fixes the grasp.')
        # Kept in every motion request, including repair: semantic_task strips
        # the appended host objective, so that text alone cannot convey limits.
        context['acquisition_probe_lift_contract'] = ACQUISITION_PROBE_LIFT_CONTRACT
        g = GarmentGrounding(session.run_dir / "workspace" / "perception_views")
        ref = g.lookup_reference("A", visual.selected_reference["reference_id"])
        surface = g.sample_local_surface("A", *ref["pixel_xy"], radius_px=3, include_nearest_reference=False)
        bounds = resolve_grasp_height(measurement=surface, table_plane_abc=None, robot_config=session.robot_config)
        context["contact_height_contract"] = {
            "estimated_surface_z_mm": bounds.surface_z_mm,
            "minimum_descent_mm": bounds.minimum_compression_mm,
            "maximum_descent_mm": bounds.maximum_compression_mm,
            "minimum_contact_z_mm": bounds.lower_z_mm,
            "physical_contact": "UNKNOWN", "observed_compression_mm": None,
            "instruction": "Choose contact_descent_mm from evidence and history. No fixed 6 mm descent. Estimated depth and configured sponge are not contact evidence."}
        payload, result, prompt, duration = self._ask("pixel_motion", context, MOTION_SCHEMA,
            self._remote_images, session.run_dir,
            "For FOLD and REPAIR_SLEEVE, the first move after closure must lift vertically at least 30 mm above the grasp. "
            "The host raises shorter positive vertical lifts to 30 mm and revalidates the full trajectory. "
            "Contact and lift photos are captured synchronously before the next robot action, for final evaluation; no model approval is required between actions. "
            "Return the complete proposed move/open_gripper/close_gripper/home sequence. Each move uses target=grasp with pixel_xy=null for the fixed selected marker, or target=pixel with [u,v] in the CURRENT upright RGB for transport destinations. height_above_grasp_mm is a proposed NONNEGATIVE relative lift above your chosen closure height; it is not a measured coordinate. yaw_deg is relative to calibrated Home. All conversions, depth checks and execution checks are local. Approach with clearance, open, descend to target=grasp and height=0, close, lift before lateral transport, lay down and release, retreat and home. Explicitly include every action; the host does not insert missing actions. In ACQUISITION_PROBE mode use only target=grasp: lift, reverse to the same contact, release and home; set requires_lift_checkpoint=true. In FOLD mode actually transport inward; in REPAIR_SLEEVE mode transport outward to unbunch, then release. If fold_state_reference is present, use its target image only as a semantic visual goal for the current step. Select grasp and transport pixels exclusively from the CURRENT Camera-A RGB/Rxxx evidence; never copy reference-image pixels, coordinates, scale, depth, or XYZ. Return contact_descent_mm, your chosen descent below the estimated surface within contact_height_contract bounds. It is a commanded geometric offset, not achieved physical compression. Do not send measured XYZ or code.")
        rgb = next(p for p in self._remote_images if p.name.lower() == "camera_a_rgb_upright.png")
        with Image.open(rgb) as image:
            size = image.size
        compile_started = time.monotonic()
        try:
            canonical_payload, source_trace = resolve_motion_sources(payload, self._remote_images,
                getattr(result, 'image_sources', ()))
            (self._call_diagnostics / 'pixel_source_resolution.json').write_text(
                json.dumps({'remote_motion': payload, 'canonical_motion': canonical_payload,
                            'resolutions': source_trace}, indent=2), encoding='utf-8')
            proposal, verification = compile_pixel_motion(canonical_payload, visual,
                GarmentGrounding(session.run_dir / "workspace" / "perception_views"),
                session.robot_config, size)
            verification['image_source_resolution'] = source_trace
        except Exception as exc:
            if isinstance(exc, WorkspaceTargetError):
                save_workspace_debug(self._call_diagnostics,
                    session.run_dir / "workspace" / "perception_views", session.robot_config, exc.trace)
            self.planner._save_invocation_log(session.run_dir, {
                "stage": "local_pixel_grounding", "remote_motion": payload,
                "diagnostics_directory": str(self._call_diagnostics),
                "error": f"{type(exc).__name__}: {exc}"}, failed=True)
            raise
        finally:
            callback = getattr(self, "_phase_callback", None)
            if callback is not None:
                callback("local_pixel_grounding_and_failure_artifacts", "finished", time.monotonic() - compile_started)
        save_workspace_debug(self._call_diagnostics,
            session.run_dir / "workspace" / "perception_views", session.robot_config,
            verification["workspace_trace"])
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
                 rollout_recording_dir=None, observer_images=(), skill_guidance=None, **kwargs):
        return self._evaluate_remote(before_images, after_images, proposal=proposal,
            run_dir=run_dir, objective=objective, rollout_recording_dir=rollout_recording_dir,
            observer_images=observer_images, acquisition=False, skill_guidance=skill_guidance,
            perception_comparison=kwargs.get('perception_comparison', False),
            rollout_evidence_images=kwargs.get('rollout_evidence_images', ()))

    def plan_height_retry(self, *, context, image_paths, run_dir, output_dir):
        from .grasp_height_retry import HEIGHT_RETRY_INSTRUCTION, HEIGHT_RETRY_SCHEMA
        payload, _, _, _ = self._ask("height_retry", context, HEIGHT_RETRY_SCHEMA,
            image_paths, run_dir, HEIGHT_RETRY_INSTRUCTION)
        return payload

    def update_experience(self, *, context, image_paths, run_dir, output_dir):
        from .fold_experience_learning import EXPERIENCE_INSTRUCTION, EXPERIENCE_UPDATE_SCHEMA
        payload, _, _, _ = self._ask("experience_update", context, EXPERIENCE_UPDATE_SCHEMA,
            image_paths, run_dir, EXPERIENCE_INSTRUCTION)
        return payload

    def evaluate_acquisition_probe(self, before_images, after_images, *, proposal, run_dir,
            rollout_recording_dir=None, rollout_evidence_images=(), observer_images=(), skill_guidance=None, **kwargs):
        return self._evaluate_remote(before_images, after_images, proposal=proposal,
            run_dir=run_dir, rollout_recording_dir=rollout_recording_dir,
            rollout_evidence_images=rollout_evidence_images, observer_images=observer_images,
            acquisition=True, skill_guidance=skill_guidance)

    def _evaluate_remote(self, before_images, after_images, *, proposal, run_dir,
            objective=None, rollout_recording_dir=None, rollout_evidence_images=(),
            observer_images=(), acquisition=False, skill_guidance=None, perception_comparison=False):
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
        snapshot_roles = {
            'camera_a_grasp_before_lift.png': 'Camera A wrist RGB at contact; inspect capture metadata: new synchronous captures are before closure, historical captures may be during closure',
            'camera_a_grasp_after_close.png': 'Camera A wrist RGB after confirmed closure, BEFORE lift',
            'camera_a_grasp_after_lift.png': 'Camera A wrist RGB requested after lift >=30 mm; capture timing comes from metadata; new captures are stationary before lateral motion, historical captures may be asynchronous',
        }
        roles = (["before"] * len(before) + ["after"] * len(after) +
                 ["rollout RGB contact sheet"] * len(video) +
                 [snapshot_roles.get(p.name.lower(),
                     'Sequential acquisition-probe move snapshot (may include reverse descent); use sequence number for chronology'
                     if p.name.lower().startswith('camera_a_lift_checkpoint_') else
                     'observer RGB evidence; use its capture label for chronology')
                  for p in observers])
        context = {"task": semantic_task(objective or "Evaluate the current garment task."),
            "skill_guidance": skill_guidance if skill_guidance is not None else self.skill_guidance,
            "acquisition_only": acquisition,
            "proposed_strategy": semantic_history(proposal.reveal_strategy),
            "expected_observation": semantic_history(proposal.expected_observation),
            "images": [{"image_index": i, "role": role, "name": path.name}
                       for i, (path, role) in enumerate(zip(images, roles))]}
        if perception_comparison:
            context['final_perception_comparison_policy'] = COMPARISON_INSTRUCTION
            def primary_index(paths, offset):
                for name in ('camera_A_rgb_upright.png', 'camera_0_A.png'):
                    for index, path in enumerate(paths):
                        if path.name == name:
                            return offset + index
                raise ExplorationPlanningError('final comparison requires unannotated Camera A perception RGB')
            context['perception_comparison_pair'] = {
                'before_image_index': primary_index(before, 0),
                'after_image_index': primary_index(after, len(before)),
                'role': 'Primary same-perception-position pair; lift stills and video are supplementary.',
            }
        schema = comparison_schema(AUTO_EVALUATION_JSON_SCHEMA) if perception_comparison else copy.deepcopy(AUTO_EVALUATION_JSON_SCHEMA)
        schema["properties"].pop("skill_update", None)
        payload, result, prompt, _ = self._ask("acquisition_evaluation" if acquisition else "evaluation",
            context, schema, images, run_dir,
            "Evaluate actual visible before/after and chronological rollout evidence, never infer success from the proposed strategy. "
            "Use supplied Camera A grasp/lift stills to assess whether fabric was acquired and lifted. "
            "Compare the same-attempt before-lift and after-lift pair: look for fabric deformation, tension, "
            "and motion relative to the static table/background, accounting for camera motion. "
            "The before-lift frame may be during closure. Similar images alone prove neither success nor failure. "
            "Current lift photos are requested after a completed lift of at least 30 mm, captured asynchronously while motion continues; "
            "do not assume they precede transport or show a stationary arm. Historical after-close stills may also be supplied. "
            "Camera A is wrist-mounted and moves with the gripper: image displacement alone is not proof of cloth motion. "
            "Closure confirmation is not proof of grasping cloth. No telemetry or depth is supplied. "
            "Mark acquisition/target UNKNOWN when images do not establish them, including occluded or missing grasp evidence. "
            "Inspect the supplied evidence using view_image first. At most TWO new image edits are allowed for a specific ambiguity; "
            "reuse saved views and finish within EIGHT model turns. Do not re-plan the fold or repeatedly rotate/crop. "
            "For acquisition-only probes, transport status must be UNKNOWN, laydown NOT_REACHED, task_progress NEUTRAL; "
            "do not claim that returning to the initial scene proves successful acquisition. "
            "Provide the full requested evaluation schema, with causal next_experiment suggestions. "
            "Do not produce skill_update. Conditional experience is analyzed separately after host outcome normalization. "
            "Keep workspace and execution safety constraints; UNKNOWN evidence is not empty grasp. "
            "Do not import opening-only success criteria into folding or claim an untested correction succeeded.")
        evaluation = validate_evaluation_payload(payload)
        if acquisition and (evaluation.transport.status != "UNKNOWN" or
                evaluation.laydown.status != "NOT_REACHED" or evaluation.task_progress.status != "NEUTRAL"):
            raise ExplorationPlanningError("acquisition evaluation incorrectly claims transport/fold progress")
        record = ClaudeEvaluationResult(prompt, result.command, result.returncode, result.stdout,
            result.stderr, _now(), evaluation, tuple(map(str, images)), tuple(map(str, refs)), tuple(errors))
        self.last_evaluation_result = record
        self._save_evaluation_log(run_dir, record.as_dict())
        return evaluation
