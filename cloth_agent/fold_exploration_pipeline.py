"""Closed-loop, video-evidence garment-folding exploration.

This pipeline is separate from the older generic opening loop and the bounded
``neat_fold_pipeline``.  It treats folding as five ordered sub-actions:

1. fold the image-left sleeve inward;
2. fold the image-right sleeve inward;
3. fold the first torso side inward;
4. fold the second torso side inward;
5. fold the bottom hem upward.

Each iteration is persisted as a complete experiment record.  Claude proposes
one action, the validated action is executed while a dual-camera recorder is
running, and a second Claude call (the evaluator) receives before/after RGB-D
images plus chronological video contact sheets.  A lightweight supervisor
tracks the five-step state and records when the garment is close to the
Camera-A frame boundary.  Boundary contact is diagnostic only in this
experiment; the pipeline does not run a recovery phase.
"""

from __future__ import annotations

import argparse
import errno
import itertools
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .auto_exploration import (
    ClaudeAutoClient,
    ClaudeEvaluationResult,
    ReferenceReselectionExhaustedError,
    _fold_sleeve_step_from_objective,
    _json_from_claude_text,
    _now as _auto_now,
    prepare_rollout_video_evidence,
)
from .config import ExperimentConfig, RobotConfig, SafetyError
from .evidence_ledger import build_evidence_record, persist_evidence_record
from .experiment import ExperimentValidationError
from .free_exploration import (
    ExplorationPlanningError,
    ExplorationProposal,
    _load_latest_perception,
    _load_or_create_session,
    exploration_source,
    global_perception_image_paths,
    split_global_lift_checkpoint_plan,
    validate_exploration_payload,
)
from .perception import PerceptionConfig, RGBDFrame, capture_two_view_rgbd
from .molmo_keypoint_pipeline import (
    KeypointSpec,
    MolmoKeypointPipelineError,
    run_molmo_keypoint_pipeline,
)
from .rollout_recorder import DualRealSenseRolloutRecorder
from .session import AgentSession
from .skill_lifecycle import RunSkillLedger, SkillStore
from .robot_api import validate_controller_trajectory, move_robot_to_perception_position


FOLD_STEPS: tuple[dict[str, str], ...] = (
    {"id": "left_sleeve", "label": "image-left sleeve inward"},
    {"id": "right_sleeve", "label": "image-right sleeve inward"},
    {"id": "left_side", "label": "first torso side inward"},
    {"id": "right_side", "label": "second torso side inward"},
    {"id": "hem_up", "label": "bottom hem upward"},
)
FOLD_STEP_IDS = tuple(item["id"] for item in FOLD_STEPS)


def _molmo_sleeve_spec(step: str) -> KeypointSpec:
    if step == "left_sleeve":
        return KeypointSpec(
            "fold_image_left_sleeve_region",
            (
                "This is an upright overhead RGB view of one T-shirt. Locate the sleeve "
                "that appears on the LEFT side of the image. Point near the visual center "
                "of that sleeve lobe so a downstream planner can reason over the whole "
                "sleeve region. Do not choose a grasp point and do not prefer an edge, "
                "seam, wrinkle, or height feature. Do not point to the other sleeve, "
                "torso, printed graphic, label, table, or robot."
            ),
            (255, 40, 220),
        )
    if step == "right_sleeve":
        return KeypointSpec(
            "fold_image_right_sleeve_region",
            (
                "This is an upright overhead RGB view of one T-shirt. Locate the sleeve "
                "that appears on the RIGHT side of the image. Point near the visual center "
                "of that sleeve lobe so a downstream planner can reason over the whole "
                "sleeve region. Do not choose a grasp point and do not prefer an edge, "
                "seam, wrinkle, or height feature. Do not point to the other sleeve, "
                "torso, printed graphic, label, table, or robot."
            ),
            (255, 40, 220),
        )
    raise ValueError(f"Molmo sleeve localization does not support step {step!r}")


def _stage_upright_molmo_perception(
    source_dir: Path,
    target_dir: Path,
) -> dict[str, Any]:
    """Rotate Camera-A RGB and calibrated maps clockwise for Molmo only."""

    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()
    raw_image_path = source / "camera_0_A.png"
    xyz_path = source / "camera_A_base_xyz_mm.npy"
    height_path = source / "camera_A_height_above_table_mm.npy"
    missing = [
        str(path)
        for path in (raw_image_path, xyz_path, height_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "cannot stage upright Molmo input; missing " + ", ".join(missing)
        )
    target.mkdir(parents=True, exist_ok=False)
    with Image.open(raw_image_path) as raw_image:
        raw_rgb = raw_image.convert("RGB")
        raw_width, raw_height = raw_rgb.size
        upright = raw_rgb.transpose(Image.Transpose.ROTATE_270)
        upright.save(target / "camera_0_A.png")
    for name in (
        "camera_A_base_xyz_mm.npy",
        "camera_A_height_above_table_mm.npy",
    ):
        array = np.load(source / name, allow_pickle=False)
        np.save(target / name, np.rot90(array, k=3, axes=(0, 1)))
    metadata = {
        "orientation": "clockwise90_upright",
        "raw_image": str(raw_image_path),
        "upright_image": str(target / "camera_0_A.png"),
        "raw_size": [raw_width, raw_height],
        "upright_size": [raw_height, raw_width],
        "upright_to_raw": "raw_x=upright_y; raw_y=raw_height-1-upright_x",
    }
    _write_json(target / "orientation.json", metadata)
    return metadata


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


class FoldDebugLogger:
    """Small run-local logger used by the closed-loop fold pipeline.

    The human-readable log is intentionally written at the same time as the
    console line so a long physical run can be inspected after the fact.  A
    second JSONL stream preserves structured fields (paths, durations, counts,
    and exception text) without putting those details into the console.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.log_path = self.output_dir / "debug.log"
        self.events_path = self.output_dir / "debug_events.jsonl"
        self._lock = threading.Lock()
        self.log_path.write_text("", encoding="utf-8")
        self.events_path.write_text("", encoding="utf-8")

    def log(self, stage: str, message: str, **fields: Any) -> None:
        elapsed = time.monotonic() - self.started
        line = f"[fold-debug +{elapsed:8.1f}s] {stage}: {message}"
        if fields:
            compact = ", ".join(f"{key}={value!r}" for key, value in fields.items())
            line += f" | {compact}"
        event = {
            "timestamp": _now(),
            "elapsed_s": elapsed,
            "stage": str(stage),
            "message": str(message),
            "fields": fields,
        }
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        print(line, flush=True)

    def exception(self, stage: str, exc: BaseException, **fields: Any) -> None:
        self.log(
            stage,
            f"{type(exc).__name__}: {exc}",
            exception_type=type(exc).__name__,
            **fields,
        )


def _compact_history(history: Sequence[Mapping[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Keep planning/supervision context bounded and focused on decisions."""

    def compact_supervisor(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        keys = (
            "status",
            "current_step",
            "completed_steps",
            "next_step",
            "garment_visibility",
            "trajectory_decision",
            "confidence",
            "reason",
            "fallback",
        )
        return {key: value.get(key) for key in keys if key in value}

    def compact_evaluation(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        result: dict[str, Any] = {}
        for key in (
            "target_selection",
            "grasp_acquisition",
            "target_structure_acquired",
            "transport",
            "laydown",
        ):
            stage = value.get(key)
            if isinstance(stage, Mapping):
                result[key] = {
                    field: stage.get(field)
                    for field in ("status", "confidence", "evidence")
                    if field in stage
                }
                evidence = result[key].get("evidence")
                if isinstance(evidence, list):
                    result[key]["evidence"] = [
                        str(item)[:240] for item in evidence[:3]
                    ]
        progress = value.get("task_progress")
        if isinstance(progress, Mapping):
            result["task_progress"] = {
                field: progress.get(field)
                for field in ("status", "confidence", "metrics")
                if field in progress
            }
        if "earliest_failure_stage" in value:
            result["earliest_failure_stage"] = value.get("earliest_failure_stage")
        next_experiment = value.get("next_experiment")
        if isinstance(next_experiment, Mapping):
            result["next_experiment"] = {
                "keep": [str(item)[:240] for item in list(next_experiment.get("keep") or [])[:6]],
                "change": [str(item)[:240] for item in list(next_experiment.get("change") or [])[:6]],
                "reason": str(next_experiment.get("reason", ""))[:600],
            }
        return result

    compact: list[dict[str, Any]] = []
    for row in list(history)[-max(1, int(limit)) :]:
        if not isinstance(row, Mapping):
            continue
        item: dict[str, Any] = {
            "iteration": row.get("iteration"),
            "mode": row.get("mode") or row.get("status"),
            "planned_step": row.get("planned_step"),
            "supervisor_before": compact_supervisor(row.get("supervisor_before")),
            "supervisor_after": compact_supervisor(row.get("supervisor_after")),
            "screen_after": {
                key: row.get("screen_after", {}).get(key)
                for key in ("visibility", "bbox_xyxy", "touching_edges")
                if isinstance(row.get("screen_after"), Mapping)
                and key in row.get("screen_after", {})
            },
            "evaluation": compact_evaluation(row.get("evaluation")),
            "trajectory": {
                "mode": (row.get("trajectory") or {}).get("mode")
                if isinstance(row.get("trajectory"), Mapping)
                else None,
                "actions": len((row.get("trajectory") or {}).get("actions", []))
                if isinstance(row.get("trajectory"), Mapping)
                else 0,
            },
            "grasp_strategy": _grasp_strategy_signature(row),
            "acquisition_learning": (
                dict(row.get("acquisition_learning"))
                if isinstance(row.get("acquisition_learning"), Mapping)
                else None
            ),
        }
        compact.append(item)
    return compact


def _proposal_actions(value: Any) -> list[Mapping[str, Any]]:
    """Extract an action list from either a record, proposal, or raw sequence."""

    if isinstance(value, ExplorationProposal):
        return list(value.actions)
    if isinstance(value, Mapping):
        proposal = value.get("proposal")
        if isinstance(proposal, Mapping) and isinstance(proposal.get("actions"), list):
            return [item for item in proposal["actions"] if isinstance(item, Mapping)]
        trajectory = value.get("trajectory")
        if isinstance(trajectory, Mapping) and isinstance(trajectory.get("actions"), list):
            return [item for item in trajectory["actions"] if isinstance(item, Mapping)]
        actions = value.get("actions")
        if isinstance(actions, list):
            return [item for item in actions if isinstance(item, Mapping)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _grasp_strategy_signature(value: Any) -> dict[str, Any] | None:
    """Summarize contact geometry so repeated physical strategies can be compared."""

    actions = _proposal_actions(value)
    close_index = next(
        (index for index, action in enumerate(actions) if action.get("name") == "close_gripper"),
        None,
    )
    if close_index is None:
        return None
    pre_moves = [
        action.get("args")
        for action in actions[:close_index]
        if action.get("name") == "move" and isinstance(action.get("args"), Mapping)
    ]
    if not pre_moves:
        return None
    grasp = pre_moves[-1]
    entry = pre_moves[-2] if len(pre_moves) >= 2 else None
    post = next(
        (
            action.get("args")
            for action in actions[close_index + 1 :]
            if action.get("name") == "move" and isinstance(action.get("args"), Mapping)
        ),
        None,
    )
    try:
        grasp_xy = [float(grasp["x"]), float(grasp["y"])]
        grasp_z = float(grasp["z"])
        yaw = float(grasp["yaw"])
    except (KeyError, TypeError, ValueError):
        return None
    entry_delta_xy = [0.0, 0.0]
    entry_lateral_mm = 0.0
    if isinstance(entry, Mapping):
        try:
            entry_delta_xy = [
                grasp_xy[0] - float(entry["x"]),
                grasp_xy[1] - float(entry["y"]),
            ]
            entry_lateral_mm = math.hypot(*entry_delta_xy)
        except (KeyError, TypeError, ValueError):
            entry_delta_xy = [0.0, 0.0]
            entry_lateral_mm = 0.0
    first_post_delta_xy = [0.0, 0.0]
    first_lift_mm = None
    if isinstance(post, Mapping):
        try:
            first_post_delta_xy = [
                float(post["x"]) - grasp_xy[0],
                float(post["y"]) - grasp_xy[1],
            ]
            first_lift_mm = float(post["z"]) - grasp_z
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "grasp_xy_mm": grasp_xy,
        "grasp_z_mm": grasp_z,
        "yaw_deg": yaw,
        "entry_delta_xy_mm": entry_delta_xy,
        "entry_lateral_mm": entry_lateral_mm,
        "entry_style": "LATERAL_ENTRY" if entry_lateral_mm >= 3.0 else "NEAR_VERTICAL_ENTRY",
        "first_post_delta_xy_mm": first_post_delta_xy,
        "first_lift_mm": first_lift_mm,
    }


def _fold_acquisition_learning_state(
    history: Sequence[Mapping[str, Any]],
    step: str,
) -> dict[str, Any]:
    """Build causal memory for one fold step without supplying a grasp answer."""

    attempts: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, Mapping) or row.get("planned_step") != step:
            continue
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        acquisition = evaluation.get("grasp_acquisition")
        if not isinstance(acquisition, Mapping):
            continue
        next_experiment = evaluation.get("next_experiment")
        attempts.append(
            {
                "iteration": row.get("iteration"),
                "status": acquisition.get("status"),
                "confidence": acquisition.get("confidence"),
                "evidence": [
                    str(item)[:240]
                    for item in list(acquisition.get("evidence") or [])[:3]
                ],
                "earliest_failure_stage": evaluation.get("earliest_failure_stage"),
                "strategy": _grasp_strategy_signature(row),
                "evaluator_next_experiment": (
                    {
                        "keep": list(next_experiment.get("keep") or [])[:6],
                        "change": list(next_experiment.get("change") or [])[:6],
                        "reason": str(next_experiment.get("reason", ""))[:600],
                    }
                    if isinstance(next_experiment, Mapping)
                    else None
                ),
            }
        )

    consecutive_failures: list[dict[str, Any]] = []
    acquisition_validated = False
    uncertain_since_last_evidence = False
    for attempt in reversed(attempts):
        status = attempt.get("status")
        if status == "SUCCESS":
            acquisition_validated = (
                not consecutive_failures and not uncertain_since_last_evidence
            )
            break
        if status == "FAILURE" and attempt.get("earliest_failure_stage") == "ACQUISITION":
            consecutive_failures.append(attempt)
            continue
        if status == "UNKNOWN":
            uncertain_since_last_evidence = True
            continue
        break
    consecutive_failures.reverse()
    failure_count = len(consecutive_failures)
    failed_signatures = [
        attempt.get("strategy")
        for attempt in consecutive_failures
        if isinstance(attempt.get("strategy"), Mapping)
    ]
    repeated_non_height_family = False
    height_only_retry_pattern = False
    if len(failed_signatures) >= 2:
        first = failed_signatures[0]
        repeated_non_height_family = all(
            math.dist(first["grasp_xy_mm"], item["grasp_xy_mm"]) < 4.0
            and _angle_delta_deg(first["yaw_deg"], item["yaw_deg"]) < 15.0
            and math.dist(
                first["entry_delta_xy_mm"], item["entry_delta_xy_mm"]
            ) < 4.0
            and first.get("entry_style") == item.get("entry_style")
            for item in failed_signatures[1:]
        )
        grasp_z_values = [float(item["grasp_z_mm"]) for item in failed_signatures]
        height_only_retry_pattern = bool(
            repeated_non_height_family
            and max(grasp_z_values) - min(grasp_z_values) >= 0.25
        )
    if acquisition_validated:
        phase = "ACQUISITION_VALIDATED"
        instruction = (
            "A recent short lift visibly acquired cloth. Preserve the supported contact "
            "family unless later transport evidence contradicts it, and continue the named fold."
        )
    elif failure_count >= 2:
        phase = "STRATEGY_DIVERSIFICATION"
        instruction = (
            "Repeated empty acquisitions show that the current contact hypothesis is wrong or "
            "incomplete. Infer a materially different contact hypothesis from RGB and the saved "
            "evidence. The next action must change contact XY, jaw alignment, or the pre-close "
            "entry path; changing only grasp Z is not an admissible new experiment. No privileged "
            "grasp structure is supplied by the host."
        )
    elif failure_count == 1:
        phase = "ACQUISITION_DIAGNOSIS"
        instruction = (
            "One empty acquisition is evidence against the whole contact hypothesis, not only "
            "against its height. Compare contact location, jaw alignment, pre-close entry path, "
            "and Z as competing causes. Plan a small short-lift test whose result distinguishes "
            "those causes; do not assume that lowering Z alone is sufficient."
        )
    else:
        phase = "BASELINE"
        instruction = (
            "No physical acquisition evidence exists for this step. Choose a contact hypothesis "
            "from the current RGB and make one conservative baseline attempt. The host does not "
            "supply a preferred grasp structure."
        )
    return {
        "schema_version": 1,
        "step": step,
        "phase": phase,
        "attempt_count": len(attempts),
        "consecutive_acquisition_failures": failure_count,
        "acquisition_validated": acquisition_validated,
        "use_lift_only_probe": bool(attempts and not acquisition_validated),
        "require_non_height_change": bool(failure_count >= 2 and not acquisition_validated),
        "uncertain_since_last_evidence": uncertain_since_last_evidence,
        "repeated_non_height_contact_family": repeated_non_height_family,
        "height_only_retry_pattern": height_only_retry_pattern,
        "instruction": instruction,
        "recent_attempts": attempts[-4:],
    }


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _validate_acquisition_strategy_change(
    proposal: ExplorationProposal,
    learning: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject repeated failed contact geometry whose only change is height."""

    current = _grasp_strategy_signature(proposal)
    if current is None:
        raise ExplorationPlanningError("cannot derive a grasp strategy signature")
    if bool(learning.get("use_lift_only_probe")):
        lift_mm = current.get("first_lift_mm")
        if not isinstance(lift_mm, (int, float)) or not 15.0 <= float(lift_mm) <= 30.0:
            raise ExplorationPlanningError(
                "acquisition probe requires an observable, reversible first lift of "
                f"15-30 mm; planned first lift={lift_mm!r}"
            )
    if not bool(learning.get("require_non_height_change")):
        return {
            "status": "NOT_REQUIRED",
            "phase": learning.get("phase"),
            "current": current,
            "comparisons": [],
        }
    comparisons: list[dict[str, Any]] = []
    repeated: list[dict[str, Any]] = []
    for attempt in learning.get("recent_attempts", []):
        if not isinstance(attempt, Mapping) or attempt.get("status") != "FAILURE":
            continue
        previous = attempt.get("strategy")
        if not isinstance(previous, Mapping):
            continue
        xy_shift = math.dist(current["grasp_xy_mm"], previous["grasp_xy_mm"])
        yaw_shift = _angle_delta_deg(current["yaw_deg"], previous["yaw_deg"])
        entry_shift = math.dist(
            current["entry_delta_xy_mm"], previous["entry_delta_xy_mm"]
        )
        post_shift = math.dist(
            current["first_post_delta_xy_mm"],
            previous.get("first_post_delta_xy_mm", [0.0, 0.0]),
        )
        changed = {
            "contact_xy": xy_shift >= 4.0,
            "jaw_alignment": yaw_shift >= 15.0,
            "entry_path": (
                entry_shift >= 4.0
                or current.get("entry_style") != previous.get("entry_style")
            ),
            "initial_lift_path": post_shift >= 4.0,
        }
        comparison = {
            "iteration": attempt.get("iteration"),
            "xy_shift_mm": xy_shift,
            "yaw_shift_deg": yaw_shift,
            "entry_vector_shift_mm": entry_shift,
            "initial_lift_xy_shift_mm": post_shift,
            "non_height_dimensions_changed": [
                name for name, value in changed.items() if value
            ],
        }
        comparisons.append(comparison)
        if not any(changed.values()):
            repeated.append(comparison)
    if repeated:
        raise ExplorationPlanningError(
            "repeated acquisition failure requires a materially different contact "
            "hypothesis; this proposal only changes grasp height relative to failed "
            f"iteration(s) {[item.get('iteration') for item in repeated]}. Change contact "
            "XY, jaw yaw, or the pre-close/initial-lift XY path before physical execution."
        )
    return {
        "status": "MATERIALLY_DIFFERENT",
        "phase": learning.get("phase"),
        "current": current,
        "comparisons": comparisons,
    }


def _select_supervisor_images(images: Sequence[Path], max_images: int = 8) -> list[Path]:
    """Keep supervisor evidence compact without hiding the important views.

    The full image bundle remains available to Claude's fold planner and is
    staged for Viser.  The state supervisor only needs a small, labelled set;
    sending every heatmap and duplicate RGB copy makes a simple state check
    unnecessarily slow and increases timeout risk.
    """

    unique: list[Path] = []
    seen: set[Path] = set()
    for raw in images:
        path = Path(raw).resolve()
        if path.is_file() and path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        return []

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        parent = "/".join(part.lower() for part in path.parts)
        if name in {"camera_0_a.png", "camera_1_b.png"}:
            rank = 0
        elif "garment_only" in name or "garment_rgb" in name:
            rank = 1
        elif "height_map_boundary" in name or "depth_heatmap_boundary" in name:
            rank = 2
        elif "height_gradient" in name or "fold_edge" in name:
            rank = 3
        elif "fused" in name:
            rank = 4
        elif "height_map" in name or "depth_heatmap" in name:
            rank = 5
        else:
            rank = 6
        # Prefer the iteration-local raw capture over a duplicate run-wide copy.
        if "before_raw" in parent or "after_raw" in parent:
            rank -= 1
        return rank, str(path)

    return sorted(unique, key=score)[: max(1, int(max_images))]


def _safe_claude(binary: str) -> str:
    resolved = shutil.which(binary) if Path(binary).name == binary else binary
    if resolved is None:
        raise RuntimeError(f"Claude CLI not found: {binary}")
    return str(resolved)


SUPERVISOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["READY", "COMPLETE", "BLOCKED"]},
        "current_step": {"type": "string", "enum": [*FOLD_STEP_IDS, "COMPLETE", "BLOCKED"]},
        "completed_steps": {
            "type": "array",
            "uniqueItems": True,
            "maxItems": 5,
            "items": {"type": "string", "enum": list(FOLD_STEP_IDS)},
        },
        "next_step": {"type": "string", "enum": [*FOLD_STEP_IDS, "COMPLETE", "BLOCKED"]},
        "garment_visibility": {"type": "string", "enum": ["FULL", "PARTIAL", "UNKNOWN"]},
        "trajectory_decision": {
            "type": "string",
            "enum": ["CONTINUE", "RECOVER_PREVIOUS_TRAJECTORY", "PLAN_INWARD_RECOVERY", "STOP"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1},
        },
        "reason": {"type": "string", "minLength": 1},
    },
    "required": [
        "status",
        "current_step",
        "completed_steps",
        "next_step",
        "garment_visibility",
        "trajectory_decision",
        "confidence",
        "evidence",
        "reason",
    ],
}


def validate_supervisor_payload(payload: Any) -> dict[str, Any]:
    """Validate the supervisor contract before it changes robot behavior."""

    if not isinstance(payload, Mapping):
        raise ExplorationPlanningError("fold supervisor response must be a JSON object")
    required = set(SUPERVISOR_SCHEMA["required"])
    if set(payload) != required:
        raise ExplorationPlanningError(
            "fold supervisor fields mismatch: "
            f"missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    result = dict(payload)
    if result["status"] not in {"READY", "COMPLETE", "BLOCKED"}:
        raise ExplorationPlanningError("fold supervisor status is invalid")
    if result["current_step"] not in {*FOLD_STEP_IDS, "COMPLETE", "BLOCKED"}:
        raise ExplorationPlanningError("fold supervisor current_step is invalid")
    if result["next_step"] not in {*FOLD_STEP_IDS, "COMPLETE", "BLOCKED"}:
        raise ExplorationPlanningError("fold supervisor next_step is invalid")
    completed = result["completed_steps"]
    if not isinstance(completed, list) or len(set(completed)) != len(completed):
        raise ExplorationPlanningError("fold supervisor completed_steps must be unique")
    if any(step not in FOLD_STEP_IDS for step in completed):
        raise ExplorationPlanningError("fold supervisor completed_steps contains an unknown step")
    if result["garment_visibility"] not in {"FULL", "PARTIAL", "UNKNOWN"}:
        raise ExplorationPlanningError("fold supervisor garment_visibility is invalid")
    if result["trajectory_decision"] not in {
        "CONTINUE",
        "RECOVER_PREVIOUS_TRAJECTORY",
        "PLAN_INWARD_RECOVERY",
        "STOP",
    }:
        raise ExplorationPlanningError("fold supervisor trajectory_decision is invalid")
    confidence = result["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ExplorationPlanningError("fold supervisor confidence must be numeric")
    if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise ExplorationPlanningError("fold supervisor confidence must be in [0,1]")
    result["confidence"] = float(confidence)
    if not isinstance(result["evidence"], list) or not result["evidence"]:
        raise ExplorationPlanningError("fold supervisor evidence must be non-empty")
    if any(not isinstance(item, str) or not item.strip() for item in result["evidence"]):
        raise ExplorationPlanningError("fold supervisor evidence must contain strings")
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ExplorationPlanningError("fold supervisor reason must be non-empty")
    result["completed_steps"] = list(result["completed_steps"])
    result["evidence"] = [str(item).strip() for item in result["evidence"]]
    result["reason"] = result["reason"].strip()
    return result


def assess_screen_visibility(
    perception: Mapping[str, Any],
    perception_path: Path,
    *,
    margin_px: int = 8,
) -> dict[str, Any]:
    """Detect a garment touching the Camera-A image border.

    This is deliberately a conservative trigger: touching any border means the
    shirt may be clipped.  This result is diagnostic only: the folding
    experiment deliberately does not substitute a recovery action.  It does
    not estimate depth or use the tabletop plane.
    """

    views = perception.get("views", [])
    camera_a = next(
        (view for view in views if isinstance(view, Mapping) and str(view.get("label", "")).upper() == "A"),
        None,
    )
    if not isinstance(camera_a, Mapping):
        return {"visibility": "UNKNOWN", "reason": "Camera A view missing"}
    mask_name = camera_a.get("garment_mask")
    image_name = camera_a.get("image")
    if not mask_name or not image_name:
        return {"visibility": "UNKNOWN", "reason": "Camera A garment mask/image missing"}
    mask_path = (perception_path.parent / str(mask_name)).resolve()
    image_path = (perception_path.parent / str(image_name)).resolve()
    if not mask_path.is_file() or not image_path.is_file():
        return {"visibility": "UNKNOWN", "reason": "Camera A garment mask/image unavailable"}
    try:
        mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    except Exception as exc:
        return {"visibility": "UNKNOWN", "reason": f"mask load failed: {type(exc).__name__}: {exc}"}
    if mask.ndim != 2 or not mask.any():
        return {"visibility": "UNKNOWN", "reason": "empty or malformed garment mask"}
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    margin = max(0, int(margin_px))
    touches = {
        "left": bool(xs.min() <= margin),
        "right": bool(xs.max() >= width - 1 - margin),
        "top": bool(ys.min() <= margin),
        "bottom": bool(ys.max() >= height - 1 - margin),
    }
    partial = any(touches.values())
    return {
        "visibility": "PARTIAL" if partial else "FULL",
        "reason": "garment mask touches Camera A frame border" if partial else "garment mask has an interior margin",
        "image": str(image_path),
        "mask": str(mask_path),
        "image_size": [int(width), int(height)],
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "touching_edges": touches,
        "margin_px": margin,
        "mask_area_px": int(mask.sum()),
    }


class FoldExperienceStore:
    """Append-only experience records plus a compact per-run summary."""

    def __init__(self, run_dir: Path):
        self.root = (Path(run_dir).resolve() / "workspace" / "fold_experience").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "experiences.jsonl"
        self.summary_path = self.root / "experience_summary.json"

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def append(self, experience: Mapping[str, Any]) -> dict[str, Any]:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(experience), ensure_ascii=False, default=str) + "\n")
        rows = self._read()
        completed: list[str] = []
        for row in rows:
            after = row.get("supervisor_after")
            if isinstance(after, Mapping):
                for step in after.get("completed_steps", []):
                    if step in FOLD_STEP_IDS and step not in completed:
                        completed.append(step)
        summary = {
            "schema_version": 1,
            "updated_at": _now(),
            "experience_count": len(rows),
            "completed_steps_in_order": completed,
            "next_step": next((step for step in FOLD_STEP_IDS if step not in completed), "COMPLETE"),
            "status_counts": {
                status: sum(1 for row in rows if row.get("status") == status)
                for status in ("FOLD", "ACQUISITION_PROBE", "RECOVERY", "FAILED")
            },
            "last_experiences": rows[-8:],
        }
        _write_json(self.summary_path, summary)
        return summary

    def history(self, limit: int = 8) -> list[dict[str, Any]]:
        return self._read()[-max(1, int(limit)) :]


def _move_action(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": "move",
        "args": {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "z": float(point["z"]),
            "yaw": float(point.get("yaw", 0.0)),
        },
    }


def build_reverse_trajectory(
    actions: Sequence[Mapping[str, Any]],
    robot_config: RobotConfig,
) -> list[dict[str, Any]]:
    """Build a grasp-at-laydown then reverse-motion recovery trajectory.

    The saved trajectory is not blindly replayed.  The recovery starts at the
    previous laydown point, closes the gripper there, reverses every post-grasp
    waypoint back to the old grasp, releases, retreats, and homes.  Static
    preflight and controller IK remain mandatory before execution.
    """

    actions = [dict(action) for action in actions]
    close_index = next((i for i, action in enumerate(actions) if action.get("name") == "close_gripper"), None)
    if close_index is None:
        raise ExplorationPlanningError("saved trajectory has no close_gripper")
    release_index = next(
        (i for i, action in enumerate(actions[close_index + 1 :], start=close_index + 1) if action.get("name") == "open_gripper"),
        None,
    )
    if release_index is None:
        raise ExplorationPlanningError("saved trajectory has no release open_gripper")
    grasp_move = next((action for action in reversed(actions[:close_index]) if action.get("name") == "move"), None)
    post_moves = [action["args"] for action in actions[close_index + 1 : release_index] if action.get("name") == "move"]
    if grasp_move is None or not post_moves:
        raise ExplorationPlanningError("saved trajectory lacks a grasp and post-grasp transport")
    laydown = post_moves[-1]
    all_z = [float(action["args"]["z"]) for action in actions if action.get("name") == "move"]
    bounds = robot_config.boundaries
    if bounds.z_max is None:
        raise ExplorationPlanningError("recovery requires configured z_max")
    recovery_high = min(
        float(bounds.z_max - robot_config.workspace_margin_mm),
        max(float(laydown["z"]) + 70.0, max(all_z) + 20.0),
    )
    if recovery_high <= float(laydown["z"]):
        raise ExplorationPlanningError("no safe recovery approach height exists")
    yaw = float(laydown.get("yaw", 0.0))
    recovery: list[dict[str, Any]] = [
        _move_action({"x": laydown["x"], "y": laydown["y"], "z": recovery_high, "yaw": yaw}),
        {"name": "open_gripper", "args": {}},
        _move_action(laydown),
        {"name": "close_gripper", "args": {}},
    ]
    for point in [*reversed(post_moves[:-1]), grasp_move["args"]]:
        recovery.append(_move_action(point))
    recovery.extend(
        [
            {"name": "open_gripper", "args": {}},
            _move_action({"x": grasp_move["args"]["x"], "y": grasp_move["args"]["y"], "z": recovery_high, "yaw": float(grasp_move["args"].get("yaw", 0.0))}),
            {"name": "home", "args": {}},
        ]
    )
    return recovery


def _proposal_from_actions(actions: Sequence[Mapping[str, Any]], *, reason: str) -> ExplorationProposal:
    payload = {
        "garment_observation": "Recovery trajectory generated from the previous saved fold trajectory.",
        "reveal_strategy": reason,
        "confidence": 0.5,
        "actions": [dict(action) for action in actions],
        "expected_observation": "The garment returns toward the previously observed camera workspace.",
        "safety_notes": ["This recovery trajectory is derived from a previously validated path and must pass fresh preflight and IK."],
    }
    # Global payload validation requires selected_grasp; regular exploration
    # payload validation is sufficient for a host-generated recovery program.
    return validate_exploration_payload(payload, max_actions=max(32, len(actions)))


def _save_frame_images(frames: Sequence[RGBDFrame], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for index, frame in enumerate(frames):
        label = str(frame.label).upper()
        rgb_path = output_dir / f"camera_{index}_{label}.png"
        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(rgb_path)
        np.save(output_dir / f"camera_{index}_{label}_depth_m.npy", np.asarray(frame.depth_m, dtype=np.float32))
        paths.append(rgb_path.resolve())
    return paths


def _clockwise90_pixel(
    x_px: float,
    y_px: float,
    *,
    raw_height: int,
) -> tuple[float, float]:
    """Map one raw Camera-A pixel into PIL's clockwise-90 display frame."""

    return float(raw_height - 1 - y_px), float(x_px)


def _build_upright_camera_a_planning_images(
    result: Mapping[str, Any],
    result_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Create the canonical RGB-only Camera-A planning view.

    Fold-step words such as ``image-left sleeve`` refer exclusively to this
    clockwise-rotated frame.  Rxxx identities remain unchanged: only their
    display pixels are rotated, while Stage 2 still looks up the original raw
    Camera-A reference and calibrated Base XYZ.
    """

    view = next(
        (
            item
            for item in result.get("views", [])
            if isinstance(item, Mapping)
            and str(item.get("label", "")).upper() == "A"
        ),
        None,
    )
    if view is None:
        raise RuntimeError("Camera A view is unavailable for upright fold planning")
    image_name = view.get("image")
    guide_name = view.get("coordinate_guide")
    if not image_name or not guide_name:
        raise RuntimeError("Camera A RGB or coordinate guide is unavailable")
    image_path = (result_path.parent / str(image_name)).resolve()
    guide_path = (result_path.parent / str(guide_name)).resolve()
    if not image_path.is_file() or not guide_path.is_file():
        raise FileNotFoundError(
            f"upright planning inputs are missing: image={image_path}, guide={guide_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        raw = source.convert("RGB")
    guide = json.loads(guide_path.read_text(encoding="utf-8"))
    if not isinstance(guide, dict):
        raise RuntimeError(f"Camera A coordinate guide is malformed: {guide_path}")
    guide_samples = guide.get("samples")
    if not isinstance(guide_samples, list) or not guide_samples:
        raise RuntimeError(f"Camera A coordinate guide has no Rxxx samples: {guide_path}")
    # Restore the original fold-planning contract: show the uniform calibrated
    # garment grid across the whole shirt.  R9xxx edge points were an
    # experimental narrowing layer and must not leak into this view, including
    # when --reuse-latest-perception loads a guide produced by an older run.
    samples = [
        sample
        for sample in guide_samples
        if not (
            isinstance(sample, Mapping)
            and sample.get("reference_source") == "fold_rgb_boundary_dense"
        )
    ]
    if not samples:
        raise RuntimeError(
            f"Camera A coordinate guide has no uniform Rxxx samples: {guide_path}"
        )
    if len(samples) != len(guide_samples):
        guide["samples"] = samples
        guide.pop("fold_boundary_reference_count", None)
        guide.pop("fold_boundary_reference_semantics", None)
        _write_json(guide_path, guide)
        # Stage 2 reads the workspace copy. Keep its selectable contract
        # identical to the full uniform grid rendered for Stage 1.
        for parent in output_dir.resolve().parents:
            workspace_guide = (
                parent / "workspace" / "perception_views" / guide_path.name
            )
            if workspace_guide.parent.is_dir():
                _write_json(workspace_guide, guide)
                break
    upright = raw.rotate(-90, expand=True)
    rgb_path = output_dir / "camera_A_rgb_upright.png"
    upright.save(rgb_path)

    overlay = upright.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    rendered = 0
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        reference_id = sample.get("reference_id")
        pixel = sample.get("pixel_xy")
        if (
            not isinstance(reference_id, str)
            or not isinstance(pixel, list)
            or len(pixel) != 2
        ):
            continue
        x_px, y_px = _clockwise90_pixel(
            float(pixel[0]),
            float(pixel[1]),
            raw_height=raw.height,
        )
        draw.ellipse(
            (x_px - 5, y_px - 5, x_px + 5, y_px + 5),
            fill=(0, 220, 220),
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            (x_px + 7, y_px - 8),
            reference_id,
            fill=(0, 0, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
        rendered += 1
    if rendered == 0:
        raise RuntimeError("Camera A upright overlay rendered no valid Rxxx references")
    overlay_path = output_dir / "camera_A_rxxx_overlay_upright.png"
    overlay.save(overlay_path)

    _write_json(
        output_dir / "camera_A_upright_mapping.json",
        {
            "rotation": "clockwise90",
            "semantic_frame": (
                "All fold-step image-left/image-right directions refer to the upright image."
            ),
            "raw_image": str(image_path),
            "coordinate_guide": str(guide_path),
            "raw_size_xy": [raw.width, raw.height],
            "upright_size_xy": [upright.width, upright.height],
            "rendered_references": rendered,
            "dense_boundary_references": [],
            "reference_mode": "uniform_full_garment",
            "reference_identity": (
                "Rxxx IDs are unchanged and ground through the raw Camera-A guide."
            ),
        },
    )
    return [rgb_path.resolve(), overlay_path.resolve()]


def _select_fold_planning_images(images: Sequence[Path]) -> list[Path]:
    """Use only the canonical upright RGB/Rxxx pair for semantic planning."""

    by_name = {Path(path).name: Path(path).resolve() for path in images}
    required = (
        "camera_A_rgb_upright.png",
        "camera_A_rxxx_overlay_upright.png",
    )
    if all(name in by_name and by_name[name].is_file() for name in required):
        return [by_name[name] for name in required]
    raise RuntimeError(
        "fold planning requires the canonical upright Camera-A RGB and Rxxx overlay"
    )


def _filter_fold_sleeve_planning_overlay(
    planning_images: Sequence[Path],
    objective: str,
    molmo_hint: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Preserve the full uniform Rxxx grid and add only a Molmo hint."""

    step = _fold_sleeve_step_from_objective(objective)
    if step is None:
        return None
    by_name = {Path(path).name: Path(path).resolve() for path in planning_images}
    rgb_path = by_name.get("camera_A_rgb_upright.png")
    overlay_path = by_name.get("camera_A_rxxx_overlay_upright.png")
    if rgb_path is None or overlay_path is None:
        raise RuntimeError("canonical upright planning images are unavailable")
    mapping_path = rgb_path.parent / "camera_A_upright_mapping.json"
    if not mapping_path.is_file():
        raise RuntimeError("Camera A upright mapping metadata is unavailable")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_path = Path(str(mapping.get("raw_image", ""))).resolve()
    guide_path = Path(str(mapping.get("coordinate_guide", ""))).resolve()
    if not raw_path.is_file() or not guide_path.is_file():
        raise RuntimeError(
            "fold sleeve overlay is missing raw RGB or coordinate guide"
        )
    guide = json.loads(guide_path.read_text(encoding="utf-8"))
    samples = guide.get("samples") if isinstance(guide, dict) else None
    if not isinstance(samples, list):
        raise RuntimeError("Camera A coordinate guide samples are unavailable")
    visible: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        if sample.get("reference_source") == "fold_rgb_boundary_dense":
            continue
        reference_id = sample.get("reference_id")
        pixel_xy = sample.get("pixel_xy")
        if not isinstance(reference_id, str) or not isinstance(pixel_xy, list):
            continue
        visible.append(
            {
                "reference_id": reference_id,
                "pixel_xy": pixel_xy,
            }
        )
    if not visible:
        raise RuntimeError(
            "no uniform Camera-A Rxxx references are available for fold planning"
        )

    molmo_fusion: dict[str, Any] = {
        "status": "MOLMO_NOT_AVAILABLE",
        "narrowed_candidates": False,
        "role": "hint_only_no_candidate_filtering",
    }
    molmo_upright = None
    if (
        isinstance(molmo_hint, Mapping)
        and molmo_hint.get("status") == "MOLMO_POINT_AVAILABLE"
        and isinstance(molmo_hint.get("upright_pixel_xy"), list)
        and len(molmo_hint["upright_pixel_xy"]) == 2
    ):
        molmo_upright = [
            int(molmo_hint["upright_pixel_xy"][0]),
            int(molmo_hint["upright_pixel_xy"][1]),
        ]
        molmo_fusion = {
            "status": "MOLMO_HINT_OVER_FULL_UNIFORM_GRID",
            "narrowed_candidates": False,
            "role": "hint_only_no_candidate_filtering",
        }

    all_overlay_path = overlay_path.with_name("camera_A_rxxx_overlay_upright_all.png")
    if not all_overlay_path.is_file():
        shutil.copy2(overlay_path, all_overlay_path)
    with Image.open(all_overlay_path) as source_overlay:
        annotated = source_overlay.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, annotated.width, 28), fill=(255, 255, 255))
    draw.text(
        (8, 8),
        f"{step.upper()} | CYAN=ALL GARMENT Rxxx | MAGENTA=MOLMO HYPOTHESIS",
        fill=(180, 0, 0),
        font=font,
    )
    if molmo_upright is not None:
        mx, my = molmo_upright
        draw.ellipse(
            (mx - 13, my - 13, mx + 13, my + 13),
            outline=(255, 0, 220),
            width=5,
        )
        draw.line((mx - 17, my, mx + 17, my), fill=(255, 0, 220), width=4)
        draw.line((mx, my - 17, mx, my + 17), fill=(255, 0, 220), width=4)
        draw.text(
            (mx + 16, my - 14),
            "MOLMO",
            fill=(255, 0, 220),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    annotated.save(overlay_path)
    report = {
        "step": step,
        "created_at": _now(),
        "source_overlay": str(all_overlay_path),
        "filtered_overlay": str(overlay_path),
        "reference_mode": "uniform_full_garment",
        "visible_reference_count": len(visible),
        # Keep the historical key for downstream debug consumers. Nothing is
        # filtered: every listed reference remains visible and selectable.
        "accepted": visible,
        "rejected": [],
        "molmo_hint": dict(molmo_hint) if isinstance(molmo_hint, Mapping) else None,
        "molmo_fusion": molmo_fusion,
    }
    _write_json(rgb_path.parent / f"camera_A_{step}_candidate_filter.json", report)
    return report


def _stage_image_artifacts(images: Sequence[Path], output_dir: Path) -> list[Path]:
    """Copy externally stored perception images into the iteration artifact tree.

    ``locate_cloth_center`` historically stores its dense overlays under the
    run-wide perception workspace.  Staging a copy makes every intermediate
    image discoverable by the per-run Viser viewer and keeps the iteration
    self-contained without changing the paths sent to Claude.
    """

    destination = output_dir / "perception_artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    seen: set[Path] = set()
    for index, raw in enumerate(images):
        path = Path(raw).resolve()
        if not path.is_file() or path in seen:
            continue
        seen.add(path)
        try:
            inside = path == output_dir.resolve() or output_dir.resolve() in path.parents
        except OSError:
            inside = False
        if inside:
            continue
        target = destination / f"{index:03d}_{path.name}"
        try:
            shutil.copy2(path, target)
        except OSError:
            continue
        staged.append(target.resolve())
    _write_json(
        output_dir / "perception_artifacts.json",
        {"source_images": [str(Path(path).resolve()) for path in images], "staged_images": [str(path) for path in staged]},
    )
    return staged


class FoldSupervisor:
    """Read-only five-step state and frame-visibility supervisor."""

    def __init__(self, binary: str = "claude", timeout_s: int = 900):
        self.binary = binary
        self.timeout_s = int(timeout_s)

    @staticmethod
    def _write_context_bundle(
        root: Path,
        *,
        images: Sequence[Path],
        video_evidence: Sequence[Path],
        history: Sequence[Mapping[str, Any]],
        screen: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist supervisor inputs locally and return a small read manifest."""

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        context_dir = root / "results" / "fold_supervisor" / f"context_{stamp}"
        context_dir.mkdir(parents=True, exist_ok=False)
        instructions_path = context_dir / "01_instructions.md"
        screen_path = context_dir / "02_screen.json"
        history_path = context_dir / "03_recent_experiences.json"
        evidence_path = context_dir / "04_evidence_manifest.json"
        manifest_path = context_dir / "manifest.json"

        instructions_path.write_text(
            "\n".join(
                [
                    "# Folding supervisor task",
                    "",
                    "Inspect one T-shirt using only the evidence listed in 04_evidence_manifest.json.",
                    "The required order has exactly five sub-actions:",
                    "1. left sleeve inward",
                    "2. right sleeve inward",
                    "3. first torso side inward",
                    "4. second torso side inward",
                    "5. bottom hem upward",
                    "",
                    "Determine which actions are visibly complete in the CURRENT images; never infer completion merely from a requested or executed plan.",
                    "Camera-A border contact is PARTIAL diagnostics only. This pipeline has no recovery phase, so border contact alone must not block folding.",
                    "Keep trajectory_decision=CONTINUE unless folding is genuinely impossible or unsafe from the supplied evidence.",
                    "Do not propose robot coordinates. Return exactly the supplied supervisor JSON schema.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        _write_json(screen_path, dict(screen))
        _write_json(history_path, _compact_history(history, 6))

        def relative(path: Path) -> str:
            return str(path.resolve().relative_to(root))

        _write_json(
            evidence_path,
            {
                "images": [relative(path) for path in images],
                "rollout_video_contact_sheets": [
                    relative(path) for path in video_evidence
                ],
                "instruction": (
                    "Read and visually inspect every listed image. Video contact sheets "
                    "are chronological evidence and must not be replaced by plan text."
                ),
            },
        )
        read_order = [
            instructions_path,
            screen_path,
            history_path,
            evidence_path,
        ]
        manifest = {
            "schema_version": 1,
            "created_at": _now(),
            "stage": "fold_supervisor",
            "read_order": [relative(path) for path in read_order],
            "files": {
                path.name: {
                    "path": relative(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in read_order
            },
        }
        _write_json(manifest_path, manifest)
        return {
            "directory": str(context_dir),
            "manifest": str(manifest_path),
            "manifest_relative": relative(manifest_path),
            "read_order": list(manifest["read_order"]),
            "files": dict(manifest["files"]),
        }

    def inspect(
        self,
        images: Sequence[Path],
        run_dir: Path,
        *,
        history: Sequence[Mapping[str, Any]],
        screen: Mapping[str, Any],
        video_evidence: Sequence[Path] = (),
    ) -> dict[str, Any]:
        safe_images = [Path(path).resolve() for path in images]
        root = Path(run_dir).resolve()
        for path in [*safe_images, *(Path(item).resolve() for item in video_evidence)]:
            if not path.is_file() or (path != root and root not in path.parents):
                raise PermissionError(f"supervisor evidence path is outside run: {path}")
        safe_video = [Path(item).resolve() for item in video_evidence]
        context_bundle = self._write_context_bundle(
            root,
            images=safe_images,
            video_evidence=safe_video,
            history=history,
            screen=screen,
        )
        prompt = (
            "Read the run-local folding-supervisor context manifest at "
            f"{context_bundle['manifest_relative']}. Read every file in its read_order, "
            "then inspect every image listed by the evidence manifest. Use no other "
            "run files. Return exactly the supervisor JSON schema."
        )
        command = [
            _safe_claude(self.binary),
            "--print",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(SUPERVISOR_SCHEMA, separators=(",", ":")),
            "--permission-mode",
            "plan",
            "--allowedTools",
            "Read",
            "--tools",
            "Read",
            "--safe-mode",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--add-dir",
            str(root),
            "--system-prompt",
            "You are a read-only folding-state supervisor. Read only the supplied evidence and return JSON.",
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_dir = root / "results" / "fold_supervisor"
            timeout_dir.mkdir(parents=True, exist_ok=True)
            timeout_path = timeout_dir / f"timeout_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
            _write_json(
                timeout_path,
                {
                    "timeout_s": self.timeout_s,
                    "command": command,
                    "stdout": getattr(exc, "stdout", "") or "",
                    "stderr": getattr(exc, "stderr", "") or "",
                    "created_at": _now(),
                },
            )
            raise TimeoutError(f"fold supervisor timed out after {self.timeout_s}s") from exc
        duration_s = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"fold supervisor exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        result = validate_supervisor_payload(_json_from_claude_text(completed.stdout))
        result["duration_s"] = duration_s
        result["context_bundle"] = context_bundle
        result["command"] = command
        result["raw_stdout"] = completed.stdout
        result["raw_stderr"] = completed.stderr
        return result


class FoldExplorationPipeline:
    """Five-step closed-loop fold experiment with video-backed evaluation."""

    def __init__(
        self,
        session: AgentSession,
        *,
        perception_config: Path,
        claude_binary: str = "claude",
        claude_timeout_s: int = 1800,
        supervisor_timeout_s: int = 900,
        max_iterations: int | None = None,
        real: bool = False,
        confirm_real: bool = False,
        record_video: bool = True,
        recording_native: bool = True,
        recording_codec: str = "mp4v",
        reuse_latest_perception: bool = False,
        screen_margin_px: int = 8,
        max_replans: int = 4,
        max_stage_retries: int = 2,
        retry_backoff_s: float = 5.0,
        molmo_sleeve_grounding: bool = True,
        molmo_confidence_threshold: float = 0.50,
        molmo_timeout_s: int = 900,
        molmo_python: Path | None = None,
        molmo_gpu_max_memory_gib: float = 17.0,
        molmo_load_in_8bit: bool = True,
        viser: bool = False,
        viser_host: str = "127.0.0.1",
        viser_port: int = 8765,
        viser_refresh_s: float = 0.5,
    ):
        self.session = session
        self.project_root = session.project_root
        self.perception_config = Path(perception_config).resolve()
        self.claude_binary = claude_binary
        self.claude_timeout_s = int(claude_timeout_s)
        self.max_iterations = max_iterations
        self.real = bool(real)
        self.confirm_real = bool(confirm_real)
        self.record_video = bool(record_video)
        self.recording_native = bool(recording_native)
        self.recording_codec = recording_codec
        self.reuse_latest_perception = bool(reuse_latest_perception)
        self.screen_margin_px = max(0, int(screen_margin_px))
        self.max_replans = max(0, int(max_replans))
        self.max_stage_retries = max(0, int(max_stage_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.molmo_sleeve_grounding = bool(molmo_sleeve_grounding)
        self.molmo_confidence_threshold = float(molmo_confidence_threshold)
        if not 0.0 <= self.molmo_confidence_threshold <= 1.0:
            raise ValueError("molmo_confidence_threshold must be between 0 and 1")
        self.molmo_timeout_s = max(30, int(molmo_timeout_s))
        self.molmo_python = Path(molmo_python).expanduser().resolve() if molmo_python else None
        self.molmo_gpu_max_memory_gib = float(molmo_gpu_max_memory_gib)
        if self.molmo_gpu_max_memory_gib <= 0:
            raise ValueError("molmo_gpu_max_memory_gib must be positive")
        self.molmo_load_in_8bit = bool(molmo_load_in_8bit)
        self.viser = bool(viser)
        self.viser_host = str(viser_host)
        self.viser_port = int(viser_port)
        self.viser_refresh_s = float(viser_refresh_s)
        if self.viser_host not in {"127.0.0.1", "localhost", "::1"}:
            raise PermissionError("fold exploration Viser must bind to loopback")
        if not 0.1 <= self.viser_refresh_s <= 30.0:
            raise ValueError("viser_refresh_s must be between 0.1 and 30 seconds")
        self.client = ClaudeAutoClient(
            binary=claude_binary,
            timeout_s=self.claude_timeout_s,
            grounding_timeout_s=min(400, max(30, self.claude_timeout_s)),
        )
        self.supervisor = FoldSupervisor(claude_binary, supervisor_timeout_s)
        self.experiences = FoldExperienceStore(session.run_dir)
        self.skill_store = SkillStore(self.project_root / "data" / "skills")
        self.skill_ledger = RunSkillLedger(session.workspace)
        self._debug_logger: FoldDebugLogger | None = None
        self._viser_process: subprocess.Popen[Any] | None = None

    def _debug(self, stage: str, message: str, **fields: Any) -> None:
        logger = self._debug_logger
        if logger is None:
            print(f"[fold-debug] {stage}: {message}", flush=True)
            return
        logger.log(stage, message, **fields)

    def _debug_exception(self, stage: str, exc: BaseException, **fields: Any) -> None:
        logger = self._debug_logger
        if logger is None:
            print(f"[fold-debug] {stage}: {type(exc).__name__}: {exc}", flush=True)
            return
        logger.exception(stage, exc, **fields)

    def _start_viser(self, output: Path) -> None:
        """Start a read-only viewer that follows this run's output directory.

        The viewer is a separate process on purpose: it keeps the browser page
        available after the robot/Claude process exits and cannot issue robot
        commands.  Its stdout/stderr are captured in the run for diagnostics.
        """

        if not self.viser:
            return
        log_path = output / "viser.log"
        command = [
            sys.executable,
            "-m",
            "cloth_agent.fold_exploration_viser",
            str(output),
            "--host",
            self.viser_host,
            "--port",
            str(self.viser_port),
            "--refresh-s",
            str(self.viser_refresh_s),
        ]
        try:
            log = log_path.open("w", encoding="utf-8")
            try:
                self._viser_process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            finally:
                log.close()
        except Exception as exc:
            # Viser is diagnostic only.  A missing dependency or an occupied
            # port must never stop a physical folding run.
            self._debug_exception("viser", exc, nonfatal=True)
            self._viser_process = None
            return
        self._debug(
            "viser",
            f"started read-only viewer at http://{self.viser_host}:{self.viser_port}",
            pid=self._viser_process.pid if self._viser_process is not None else None,
            log=str(log_path),
            source=str(output),
        )

    def _capture(self, config: PerceptionConfig, output_dir: Path, *, reuse: bool) -> tuple[dict[str, Any], Path, list[Path]]:
        started = time.monotonic()
        self._debug("perception", "capture started", reuse=reuse, output_dir=str(output_dir))
        if reuse:
            saved, saved_path = _load_latest_perception(self.session)
            if saved is None or saved_path is None:
                raise RuntimeError("no saved perception available for reuse")
            upright = _build_upright_camera_a_planning_images(
                saved,
                saved_path,
                output_dir,
            )
            images = [*upright, *global_perception_image_paths(saved, saved_path)]
            staged = _stage_image_artifacts(images, output_dir)
            self._debug(
                "perception",
                "reused latest perception",
                result=str(saved_path),
                images=len(images),
                upright_planning_images=[path.name for path in upright],
                staged_images=len(staged),
                duration_s=round(time.monotonic() - started, 3),
            )
            return saved, saved_path, images
        if self.real:
            self._debug("perception", "moving robot to calibrated perception pose")
            move_robot_to_perception_position(self.session.robot_config)
        self._debug("perception", "capturing synchronized Camera A/B RGB-D")
        frames = capture_two_view_rgbd(config)
        self._debug("perception", "running garment localization and depth fusion", frames=len(frames))
        self.session.locate_cloth_center(config, frames=frames)
        saved, saved_path = _load_latest_perception(self.session)
        if saved is None or saved_path is None:
            raise RuntimeError("perception completed without saved result")
        raw = _save_frame_images(frames, output_dir)
        upright = _build_upright_camera_a_planning_images(
            saved,
            saved_path,
            output_dir,
        )
        images = [*upright, *raw, *global_perception_image_paths(saved, saved_path)]
        staged = _stage_image_artifacts(images, output_dir)
        self._debug(
            "perception",
            "capture completed",
            result=str(saved_path),
            raw_images=len(raw),
            images=len(images),
            upright_planning_images=[path.name for path in upright],
            staged_images=len(staged),
            duration_s=round(time.monotonic() - started, 3),
        )
        return saved, saved_path, images

    def _capture_with_retries(
        self,
        config: PerceptionConfig,
        output_dir: Path,
        *,
        reuse: bool,
        stage: str,
    ) -> tuple[dict[str, Any], Path, list[Path]]:
        """Retry camera/perception acquisition without retrying robot motion."""

        errors: list[dict[str, Any]] = []
        attempts = self.max_stage_retries + 1
        for attempt in range(1, attempts + 1):
            attempt_dir = output_dir if attempt == 1 else output_dir.with_name(
                f"{output_dir.name}_retry_{attempt:02d}"
            )
            try:
                return self._capture(config, attempt_dir, reuse=reuse)
            except Exception as exc:
                errors.append(
                    {
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                self._debug_exception(
                    "perception",
                    exc,
                    stage=stage,
                    attempt=attempt,
                    max_attempts=attempts,
                )
                if attempt < attempts:
                    delay = self.retry_backoff_s * attempt
                    if delay:
                        self._debug("perception", "waiting before capture retry", delay_s=delay)
                        time.sleep(delay)
        _write_json(
            output_dir.parent / f"{output_dir.name}_attempts.json",
            {"stage": stage, "attempts": errors},
        )
        raise RuntimeError(
            f"{stage} failed after {attempts} capture attempt(s): "
            + errors[-1]["error"]
        )

    def _source_for(self, proposal: ExplorationProposal, name: str) -> Path:
        source_path = self.session.workspace / name
        source_path.write_text(exploration_source(proposal), encoding="utf-8")
        return source_path

    def _execute(
        self,
        source_path: Path,
        config: PerceptionConfig,
        iteration_dir: Path,
        *,
        label: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.monotonic()
        self._debug("execution", "starting trajectory execution", label=label, source=str(source_path))
        recording_dir = iteration_dir / "rollout_recording"
        recorder: DualRealSenseRolloutRecorder | None = None
        thread: threading.Thread | None = None
        recording_result: dict[str, Any] = {}
        recording_errors: list[str] = []
        if self.real and self.record_video:
            self._debug("recording", "starting dual-camera rollout recorder", directory=str(recording_dir))
            try:
                recorder = DualRealSenseRolloutRecorder(
                    config,
                    recording_dir,
                    record_bag=self.recording_native,
                    record_depth_video=True,
                    record_composite=True,
                    codec=self.recording_codec,
                )
                recorder.start()
            except Exception as exc:
                # Video is evidence, not a robot safety gate.  Camera
                # contention or an encoder failure must not prevent the
                # already validated physical trajectory from running.
                recording_errors.append(f"recorder start: {type(exc).__name__}: {exc}")
                self._debug_exception("recording", exc, nonfatal=True)
                failed_recorder = recorder
                recorder = None
                if failed_recorder is not None:
                    try:
                        failed_recorder.close()
                    except Exception as close_exc:
                        recording_errors.append(
                            f"recorder cleanup: {type(close_exc).__name__}: {close_exc}"
                        )

            def record() -> None:
                try:
                    recording_result["manifest"] = recorder.record()
                except BaseException as exc:
                    recording_errors.append(f"{type(exc).__name__}: {exc}")

            thread = threading.Thread(target=record, daemon=True, name=f"fold-record-{label}")
            thread.start()
            time.sleep(0.25)
        try:
            self._debug("execution", "sending validated trajectory to session runner", real=self.real)
            execution = self.session.run_experiment(
                source_path.name,
                real=self.real,
                confirmed=self.confirm_real,
                notes=f"Closed-loop five-step folding {label}.",
            )
        finally:
            if recorder is not None:
                recorder.request_stop("fold_action_completed")
            if thread is not None:
                thread.join(timeout=300.0)
            if recorder is not None and thread is not None and thread.is_alive():
                recording_errors.append("recording thread did not stop within 300 seconds")
                recorder.close()
                thread.join(timeout=3.0)
        recording = {
            "status": "failed" if recording_errors else ("completed" if recorder is not None else "disabled"),
            "directory": str(recording_dir.resolve()) if recorder is not None else None,
            "manifest": recording_result.get("manifest"),
            "errors": recording_errors,
        }
        self._debug(
            "execution",
            "trajectory execution completed",
            label=label,
            execution_status=execution.get("status") if isinstance(execution, Mapping) else None,
            recording_status=recording["status"],
            recording_errors=len(recording_errors),
            duration_s=round(time.monotonic() - started, 3),
        )
        return execution, recording

    def _locate_sleeve_with_molmo(
        self,
        *,
        step: str,
        iteration: int,
        iteration_dir: Path,
    ) -> dict[str, Any] | None:
        """Return one fallible Molmo sleeve-region hint before Claude selects Rxxx."""

        if not self.molmo_sleeve_grounding or step not in {
            "left_sleeve",
            "right_sleeve",
        }:
            return None
        artifact_dir = iteration_dir / "molmo_sleeve_locator"
        source_perception_dir = (
            self.session.run_dir.resolve() / "workspace" / "perception_views"
        )
        molmo_perception_dir = iteration_dir / "molmo_sleeve_input_upright"
        self._debug(
            "molmo",
            "target-specific sleeve localization started",
            iteration=iteration,
            step=step,
            confidence_threshold=self.molmo_confidence_threshold,
            artifact_dir=str(artifact_dir),
        )
        started = time.monotonic()

        def worker_line(line: str) -> None:
            message = str(line).strip()
            if message:
                self._debug("molmo-worker", message, iteration=iteration, step=step)

        try:
            orientation = _stage_upright_molmo_perception(
                source_perception_dir,
                molmo_perception_dir,
            )
            manifest = run_molmo_keypoint_pipeline(
                project_root=self.project_root,
                perception_dir=molmo_perception_dir,
                artifact_dir=artifact_dir,
                confidence_threshold=self.molmo_confidence_threshold,
                molmo_python=self.molmo_python,
                keypoint_specs=(_molmo_sleeve_spec(step),),
                cameras=("A",),
                query_batch_size=1,
                max_crops=1,
                max_new_tokens=48,
                gpu_max_memory_gib=self.molmo_gpu_max_memory_gib,
                allow_cpu_offload=False,
                load_in_8bit=self.molmo_load_in_8bit,
                direct_keypoints=True,
                timeout_s=self.molmo_timeout_s,
                local_files_only=True,
                install=False,
                worker_line_callback=worker_line,
            )
            references = (
                manifest.get("references", [])
                if isinstance(manifest, Mapping)
                else []
            )
            reference = (
                references[0]
                if isinstance(references, list) and references
                else None
            )
            if not isinstance(reference, Mapping):
                hint = {
                    "status": "NO_VALID_MOLMO_SLEEVE_POINT",
                    "step": step,
                    "manifest_status": manifest.get("status") if isinstance(manifest, Mapping) else None,
                    "confidence_threshold": self.molmo_confidence_threshold,
                    "artifact_dir": str(artifact_dir),
                    "duration_s": time.monotonic() - started,
                }
            else:
                upright_source_pixel = reference.get(
                    "source_pixel_xy"
                ) or reference.get("pixel_xy")
                if (
                    not isinstance(upright_source_pixel, list)
                    or len(upright_source_pixel) != 2
                ):
                    raise MolmoKeypointPipelineError(
                        "accepted Molmo sleeve reference has no source pixel"
                    )
                upright_pixel = [
                    int(upright_source_pixel[0]),
                    int(upright_source_pixel[1]),
                ]
                raw_height = int(orientation["raw_size"][1])
                camera_raw_pixel = [
                    int(upright_pixel[1]),
                    int(raw_height - 1 - upright_pixel[0]),
                ]
                hint = {
                    "status": "MOLMO_POINT_AVAILABLE",
                    "step": step,
                    "camera": "A",
                    "raw_pixel_xy": camera_raw_pixel,
                    "upright_pixel_xy": upright_pixel,
                    "confidence": float(reference.get("confidence", 0.0)),
                    "confidence_threshold": self.molmo_confidence_threshold,
                    "name": reference.get("name"),
                    "base_xyz_mm": reference.get("base_xyz_mm"),
                    "role": "fallible_semantic_sleeve_region_not_grasp_point",
                    "molmo_input_orientation": "clockwise90_upright",
                    "artifact_dir": str(artifact_dir),
                    "accepted_overlay": (
                        manifest.get("views", [{}])[0].get("accepted_overlay")
                        if isinstance(manifest.get("views"), list)
                        and manifest.get("views")
                        else None
                    ),
                    "duration_s": time.monotonic() - started,
                }
            _write_json(iteration_dir / "molmo_sleeve_hint.json", hint)
            self._debug(
                "molmo",
                "target-specific sleeve localization completed",
                iteration=iteration,
                step=step,
                status=hint["status"],
                raw_pixel=hint.get("raw_pixel_xy"),
                upright_pixel=hint.get("upright_pixel_xy"),
                confidence=hint.get("confidence"),
                duration_s=round(time.monotonic() - started, 3),
            )
            return hint
        except Exception as exc:
            hint = {
                "status": "MOLMO_FALLBACK_TO_HOST_RGB",
                "step": step,
                "error": f"{type(exc).__name__}: {exc}",
                "artifact_dir": str(artifact_dir),
                "duration_s": time.monotonic() - started,
            }
            _write_json(iteration_dir / "molmo_sleeve_hint.json", hint)
            self._debug_exception(
                "molmo",
                exc,
                iteration=iteration,
                step=step,
                nonfatal=True,
                fallback="host_rgb_boundary_candidates",
            )
            return hint

    def _plan_recovery(
        self,
        screen: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        images: Sequence[Path],
    ) -> ExplorationProposal:
        objective = (
            "RECOVERY ONLY: the shirt is partially outside the Camera A frame. "
            "Plan one conservative inward grasp-and-transport action that brings the visible garment "
            "back toward the center of the camera/table workspace. Do not fold a new part, do not "
            "make a long pull, and release only after the inward transport."
        )
        objective += "\nDeterministic screen evidence:\n" + json.dumps(dict(screen), ensure_ascii=False, indent=2)
        started = time.monotonic()
        try:
            proposal = self.client.plan(
                list(images),
                self.session,
                objective,
                history=_compact_history(history),
                reference_policy="uniform",
            )
        except BaseException as exc:
            self._debug_exception("planning", exc, mode="RECOVERY", duration_s=round(time.monotonic() - started, 3))
            raise
        self._debug(
            "planning",
            "Claude recovery proposal returned",
            mode="RECOVERY",
            actions=len(proposal.actions),
            duration_s=round(time.monotonic() - started, 3),
        )
        return proposal

    def _plan_fold_with_retries(
        self,
        images: Sequence[Path],
        objective: str,
        history: Sequence[Mapping[str, Any]],
        *,
        feedback: str | None = None,
        iteration: int,
        attempt_kind: str = "fold",
        molmo_hint: Mapping[str, Any] | None = None,
    ) -> ExplorationProposal:
        """Retry a Claude planning invocation before declaring the run failed.

        A planning timeout or transient CLI failure is not a physical failure;
        retry it a bounded number of times.  If all attempts fail we stop before
        preflight/execution because there is no safe robot program to run.
        """

        attempts = self.max_stage_retries + 1
        last_error: Exception | None = None
        planning_images = _select_fold_planning_images(images)
        candidate_filter = _filter_fold_sleeve_planning_overlay(
            planning_images,
            objective,
            molmo_hint=molmo_hint,
        )
        self._debug(
            "planning",
            "using canonical RGB-only Camera-A evidence",
            iteration=iteration,
            images=[path.name for path in planning_images],
            semantic_orientation="clockwise90_upright",
            visible_uniform_reference_count=(
                candidate_filter.get("visible_reference_count")
                if candidate_filter is not None
                else None
            ),
            reference_mode=(
                candidate_filter.get("reference_mode")
                if candidate_filter is not None
                else None
            ),
            molmo_fusion=(
                candidate_filter.get("molmo_fusion")
                if candidate_filter is not None
                else None
            ),
        )
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                proposal = self.client.plan(
                    planning_images,
                    self.session,
                    objective,
                    feedback=feedback,
                    history=_compact_history(history),
                    reference_policy="uniform",
                )
                self._debug(
                    "planning",
                    "Claude proposal returned",
                    iteration=iteration,
                    attempt=attempt,
                    attempt_kind=attempt_kind,
                    actions=len(proposal.actions),
                    timing=getattr(self.client, "last_plan_timing", {}),
                    selected_reference_validation=getattr(
                        self.client,
                        "last_reference_validation",
                        None,
                    ),
                    grounding_verification=getattr(
                        self.client,
                        "last_grounding_verification",
                        None,
                    ),
                    duration_s=round(time.monotonic() - started, 3),
                )
                return proposal
            except Exception as exc:
                last_error = exc
                self._debug_exception(
                    "planning",
                    exc,
                    iteration=iteration,
                    attempt=attempt,
                    attempt_kind=attempt_kind,
                    max_attempts=attempts,
                    duration_s=round(time.monotonic() - started, 3),
                )
                if isinstance(exc, OSError) and exc.errno == errno.E2BIG:
                    raise RuntimeError(
                        "Claude planning command exceeded the OS argument-size limit; "
                        "this deterministic launch failure will not be retried"
                    ) from exc
                if "Argument list too long" in str(exc):
                    raise RuntimeError(
                        "Claude planning command exceeded the OS argument-size limit; "
                        "this deterministic launch failure will not be retried"
                    ) from exc
                if isinstance(exc, ReferenceReselectionExhaustedError):
                    raise RuntimeError(
                        "Claude fold planning has no executable reference under the "
                        "current deterministic gates; this unchanged frame will not be "
                        f"retried: {exc}"
                    ) from exc
                if attempt < attempts:
                    delay = self.retry_backoff_s * attempt
                    if delay:
                        self._debug("planning", "waiting before Claude planning retry", delay_s=delay)
                        time.sleep(delay)
        error_text = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown planning failure"
        raise RuntimeError(
            f"Claude {attempt_kind} planning failed after {attempts} attempt(s): {error_text}"
        ) from last_error

    def _supervisor(self, images: Sequence[Path], screen: Mapping[str, Any], history: Sequence[Mapping[str, Any]], *, video: Sequence[Path] = ()) -> dict[str, Any]:
        started = time.monotonic()
        selected_images = _select_supervisor_images(images)
        self._debug(
            "supervisor",
            "inspection started",
            images=len(images),
            selected_images=len(selected_images),
            selected_image_names=[path.name for path in selected_images],
            video_images=len(video),
            visibility=screen.get("visibility"),
        )
        # Supervisor calls are read-only but can be very expensive.  Keep at
        # most one retry by default so a hung Claude process cannot consume an
        # entire unattended night on a single pre-action state check.
        attempts = min(2, self.max_stage_retries + 1)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                result = self.supervisor.inspect(
                    selected_images,
                    self.session.run_dir,
                    history=history,
                    screen=screen,
                    video_evidence=video,
                )
                break
            except Exception as exc:
                last_error = exc
                self._debug_exception(
                    "supervisor",
                    exc,
                    attempt=attempt,
                    max_attempts=attempts,
                    duration_s=round(time.monotonic() - started, 3),
                )
                if attempt < attempts:
                    delay = self.retry_backoff_s * attempt
                    if delay:
                        self._debug("supervisor", "waiting before supervisor retry", delay_s=delay)
                        time.sleep(delay)
        else:
            result = self._fallback_supervisor(screen, history, last_error)
            self._debug(
                "supervisor",
                "using deterministic fallback after Claude supervisor failures",
                attempts=attempts,
            )
        self._debug(
            "supervisor",
            "inspection completed",
            status=result.get("status"),
            current_step=result.get("current_step"),
            next_step=result.get("next_step"),
            decision=result.get("trajectory_decision"),
            confidence=result.get("confidence"),
            duration_s=round(time.monotonic() - started, 3),
        )
        result.setdefault("fallback", False)
        # Recovery has deliberately been removed from this experiment.  Older
        # Claude supervisor prompts could still return one of the historical
        # recovery decisions when the mask touched an image border; normalize
        # that legacy response so a partial sleeve does not terminate the run.
        if result.get("trajectory_decision") in {
            "RECOVER_PREVIOUS_TRAJECTORY",
            "PLAN_INWARD_RECOVERY",
        }:
            old_decision = result["trajectory_decision"]
            result["trajectory_decision"] = "CONTINUE"
            if result.get("status") == "BLOCKED":
                result["status"] = "READY"
            result["reason"] = (
                f"Recovery decision {old_decision} ignored because recovery is disabled for this run. "
                + str(result.get("reason", ""))
            ).strip()
            self._debug(
                "supervisor",
                "normalized legacy recovery decision; continuing fold planning",
                old_decision=old_decision,
                new_decision="CONTINUE",
            )
        elif (
            result.get("status") == "BLOCKED"
            and result.get("next_step") in FOLD_STEP_IDS
            and result.get("trajectory_decision") == "CONTINUE"
        ):
            # Keep the run moving when Claude emits an internally inconsistent
            # BLOCKED/CONTINUE pair.  The fold planner still has to pass all
            # static and controller checks before anything physical happens.
            result["status"] = "READY"
            self._debug(
                "supervisor",
                "normalized inconsistent BLOCKED/CONTINUE response",
                next_step=result.get("next_step"),
            )
        return result

    def _fallback_supervisor(
        self,
        screen: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        error: BaseException | None,
    ) -> dict[str, Any]:
        """Produce a conservative state when the read-only supervisor is down.

        This fallback never fabricates a completed fold.  It keeps the earliest
        step not confirmed by prior supervisor records, allowing Claude's
        planner and the normal preflight/IK gates to continue the experiment.
        """

        completed: list[str] = []
        for row in history:
            after = row.get("supervisor_after") if isinstance(row, Mapping) else None
            if isinstance(after, Mapping):
                for step in after.get("completed_steps", []):
                    if step in FOLD_STEP_IDS and step not in completed:
                        completed.append(step)
            # When the after-supervisor is unavailable, do not repeat the
            # exact same already-executed physical fold indefinitely.  Advance
            # the sequencing ledger using the recorded planned step, while the
            # fallback flag/confidence=0 makes clear this is not visual proof.
            planned_step = row.get("planned_step") if isinstance(row, Mapping) else None
            if (
                not isinstance(after, Mapping)
                and planned_step in FOLD_STEP_IDS
                and planned_step not in completed
            ):
                completed.append(str(planned_step))
        next_step = next((step for step in FOLD_STEP_IDS if step not in completed), "COMPLETE")
        visibility = str(screen.get("visibility", "UNKNOWN"))
        if visibility not in {"FULL", "PARTIAL", "UNKNOWN"}:
            visibility = "UNKNOWN"
        error_text = f"{type(error).__name__}: {error}" if error else "unknown supervisor failure"
        return {
            "status": "COMPLETE" if next_step == "COMPLETE" else "READY",
            "current_step": next_step,
            "completed_steps": completed,
            "next_step": next_step,
            "garment_visibility": visibility,
            "trajectory_decision": "CONTINUE",
            "confidence": 0.0,
            "evidence": [
                "Claude supervisor was unavailable after retries; no fold completion was inferred.",
                "Previously executed planned steps are used only to prevent an unattended loop from repeating the same manipulation.",
                f"Fallback reason: {error_text}",
            ],
            "reason": "Continue with the earliest previously unconfirmed fold step; supervisor state is unknown.",
            "fallback": True,
        }

    def _evaluate_with_retries(self, *args: Any, iteration: int, **kwargs: Any) -> tuple[Any, ClaudeEvaluationResult | None]:
        attempts = self.max_stage_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                evaluation = self.client.evaluate(*args, **kwargs)
                return evaluation, self.client.last_evaluation_result
            except Exception as exc:
                last_error = exc
                self._debug_exception(
                    "evaluation",
                    exc,
                    iteration=iteration,
                    attempt=attempt,
                    max_attempts=attempts,
                )
                if attempt < attempts:
                    delay = self.retry_backoff_s * attempt
                    if delay:
                        self._debug("evaluation", "waiting before evaluation retry", delay_s=delay)
                        time.sleep(delay)
        error_text = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown evaluation failure"
        self._debug("evaluation", "using UNKNOWN evaluation fallback", iteration=iteration, attempts=attempts)
        return {
            "target_selection": {"status": "UNKNOWN", "confidence": 0.0, "evidence": [error_text]},
            "grasp_acquisition": {"status": "UNKNOWN", "confidence": 0.0, "evidence": [error_text]},
            "target_structure_acquired": {"status": "UNKNOWN", "confidence": 0.0, "evidence": [error_text]},
            "transport": {"status": "UNKNOWN", "confidence": 0.0, "evidence": [error_text]},
            "laydown": {"status": "UNKNOWN", "confidence": 0.0, "evidence": [error_text]},
            "task_progress": {
                "status": "NEUTRAL",
                "confidence": 0.0,
                "metrics": {
                    "visible_area_delta": "UNKNOWN",
                    "overlap_delta": "UNKNOWN",
                    "relief_delta": "UNKNOWN",
                    "boundary_change": "UNKNOWN",
                },
            },
            "earliest_failure_stage": "UNKNOWN",
            "next_experiment": {
                "keep": [],
                "change": ["repeat visual evaluation"],
                "reason": "Claude evaluation failed after retries; retain the trajectory evidence and re-evaluate next iteration.",
            },
            "fallback": True,
        }, None

    def run(self) -> dict[str, Any]:
        if self.real and not self.confirm_real:
            raise PermissionError("physical folding requires --real and --confirm-real")
        if self.max_iterations == 0:
            limit: int | None = None
        else:
            limit = self.max_iterations
        if limit is not None and limit < 1:
            raise ValueError("max_iterations must be positive or zero for continuous mode")
        output = self.session.results / "fold_exploration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output.mkdir(parents=True, exist_ok=False)
        self._debug_logger = FoldDebugLogger(output)
        self._debug(
            "run",
            "created output directory",
            output=str(output),
            run_dir=str(self.session.run_dir),
            real=self.real,
            record_video=self.record_video,
            max_iterations="continuous" if limit is None else limit,
            max_replans=self.max_replans,
            max_stage_retries=self.max_stage_retries,
            retry_backoff_s=self.retry_backoff_s,
            claude_timeout_s=self.claude_timeout_s,
            supervisor_timeout_s=self.supervisor.timeout_s,
        )
        self._start_viser(output)
        try:
            config = PerceptionConfig.load(self.project_root, self.perception_config)
        except BaseException as exc:
            self._debug_exception("run", exc, stage_detail="loading perception configuration")
            raise
        self._debug("run", "loaded perception configuration", path=str(self.perception_config))
        history: list[dict[str, Any]] = self.experiences.history()
        summary: dict[str, Any] = {
            "schema_version": 1,
            "created_at": _now(),
            "status": "RUNNING",
            "objective": "Fold a T-shirt in five ordered steps with video-backed closed-loop supervision.",
            "fold_steps": list(FOLD_STEPS),
            "run_dir": str(self.session.run_dir),
            "output_dir": str(output),
            "physical_execution": self.real,
            "video_recording": self.record_video,
            "timeouts": {
                "claude_s": self.claude_timeout_s,
                "supervisor_s": self.supervisor.timeout_s,
                "recording_join_s": 300,
            },
            "retry_policy": {
                "max_stage_retries": self.max_stage_retries,
                "retry_backoff_s": self.retry_backoff_s,
                "supervisor_fallback": True,
                "evaluation_fallback": True,
            },
            "molmo_sleeve_grounding": {
                "enabled": self.molmo_sleeve_grounding,
                "role": "fallible_semantic_sleeve_region_before_Claude_Rxxx_selection",
                "camera": "A",
                "confidence_threshold": self.molmo_confidence_threshold,
                "timeout_s": self.molmo_timeout_s,
                "gpu_max_memory_gib": self.molmo_gpu_max_memory_gib,
                "cpu_offload": False,
                "load_in_8bit": self.molmo_load_in_8bit,
                "fallback": "host_validated_outer_sleeve_region_candidates",
            },
            "debug": {
                "log": str((output / "debug.log").resolve()),
                "events": str((output / "debug_events.jsonl").resolve()),
            },
            "viser": {
                "enabled": self.viser,
                "host": self.viser_host if self.viser else None,
                "port": self.viser_port if self.viser else None,
                "url": f"http://{self.viser_host}:{self.viser_port}" if self.viser else None,
                "log": str((output / "viser.log").resolve()) if self.viser else None,
                "read_only": True,
            },
            "iterations": [],
        }
        _write_json(output / "summary.json", summary)
        iteration = 0
        try:
            while limit is None or iteration < limit:
                iteration += 1
                iteration_dir = output / f"iteration_{iteration:03d}"
                iteration_dir.mkdir(parents=True, exist_ok=False)
                iteration_started = time.monotonic()
                self._debug("iteration", f"starting iteration {iteration}", iteration=iteration)
                before, before_path, before_images = self._capture_with_retries(
                    config,
                    iteration_dir / "before_raw",
                    reuse=self.reuse_latest_perception and iteration == 1,
                    stage="before perception",
                )
                screen_before = assess_screen_visibility(before, before_path, margin_px=self.screen_margin_px)
                self._debug(
                    "screen",
                    "before visibility assessed",
                    iteration=iteration,
                    visibility=screen_before.get("visibility"),
                    bbox=screen_before.get("bbox_xyxy"),
                    touching_edges=screen_before.get("touching_edges"),
                )
                supervisor_before = self._supervisor(before_images, screen_before, history)
                _write_json(iteration_dir / "supervisor_before.json", supervisor_before)
                self._debug(
                    "supervisor",
                    "saved before decision",
                    iteration=iteration,
                    path=str(iteration_dir / "supervisor_before.json"),
                )
                if supervisor_before.get("fallback"):
                    self._debug(
                        "supervisor",
                        "before decision came from fallback; continuing with conservative earliest step",
                        iteration=iteration,
                    )
                if supervisor_before["status"] == "COMPLETE" or supervisor_before["next_step"] == "COMPLETE":
                    summary["status"] = "COMPLETE"
                    summary["completed_at"] = _now()
                    _write_json(output / "summary.json", summary)
                    return summary
                if supervisor_before["status"] == "BLOCKED" or supervisor_before["next_step"] == "BLOCKED":
                    summary["status"] = "BLOCKED"
                    summary["blocked_reason"] = supervisor_before["reason"]
                    summary["completed_at"] = _now()
                    _write_json(output / "summary.json", summary)
                    return summary
                # This experiment intentionally has no recovery branch.  The
                # screen result is retained in the record/debug stream, while
                # folding proceeds from the supervisor's next ordered step.
                mode = "FOLD"
                proposal: ExplorationProposal
                next_step = supervisor_before["next_step"]
                acquisition_learning = _fold_acquisition_learning_state(
                    history,
                    next_step,
                )
                _write_json(
                    iteration_dir / "acquisition_learning_before.json",
                    acquisition_learning,
                )
                self._debug(
                    "learning",
                    "built acquisition learning state",
                    iteration=iteration,
                    step=next_step,
                    phase=acquisition_learning.get("phase"),
                    consecutive_failures=acquisition_learning.get(
                        "consecutive_acquisition_failures"
                    ),
                    lift_only_probe=acquisition_learning.get("use_lift_only_probe"),
                    require_non_height_change=acquisition_learning.get(
                        "require_non_height_change"
                    ),
                    height_only_retry_pattern=acquisition_learning.get(
                        "height_only_retry_pattern"
                    ),
                )
                molmo_hint = self._locate_sleeve_with_molmo(
                    step=next_step,
                    iteration=iteration,
                    iteration_dir=iteration_dir,
                )
                step_label = next((item["label"] for item in FOLD_STEPS if item["id"] == next_step), "the next incomplete fold step")
                objective = (
                    "Fold this shirt using exactly five ordered steps: left sleeve inward, right sleeve inward, "
                    "first torso side inward, second torso side inward, bottom hem upward. "
                    f"The supervisor says the next incomplete step is {next_step} ({step_label}). "
                    "Plan exactly one action for that earliest incomplete step; do not skip ahead. "
                    "Use current RGB as the primary evidence and the calibrated references only for grounding. "
                    "The host does not reveal a privileged grasp structure; infer the contact hypothesis from "
                    "the current RGB and saved physical outcomes."
                )
                objective += "\nSupervisor state:\n" + json.dumps(supervisor_before, ensure_ascii=False, indent=2)
                objective += (
                    "\nAcquisition learning state (physical evidence, not a grasp answer):\n"
                    + json.dumps(acquisition_learning, ensure_ascii=False, indent=2)
                )
                if isinstance(molmo_hint, Mapping):
                    objective += (
                        "\nMolmo sleeve-region hypothesis (fallible topology guide, "
                        "never a direct grasp point):\n"
                        + json.dumps(molmo_hint, ensure_ascii=False, indent=2)
                    )
                self._debug("planning", "asking Claude for fold proposal", iteration=iteration, next_step=next_step)
                proposal = self._plan_fold_with_retries(
                    before_images,
                    objective,
                    history,
                    iteration=iteration,
                    molmo_hint=molmo_hint,
                )
                planning_attempts: list[dict[str, Any]] = []
                source_path = self.session.workspace / f"_fold_experiment_{iteration:03d}.py"
                preflight = None
                controller = None
                acquisition_strategy_validation: dict[str, Any] | None = None
                acquisition_probe_plan: dict[str, Any] | None = None
                # Deterministic failures (bad schema, grounding, workspace,
                # preflight, IK) are fed back to Claude immediately.  No
                # physical command is sent until one attempt passes all gates.
                plan_feedback: str | None = None
                for plan_attempt in range(1, self.max_replans + 2):
                    attempt_started = time.monotonic()
                    self._debug(
                        "planning",
                        "validating Claude proposal",
                        iteration=iteration,
                        attempt=plan_attempt,
                        mode=mode,
                    )
                    try:
                        if plan_attempt > 1:
                            self._debug("planning", "requesting Claude replan with validation feedback", iteration=iteration, attempt=plan_attempt)
                            proposal = self._plan_fold_with_retries(
                                before_images,
                                objective,
                                history,
                                feedback=plan_feedback,
                                iteration=iteration,
                                attempt_kind=f"fold-replan-{plan_attempt}",
                                molmo_hint=molmo_hint,
                            )
                        acquisition_strategy_validation = (
                            _validate_acquisition_strategy_change(
                                proposal,
                                acquisition_learning,
                            )
                        )
                        execution_proposal = proposal
                        acquisition_probe_plan = None
                        if bool(acquisition_learning.get("use_lift_only_probe")):
                            lift_plan = split_global_lift_checkpoint_plan(
                                proposal,
                                checkpoint_count=2,
                            )
                            acquisition_probe_plan = lift_plan.as_dict()
                            execution_proposal = replace(
                                proposal,
                                actions=lift_plan.actions,
                                expected_observation=(
                                    "Acquisition-only experiment: the gripper should visibly "
                                    "support cloth during the two short lift checkpoints, then "
                                    "reverse the lift, release at the original contact, and leave "
                                    "the garment state recoverable. No fold transport is executed."
                                ),
                                requires_lift_checkpoint=False,
                            )
                        source_path.write_text(
                            exploration_source(execution_proposal),
                            encoding="utf-8",
                        )
                        candidate_preflight = self.session.runner.preflight(source_path.name)
                        if candidate_preflight.error:
                            raise ExperimentValidationError(candidate_preflight.error)
                        candidate_controller = validate_controller_trajectory(
                            self.session.robot_config, candidate_preflight.actions
                        )
                        preflight, controller = candidate_preflight, candidate_controller
                        proposal = execution_proposal
                        mode = (
                            "ACQUISITION_PROBE"
                            if acquisition_probe_plan is not None
                            else "FOLD"
                        )
                        planning_attempts.append(
                            {
                                "attempt": plan_attempt,
                                "status": "ACCEPTED",
                                "mode": mode,
                                "acquisition_strategy_validation": (
                                    acquisition_strategy_validation
                                ),
                            }
                        )
                        self._debug(
                            "planning",
                            "proposal accepted after preflight and controller IK",
                            iteration=iteration,
                            attempt=plan_attempt,
                            actions=len(candidate_preflight.actions),
                            mode=mode,
                            duration_s=round(time.monotonic() - attempt_started, 3),
                        )
                        break
                    except Exception as exc:
                        attempt = {
                            "attempt": plan_attempt,
                            "status": "REJECTED_BEFORE_EXECUTION",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        planning_attempts.append(attempt)
                        self._debug_exception(
                            "planning",
                            exc,
                            iteration=iteration,
                            attempt=plan_attempt,
                            duration_s=round(time.monotonic() - attempt_started, 3),
                        )
                        if plan_attempt >= self.max_replans + 1:
                            raise
                        plan_feedback = (
                            f"{type(exc).__name__}: {exc}\n"
                            "No robot command was sent. Correct this deterministic failure "
                            "and return a materially different, controller-valid fold proposal."
                        )
                if preflight is None or controller is None:
                    raise RuntimeError("fold planning ended without a validated proposal")
                trajectory = {
                    "mode": mode,
                    "actions": preflight.actions,
                    "source": str(source_path),
                    "created_at": _now(),
                }
                _write_json(iteration_dir / "trajectory.json", trajectory)
                planning_diagnostics: dict[str, Any] = {
                    "mode": mode,
                    "attempts": planning_attempts,
                    "client_timing": getattr(self.client, "last_plan_timing", {}),
                    "rejected_visual_references": getattr(self.client, "last_rejected_visual_references", []),
                    "selected_reference_validation": getattr(
                        self.client,
                        "last_reference_validation",
                        None,
                    ),
                    "molmo_sleeve_hint": molmo_hint,
                    "acquisition_learning": acquisition_learning,
                    "acquisition_strategy_validation": (
                        acquisition_strategy_validation
                    ),
                    "acquisition_probe_plan": acquisition_probe_plan,
                }
                visual_result = getattr(self.client, "last_visual_plan_result", None)
                plan_result = getattr(self.client, "last_plan_result", None)
                if visual_result is not None and hasattr(visual_result, "as_dict"):
                    planning_diagnostics["visual_plan_result"] = visual_result.as_dict()
                if plan_result is not None and hasattr(plan_result, "as_dict"):
                    planning_diagnostics["plan_result"] = plan_result.as_dict()
                _write_json(iteration_dir / "planning_diagnostics.json", planning_diagnostics)
                self._debug(
                    "trajectory",
                    "saved validated trajectory",
                    iteration=iteration,
                    mode=mode,
                    actions=len(preflight.actions),
                    path=str(iteration_dir / "trajectory.json"),
                )
                execution, recording = self._execute(
                    source_path,
                    config,
                    iteration_dir,
                    label=f"iteration_{iteration:03d}_{mode.lower()}",
                )
                _write_json(iteration_dir / "execution.json", execution)
                _write_json(iteration_dir / "recording.json", recording)
                after, after_path, after_images = self._capture_with_retries(
                    config,
                    iteration_dir / "after_raw",
                    reuse=False,
                    stage="after perception",
                )
                screen_after = assess_screen_visibility(after, after_path, margin_px=self.screen_margin_px)
                self._debug(
                    "screen",
                    "after visibility assessed",
                    iteration=iteration,
                    visibility=screen_after.get("visibility"),
                    bbox=screen_after.get("bbox_xyxy"),
                    touching_edges=screen_after.get("touching_edges"),
                )
                video_images: list[Path] = []
                video_refs: list[Path] = []
                video_errors: list[str] = []
                if recording.get("status") == "completed" and recording.get("directory"):
                    self._debug("recording", "building rollout video contact sheets", iteration=iteration)
                    try:
                        video_images, video_refs, video_errors = prepare_rollout_video_evidence(Path(recording["directory"]))
                        self._debug(
                            "recording",
                            "rollout video evidence ready",
                            iteration=iteration,
                            contact_sheets=len(video_images),
                            references=len(video_refs),
                            errors=len(video_errors),
                        )
                    except Exception as exc:
                        video_errors.append(f"{type(exc).__name__}: {exc}")
                        self._debug_exception("recording", exc, iteration=iteration, operation="contact_sheet", nonfatal=True)
                self._debug("evaluation", "asking Claude to judge before/after and video", iteration=iteration)
                evaluation_objective = (
                    "This iteration intentionally tested grasp acquisition only with two "
                    "short lift checkpoints followed by the reverse path and release at the "
                    "original contact. Judge whether cloth followed the gripper from the "
                    "chronological video. Do not expect or credit fold transport or a changed "
                    "after image; task progress may remain NEUTRAL even when acquisition succeeds."
                    if mode == "ACQUISITION_PROBE"
                    else "Fold the shirt into the five ordered steps and a neat compact stack."
                )
                evaluation, evaluation_result = self._evaluate_with_retries(
                    list(before_images),
                    list(after_images),
                    iteration=iteration,
                    proposal=proposal,
                    objective=evaluation_objective,
                    run_dir=self.session.run_dir,
                    rollout_recording_dir=(Path(recording["directory"]) if recording.get("status") == "completed" else None),
                    skill_guidance=self.skill_store.prompt(),
                )
                self._debug(
                    "evaluation",
                    "Claude evaluation returned",
                    iteration=iteration,
                    task_progress=(evaluation.as_dict().get("task_progress") if hasattr(evaluation, "as_dict") else evaluation.get("task_progress") if isinstance(evaluation, Mapping) else None),
                )
                supervisor_after = self._supervisor(
                    after_images,
                    screen_after,
                    [
                        *history,
                        {
                            "iteration": iteration,
                            "mode": mode,
                            "planned_step": next_step,
                            "evaluation": evaluation.as_dict() if hasattr(evaluation, "as_dict") else evaluation,
                        },
                    ],
                    video=video_images,
                )
                _write_json(iteration_dir / "evaluation.json", evaluation.as_dict() if hasattr(evaluation, "as_dict") else evaluation)
                if evaluation_result is not None:
                    _write_json(iteration_dir / "claude_evaluation_result.json", evaluation_result.as_dict())
                _write_json(iteration_dir / "supervisor_after.json", supervisor_after)
                self._debug(
                    "supervisor",
                    "saved after decision",
                    iteration=iteration,
                    path=str(iteration_dir / "supervisor_after.json"),
                )
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "planned_step": next_step,
                    "status": mode,
                    "proposal": proposal.as_dict(),
                    "trajectory": trajectory,
                    "preflight": asdict(preflight),
                    "controller_ik": asdict(controller),
                    "planning_attempts": planning_attempts,
                    "execution": execution,
                    "recording": recording,
                    "before_images": [str(path) for path in before_images],
                    "after_images": [str(path) for path in after_images],
                    "video_evidence": [str(path) for path in video_images],
                    "video_references": [str(path) for path in video_refs],
                    "video_errors": video_errors,
                    "screen_before": screen_before,
                    "screen_after": screen_after,
                    "supervisor_before": supervisor_before,
                    "molmo_sleeve_hint": molmo_hint,
                    "acquisition_learning": acquisition_learning,
                    "acquisition_strategy_validation": (
                        acquisition_strategy_validation
                    ),
                    "acquisition_probe_plan": acquisition_probe_plan,
                    "planning_diagnostics": planning_diagnostics,
                    "supervisor_after": supervisor_after,
                    "evaluation": evaluation.as_dict() if hasattr(evaluation, "as_dict") else evaluation,
                    "evaluation_raw": evaluation_result.as_dict() if evaluation_result is not None else None,
                    "completed_at": _now(),
                }
                _write_json(iteration_dir / "record.json", record)
                try:
                    evidence = build_evidence_record(record, iteration=iteration, run_dir=self.session.run_dir)
                except Exception as exc:
                    # Evidence indexing must not discard a completed physical
                    # iteration.  Preserve the error beside the iteration and
                    # continue with the raw record/experience summary.
                    self._debug_exception("evidence", exc, iteration=iteration, nonfatal=True)
                    evidence = {
                        "schema_version": 1,
                        "iteration": iteration,
                        "status": "UNAVAILABLE",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    _write_json(iteration_dir / "evidence_error.json", evidence)
                record["evidence"] = evidence
                try:
                    record["evidence_artifacts"] = persist_evidence_record(
                        self.session.run_dir,
                        evidence,
                        iteration_dir=iteration_dir,
                    )
                except Exception as exc:
                    self._debug_exception("evidence", exc, iteration=iteration, operation="persist", nonfatal=True)
                    record["evidence_artifacts"] = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
                try:
                    self.skill_ledger.append_experience(
                        {
                            "created_at": _now(),
                            "iteration": iteration,
                            "mode": mode,
                            "supervisor_before": supervisor_before,
                            "supervisor_after": supervisor_after,
                            "evaluation": record["evaluation"],
                            "trajectory": trajectory,
                            "video": recording,
                            "evidence": evidence,
                        }
                    )
                except Exception as exc:
                    self._debug_exception("skills", exc, iteration=iteration, nonfatal=True)
                    _write_json(iteration_dir / "skill_append_error.json", {"error": f"{type(exc).__name__}: {exc}"})
                experience_summary = self.experiences.append(record)
                record["experience_summary"] = experience_summary
                _write_json(iteration_dir / "record.json", record)
                self._debug(
                    "experience",
                    "persisted iteration experience",
                    iteration=iteration,
                    summary_path=str(self.experiences.summary_path),
                    experience_count=experience_summary.get("experience_count"),
                    next_step=experience_summary.get("next_step"),
                )
                history.append(record)
                summary["iterations"].append(
                    {
                        "iteration": iteration,
                        "mode": mode,
                        "next_step": supervisor_after["next_step"],
                        "visibility": supervisor_after["garment_visibility"],
                        "evaluation_status": (record["evaluation"].get("task_progress", {}) or {}).get("status") if isinstance(record["evaluation"], Mapping) else None,
                    }
                )
                _write_json(output / "summary.json", summary)
                self._debug(
                    "iteration",
                    "iteration completed",
                    iteration=iteration,
                    mode=mode,
                    next_step=supervisor_after.get("next_step"),
                    duration_s=round(time.monotonic() - iteration_started, 3),
                )
                if supervisor_after["status"] == "COMPLETE" or supervisor_after["next_step"] == "COMPLETE":
                    summary["status"] = "COMPLETE"
                    summary["completed_at"] = _now()
                    _write_json(output / "summary.json", summary)
                    return summary
                if supervisor_after["trajectory_decision"] == "STOP":
                    summary["status"] = "SUPERVISOR_STOPPED"
                    summary["stop_reason"] = supervisor_after["reason"]
                    summary["completed_at"] = _now()
                    _write_json(output / "summary.json", summary)
                    return summary
                self.reuse_latest_perception = False
                if source_path.is_file():
                    try:
                        source_path.unlink()
                    except OSError as exc:
                        self._debug_exception("cleanup", exc, path=str(source_path), nonfatal=True)
            summary["status"] = "MAX_ITERATIONS_REACHED"
            summary["completed_at"] = _now()
            _write_json(output / "summary.json", summary)
            self._debug("run", "reached configured iteration limit", status=summary["status"], iterations=iteration)
            return summary
        except BaseException as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["failed_at"] = _now()
            _write_json(output / "summary.json", summary)
            self._debug_exception("run", exc, iteration=iteration, summary_path=str(output / "summary.json"))
            raise
        finally:
            try:
                synthesis = self.skill_ledger.finalize(self.skill_store)
                _write_json(output / "skill_synthesis.json", synthesis)
            except Exception as exc:
                _write_json(output / "skill_synthesis_error.json", {"error": f"{type(exc).__name__}: {exc}"})
                self._debug_exception("skills", exc)
            self._debug("run", "pipeline finished", status=summary.get("status"), iterations=iteration)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--run-id")
    parser.add_argument("--robot-config", type=Path, default=Path("config/robot.example.json"))
    parser.add_argument("--perception-config", type=Path, default=Path("config/perception.free_exploration.json"))
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=1800)
    parser.add_argument("--supervisor-timeout-s", type=int, default=900)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means continuous until supervisor COMPLETE or a hard failure")
    parser.add_argument("--max-replans", type=int, default=4)
    parser.add_argument("--max-stage-retries", type=int, default=2, help="extra retries for capture, supervisor, and evaluation failures")
    parser.add_argument("--retry-backoff-s", type=float, default=5.0)
    parser.add_argument(
        "--no-molmo-sleeve-grounding",
        action="store_true",
        help="disable the target-specific Molmo sleeve-region hint and use host RGB candidates only",
    )
    parser.add_argument("--molmo-confidence-threshold", type=float, default=0.50)
    parser.add_argument("--molmo-timeout-s", type=int, default=900)
    parser.add_argument("--molmo-python", type=Path)
    parser.add_argument("--molmo-gpu-max-memory-gib", type=float, default=17.0)
    parser.add_argument(
        "--molmo-full-precision",
        action="store_true",
        help="disable the default bitsandbytes 8-bit Molmo loading",
    )
    parser.add_argument("--reuse-latest-perception", action="store_true")
    parser.add_argument("--screen-margin-px", type=int, default=8)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--recording-no-native", action="store_true")
    parser.add_argument("--recording-codec", default="mp4v")
    parser.add_argument("--viser", action="store_true", help="start a read-only Viser viewer for every run artifact")
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8765)
    parser.add_argument("--viser-refresh-s", type=float, default=0.5)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--confirm-real", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    perception = args.perception_config if args.perception_config.is_absolute() else root / args.perception_config
    robot_path = args.robot_config if args.robot_config.is_absolute() else root / args.robot_config
    session = _load_or_create_session(
        root,
        args.run_dir.resolve() if args.run_dir else None,
        args.run_id,
        robot_path.resolve(),
    )
    summary = FoldExplorationPipeline(
        session,
        perception_config=perception.resolve(),
        claude_binary=args.claude_binary,
        claude_timeout_s=args.claude_timeout_s,
        supervisor_timeout_s=args.supervisor_timeout_s,
        max_iterations=args.max_iterations,
        real=args.real,
        confirm_real=args.confirm_real,
        record_video=not args.no_video,
        recording_native=not args.recording_no_native,
        recording_codec=args.recording_codec,
        reuse_latest_perception=args.reuse_latest_perception,
        screen_margin_px=args.screen_margin_px,
        max_replans=args.max_replans,
        max_stage_retries=args.max_stage_retries,
        retry_backoff_s=args.retry_backoff_s,
        molmo_sleeve_grounding=not args.no_molmo_sleeve_grounding,
        molmo_confidence_threshold=args.molmo_confidence_threshold,
        molmo_timeout_s=args.molmo_timeout_s,
        molmo_python=args.molmo_python,
        molmo_gpu_max_memory_gib=args.molmo_gpu_max_memory_gib,
        molmo_load_in_8bit=not args.molmo_full_precision,
        viser=args.viser,
        viser_host=args.viser_host,
        viser_port=args.viser_port,
        viser_refresh_s=args.viser_refresh_s,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"COMPLETE", "MAX_ITERATIONS_REACHED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
