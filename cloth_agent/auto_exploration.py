"""Standalone automatic Claude/RealSense/xArm exploration loop.

The existing manual free-exploration console remains unchanged.  This module
adds an opt-in state machine for one cautious real rollout at a time:

``Viser RGB-D preview -> RGB-D perception -> Claude plan -> preflight/IK -> execute ->
Viser RGB-D preview -> before/after Claude evaluation -> next iteration``.

Pre-execution validation failures are returned to Claude for a bounded
replanning attempt. With the opt-in recovery flags, a failed pre-execution
planning/evaluation phase can be checkpointed and followed by a fresh
perception iteration. The loop never retries after unknown physical state,
incomplete execution, or failed mandatory return-Home, and never interrupts
an in-progress physical command.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import SafetyError
from .experiment import ExperimentValidationError
from .evidence_ledger import build_evidence_record, persist_evidence_record
from .free_exploration import (
    ClaudeExplorationClient,
    DEFAULT_EXPLORATION_OBJECTIVE,
    ExplorationPlanningError,
    ExplorationTimeoutError,
    ClaudeExplorationResult,
    ExplorationProposal,
    evaluation_depth_ranges,
    evaluation_perception_image_paths,
    _json_from_claude_text,
    _proposal_markdown,
    _voxel_balance_cloud,
    exploration_source,
    grounding_mcp_config,
    is_default_exploration_objective,
    GROUNDING_MCP_TOOLS,
    perception_image_paths,
    validate_exploration_payload,
)
from .kinematics import AnimationFrame, XArm6Kinematics
from .garment_grounding_mcp import GarmentGrounding, GroundingToolError
from .molmo_keypoint_pipeline import (
    DEFAULT_CONFIDENCE_THRESHOLD as DEFAULT_MOLMO_KEYPOINT_CONFIDENCE_THRESHOLD,
    load_keypoint_specs,
    run_molmo_keypoint_pipeline,
    validate_confidence_threshold,
)
from .perception import (
    CameraSpec,
    GarmentCenterWorkspace,
    PerceptionConfig,
    RGBDFrame,
    RealSenseRGBD,
    _scalar_heatmap_rgb,
    camera_height_map_mm,
    capture_two_view_rgbd,
    load_extrinsics,
)
from .persistent_claude import PersistentClaudeSession
from .robot_api import RobotExecutionError, validate_controller_trajectory
from .rollout_recorder import DualRealSenseRolloutRecorder
from .report_figure import compose_camera_perception_report
from .session import AgentSession
from .skill_lifecycle import RunSkillLedger, SkillProposal, SkillStore
from .skills import available_skill_names
from .viewer import (
    _frame_point_cloud,
    _load_fused_point_cloud,
    _load_latest_perception,
)


AUTO_EVALUATION_FIELDS = frozenset(
    {
        "target_selection",
        "grasp_acquisition",
        "target_structure_acquired",
        "transport",
        "laydown",
        "task_progress",
        "earliest_failure_stage",
        "next_experiment",
    }
)
AUTO_EVALUATION_OPTIONAL_FIELDS = frozenset({"skill_update"})
AUTO_EVALUATION_STAGE_FIELDS = frozenset({"status", "confidence", "evidence"})
AUTO_EVALUATION_PROGRESS_FIELDS = frozenset({"status", "confidence", "metrics"})
AUTO_EVALUATION_METRIC_FIELDS = frozenset(
    {
        "visible_area_delta",
        "overlap_delta",
        "relief_delta",
        "boundary_change",
    }
)
AUTO_EVALUATION_NEXT_EXPERIMENT_FIELDS = frozenset({"keep", "change", "reason"})
AUTO_EVALUATION_DELTA_VALUES = frozenset(
    {"INCREASED", "DECREASED", "UNCHANGED", "UNKNOWN"}
)
AUTO_EVALUATION_STAGE_STATUSES = {
    "target_selection": frozenset({"SUPPORTED", "CONTRADICTED", "UNKNOWN"}),
    "grasp_acquisition": frozenset({"SUCCESS", "FAILURE", "UNKNOWN"}),
    "target_structure_acquired": frozenset(
        {"SUPPORTED", "CONTRADICTED", "UNKNOWN"}
    ),
    "transport": frozenset(
        {"GOOD", "BAD_DIRECTION", "INSUFFICIENT", "OVERPULL", "UNKNOWN"}
    ),
    "laydown": frozenset({"SUCCESS", "FAILURE", "NOT_REACHED", "UNKNOWN"}),
}
AUTO_EVALUATION_PROGRESS_STATUSES = frozenset({"IMPROVED", "NEUTRAL", "REGRESSED"})
AUTO_EVALUATION_FAILURE_STAGES = frozenset(
    {"ACQUISITION", "TARGET", "TRANSPORT", "LAYDOWN", "NONE", "UNKNOWN"}
)
AUTO_EVALUATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        name: {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {"type": "string", "enum": sorted(statuses)},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": ["status", "confidence", "evidence"],
        }
        for name, statuses in AUTO_EVALUATION_STAGE_STATUSES.items()
    },
    "required": list(AUTO_EVALUATION_FIELDS),
}
AUTO_EVALUATION_JSON_SCHEMA["properties"].update(
    {
        "task_progress": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "status": {
                    "type": "string",
                    "enum": sorted(AUTO_EVALUATION_PROGRESS_STATUSES),
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "metrics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "visible_area_delta": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "string",
                                    "enum": sorted(AUTO_EVALUATION_DELTA_VALUES),
                                },
                            ]
                        },
                        "overlap_delta": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "string",
                                    "enum": sorted(AUTO_EVALUATION_DELTA_VALUES),
                                },
                            ]
                        },
                        "relief_delta": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "string",
                                    "enum": sorted(AUTO_EVALUATION_DELTA_VALUES),
                                },
                            ]
                        },
                        "boundary_change": {"type": "string", "minLength": 1},
                    },
                    "required": list(AUTO_EVALUATION_METRIC_FIELDS),
                },
            },
            "required": ["status", "confidence", "metrics"],
        },
        "earliest_failure_stage": {
            "type": "string",
            "enum": sorted(AUTO_EVALUATION_FAILURE_STAGES),
        },
        "next_experiment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keep": {
                    "type": "array",
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "change": {
                    "type": "array",
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["keep", "change", "reason"],
        },
        "skill_update": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operation": {"type": "string", "enum": ["create", "modify"]},
                "name": {"type": "string", "minLength": 1},
                "base_skill": {"type": ["string", "null"]},
                "purpose": {"type": "string", "minLength": 1},
                "guidance": {"type": "string", "minLength": 1},
                "rationale": {"type": "string", "minLength": 1},
                "evidence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 1},
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": [
                "operation",
                "name",
                "purpose",
                "guidance",
                "rationale",
                "evidence",
                "confidence",
            ],
        },
    }
)
ACQUISITION_EVALUATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "grasp_acquisition": AUTO_EVALUATION_JSON_SCHEMA["properties"][
            "grasp_acquisition"
        ],
        "target_structure_acquired": AUTO_EVALUATION_JSON_SCHEMA["properties"][
            "target_structure_acquired"
        ],
        "garment_state_change": {
            "type": "string",
            "enum": ["CHANGED", "UNCHANGED", "UNKNOWN"],
        },
        "next_experiment": AUTO_EVALUATION_JSON_SCHEMA["properties"][
            "next_experiment"
        ],
    },
    "required": [
        "grasp_acquisition",
        "target_structure_acquired",
        "garment_state_change",
        "next_experiment",
    ],
}
VISUAL_PLAN_REQUIRED_FIELDS = frozenset(
    {
        "garment_observation",
        "opening_strategy",
        "confidence",
        "selected_reference",
        "motion_intent",
        "expected_observation",
        "safety_notes",
    }
)
VISUAL_PLAN_OPTIONAL_FIELDS = frozenset({"skill_invocations"})
VISUAL_PLAN_FIELDS = VISUAL_PLAN_REQUIRED_FIELDS | VISUAL_PLAN_OPTIONAL_FIELDS
VISUAL_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "garment_observation": {"type": "string", "minLength": 1, "maxLength": 1600},
        "opening_strategy": {"type": "string", "minLength": 1, "maxLength": 1800},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "selected_reference": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "camera": {"type": "string", "enum": ["A", "B"]},
                "reference_id": {"type": "string", "pattern": "^R[0-9]{3,}$"},
                "reason": {"type": "string", "minLength": 1, "maxLength": 1400},
            },
            "required": ["camera", "reference_id", "reason"],
        },
        "motion_intent": {"type": "string", "minLength": 1, "maxLength": 1800},
        "expected_observation": {"type": "string", "minLength": 1, "maxLength": 1400},
        "safety_notes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {"type": "string", "minLength": 1, "maxLength": 400},
        },
        "skill_invocations": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 500},
                },
                "required": ["name", "reason"],
            },
        },
    },
    "required": sorted(VISUAL_PLAN_REQUIRED_FIELDS),
}
DEFAULT_AUTO_OBJECTIVE = DEFAULT_EXPLORATION_OBJECTIVE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _depth_heatmap_preview(
    depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Fallback preview while a live frame has no fitted table plane yet."""

    depth = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > min_depth_m) & (depth < max_depth_m)
    return _scalar_heatmap_rgb(depth, valid)


def _height_map_heatmap_preview(
    frame: RGBDFrame,
    config: PerceptionConfig,
) -> np.ndarray:
    """Return a camera preview colored by surface height above the table."""

    try:
        height_map, valid, _ = camera_height_map_mm(frame, config)
        return _scalar_heatmap_rgb(
            height_map,
            valid,
            higher_is_bright=True,
        )
    except Exception:
        # A live frame can temporarily lack enough table points while the
        # sensor is starting or the arm occludes the scene.  Keep the preview
        # usable until the next frame; saved dense-fusion maps remain strict.
        return _depth_heatmap_preview(
            frame.depth_m,
            min_depth_m=config.min_depth_m,
            max_depth_m=config.max_depth_m,
        )


class AutoExplorationError(RuntimeError):
    """Raised when an automatic-loop contract or runtime phase fails."""


class SelectedReferenceNotExecutableError(ExplorationPlanningError):
    """Raised when a Stage-1 Rxxx cannot be used inside the robot workspace."""

    def __init__(
        self,
        camera: str,
        reference_id: str,
        reason: str,
        *,
        measurement: dict[str, Any] | None = None,
    ):
        self.camera = camera
        self.reference_id = reference_id
        self.reason = reason
        self.measurement = measurement
        super().__init__(
            f"selected reference {camera}/{reference_id} is not executable: {reason}"
        )


class ReferenceReselectionExhaustedError(ExplorationPlanningError):
    """Raised after bounded Stage-1 reference reselection is exhausted."""


@dataclass(frozen=True)
class GarmentWorkspaceRecovery:
    """Deterministic recovery request derived from the robust garment center."""

    center_xy_mm: tuple[float, float]
    bounds_mm: dict[str, float]
    violations: tuple[str, ...]
    full_reentry_target_xy_mm: tuple[float, float]
    requested_target_xy_mm: tuple[float, float]
    requested_translation_xy_mm: tuple[float, float]
    requested_distance_mm: float
    required_min_progress_mm: float

    @property
    def required(self) -> bool:
        return bool(self.violations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "RECOVERY_REQUIRED" if self.required else "IN_BOUNDS",
            "center_xy_mm": list(self.center_xy_mm),
            "bounds_mm": dict(self.bounds_mm),
            "violations": list(self.violations),
            "full_reentry_target_xy_mm": list(self.full_reentry_target_xy_mm),
            "requested_target_xy_mm": list(self.requested_target_xy_mm),
            "requested_translation_xy_mm": list(
                self.requested_translation_xy_mm
            ),
            "requested_distance_mm": self.requested_distance_mm,
            "required_min_progress_mm": self.required_min_progress_mm,
            "center_definition": "dense A/B fused garment robust median XY",
        }


def assess_garment_workspace(
    center_base_mm: Sequence[float],
    workspace: GarmentCenterWorkspace,
) -> GarmentWorkspaceRecovery:
    """Return an inward, step-limited recovery request for an out-of-range center."""

    center = np.asarray(center_base_mm, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise AutoExplorationError(
            "garment workspace check requires a finite center_base_mm XYZ"
        )
    x_mm, y_mm = float(center[0]), float(center[1])
    violations: list[str] = []
    if x_mm < workspace.x_min:
        violations.append("x_below_min")
    elif x_mm > workspace.x_max:
        violations.append("x_above_max")
    if y_mm < workspace.y_min:
        violations.append("y_below_min")
    elif y_mm > workspace.y_max:
        violations.append("y_above_max")

    if not violations:
        target = np.asarray([x_mm, y_mm], dtype=np.float64)
        requested = np.zeros(2, dtype=np.float64)
        requested_target = target.copy()
        requested_distance = 0.0
        required_progress = 0.0
    else:
        target = np.asarray(
            [
                np.clip(
                    x_mm,
                    workspace.x_min + workspace.reentry_margin_mm,
                    workspace.x_max - workspace.reentry_margin_mm,
                ),
                np.clip(
                    y_mm,
                    workspace.y_min + workspace.reentry_margin_mm,
                    workspace.y_max - workspace.reentry_margin_mm,
                ),
            ],
            dtype=np.float64,
        )
        full_translation = target - np.asarray([x_mm, y_mm], dtype=np.float64)
        full_distance = float(np.linalg.norm(full_translation))
        step_distance = min(full_distance, workspace.max_recovery_step_mm)
        requested = full_translation / full_distance * step_distance
        requested_target = np.asarray([x_mm, y_mm], dtype=np.float64) + requested
        requested_distance = float(step_distance)
        required_progress = 0.75 * min(
            requested_distance, workspace.min_recovery_step_mm
        )

    return GarmentWorkspaceRecovery(
        center_xy_mm=(x_mm, y_mm),
        bounds_mm={
            "x_min": workspace.x_min,
            "x_max": workspace.x_max,
            "y_min": workspace.y_min,
            "y_max": workspace.y_max,
            "reentry_margin_mm": workspace.reentry_margin_mm,
        },
        violations=tuple(violations),
        full_reentry_target_xy_mm=(float(target[0]), float(target[1])),
        requested_target_xy_mm=(
            float(requested_target[0]),
            float(requested_target[1]),
        ),
        requested_translation_xy_mm=(
            float(requested[0]),
            float(requested[1]),
        ),
        requested_distance_mm=requested_distance,
        required_min_progress_mm=float(required_progress),
    )


def validate_garment_recovery_actions(
    actions: Sequence[dict[str, Any]],
    recovery: GarmentWorkspaceRecovery,
) -> dict[str, Any]:
    """Require the first grasp-to-release transport to move materially inward."""

    if not recovery.required:
        return {"status": "NOT_REQUIRED"}
    close_index = next(
        (
            index
            for index, action in enumerate(actions)
            if action.get("name") == "close_gripper"
        ),
        None,
    )
    if close_index is None:
        raise ExplorationPlanningError(
            "workspace recovery plan has no close_gripper action"
        )
    grasp_move = next(
        (
            action
            for action in reversed(actions[:close_index])
            if action.get("name") == "move"
        ),
        None,
    )
    release_index = next(
        (
            index
            for index, action in enumerate(
                actions[close_index + 1 :], start=close_index + 1
            )
            if action.get("name") == "open_gripper"
        ),
        None,
    )
    if grasp_move is None or release_index is None:
        raise ExplorationPlanningError(
            "workspace recovery requires a grounded grasp and later release"
        )
    if any(
        action.get("name") == "home"
        for action in actions[close_index + 1 : release_index]
    ):
        raise ExplorationPlanningError(
            "workspace recovery cannot return Home while holding the garment"
        )
    release_move = next(
        (
            action
            for action in reversed(actions[close_index + 1 : release_index])
            if action.get("name") == "move"
        ),
        None,
    )
    if release_move is None:
        raise ExplorationPlanningError(
            "workspace recovery requires a post-grasp move before release"
        )
    grasp_xy = np.asarray(
        [grasp_move["args"]["x"], grasp_move["args"]["y"]],
        dtype=np.float64,
    )
    release_xy = np.asarray(
        [release_move["args"]["x"], release_move["args"]["y"]],
        dtype=np.float64,
    )
    actual = release_xy - grasp_xy
    requested = np.asarray(
        recovery.requested_translation_xy_mm, dtype=np.float64
    )
    requested_distance = float(np.linalg.norm(requested))
    actual_distance = float(np.linalg.norm(actual))
    if requested_distance <= 0:
        raise ExplorationPlanningError(
            "workspace recovery request has no inward translation"
        )
    direction = requested / requested_distance
    inward_progress = float(np.dot(actual, direction))
    lateral_error = float(
        np.linalg.norm(actual - inward_progress * direction)
    )
    if inward_progress < recovery.required_min_progress_mm:
        raise ExplorationPlanningError(
            "workspace recovery transport is too small or points outward: "
            f"inward_progress={inward_progress:.1f} mm, required>="
            f"{recovery.required_min_progress_mm:.1f} mm"
        )
    if actual_distance > requested_distance + max(30.0, 0.25 * requested_distance):
        raise ExplorationPlanningError(
            "workspace recovery transport exceeds the requested safe step: "
            f"actual={actual_distance:.1f} mm, requested={requested_distance:.1f} mm"
        )
    if lateral_error > max(25.0, 0.75 * inward_progress):
        raise ExplorationPlanningError(
            "workspace recovery transport has excessive lateral drift: "
            f"inward={inward_progress:.1f} mm, lateral={lateral_error:.1f} mm"
        )
    predicted_center = np.asarray(recovery.center_xy_mm) + actual
    return {
        "status": "VALIDATED_INWARD_TRANSPORT",
        "grasp_xy_mm": grasp_xy.tolist(),
        "release_xy_mm": release_xy.tolist(),
        "actual_translation_xy_mm": actual.tolist(),
        "actual_distance_mm": actual_distance,
        "inward_progress_mm": inward_progress,
        "lateral_error_mm": lateral_error,
        "predicted_center_xy_mm": predicted_center.tolist(),
    }


def _planning_mode_from_history(
    history: Sequence[dict[str, Any]] | None,
) -> tuple[str, str]:
    """Choose probe-versus-expansion behavior from the last evaluator result."""

    last_evaluation: dict[str, Any] | None = None
    for item in reversed(list(history or [])):
        if (
            isinstance(item, dict)
            and item.get("iteration_mode") == "workspace_recovery"
        ):
            continue
        candidate = item.get("evaluation") if isinstance(item, dict) else None
        if isinstance(candidate, dict) and candidate:
            last_evaluation = candidate
            break
    if last_evaluation is None:
        return (
            "EXPLORATION",
            ""
            "MODE = EXPLORATION: no previous grasp/layer hypothesis is validated. "
            "Use a small, reversible probe that is just large enough to distinguish "
            "whether the selected layer moves; keep the net lateral displacement roughly "
            "10–30 mm or no more than one third of the visibly safe distance. Do not make "
            "a long committed pull before acquisition and target-layer motion are evidenced.",
        )

    target_selection = (last_evaluation.get("target_selection") or {}).get("status")
    grasp = (last_evaluation.get("grasp_acquisition") or {}).get("status")
    target_layer = (last_evaluation.get("target_structure_acquired") or {}).get("status")
    transport = (last_evaluation.get("transport") or {}).get("status")
    validated = (
        target_selection == "SUPPORTED"
        and grasp == "SUCCESS"
        and target_layer == "SUPPORTED"
    )
    if validated and transport in {"GOOD", "INSUFFICIENT"}:
        return (
            "VALIDATED_EXPANSION",
            ""
            "MODE = VALIDATED_EXPANSION: the previous evaluator supported target selection, "
            "grasp acquisition, and target-layer motion. Preserve the validated grasp anchor "
            "and grasp depth; do not restart with another tiny probe. Commit to the full "
            "outward transport and laydown that the geometry supports, normally covering most "
            "of the visible safe distance and at least about 40 mm when workspace and garment "
            "scale permit. Use waypoints to shape the path, not to reduce its net displacement.",
        )
    if validated and transport in {"BAD_DIRECTION", "OVERPULL"}:
        return (
            "VALIDATED_TRANSPORT_CORRECTION",
            ""
            "MODE = VALIDATED_TRANSPORT_CORRECTION: acquisition and target-layer motion were "
            "supported, but the previous transport direction or magnitude was wrong. Preserve "
            "the validated grasp anchor and depth, change the transport direction/profile, and "
            "make a deliberate correction across a meaningful distance; do not repeat the same "
            "short motion or re-probe acquisition.",
        )
    return (
        "EXPLORATION",
        ""
        "MODE = EXPLORATION: the previous result did not validate both acquisition and target "
        "layer motion. Use a small, reversible probe that is just large enough to distinguish "
        "the layer response; keep the net lateral displacement roughly 10–30 mm or no more than "
        "one third of the visibly safe distance. Do not commit a long pull until the hypothesis "
        "is supported.",
    )


def _is_preexecution_replan_error(exc: BaseException) -> bool:
    """Return whether a pre-motion failure is useful feedback for replanning."""

    if isinstance(exc, (ExplorationTimeoutError, ReferenceReselectionExhaustedError)):
        return False
    return isinstance(
        exc,
        (
            ExperimentValidationError,
            SafetyError,
            RobotExecutionError,
            AutoExplorationError,
            ExplorationPlanningError,
        ),
    ) and "physical rollout did not complete" not in str(exc)


def _is_recoverable_viewer_error(
    exc: BaseException,
    record: dict[str, Any],
) -> bool:
    """Return whether the viewer loop may start a fresh iteration safely.

    A fresh perception/replan is safe before any robot command.  After a
    completed rollout it is also safe to recover from a Claude/evidence
    failure only when the mandatory return-home phase completed.  Unknown
    robot state, incomplete physical execution, and controller errors remain
    hard stops.
    """

    message = str(exc).lower()
    hard_tokens = (
        "physical rollout did not complete",
        "robotexecutionerror",
        "robot error",
        "xarm",
        "set_position",
        "set_servo_angle",
        "return home",
        "home failed",
    )
    if any(token in message for token in hard_tokens):
        return False
    execution = record.get("execution")
    if not isinstance(execution, dict):
        return isinstance(
            exc,
            (
                ExplorationTimeoutError,
                ReferenceReselectionExhaustedError,
                ExplorationPlanningError,
                ExperimentValidationError,
                SafetyError,
                AutoExplorationError,
            ),
        )
    if not execution.get("execution_completed"):
        return False
    home = record.get("mandatory_return_home")
    if not isinstance(home, dict) or not home.get("completed"):
        return False
    return isinstance(exc, (ExplorationTimeoutError, AutoExplorationError))


def grasp_targets_from_actions(
    actions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return each grounded grasp target from the move before close_gripper.

    A proposal may contain more than one regrasp.  Each target is reported in
    execution order.  Plans without a finite preceding Cartesian move return an
    empty list so the UI can display ``unknown`` rather than inventing a point.
    """

    targets: list[dict[str, Any]] = []
    latest_move: dict[str, Any] | None = None
    latest_move_index: int | None = None
    for action_index, action in enumerate(actions):
        name = action.get("name")
        if name == "home":
            latest_move = None
            latest_move_index = None
            continue
        if name == "move":
            args = action.get("args", {})
            try:
                values = {key: float(args[key]) for key in ("x", "y", "z", "yaw")}
            except (KeyError, TypeError, ValueError):
                latest_move = None
                latest_move_index = None
                continue
            if not all(math.isfinite(value) for value in values.values()):
                latest_move = None
                latest_move_index = None
                continue
            latest_move = values
            latest_move_index = action_index
            continue
        if name != "close_gripper":
            continue
        if latest_move is not None and latest_move_index is not None:
            targets.append(
                {
                    "target_index": len(targets) + 1,
                    "move_action_index": latest_move_index + 1,
                    "close_action_index": action_index + 1,
                    **latest_move,
                }
            )
        latest_move = None
        latest_move_index = None
    return targets


def _project_base_target_to_frame(
    frame: RGBDFrame,
    target: dict[str, Any],
) -> tuple[float, float] | None:
    """Project one base-frame target into a calibrated RGB frame."""

    base_from_camera = np.asarray(frame.X_base_camera, dtype=np.float64)
    intrinsics = np.asarray(frame.intrinsics, dtype=np.float64)
    if base_from_camera.shape != (4, 4) or intrinsics.shape != (3, 3):
        return None
    xyz_base_m = np.asarray(
        [target["x"], target["y"], target["z"], 1000.0], dtype=np.float64
    ) / 1000.0
    try:
        xyz_camera = np.linalg.inv(base_from_camera) @ xyz_base_m
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(xyz_camera)) or xyz_camera[2] <= 0.0:
        return None
    x_px = intrinsics[0, 0] * xyz_camera[0] / xyz_camera[2] + intrinsics[0, 2]
    y_px = intrinsics[1, 1] * xyz_camera[1] / xyz_camera[2] + intrinsics[1, 2]
    if not math.isfinite(float(x_px)) or not math.isfinite(float(y_px)):
        return None
    return float(x_px), float(y_px)


def target_overlay_image(
    frame: RGBDFrame,
    targets: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw grasp target crosshairs on one RGB frame and report projections."""

    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    projections: list[dict[str, Any]] = []
    if not targets:
        draw.rectangle((8, 8, 226, 34), fill=(0, 0, 0), outline=(255, 255, 255), width=1)
        draw.text((14, 14), "GRASP TARGET: unknown", fill=(255, 210, 40))
        return np.asarray(image), projections

    palette = [(255, 45, 45), (255, 65, 210), (255, 155, 25)]
    for target in targets:
        target_index = int(target["target_index"])
        pixel = _project_base_target_to_frame(frame, target)
        visible = False
        projection: dict[str, Any] = {
            "target_index": target_index,
            "pixel": None,
            "visible": False,
        }
        if pixel is not None:
            x_px, y_px = pixel
            projection["pixel"] = [x_px, y_px]
            visible = 0 <= x_px < image.width and 0 <= y_px < image.height
            projection["visible"] = visible
            if visible:
                color = palette[(target_index - 1) % len(palette)]
                radius = 16
                draw.ellipse(
                    (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                    outline=(255, 255, 255),
                    width=6,
                )
                draw.ellipse(
                    (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                    outline=color,
                    width=3,
                )
                draw.line((x_px - 24, y_px, x_px + 24, y_px), fill=color, width=3)
                draw.line((x_px, y_px - 24, x_px, y_px + 24), fill=color, width=3)
                draw.text((x_px + 19, y_px - 23), f"T{target_index}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        projections.append(projection)

    header_height = 18 + 17 * len(targets)
    draw.rectangle(
        (8, 8, min(image.width - 8, 425), min(image.height - 8, header_height)),
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        width=1,
    )
    for row, target in enumerate(targets):
        projection = projections[row]
        visibility = "visible" if projection["visible"] else "off-image"
        draw.text(
            (14, 13 + row * 17),
            (
                f"T{target['target_index']} base=({target['x']:.1f}, {target['y']:.1f}, "
                f"{target['z']:.1f})mm yaw={target['yaw']:.1f}deg {visibility}"
            ),
            fill=(255, 255, 255),
        )
    return np.asarray(image), projections


@dataclass(frozen=True)
class StageEvaluation:
    status: str
    confidence: float
    evidence: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ProgressMetrics:
    visible_area_delta: float | str
    overlap_delta: float | str
    relief_delta: float | str
    boundary_change: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "visible_area_delta": self.visible_area_delta,
            "overlap_delta": self.overlap_delta,
            "relief_delta": self.relief_delta,
            "boundary_change": self.boundary_change,
        }


@dataclass(frozen=True)
class TaskProgressEvaluation:
    status: str
    confidence: float
    metrics: ProgressMetrics

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class NextExperiment:
    keep: tuple[str, ...]
    change: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "keep": list(self.keep),
            "change": list(self.change),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ExplorationEvaluation:
    target_selection: StageEvaluation
    grasp_acquisition: StageEvaluation
    target_structure_acquired: StageEvaluation
    transport: StageEvaluation
    laydown: StageEvaluation
    task_progress: TaskProgressEvaluation
    earliest_failure_stage: str
    next_experiment: NextExperiment
    skill_update: SkillProposal | None = None

    @property
    def useful(self) -> bool:
        """Compatibility summary for old dashboards and historical callers."""

        return self.task_progress.status == "IMPROVED"

    @property
    def confidence(self) -> float:
        return self.task_progress.confidence

    @property
    def stop(self) -> bool:
        """An empty change list means no safe grounded next experiment exists."""

        return not self.next_experiment.change

    @property
    def reason(self) -> str:
        return self.next_experiment.reason

    @property
    def next_objective(self) -> str:
        if self.stop:
            return f"Stop: {self.next_experiment.reason}"
        keep = ", ".join(self.next_experiment.keep) or "none"
        change = ", ".join(self.next_experiment.change)
        return (
            f"Next experiment: keep [{keep}]; change [{change}]. "
            f"Reason: {self.next_experiment.reason}"
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "target_selection": self.target_selection.as_dict(),
            "grasp_acquisition": self.grasp_acquisition.as_dict(),
            "target_structure_acquired": self.target_structure_acquired.as_dict(),
            "transport": self.transport.as_dict(),
            "laydown": self.laydown.as_dict(),
            "task_progress": self.task_progress.as_dict(),
            "earliest_failure_stage": self.earliest_failure_stage,
            "next_experiment": self.next_experiment.as_dict(),
        }
        if self.skill_update is not None:
            payload["skill_update"] = self.skill_update.as_dict()
        return payload


@dataclass(frozen=True)
class ClaudeEvaluationResult:
    """Raw Claude evaluator call plus its validated structured judgement."""

    prompt: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    created_at: str
    evaluation: ExplorationEvaluation
    evidence_images: tuple[str, ...] = ()
    video_references: tuple[str, ...] = ()
    video_evidence_errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "created_at": self.created_at,
            "evaluation": self.evaluation.as_dict(),
            "evidence_images": list(self.evidence_images),
            "video_references": list(self.video_references),
            "video_evidence_errors": list(self.video_evidence_errors),
        }


@dataclass(frozen=True)
class VisualPlanDecision:
    """Stage-one visual decision with one selected but not yet grounded Rxxx."""

    garment_observation: str
    opening_strategy: str
    confidence: float
    selected_reference: dict[str, str]
    motion_intent: str
    expected_observation: str
    safety_notes: tuple[str, ...]
    skill_invocations: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "garment_observation": self.garment_observation,
            "opening_strategy": self.opening_strategy,
            "confidence": self.confidence,
            "selected_reference": dict(self.selected_reference),
            "motion_intent": self.motion_intent,
            "expected_observation": self.expected_observation,
            "safety_notes": list(self.safety_notes),
            "skill_invocations": [dict(item) for item in self.skill_invocations],
        }


@dataclass(frozen=True)
class ClaudeVisualPlanResult:
    """Raw stage-one Claude call plus its validated visual decision."""

    prompt: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    created_at: str
    duration_s: float
    decision: VisualPlanDecision

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "created_at": self.created_at,
            "duration_s": self.duration_s,
            "decision": self.decision.as_dict(),
        }


def _json_default(value: Any) -> Any:
    """Serialize paths, NumPy values, and dataclasses in run records."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _support_layer_context(session: AgentSession) -> dict[str, Any]:
    """Build a compact, auditable support-layer context for Claude.

    The physical support setup is configuration, not something Claude should
    have to infer from an RGB image.  Include both the operator declaration and
    the latest local-ring diagnostics so the model understands why a deeper
    press is or is not authorized.  The host remains authoritative for the
    final numeric grasp height.
    """

    robot = session.robot_config
    configured = str(getattr(robot, "support_layer_type", "none")).strip().lower()
    confirmed = bool(getattr(robot, "support_layer_confirmed", False))
    threshold = float(
        getattr(robot, "support_layer_presence_threshold_mm", 3.0)
    )
    ring_by_camera: dict[str, Any] = {}
    ring_confirmed = False
    workspace = getattr(session, "workspace", None)
    if workspace is None:
        workspace = Path(session.run_dir) / "workspace"
    perception_dir = Path(workspace) / "perception_views"
    for label in ("A", "B"):
        path = perception_dir / f"camera_{label}_local_support.json"
        payload: dict[str, Any] = {}
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    payload = raw
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                payload = {}
        elevation = payload.get("ring_elevation_median_mm")
        detected = bool(
            payload.get("valid") is True
            and isinstance(elevation, (int, float))
            and not isinstance(elevation, bool)
            and math.isfinite(float(elevation))
            and float(elevation) >= threshold
        )
        ring_confirmed = ring_confirmed or detected
        ring_by_camera[label] = {
            "valid": payload.get("valid") is True,
            "ring_elevation_median_mm": elevation,
            "presence_threshold_mm": threshold,
            "detected": detected,
            "source": path.name if path.is_file() else None,
        }
    active = bool(configured == "sponge" and (confirmed or ring_confirmed))
    activation_source = None
    if active:
        activation_source = "declared_configuration" if confirmed else "local_support_ring"
    return {
        "type": configured,
        "declared_present": bool(configured == "sponge"),
        "confirmed": confirmed,
        "active": active,
        "activation_source": activation_source,
        "thickness_mm": float(getattr(robot, "support_layer_thickness_mm", 0.0)),
        "press_mm": float(getattr(robot, "support_layer_press_mm", 0.0)),
        "max_compression_mm": float(
            getattr(robot, "support_layer_max_compression_mm", 0.0)
        ),
        "hard_table_clearance_mm": float(
            getattr(robot, "support_layer_hard_clearance_mm", 0.0)
        ),
        "presence_threshold_mm": threshold,
        "ring_by_camera": ring_by_camera,
        "policy": (
            "A confirmed sponge permits the configured deeper press; the host "
            "computes the final grasp Z from the selected local surface and keeps "
            "the robot lower bound authoritative."
            if active
            else "No deeper sponge allowance is active; retain the ordinary grasp policy."
        ),
    }


def _command_argument_diagnostics(command: Sequence[str]) -> dict[str, Any]:
    """Return byte-size diagnostics for an argv before spawning a process."""

    encoded_sizes = [len(str(item).encode("utf-8")) for item in command]
    longest_index = int(np.argmax(encoded_sizes)) if encoded_sizes else -1
    return {
        "argument_count": len(encoded_sizes),
        "total_argument_bytes": int(sum(encoded_sizes)),
        "largest_argument_bytes": (
            int(encoded_sizes[longest_index]) if longest_index >= 0 else 0
        ),
        "largest_argument_index": longest_index,
    }


def _normalize_final_grounding_payload(
    payload: Any,
) -> tuple[Any, list[str]]:
    """Repair harmless non-action schema overflow in final grounding output."""

    normalizations: list[str] = []
    if not isinstance(payload, dict):
        return payload, normalizations
    raw_safety_notes = payload.get("safety_notes")
    if (
        isinstance(raw_safety_notes, list)
        and len(raw_safety_notes) > 10
        and all(
            isinstance(note, str) and note.strip()
            for note in raw_safety_notes
        )
    ):
        payload = dict(payload)
        payload["safety_notes"] = raw_safety_notes[:10]
        normalizations.append(
            "truncated safety_notes to the schema maximum of 10"
        )
    return payload, normalizations


def _write_final_grounding_context(
    root: Path,
    *,
    context: Mapping[str, Any],
    selected_reference: Mapping[str, Any],
    mode_action_instruction: str,
) -> dict[str, Any]:
    """Persist large Stage-2 inputs as separate auditable run-local logs."""

    root = Path(root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    context_dir = root / "results" / "claude_context" / f"final_grounding_{stamp}"
    context_dir.mkdir(parents=True, exist_ok=False)

    objective_path = context_dir / "01_objective.md"
    visual_path = context_dir / "02_visual_plan.json"
    history_path = context_dir / "03_previous_physical_outcomes.json"
    robot_path = context_dir / "04_robot_and_mode_context.json"
    contract_path = context_dir / "05_final_grounding_contract.md"
    manifest_path = context_dir / "manifest.json"

    objective_path.write_text(str(context.get("objective", "")).strip() + "\n", encoding="utf-8")
    _write_json(visual_path, context.get("visual_plan", {}))
    _write_json(history_path, context.get("previous_physical_outcomes", []))
    _write_json(
        robot_path,
        {
            key: value
            for key, value in context.items()
            if key not in {"objective", "visual_plan", "previous_physical_outcomes"}
        },
    )
    selected_camera = str(selected_reference.get("camera", ""))
    selected_id = str(selected_reference.get("reference_id", ""))
    fold_step = _fold_sleeve_step_from_objective(str(context.get("objective", "")))
    grasp_anchor_lines = [
        "For ordinary tasks, the move immediately before close_gripper must use the selected Rxxx Base XY within 2 mm.",
    ]
    if fold_step is not None:
        grasp_anchor_lines.extend(
            [
                f"For this {fold_step} sleeve step, Rxxx is a calibrated semantic anchor, not a mandatory closure center.",
                f"A deliberate learned contact offset up to {_FOLD_GRASP_ANCHOR_OFFSET_MAX_MM:.1f} mm from the anchor is allowed on either side of the observed garment-mask boundary.",
                "The garment mask is perceptual evidence, not a robot safety boundary: an edge-straddle hypothesis may place the TCP center slightly over visible table so one jaw is outside cloth and one jaw is inside. Keep the offset small and explicit in reveal_strategy/safety_notes; the host still requires a calibrated Camera-A pixel, robot workspace validity, safe Z, preflight, and controller IK.",
                "Approach and pre-scuff waypoints may shape the entry, but the bounded offset rule applies to the final move immediately before close_gripper.",
            ]
        )
    contract_path.write_text(
        "\n".join(
            [
                "# Stage 2 final grounding contract",
                "",
                "The Stage-1 visual decision is fixed. Do not revisit images, compare alternatives, or change the selected reference.",
                f"Call `lookup_reference` exactly once with camera={selected_camera} and reference_id={selected_id}.",
                "Use that returned measurement to ground the grasp and compose the final numeric RobotAPI proposal.",
                "Do not call any other MCP tool.",
                "The supplied support-layer context is physical setup information, not a visual hypothesis. If it is active or explicitly confirmed, a deeper compressive bite up to the configured press/max-compression values is allowed; do not reject it using a hard-table assumption.",
                "The host will replace the move immediately before close_gripper with the shared grasp-height resolution from the selected local surface. Treat the host-resolved Z as authoritative and do not compensate by inventing a second Z elsewhere in the trajectory.",
                "Y workspace is yaw-dependent: action yaw is relative to Home; yaw=0 keeps "
                "the configured Y bounds unchanged, while the outward allowance is "
                "0.5 * gripper_width_mm * abs(sin(radians(yaw))) (at +/-90 degrees this "
                "is half the effective gripper width). This allowance applies only to "
                "the TCP center and only in Y; host validation recomputes it from every "
                "actual waypoint and controller IK remains authoritative.",
                "If using the yaw-dependent Y extension, rotate to the chosen relative yaw "
                "at an in-bounds waypoint before crossing the original yaw=0 Y limit; "
                "never move outside the yaw=0 envelope and rotate afterward. Interpolated "
                "waypoints are checked with their instantaneous yaw allowance.",
                "Always release before the action list ends and keep at most 12 actions.",
                "By default, the first move after close_gripper must be a vertical lift for a hold check. If the proposal explicitly sets requires_lift_checkpoint=false, it may instead close while translating for a clearly stated rolling/buckling experiment; this opts out only of the hold-ordering rule, not workspace, Z-range, preflight, or IK validation.",
                "If the objective contains the explicit ACQUISITION PROBE contract, the returned action list must itself contain only a reversible near-vertical lift probe with no post-close lateral transport; the host will not silently rewrite a full fold plan.",
                "The objective file is authoritative; do not replace the requested task with generic garment opening.",
                "",
                "Return exactly these fields and no others:",
                "- garment_observation: string",
                "- reveal_strategy: string",
                "- confidence: number in [0,1]",
                "- actions: non-empty list of {name,args}",
                "- expected_observation: string",
                "- safety_notes: list containing 1 to 10 non-empty strings",
                "- optional skill_invocations: list of {name,reason}",
                "- requires_lift_checkpoint: boolean; set false only when this experiment intentionally closes while translating (for example a rolling/buckling bite) and explain that choice in reveal_strategy or safety_notes",
                "",
                "For move, args must contain exactly numeric x,y,z,yaw in millimetres/degrees.",
                "The only permitted actions are move, open_gripper, close_gripper, and home.",
                "Use multiple waypoints to shape the path, not to silently change the intended net displacement.",
                "",
                "## Selected-reference anchor rule",
                *grasp_anchor_lines,
                "",
                "## Planning-mode action rule",
                mode_action_instruction,
                "",
            ]
        ),
        encoding="utf-8",
    )

    read_order = [
        objective_path,
        visual_path,
        history_path,
        robot_path,
        contract_path,
    ]
    manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "stage": "final_grounding",
        "selected_reference": dict(selected_reference),
        "read_order": [_run_relative(path, root) for path in read_order],
        "files": {
            path.name: {
                "path": _run_relative(path, root),
                "size_bytes": path.stat().st_size,
            }
            for path in read_order
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "directory": str(context_dir),
        "manifest": str(manifest_path),
        "manifest_relative": _run_relative(manifest_path, root),
        "read_order": list(manifest["read_order"]),
        "files": dict(manifest["files"]),
    }


def _run_relative(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _evaluation_exact_fields(
    payload: Any,
    fields: frozenset[str],
    *,
    context: str,
    optional_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AutoExplorationError(f"{context} must be a JSON object")
    missing = fields.difference(payload)
    unknown = set(payload).difference(fields | optional_fields)
    if missing:
        raise AutoExplorationError(f"{context} is missing fields: {sorted(missing)}")
    if unknown:
        raise AutoExplorationError(f"{context} has unknown fields: {sorted(unknown)}")
    return payload


def _evaluation_confidence(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AutoExplorationError(f"{context} confidence must be numeric")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise AutoExplorationError(f"{context} confidence must be between 0 and 1")
    return confidence


def _evaluation_strings(
    value: Any,
    *,
    context: str,
    allow_empty: bool = False,
    limit: int = 12,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > limit:
        raise AutoExplorationError(f"{context} must be a list of at most {limit} strings")
    if not value and not allow_empty:
        raise AutoExplorationError(f"{context} must contain at least one evidence string")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AutoExplorationError(f"every {context} item must be a non-empty string")
        normalized = item.strip()
        if normalized in result:
            raise AutoExplorationError(f"{context} contains a duplicate item: {normalized}")
        result.append(normalized)
    return tuple(result)


def _validate_stage_evaluation(name: str, payload: Any) -> StageEvaluation:
    value = _evaluation_exact_fields(
        payload,
        AUTO_EVALUATION_STAGE_FIELDS,
        context=f"evaluation.{name}",
    )
    status = value["status"]
    allowed = AUTO_EVALUATION_STAGE_STATUSES[name]
    if not isinstance(status, str) or status not in allowed:
        raise AutoExplorationError(
            f"evaluation.{name}.status must be one of {sorted(allowed)}"
        )
    return StageEvaluation(
        status=status,
        confidence=_evaluation_confidence(value["confidence"], context=f"evaluation.{name}"),
        evidence=_evaluation_strings(
            value["evidence"], context=f"evaluation.{name}.evidence"
        ),
    )


def _validate_metric_delta(value: Any, *, context: str) -> float | str:
    if isinstance(value, bool):
        raise AutoExplorationError(
            f"{context} must be a finite number or one of {sorted(AUTO_EVALUATION_DELTA_VALUES)}"
        )
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in AUTO_EVALUATION_DELTA_VALUES:
            return normalized
    raise AutoExplorationError(
        f"{context} must be a finite number or one of {sorted(AUTO_EVALUATION_DELTA_VALUES)}"
    )


def validate_evaluation_payload(payload: Any) -> ExplorationEvaluation:
    """Validate the stage-wise before/after judgement before another iteration."""

    value = _evaluation_exact_fields(
        payload,
        AUTO_EVALUATION_FIELDS,
        context="Claude evaluation",
        optional_fields=AUTO_EVALUATION_OPTIONAL_FIELDS,
    )
    stages = {
        name: _validate_stage_evaluation(name, value[name])
        for name in AUTO_EVALUATION_STAGE_STATUSES
    }

    progress_value = _evaluation_exact_fields(
        value["task_progress"],
        AUTO_EVALUATION_PROGRESS_FIELDS,
        context="evaluation.task_progress",
    )
    progress_status = progress_value["status"]
    if (
        not isinstance(progress_status, str)
        or progress_status not in AUTO_EVALUATION_PROGRESS_STATUSES
    ):
        raise AutoExplorationError(
            "evaluation.task_progress.status must be one of "
            f"{sorted(AUTO_EVALUATION_PROGRESS_STATUSES)}"
        )
    metric_value = _evaluation_exact_fields(
        progress_value["metrics"],
        AUTO_EVALUATION_METRIC_FIELDS,
        context="evaluation.task_progress.metrics",
    )
    boundary_change = metric_value["boundary_change"]
    if not isinstance(boundary_change, str) or not boundary_change.strip():
        raise AutoExplorationError(
            "evaluation.task_progress.metrics.boundary_change must be a non-empty string"
        )
    metrics = ProgressMetrics(
        visible_area_delta=_validate_metric_delta(
            metric_value["visible_area_delta"],
            context="evaluation.task_progress.metrics.visible_area_delta",
        ),
        overlap_delta=_validate_metric_delta(
            metric_value["overlap_delta"],
            context="evaluation.task_progress.metrics.overlap_delta",
        ),
        relief_delta=_validate_metric_delta(
            metric_value["relief_delta"],
            context="evaluation.task_progress.metrics.relief_delta",
        ),
        boundary_change=boundary_change.strip(),
    )

    failure_stage = value["earliest_failure_stage"]
    if not isinstance(failure_stage, str) or failure_stage not in AUTO_EVALUATION_FAILURE_STAGES:
        raise AutoExplorationError(
            "evaluation.earliest_failure_stage must be one of "
            f"{sorted(AUTO_EVALUATION_FAILURE_STAGES)}"
        )

    next_value = _evaluation_exact_fields(
        value["next_experiment"],
        AUTO_EVALUATION_NEXT_EXPERIMENT_FIELDS,
        context="evaluation.next_experiment",
    )
    keep = _evaluation_strings(
        next_value["keep"],
        context="evaluation.next_experiment.keep",
        allow_empty=True,
    )
    change = _evaluation_strings(
        next_value["change"],
        context="evaluation.next_experiment.change",
        allow_empty=True,
    )
    overlap = set(keep).intersection(change)
    if overlap:
        raise AutoExplorationError(
            "evaluation.next_experiment cannot both keep and change: "
            f"{sorted(overlap)}"
        )
    reason = next_value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise AutoExplorationError(
            "evaluation.next_experiment.reason must be a non-empty string"
        )

    try:
        skill_update = SkillStore.parse_update(value.get("skill_update"))
    except ValueError as exc:
        raise AutoExplorationError(str(exc)) from exc

    return ExplorationEvaluation(
        target_selection=stages["target_selection"],
        grasp_acquisition=stages["grasp_acquisition"],
        target_structure_acquired=stages["target_structure_acquired"],
        transport=stages["transport"],
        laydown=stages["laydown"],
        task_progress=TaskProgressEvaluation(
            status=progress_status,
            confidence=_evaluation_confidence(
                progress_value["confidence"], context="evaluation.task_progress"
            ),
            metrics=metrics,
        ),
        earliest_failure_stage=failure_stage,
        next_experiment=NextExperiment(
            keep=keep,
            change=change,
            reason=reason.strip(),
        ),
        skill_update=skill_update,
    )


def validate_visual_plan_payload(
    payload: Any,
    *,
    allowed_skill_names: Sequence[str] | None = None,
) -> VisualPlanDecision:
    """Validate stage-one output before exact Rxx grounding is permitted."""

    if not isinstance(payload, dict):
        raise ExplorationPlanningError("visual plan must be a JSON object")
    # Normalize two common harmless Claude aliases/explanatory additions before
    # enforcing the executable contract.  Motion fields, camera, and Rxxx are
    # never inferred or changed here.
    payload = dict(payload)
    raw_reference = payload.get("selected_reference")
    if isinstance(raw_reference, dict) and {
        "camera",
        "reference_id",
        "reason",
    }.issubset(raw_reference):
        payload["selected_reference"] = {
            key: raw_reference[key]
            for key in ("camera", "reference_id", "reason")
        }
    raw_skills = payload.get("skill_invocations")
    if isinstance(raw_skills, list):
        normalized_skills: list[Any] = []
        for item in raw_skills:
            if isinstance(item, dict) and "name" not in item and "skill" in item:
                item = {**item, "name": item["skill"]}
                item.pop("skill", None)
            normalized_skills.append(item)
        payload["skill_invocations"] = normalized_skills
    missing = VISUAL_PLAN_REQUIRED_FIELDS.difference(payload)
    unknown = set(payload).difference(VISUAL_PLAN_FIELDS)
    if missing:
        raise ExplorationPlanningError(
            f"visual plan is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise ExplorationPlanningError(
            f"visual plan has unknown fields: {sorted(unknown)}"
        )
    strings: dict[str, str] = {}
    for name in (
        "garment_observation",
        "opening_strategy",
        "motion_intent",
        "expected_observation",
    ):
        value = payload[name]
        if not isinstance(value, str) or not value.strip():
            raise ExplorationPlanningError(
                f"visual plan field {name} must be a non-empty string"
            )
        strings[name] = value.strip()
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ExplorationPlanningError("visual plan confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ExplorationPlanningError(
            "visual plan confidence must be between 0 and 1"
        )
    reference = payload["selected_reference"]
    if not isinstance(reference, dict) or set(reference) != {
        "camera",
        "reference_id",
        "reason",
    }:
        raise ExplorationPlanningError(
            "selected_reference must contain exactly camera, reference_id, and reason"
        )
    camera = str(reference["camera"]).strip().upper()
    reference_id = str(reference["reference_id"]).strip().upper()
    reason = reference["reason"]
    if camera not in {"A", "B"}:
        raise ExplorationPlanningError("selected reference camera must be A or B")
    if not re.fullmatch(r"R\d{3,}", reference_id):
        raise ExplorationPlanningError(
            "selected reference_id must look like R026"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ExplorationPlanningError(
            "selected reference reason must be a non-empty string"
        )
    notes = payload["safety_notes"]
    if not isinstance(notes, list) or not 1 <= len(notes) <= 10:
        raise ExplorationPlanningError(
            "visual plan safety_notes must contain 1 to 10 strings"
        )
    safety_notes = tuple(str(note).strip() for note in notes)
    if any(not note for note in safety_notes):
        raise ExplorationPlanningError(
            "every visual plan safety note must be non-empty"
        )
    raw_skills = payload.get("skill_invocations", [])
    approved_skill_names = set(allowed_skill_names or available_skill_names())
    if not isinstance(raw_skills, list):
        raise ExplorationPlanningError("visual plan skill_invocations must be a list")
    skills: list[dict[str, str]] = []
    for item in raw_skills:
        if not isinstance(item, dict) or set(item) != {"name", "reason"}:
            raise ExplorationPlanningError(
                "each visual skill invocation needs exactly name and reason"
            )
        name, skill_reason = item["name"], item["reason"]
        if (
            not isinstance(name, str)
            or name.strip().lower() not in approved_skill_names
            or not isinstance(skill_reason, str)
            or not skill_reason.strip()
        ):
            raise ExplorationPlanningError(
                "visual plan skill invocation must use an approved skill with a "
                "non-empty reason"
            )
        skills.append({"name": name.strip().lower(), "reason": skill_reason.strip()})
    return VisualPlanDecision(
        garment_observation=strings["garment_observation"],
        opening_strategy=strings["opening_strategy"],
        confidence=confidence,
        selected_reference={
            "camera": camera,
            "reference_id": reference_id,
            "reason": reason.strip(),
        },
        motion_intent=strings["motion_intent"],
        expected_observation=strings["expected_observation"],
        safety_notes=safety_notes,
        skill_invocations=tuple(skills),
    )


# The fold pipeline used to describe the supervisor output as
# ``next incomplete step is left_sleeve``.  The supervisor contract now uses
# ``current_step`` to mean the action that should be executed immediately, so
# the parser must accept both forms.  Keep the accepted syntax deliberately
# narrow: this value is used to activate deterministic sleeve-reference
# prevalidation before spending a Claude planning call.
_FOLD_NEXT_STEP_RE = re.compile(
    r"(?:next\s+incomplete\s+step\s+is|current_step\s*[\"']?\s*(?:is|=|:))"
    r"\s*[\"']?"
    r"(left_sleeve|right_sleeve)\b",
    re.IGNORECASE,
)
_EXACT_REFERENCE_GRASP_TOLERANCE_MM = 2.0
_FOLD_GRASP_ANCHOR_OFFSET_MAX_MM = 10.0
_FOLD_GRASP_OFFSET_PIXEL_MATCH_MAX_MM = 2.0


def _fold_sleeve_step_from_objective(objective: str | None) -> str | None:
    if not objective:
        return None
    match = _FOLD_NEXT_STEP_RE.search(objective)
    return match.group(1).lower() if match else None


def _fold_sleeve_reference_geometry(
    garment_mask_raw: np.ndarray,
    raw_pixel_xy: Sequence[int | float],
    *,
    step: str,
    rgb_raw: np.ndarray | None = None,
    max_interior_distance_px: float = 14.0,
    require_free_edge: bool = True,
) -> dict[str, Any]:
    """Validate a sleeve anchor in the canonical clockwise-90 RGB frame.

    This is deliberately a coarse deterministic gate, not a semantic segmenter.
    It rejects the failure class seen in production: a torso/print interior point
    described as a sleeve. The caller may retain the historical free-edge gate,
    or may admit the whole outer sleeve region for acquisition-learning runs in
    which useful contact structure must be inferred from physical outcomes.
    """

    if step not in {"left_sleeve", "right_sleeve"}:
        raise ValueError(f"unsupported sleeve step: {step}")
    mask_raw = np.asarray(garment_mask_raw, dtype=bool)
    if mask_raw.ndim != 2 or not np.any(mask_raw):
        raise ValueError("Camera A garment mask is empty or invalid")
    if len(raw_pixel_xy) != 2:
        raise ValueError("selected reference has no valid raw Camera-A pixel")
    x_raw = int(round(float(raw_pixel_xy[0])))
    y_raw = int(round(float(raw_pixel_xy[1])))
    raw_height, raw_width = mask_raw.shape
    if not (0 <= x_raw < raw_width and 0 <= y_raw < raw_height):
        raise ValueError(
            f"raw pixel [{x_raw}, {y_raw}] lies outside Camera A {raw_width}x{raw_height}"
        )

    mask = np.rot90(mask_raw, k=3)
    x_upright = raw_height - 1 - y_raw
    y_upright = x_raw
    if not bool(mask[y_upright, x_upright]):
        raise ValueError(
            f"upright pixel [{x_upright}, {y_upright}] is outside the garment mask"
        )

    ys, xs = np.nonzero(mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    width = max(1.0, float(x_max - x_min))
    height = max(1.0, float(y_max - y_min))
    outer_fraction = 0.18
    if step == "left_sleeve":
        side_limit = float(x_min) + outer_fraction * width
        side_ok = float(x_upright) <= side_limit
        side_rule = f"x <= {side_limit:.1f}"
    else:
        side_limit = float(x_max) - outer_fraction * width
        side_ok = float(x_upright) >= side_limit
        side_rule = f"x >= {side_limit:.1f}"

    sleeve_y_min = float(y_min) + 0.15 * height
    sleeve_y_max = float(y_min) + 0.58 * height
    sleeve_height_ok = sleeve_y_min <= float(y_upright) <= sleeve_y_max
    sleeve_height_rule = f"{sleeve_y_min:.1f} <= y <= {sleeve_y_max:.1f}"

    from scipy.ndimage import distance_transform_edt

    interior_distance_px = float(distance_transform_edt(mask)[y_upright, x_upright])
    max_interior_distance_px = float(max_interior_distance_px)
    if (
        not math.isfinite(max_interior_distance_px)
        or max_interior_distance_px <= 0.0
    ):
        raise ValueError("max_interior_distance_px must be positive and finite")
    boundary_ok = interior_distance_px <= max_interior_distance_px
    edge_contrast_ok: bool | None = None
    edge_contrast_rgb_norm: float | None = None
    minimum_edge_contrast_rgb_norm = 80.0
    if rgb_raw is not None:
        rgb_array = np.asarray(rgb_raw)
        if rgb_array.shape[:2] != mask_raw.shape or rgb_array.ndim != 3:
            raise ValueError("Camera A RGB shape does not match its garment mask")
        rgb = np.rot90(rgb_array[..., :3], k=3).astype(np.float64)
        local_radius = 4
        x0 = max(0, x_upright - local_radius)
        x1 = min(mask.shape[1], x_upright + local_radius + 1)
        y0 = max(0, y_upright - local_radius)
        y1 = min(mask.shape[0], y_upright + local_radius + 1)
        local_mask = mask[y0:y1, x0:x1]
        local_pixels = rgb[y0:y1, x0:x1][local_mask]
        search_radius = max(20, int(math.ceil(interior_distance_px)) + 10)
        sx0 = max(0, x_upright - search_radius)
        sx1 = min(mask.shape[1], x_upright + search_radius + 1)
        sy0 = max(0, y_upright - search_radius)
        sy1 = min(mask.shape[0], y_upright + search_radius + 1)
        outside_pixels = rgb[sy0:sy1, sx0:sx1][~mask[sy0:sy1, sx0:sx1]]
        if len(local_pixels) == 0 or len(outside_pixels) == 0:
            edge_contrast_ok = False
            edge_contrast_rgb_norm = 0.0
        else:
            outside_rgb = np.median(outside_pixels, axis=0)
            selected_rgb = rgb[y_upright, x_upright]
            edge_contrast_rgb_norm = float(np.linalg.norm(selected_rgb - outside_rgb))
            edge_contrast_ok = (
                edge_contrast_rgb_norm >= minimum_edge_contrast_rgb_norm
            )
    diagnostic = {
        "step": step,
        "rotation": "clockwise90",
        "raw_pixel_xy": [x_raw, y_raw],
        "upright_pixel_xy": [x_upright, y_upright],
        "upright_garment_bbox_xyxy": [x_min, y_min, x_max, y_max],
        "requested_side_rule": side_rule,
        "requested_side_ok": side_ok,
        "sleeve_height_rule": sleeve_height_rule,
        "sleeve_height_ok": sleeve_height_ok,
        "interior_distance_px": interior_distance_px,
        "max_interior_distance_px": max_interior_distance_px,
        "free_edge_band_ok": boundary_ok,
        "edge_contrast_rgb_norm": edge_contrast_rgb_norm,
        "minimum_edge_contrast_rgb_norm": minimum_edge_contrast_rgb_norm,
        "edge_contrast_ok": edge_contrast_ok,
        "free_edge_required": bool(require_free_edge),
    }
    failures: list[str] = []
    if not side_ok:
        failures.append(
            f"upright x={x_upright} is not in the outer {step} side band ({side_rule})"
        )
    if not sleeve_height_ok:
        failures.append(
            f"upright y={y_upright} is outside the sleeve-height band "
            f"({sleeve_height_rule})"
        )
    if require_free_edge and not boundary_ok:
        failures.append(
            f"point is {interior_distance_px:.1f}px inside the garment, beyond the "
            f"{max_interior_distance_px:.1f}px sleeve free-edge band"
        )
    if require_free_edge and edge_contrast_ok is False:
        failures.append(
            f"RGB edge contrast {edge_contrast_rgb_norm:.1f} is below the "
            f"{minimum_edge_contrast_rgb_norm:.1f} threshold; marker may be mask leakage "
            "or flat table rather than a visible cuff boundary"
        )
    if failures:
        raise ValueError("; ".join(failures))
    return diagnostic


def _validate_fold_grasp_anchor_offset(
    session: AgentSession,
    measurement: Mapping[str, Any],
    actual_xy: Sequence[int | float],
    *,
    step: str,
    max_offset_mm: float = _FOLD_GRASP_ANCHOR_OFFSET_MAX_MM,
) -> dict[str, Any]:
    """Validate a small closure offset from a calibrated sleeve anchor.

    Rxxx remains the measured semantic anchor. A fold planner may use a small,
    deliberate contact offset supported by its physical-outcome history, but
    the offset must map back to a calibrated Camera-A pixel and remain within
    the bounded physical radius. The garment mask is diagnostic rather than a
    safety boundary: a closure center just outside cloth is legal for an
    edge-straddle experiment and may simply produce an observable empty grasp.
    """

    if step not in {"left_sleeve", "right_sleeve"}:
        raise ValueError(f"unsupported fold grasp-offset step: {step}")
    expected_xy = np.asarray(measurement.get("base_xyz_mm", [])[:2], dtype=np.float64)
    actual = np.asarray(actual_xy, dtype=np.float64)
    if expected_xy.shape != (2,) or actual.shape != (2,):
        raise ValueError("fold grasp offset requires finite anchor and actual Base XY")
    if not np.all(np.isfinite(expected_xy)) or not np.all(np.isfinite(actual)):
        raise ValueError("fold grasp offset contains non-finite Base XY")
    offset_xy = actual - expected_xy
    offset_mm = float(np.linalg.norm(offset_xy))
    if offset_mm > float(max_offset_mm):
        raise ValueError(
            f"closure offset {offset_mm:.1f} mm exceeds the "
            f"{float(max_offset_mm):.1f} mm sleeve-anchor limit"
        )

    pixel_xy = measurement.get("pixel_xy")
    if not isinstance(pixel_xy, Sequence) or len(pixel_xy) != 2:
        raise ValueError("selected sleeve anchor has no raw Camera-A pixel")
    selected_x = int(round(float(pixel_xy[0])))
    selected_y = int(round(float(pixel_xy[1])))
    perception_dir = (
        session.run_dir.resolve() / "workspace" / "perception_views"
    )
    xyz_path = perception_dir / "camera_A_base_xyz_mm.npy"
    mask_path = perception_dir / "camera_A_garment_mask.npy"
    if not xyz_path.is_file() or not mask_path.is_file():
        raise ValueError(
            "Camera-A dense Base XYZ or garment mask is unavailable for "
            "fold grasp-offset validation"
        )
    xyz_map = np.load(xyz_path, allow_pickle=False)
    garment_mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    if xyz_map.shape[:2] != garment_mask.shape or xyz_map.ndim != 3 or xyz_map.shape[2] < 2:
        raise ValueError("Camera-A dense Base XYZ and garment mask shapes do not match")
    height, width = garment_mask.shape
    if not (0 <= selected_x < width and 0 <= selected_y < height):
        raise ValueError("selected sleeve anchor pixel lies outside Camera A")

    search_radius_px = 48
    x0 = max(0, selected_x - search_radius_px)
    x1 = min(width, selected_x + search_radius_px + 1)
    y0 = max(0, selected_y - search_radius_px)
    y1 = min(height, selected_y + search_radius_px + 1)
    local_xy = np.asarray(xyz_map[y0:y1, x0:x1, :2], dtype=np.float64)
    finite = np.all(np.isfinite(local_xy), axis=2)
    if not np.any(finite):
        raise ValueError("no finite Camera-A Base XY exists near the sleeve anchor")
    distances = np.linalg.norm(local_xy - actual.reshape(1, 1, 2), axis=2)
    distances[~finite] = np.inf
    local_y, local_x = np.unravel_index(int(np.argmin(distances)), distances.shape)
    map_match_error_mm = float(distances[local_y, local_x])
    actual_x = int(x0 + local_x)
    actual_y = int(y0 + local_y)
    if map_match_error_mm > _FOLD_GRASP_OFFSET_PIXEL_MATCH_MAX_MM:
        raise ValueError(
            f"offset closure Base XY has no calibrated Camera-A pixel within "
            f"{_FOLD_GRASP_OFFSET_PIXEL_MATCH_MAX_MM:.1f} mm; nearest error is "
            f"{map_match_error_mm:.1f} mm"
        )
    closure_inside_mask = bool(garment_mask[actual_y, actual_x])
    closure_outside_distance_px = 0.0
    if not closure_inside_mask:
        from scipy.ndimage import distance_transform_edt

        closure_outside_distance_px = float(
            distance_transform_edt(~garment_mask)[actual_y, actual_x]
        )

    selected_upright_x = float(height - 1 - selected_y)
    actual_upright_x = float(height - 1 - actual_y)
    if step == "left_sleeve":
        inboard_delta_px = actual_upright_x - selected_upright_x
        inboard_rule = "upright +x toward garment center"
    else:
        inboard_delta_px = selected_upright_x - actual_upright_x
        inboard_rule = "upright -x toward garment center"
    return {
        "mode": "bounded_sleeve_anchor_offset",
        "step": step,
        "anchor_reference_id": measurement.get("reference_id"),
        "anchor_base_xy_mm": expected_xy.tolist(),
        "closure_base_xy_mm": actual.tolist(),
        "offset_xy_mm": offset_xy.tolist(),
        "offset_distance_mm": offset_mm,
        "max_offset_mm": float(max_offset_mm),
        "anchor_raw_pixel_xy": [selected_x, selected_y],
        "closure_raw_pixel_xy": [actual_x, actual_y],
        "closure_pixel_map_error_mm": map_match_error_mm,
        "inboard_delta_upright_px": inboard_delta_px,
        "inboard_rule": inboard_rule,
        "closure_inside_garment_mask": closure_inside_mask,
        "closure_outside_mask_distance_px": closure_outside_distance_px,
        "mask_policy": "DIAGNOSTIC_ONLY_EDGE_STRADDLE_ALLOWED",
    }


class ClaudeAutoClient:
    """Claude adapter that plans actions and judges before/after images."""

    def __init__(
        self,
        binary: str = "claude",
        timeout_s: int = 400,
        grounding_timeout_s: int = 120,
        max_reference_reselections: int = 2,
        skill_guidance: str | None = None,
        skill_names: Sequence[str] | None = None,
        persistent_session: PersistentClaudeSession | None = None,
    ):
        if max_reference_reselections < 0 or max_reference_reselections > 10:
            raise ValueError("max_reference_reselections must be between 0 and 10")
        self.binary = binary
        self.timeout_s = timeout_s
        self.grounding_timeout_s = grounding_timeout_s
        self.max_reference_reselections = max_reference_reselections
        self.skill_guidance = skill_guidance
        self.skill_names = tuple(skill_names or available_skill_names())
        self.persistent_session = persistent_session
        self.planner = ClaudeExplorationClient(binary=binary, timeout_s=timeout_s)
        self.last_plan_result: ClaudeExplorationResult | None = None
        self.last_visual_plan_result: ClaudeVisualPlanResult | None = None
        self.last_plan_timing: dict[str, float] = {}
        self.last_rejected_visual_references: list[dict[str, Any]] = []
        self.last_reference_validation: dict[str, Any] | None = None
        self.last_reference_candidate_report: dict[str, Any] | None = None
        self.last_grounding_verification: dict[str, Any] | None = None
        self.last_evaluation_result: ClaudeEvaluationResult | None = None

    def _prepare_command(self, command: Sequence[str], *, stage: str) -> list[str]:
        if self.persistent_session is None:
            return [str(item) for item in command]
        return self.persistent_session.prepare_command(command, stage=stage)

    def _record_successful_turn(self, *, stage: str, stdout: str) -> None:
        if self.persistent_session is not None:
            self.persistent_session.record_success(stage=stage, stdout=stdout)

    @staticmethod
    def _save_evaluation_log(root: Path, payload: dict[str, Any], *, failed: bool = False) -> None:
        log_dir = root / "results" / "claude_auto"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"{stamp}_evaluation{'_failed' if failed else ''}.json"
        _write_json(log_dir / name, payload)

    @staticmethod
    def _save_visual_log(root: Path, payload: dict[str, Any], *, failed: bool = False) -> None:
        log_dir = root / "results" / "claude_visual"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"{stamp}_visual_plan{'_failed' if failed else ''}.json"
        _write_json(log_dir / name, payload)

    def _binary(self) -> str:
        binary = self.binary
        if Path(binary).name == binary:
            resolved = shutil.which(binary)
            if resolved is None:
                raise AutoExplorationError(f"Claude CLI not found: {binary}")
            binary = resolved
        return str(binary)

    @staticmethod
    def _safe_images(image_paths: Sequence[Path], root: Path) -> list[Path]:
        safe: list[Path] = []
        for raw in image_paths:
            path = Path(raw).resolve()
            if path != root and root not in path.parents:
                raise PermissionError("Claude images must stay inside the current run")
            if not path.is_file():
                raise FileNotFoundError(path)
            safe.append(path)
        if not safe:
            raise ExplorationPlanningError("at least one garment image is required")
        return safe

    def _visual_plan(
        self,
        image_paths: Sequence[Path],
        base_prompt: str,
        run_dir: Path,
    ) -> ClaudeVisualPlanResult:
        root = run_dir.resolve()
        safe_images = self._safe_images(image_paths, root)
        def image_label(path: Path) -> str:
            name = path.name.lower()
            if name == "camera_a_flat_reference.png":
                return "REFERENCE | FLAT GARMENT | RAW RGB"
            if name == "camera_a_flat_reference_anchors.png":
                return "REFERENCE | FLAT GARMENT | ANNOTATED ANCHORS"
            if name == "camera_a_rgb_upright.png":
                return (
                    "CURRENT CAMERA A | RGB ONLY | CANONICAL UPRIGHT "
                    "(CLOCKWISE 90-DEG ROTATION)"
                )
            if name == "camera_a_rxxx_overlay_upright.png":
                return (
                    "CURRENT CAMERA A | Rxxx OVERLAY | SAME CANONICAL UPRIGHT FRAME"
                )
            return "CURRENT SCENE | RGB/GEOMETRY"

        image_text = "\n".join(
            f"- {image_label(path)}: {path}" for path in safe_images
        )
        prompt = (
            f"{base_prompt}\n\n"
            "STAGE 1 — VISUAL PLANNING ONLY. Preserve the original image-reasoning "
            "workflow. No MCP server or coordinate lookup tool is available in this "
            "stage. Inspect the supplied RGB, height, boundary, gradient, and Rxxx "
            "overlay images. Select exactly one visually justified Camera A/B Rxxx "
            "reference for the eventual grasp. Do not emit numeric RobotAPI actions "
            "yet; the exact selected Rxxx measurement and final run will be produced "
            "by stage 2. Describe enough motion intent for stage 2 to choose approach, "
            "grasp height, lift, retreat, laydown, release, and yaw. Follow the explicit "
            "probe-versus-expansion MODE supplied below: exploration may be a small reversible "
            "probe, while a validated hypothesis should be expanded into meaningful transport.\n\n"
            f"Garment images to inspect:\n{image_text}\n\n"
            "When the canonical upright Camera-A RGB and Rxxx overlay are supplied, "
            "they are the only authoritative frame for image-left/image-right garment "
            "semantics: LEFT means the viewer's left side of the displayed image "
            "(smaller upright x), and RIGHT means the viewer's right side (larger "
            "upright x). Do not use the wearer's anatomical left/right and do not "
            "mirror the image. They show the same rotated pixels and the same Rxxx identities. "
            "Do not reinterpret an Rxxx from a sideways/raw orientation. Before naming "
            "a sleeve reference, visually verify that its marker is on the requested "
            "sleeve fabric, not the torso interior, chest print, opposite sleeve, label, "
            "or table. Do not assume that any particular edge, interior point, seam, or "
            "wrinkle is privileged; use the supplied physical-outcome history to justify "
            "the current contact hypothesis.\n\n"
            "Visual evidence priority: inspect the raw flat-garment reference first to "
            "establish topology, printed-pattern correspondence, and which current layer "
            "is covering which region. Use current RGB to localize that structure and use "
            "height/depth/gradient images only to verify relief, boundaries, and graspability. "
            "Do not select an Rxxx reference solely because it is the brightest or highest "
            "heatmap region; explain the corresponding reference pattern/region and the "
            "expected newly exposed garment area first.\n\n"
            "Return exactly one JSON object with these fields and no others: "
            "garment_observation (string), opening_strategy (string), confidence "
            "(number 0..1), selected_reference ({camera: A|B, reference_id: Rxxx, "
            "reason: string}), motion_intent (string), expected_observation (string), "
            "safety_notes (list containing 1 to 10 non-empty strings), and optional skill_invocations "
            "(objects containing exactly name and reason; name is an approved skill "
            "such as laydown or flatten-garment). The selected_reference object must "
            "contain exactly camera, reference_id, and reason; put all supporting "
            "visual context inside reason instead of adding fields. Do not return "
            "actions, XYZ coordinates, Python, or a run function in this stage."
        )
        command = [
            self._binary(),
            "--print",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(VISUAL_PLAN_JSON_SCHEMA, separators=(",", ":")),
            "--permission-mode",
            "plan",
            "--allowedTools",
            "Read",
            "--tools",
            "Read",
            "--add-dir",
            str(root),
            "--safe-mode",
            "--no-session-persistence",
            "--system-prompt",
            (
                "You are the visual-planning stage of a cautious garment-task "
                "robotics agent. Read only the supplied run images. Select one final "
                "Camera/Rxxx reference and return the requested JSON decision. Do not "
                "write files, execute commands, call MCP tools, or control a robot."
            ),
        ]
        command = self._prepare_command(command, stage="visual_planning")
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
            duration_s = time.monotonic() - started
            if self.persistent_session is not None:
                self.persistent_session.rollover(
                    reason="claude_timeout", stage="visual_planning"
                )
            error = (
                f"ExplorationTimeoutError: Claude visual planning timed out after "
                f"{self.timeout_s} seconds"
            )
            self._save_visual_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": None,
                    "stdout": getattr(exc, "stdout", "") or "",
                    "stderr": getattr(exc, "stderr", "") or "",
                    "duration_s": duration_s,
                    "error": error,
                    "created_at": _now(),
                },
                failed=True,
            )
            raise ExplorationTimeoutError(
                f"Claude visual planning timed out after {self.timeout_s} seconds"
            ) from exc
        except OSError as exc:
            raise ExplorationPlanningError(
                f"Claude visual planning invocation failed: {exc}"
            ) from exc
        duration_s = time.monotonic() - started
        if completed.returncode != 0:
            if self.persistent_session is not None:
                if self.persistent_session.is_session_conflict_error(
                    completed.stdout, completed.stderr
                ):
                    self.persistent_session.rollover(
                        reason="claude_session_conflict", stage="visual_planning"
                    )
                elif self.persistent_session.is_context_limit_error(
                    completed.stdout, completed.stderr
                ):
                    self.persistent_session.rollover(
                        reason="claude_context_limit", stage="visual_planning"
                    )
            self._save_visual_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "duration_s": duration_s,
                    "error": "non-zero Claude visual-planning return code",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise ExplorationPlanningError(
                f"Claude visual planning exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        self._record_successful_turn(
            stage="visual_planning",
            stdout=completed.stdout,
        )
        try:
            decision = validate_visual_plan_payload(
                _json_from_claude_text(completed.stdout),
                allowed_skill_names=self.skill_names,
            )
        except BaseException as exc:
            self._save_visual_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "duration_s": duration_s,
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise
        result = ClaudeVisualPlanResult(
            prompt=prompt,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            created_at=_now(),
            duration_s=duration_s,
            decision=decision,
        )
        self._save_visual_log(root, result.as_dict())
        return result

    @staticmethod
    def _validate_measurement_for_stage2(
        camera: str,
        reference_id: str,
        measurement: Mapping[str, Any],
        session: AgentSession,
        objective: str | None = None,
    ) -> dict[str, Any]:
        """Apply the deterministic Stage-2 gates to one saved measurement."""

        measurement = dict(measurement)

        xyz = np.asarray(measurement.get("base_xyz_mm", []), dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise SelectedReferenceNotExecutableError(
                camera,
                reference_id,
                "saved calibrated Base XYZ is missing or non-finite",
                measurement=measurement,
            )

        robot_config = session.robot_config
        bounds = robot_config.boundaries
        margin = float(robot_config.workspace_margin_mm)
        # Stage 2 runs before Claude has supplied action waypoints/yaw.  Do
        # not reject an otherwise valid edge reference merely because the
        # eventual gripper orientation may permit the configured maximum
        # half-width extension.  The final action is rechecked with its actual
        # relative yaw by RobotAPI and the controller trajectory validator.
        potential_y_low, potential_y_high = robot_config.y_workspace_bounds_mm(90.0)
        violations: list[str] = []
        try:
            bounds.validate_lateral(float(xyz[0]), float(xyz[1]), margin)
        except SafetyError as exc:
            violations.append(str(exc))
        for axis, value in (("x", float(xyz[0])), ("y", float(xyz[1]))):
            low = getattr(bounds, f"{axis}_min")
            high = getattr(bounds, f"{axis}_max")
            if axis == "y":
                low = potential_y_low
                high = potential_y_high
            if low is not None and value < float(low) + margin:
                violations.append(
                    f"{axis}={value:.3f} is below the safe lower bound "
                    f"{float(low) + margin:.3f} mm"
                )
            if high is not None and value > float(high) - margin:
                violations.append(
                    f"{axis}={value:.3f} is above the safe upper bound "
                    f"{float(high) - margin:.3f} mm"
                )
        if violations:
            raise SelectedReferenceNotExecutableError(
                camera,
                reference_id,
                "; ".join(violations),
                measurement=measurement,
            )

        measurement["workspace_candidate_validation"] = {
            "mode": "maximum_yaw_extension_before_final_action_yaw_is_known",
            "relative_yaw_deg_assumed": 90.0,
            "y_extension_mm": robot_config.y_workspace_extension_mm(90.0),
            "effective_y_bounds_mm": [potential_y_low, potential_y_high],
            "final_action_validation": "actual relative yaw is checked by host and controller IK",
        }

        fold_step = _fold_sleeve_step_from_objective(objective)
        if fold_step is not None:
            if camera != "A":
                raise SelectedReferenceNotExecutableError(
                    camera,
                    reference_id,
                    (
                        f"{fold_step} must be selected in the canonical upright Camera-A "
                        "RGB/Rxxx frame; Camera B is observation-only"
                    ),
                    measurement=measurement,
                )
            mask_path = (
                session.run_dir.resolve()
                / "workspace"
                / "perception_views"
                / "camera_A_garment_mask.npy"
            )
            if not mask_path.is_file():
                raise SelectedReferenceNotExecutableError(
                    camera,
                    reference_id,
                    f"Camera A garment mask is unavailable for {fold_step} semantic validation",
                    measurement=measurement,
                )
            try:
                from PIL import Image

                rgb_path = mask_path.with_name("camera_0_A.png")
                rgb_raw = None
                if rgb_path.is_file():
                    with Image.open(rgb_path) as rgb_image:
                        rgb_raw = np.asarray(rgb_image.convert("RGB"))
                maximum_interior_distance_px = 14.0
                if measurement.get("measurement_kind") == (
                    "uniform_calibrated_reference"
                ):
                    try:
                        uniform_stride_px = float(
                            measurement.get("sample_stride_px", 48.0)
                        )
                    except (TypeError, ValueError):
                        uniform_stride_px = 48.0
                    # A uniform grid cannot guarantee a point within the old
                    # 14 px edge band: with a 48 px stride, the nearest visible
                    # cloth sample can naturally be about 24 px inside. Match
                    # the gate to the sampling contract while retaining the
                    # independent outer-side and sleeve-height checks.
                    maximum_interior_distance_px = min(
                        32.0,
                        max(14.0, 0.6 * uniform_stride_px),
                    )
                semantic = _fold_sleeve_reference_geometry(
                    np.load(mask_path),
                    measurement.get("pixel_xy", []),
                    step=fold_step,
                    rgb_raw=rgb_raw,
                    max_interior_distance_px=maximum_interior_distance_px,
                    require_free_edge=False,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise SelectedReferenceNotExecutableError(
                    camera,
                    reference_id,
                    (
                        f"semantic sleeve-region validation failed for {fold_step}: {exc}. "
                        "Choose a visible Rxxx on the requested outer sleeve region, not "
                        "the torso, print interior, opposite sleeve, or table"
                    ),
                    measurement=measurement,
                ) from exc
            measurement = dict(measurement)
            measurement["fold_semantic_validation"] = semantic
        return measurement

    @staticmethod
    def _validate_reference_for_stage2(
        visual: VisualPlanDecision,
        session: AgentSession,
        objective: str | None = None,
    ) -> dict[str, Any]:
        """Reject an unexecutable Stage-1 reference before spending a Stage-2 call."""

        selected = visual.selected_reference
        camera = selected["camera"]
        reference_id = selected["reference_id"]
        try:
            measurement = GarmentGrounding(
                session.run_dir.resolve() / "workspace" / "perception_views"
            ).lookup_reference(camera, reference_id)
        except GroundingToolError as exc:
            raise SelectedReferenceNotExecutableError(
                camera,
                reference_id,
                f"saved calibrated measurement is unavailable: {exc}",
            ) from exc
        return ClaudeAutoClient._validate_measurement_for_stage2(
            camera,
            reference_id,
            measurement,
            session,
            objective,
        )

    @staticmethod
    def _fold_reference_candidate_report(
        session: AgentSession,
        objective: str | None,
    ) -> dict[str, Any] | None:
        """Prevalidate every visible uniform Rxxx for a named sleeve step.

        The full uniform overlay remains visible for semantic context, but Claude
        must not spend calls selecting references that the host can already prove
        will fail workspace or sleeve-anchor validation.
        """

        step = _fold_sleeve_step_from_objective(objective)
        if step is None:
            return None
        perception_dir = (
            session.run_dir.resolve() / "workspace" / "perception_views"
        )
        grounding = GarmentGrounding(perception_dir)
        try:
            guide = grounding._guide("A")
        except GroundingToolError as exc:
            raise ReferenceReselectionExhaustedError(
                f"deterministic {step} reference prevalidation could not read "
                f"Camera-A references: {exc}"
            ) from exc

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for sample in guide.get("samples", []):
            if not isinstance(sample, Mapping):
                continue
            if sample.get("reference_source") == "fold_rgb_boundary_dense":
                continue
            reference_id = str(sample.get("reference_id", "")).strip().upper()
            if not reference_id:
                continue
            try:
                measurement = grounding.lookup_reference("A", reference_id)
                validated = ClaudeAutoClient._validate_measurement_for_stage2(
                    "A",
                    reference_id,
                    measurement,
                    session,
                    objective,
                )
            except (GroundingToolError, SelectedReferenceNotExecutableError) as exc:
                reason = exc.reason if isinstance(
                    exc, SelectedReferenceNotExecutableError
                ) else str(exc)
                rejected.append(
                    {
                        "reference_id": reference_id,
                        "pixel_xy": sample.get("pixel_xy"),
                        "reason": reason,
                    }
                )
                continue
            semantic = validated.get("fold_semantic_validation", {})
            accepted.append(
                {
                    "reference_id": reference_id,
                    "pixel_xy": validated.get("pixel_xy"),
                    "base_xyz_mm": validated.get("base_xyz_mm"),
                    "interior_distance_px": semantic.get("interior_distance_px"),
                    "edge_contrast_rgb_norm": semantic.get(
                        "edge_contrast_rgb_norm"
                    ),
                }
            )
        return {
            "step": step,
            "reference_mode": "uniform_full_garment",
            "workspace_y_extension_policy": {
                "candidate_prevalidation": "maximum possible extension at |yaw|=90; final action yaw is revalidated",
                "gripper_width_mm": session.robot_config.gripper_width_mm,
                "maximum_extension_mm": session.robot_config.y_workspace_extension_mm(90.0),
            },
            "visible_reference_count": len(accepted) + len(rejected),
            "executable_reference_count": len(accepted),
            "executable_reference_ids": [
                item["reference_id"] for item in accepted
            ],
            "accepted": accepted,
            "rejected": rejected,
        }

    def _ground_final_plan(
        self,
        visual: VisualPlanDecision,
        session: AgentSession,
        objective: str,
        history: Sequence[dict[str, Any]] | None = None,
        workspace_recovery: GarmentWorkspaceRecovery | None = None,
    ) -> ClaudeExplorationResult:
        root = session.run_dir.resolve()
        selected = visual.selected_reference
        planning_mode, planning_mode_instruction = _planning_mode_from_history(history)
        if workspace_recovery is not None and workspace_recovery.required:
            planning_mode = "WORKSPACE_RECOVERY"
            planning_mode_instruction = (
                "MODE = WORKSPACE_RECOVERY: the robust garment center is outside the "
                "configured operating rectangle. This safety-priority iteration must "
                "move the grasped garment inward by the requested Base-frame XY "
                "translation before release. Do not spend this action on opening, "
                "probing, or outward expansion. Use a safe lift/transfer/laydown path, "
                "keep the net grasp-to-release transport aligned with the requested "
                "translation, and do not exceed the requested step."
            )
        context = {
            "objective": objective,
            "visual_plan": visual.as_dict(),
            "previous_physical_outcomes": list(history or [])[-8:],
            "planning_mode": planning_mode,
            "planning_mode_instruction": planning_mode_instruction,
            "validated_center_reference": {
                "x_mm": session.experiment_config.cloth_center_x,
                "y_mm": session.experiment_config.cloth_center_y,
                "surface_z_mm": session.experiment_config.grasp_z,
            },
            "workspace_bounds_mm": asdict(session.robot_config.boundaries),
            "yaw_dependent_workspace": {
                "relative_yaw_definition": (
                    "action yaw is relative to the calibrated Home TCP orientation"
                ),
                "gripper_width_mm": float(session.robot_config.gripper_width_mm),
                "y_extension_formula": (
                    "0.5 * gripper_width_mm * abs(sin(radians(relative_yaw_deg)))"
                ),
                "y_extension_at_0_deg_mm": session.robot_config.y_workspace_extension_mm(0.0),
                "y_extension_at_90_deg_mm": session.robot_config.y_workspace_extension_mm(90.0),
                "policy": (
                    "yaw=0 keeps the calibrated Y bounds; +/-90 degrees permits "
                    "up to half the configured effective gripper width for the TCP "
                    "center only. Final host and controller validation remain required."
                ),
            },
            "fixed_orientation_deg": {
                "roll": session.robot_config.orientation_roll_deg,
                "pitch": session.robot_config.orientation_pitch_deg,
            },
            "support_layer": _support_layer_context(session),
            "grasp_height_policy": {
                "surface_compression_mm": session.robot_config.grasp_surface_compression_mm,
                "min_compression_mm": session.robot_config.grasp_min_compression_mm,
                "max_compression_mm": session.robot_config.grasp_max_compression_mm,
                "table_clearance_mm": session.robot_config.grasp_table_clearance_mm,
                "use_table_clearance_floor": session.robot_config.grasp_use_table_clearance_floor,
                "final_z_authority": "host_resolve_grasp_height",
            },
        }
        if workspace_recovery is not None:
            context["garment_workspace_recovery"] = workspace_recovery.as_dict()
        if is_default_exploration_objective(objective):
            mode_action_instruction = (
                "Follow the planning mode in the context: in EXPLORATION, use a small "
                "reversible probe sufficient to distinguish the layer response; in "
                "VALIDATED_EXPANSION, preserve the validated grasp anchor/depth and complete "
                "a meaningful outward transport, normally covering most of the visible safe "
                "distance and at least about 40 mm when scale and workspace permit; in "
                "VALIDATED_TRANSPORT_CORRECTION, preserve acquisition and change only the "
                "proven transport direction/profile with a deliberate correction."
            )
        else:
            mode_action_instruction = (
                "Follow the user objective and the planning mode in the context. In the initial "
                "task-directed exploration, use only the smallest reversible action needed to "
                "resolve a concrete uncertainty about the named target or its safe manipulation. "
                "After the target is validated, preserve that target and make meaningful progress "
                "toward the requested state. Do not turn the user task into a generic garment "
                "opening, spreading, or outward-transport objective."
            )
        if "EXECUTION CONTRACT — ACQUISITION PROBE" in objective:
            mode_action_instruction = (
                "Follow the explicit ACQUISITION PROBE contract in the objective. "
                "Claude must return the reversible probe directly: approach/open/close, "
                "one or more near-vertical post-close lift checkpoints, reversal, "
                "release, and optional home. Do not include post-close lateral transport "
                "or fold laydown. The host will validate this contract but will not "
                "rewrite a full fold trajectory into a different probe."
            )
        context_bundle = _write_final_grounding_context(
            root,
            context=context,
            selected_reference=selected,
            mode_action_instruction=mode_action_instruction,
        )
        prompt = (
            "STAGE 2 — FINAL RXX GROUNDING AND RUN GENERATION. "
            f"Read `{context_bundle['manifest_relative']}` first with the Read tool, "
            "then read every file listed in its read_order completely and in order. "
            "Those run-local files are the complete authoritative planning context. "
            f"After reading them, call lookup_reference exactly once for "
            f"{selected['camera']}/{selected['reference_id']} and return only the required "
            "final JSON proposal. Do not inspect images or any file not listed by the manifest."
        )
        mcp_config = grounding_mcp_config(root)
        enabled_tools = ("Read", *GROUNDING_MCP_TOOLS)
        command = [
            self._binary(),
            "--print",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            ",".join(enabled_tools),
            "--tools",
            "Read",
            "--mcp-config",
            json.dumps(mcp_config, ensure_ascii=False, separators=(",", ":")),
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--add-dir",
            str(root),
            "--system-prompt",
            (
                "You are the final grounding/compiler stage of a garment-task "
                "robotics agent. Read only the manifest and context files explicitly "
                "named by the bootstrap prompt. The visual decision is fixed. Call the "
                "single exact Rxxx lookup once, then return only the final JSON proposal. "
                "Do not read images, write files, execute commands, or control a robot."
            ),
        ]
        command = self._prepare_command(command, stage="final_grounding")
        command_diagnostics = _command_argument_diagnostics(command)
        if command_diagnostics["largest_argument_bytes"] >= 120_000:
            raise ExplorationPlanningError(
                "Claude final-grounding command still contains an oversized argument "
                f"before process launch: {command_diagnostics}"
            )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                timeout=self.grounding_timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            error = (
                f"ExplorationTimeoutError: Claude final grounding timed out after "
                f"{self.grounding_timeout_s} seconds"
            )
            if self.persistent_session is not None:
                self.persistent_session.rollover(
                    reason="claude_timeout", stage="final_grounding"
                )
            self.planner._save_invocation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": None,
                    "stdout": getattr(exc, "stdout", "") or "",
                    "stderr": getattr(exc, "stderr", "") or "",
                    "duration_s": time.monotonic() - started,
                    "error": error,
                    "created_at": _now(),
                    "stage": "final_grounding",
                    "context_bundle": context_bundle,
                    "command_diagnostics": command_diagnostics,
                },
                failed=True,
            )
            raise ExplorationTimeoutError(
                f"Claude final grounding timed out after "
                f"{self.grounding_timeout_s} seconds"
            ) from exc
        duration_s = time.monotonic() - started
        if completed.returncode != 0:
            if self.persistent_session is not None:
                if self.persistent_session.is_session_conflict_error(
                    completed.stdout, completed.stderr
                ):
                    self.persistent_session.rollover(
                        reason="claude_session_conflict", stage="final_grounding"
                    )
                elif self.persistent_session.is_context_limit_error(
                    completed.stdout, completed.stderr
                ):
                    self.persistent_session.rollover(
                        reason="claude_context_limit", stage="final_grounding"
                    )
            self.planner._save_invocation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "duration_s": duration_s,
                    "error": "non-zero Claude final-grounding return code",
                    "created_at": _now(),
                    "stage": "final_grounding",
                    "context_bundle": context_bundle,
                    "command_diagnostics": command_diagnostics,
                },
                failed=True,
            )
            raise ExplorationPlanningError(
                f"Claude final grounding exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        self._record_successful_turn(
            stage="final_grounding",
            stdout=completed.stdout,
        )
        payload_normalizations: list[str] = []
        try:
            grounded_payload, payload_normalizations = (
                _normalize_final_grounding_payload(
                    _json_from_claude_text(completed.stdout)
                )
            )
            proposal = validate_exploration_payload(
                grounded_payload,
                allowed_skill_names=self.skill_names,
            )
            measurement = GarmentGrounding(
                root / "workspace" / "perception_views"
            ).lookup_reference(selected["camera"], selected["reference_id"])
            targets = grasp_targets_from_actions(proposal.actions)
            if not targets:
                raise ExplorationPlanningError(
                    "final grounded proposal has no move immediately before close_gripper"
                )
            expected_xy = np.asarray(
                measurement["base_xyz_mm"][:2], dtype=np.float64
            )
            actual_xy = np.asarray(
                [targets[0]["x"], targets[0]["y"]], dtype=np.float64
            )
            grounding_error_mm = float(np.linalg.norm(actual_xy - expected_xy))
            fold_step = _fold_sleeve_step_from_objective(objective)
            grasp_anchor_offset_validation: dict[str, Any]
            if grounding_error_mm <= _EXACT_REFERENCE_GRASP_TOLERANCE_MM:
                grasp_anchor_offset_validation = {
                    "mode": "exact_anchor",
                    "step": fold_step,
                    "anchor_reference_id": measurement.get("reference_id"),
                    "anchor_base_xy_mm": expected_xy.tolist(),
                    "closure_base_xy_mm": actual_xy.tolist(),
                    "offset_xy_mm": (actual_xy - expected_xy).tolist(),
                    "offset_distance_mm": grounding_error_mm,
                    "max_offset_mm": _EXACT_REFERENCE_GRASP_TOLERANCE_MM,
                }
            elif fold_step is not None:
                try:
                    grasp_anchor_offset_validation = (
                        _validate_fold_grasp_anchor_offset(
                            session,
                            measurement,
                            actual_xy,
                            step=fold_step,
                        )
                    )
                except ValueError as exc:
                    raise ExplorationPlanningError(
                        "final fold grasp offset from the selected Rxxx anchor is invalid: "
                        f"selected={selected['camera']}/{selected['reference_id']} "
                        f"expected_xy={expected_xy.tolist()} actual_xy={actual_xy.tolist()} "
                        f"error={grounding_error_mm:.1f} mm; {exc}"
                    ) from exc
            else:
                raise ExplorationPlanningError(
                    "final grasp XY does not use the visually selected Rxxx measurement: "
                    f"selected={selected['camera']}/{selected['reference_id']} "
                    f"expected_xy={expected_xy.tolist()} actual_xy={actual_xy.tolist()} "
                    f"error={grounding_error_mm:.1f} mm; ordinary tasks allow at most "
                    f"{_EXACT_REFERENCE_GRASP_TOLERANCE_MM:.1f} mm"
                )
            recovery_validation = (
                validate_garment_recovery_actions(
                    proposal.actions, workspace_recovery
                )
                if workspace_recovery is not None
                else None
            )
        except BaseException as exc:
            self.planner._save_invocation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "duration_s": duration_s,
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                    "stage": "final_grounding",
                    "selected_reference": dict(selected),
                    "context_bundle": context_bundle,
                    "command_diagnostics": command_diagnostics,
                    "payload_normalizations": payload_normalizations,
                },
                failed=True,
            )
            raise
        result = ClaudeExplorationResult(
            prompt=prompt,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            created_at=_now(),
            proposal=proposal,
        )
        payload = {
            "prompt": result.prompt,
            "command": list(result.command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "created_at": result.created_at,
            "duration_s": duration_s,
            "stage": "final_grounding",
            "selected_reference": dict(selected),
            "context_bundle": context_bundle,
            "command_diagnostics": command_diagnostics,
            "payload_normalizations": payload_normalizations,
            "grounding_verification": {
                "measurement": measurement,
                "grasp_xy_error_mm": grounding_error_mm,
                "grasp_anchor_offset": grasp_anchor_offset_validation,
            },
            "proposal": proposal.as_dict(),
        }
        self.last_grounding_verification = dict(payload["grounding_verification"])
        if recovery_validation is not None:
            payload["garment_workspace_recovery_validation"] = recovery_validation
        self.planner._save_invocation_log(root, payload)
        return result

    def repair_last_grounding_plan(
        self,
        session: AgentSession,
        objective: str,
        *,
        feedback: str,
        history: Sequence[dict[str, Any]] | None = None,
    ) -> ExplorationProposal:
        """Recompile the last visual decision once after a host rejection.

        A failure discovered after visual selection usually requires a compact
        trajectory correction, not another image-reasoning pass. The selected
        Rxxx stays fixed for this repair. If it still fails, the pipeline asks
        for a fresh visual plan on the following attempt.
        """

        visual_result = self.last_visual_plan_result
        if visual_result is None:
            raise ExplorationPlanningError(
                "cannot repair final grounding without a previous visual decision"
            )
        repair_objective = (
            f"{objective}\n\n"
            "HOST VALIDATION CORRECTION. No physical command was sent. Keep the "
            "previous visual Rxxx decision fixed for this single compiler repair, "
            "and change only the grounded contact/trajectory fields needed to fix:\n"
            f"{feedback}"
        )
        started = time.monotonic()
        response = self._ground_final_plan(
            visual_result.decision,
            session,
            repair_objective,
            history=list(history or []),
        )
        duration_s = time.monotonic() - started
        self.last_plan_result = response
        self.last_plan_timing = {
            "visual_planning_s": 0.0,
            "visual_planning_attempts": 0,
            "visual_reselection_count": 0,
            "final_grounding_s": duration_s,
            "total_planning_s": duration_s,
            "repair_only": True,
        }
        return response.proposal

    def plan(
        self,
        image_paths: list[Path],
        session: AgentSession,
        objective: str,
        feedback: str | None = None,
        history: list[dict[str, Any]] | None = None,
        phase_callback: Callable[[str, str, float], None] | None = None,
        reference_policy: str = "uniform",
        workspace_recovery: GarmentWorkspaceRecovery | None = None,
    ) -> ExplorationProposal:
        previous_visual_result = self.last_visual_plan_result
        self.last_plan_result = None
        self.last_visual_plan_result = None
        self.last_plan_timing = {}
        self.last_rejected_visual_references = []
        self.last_reference_validation = None
        self.last_reference_candidate_report = None
        self.last_grounding_verification = None
        prompt_objective = objective
        if feedback:
            prompt_objective += (
                "\n\nThe previous candidate was rejected before physical execution. "
                "Use the failure report below to generate a materially different, "
                "more conservative proposal. Do not repeat the rejected pose or "
                "assume that local XYZ bounds imply IK reachability.\n"
                f"Failure report:\n{feedback}"
            )
        history_items = list(history or [])[-8:]
        planning_mode, planning_mode_instruction = _planning_mode_from_history(history_items)
        if not is_default_exploration_objective(objective):
            if planning_mode == "VALIDATED_EXPANSION":
                planning_mode_instruction = (
                    "MODE = VALIDATED_TASK_PROGRESS: preserve the validated named target and "
                    "make a meaningful next move toward the user objective. Do not substitute "
                    "a generic outward-spreading action."
                )
            else:
                planning_mode_instruction = (
                    "MODE = TASK_DIRECTED_EXPLORATION: use only the smallest reversible action "
                    "needed to resolve a concrete uncertainty about the named target or its "
                    "safe manipulation, then continue toward the user objective. Do not spend "
                    "the iteration on an unrelated generic garment-opening probe."
                )
        if reference_policy == "uniform":
            reference_instruction = (
                "Rxxx markers are uniform coordinate references, not ranked candidates. "
                "Choose the single visible reference that best supports the next opening "
                "motion"
            )
        elif reference_policy == "molmo_confidence_filtered_keypoints":
            reference_instruction = (
                "Rxxx markers in the Molmo keypoint-reference overlays are the only "
                "allowed grasp references. Every shown marker passed the configured "
                "Molmo confidence threshold and calibrated-geometry gate. Do not select "
                "an unmarked pixel, a below-threshold/rejected keypoint, or a reference "
                "from an older uniform overlay. Choose one of the currently shown Rxxx "
                "keypoints"
            )
        else:
            raise ValueError(f"unknown reference_policy: {reference_policy}")
        if is_default_exploration_objective(objective):
            objective_instruction = (
                "The built-in fallback task is to make the garment as open and spread on the "
                "table as safely possible. Prefer a reference that supports separating overlap "
                "and a controlled laydown."
            )
        else:
            objective_instruction = (
                f"The sole user task objective is: {objective}\n"
                "Plan specifically for this objective. Do not replace it with a generic garment "
                "opening or spreading objective. If it names a garment part or region, select a "
                "reference that corresponds to that named target; do not treat an unrelated "
                "convenient fold as an acceptable substitute."
            )
        fold_sleeve_step = _fold_sleeve_step_from_objective(objective)
        fold_reference_instruction = ""
        if fold_sleeve_step is not None:
            candidate_report = self._fold_reference_candidate_report(
                session,
                objective,
            )
            self.last_reference_candidate_report = candidate_report
            _write_json(
                session.run_dir.resolve()
                / "workspace"
                / "perception_views"
                / f"camera_A_{fold_sleeve_step}_executable_references.json",
                candidate_report,
            )
            executable_reference_ids = list(
                candidate_report.get("executable_reference_ids", [])
                if isinstance(candidate_report, Mapping)
                else []
            )
            if not executable_reference_ids:
                rejected = list(candidate_report.get("rejected", []))
                reason_counts: dict[str, int] = {}
                for item in rejected:
                    reason = str(item.get("reason", "unknown rejection"))
                    if "safe upper bound" in reason or "safe lower bound" in reason or "left/right boundaries" in reason:
                        key = "workspace"
                    elif "RGB edge contrast" in reason:
                        key = "rgb_edge_contrast"
                    elif "free-edge band" in reason:
                        key = "free_edge_band"
                    else:
                        key = "other"
                    reason_counts[key] = reason_counts.get(key, 0) + 1
                payload = {
                    "stage": "reference_candidate_prevalidation",
                    "created_at": _now(),
                    "failed": True,
                    "candidate_report": candidate_report,
                    "reason_counts": reason_counts,
                }
                self._save_visual_log(session.run_dir.resolve(), payload, failed=True)
                raise ReferenceReselectionExhaustedError(
                    "deterministic prevalidation found no executable Camera-A "
                    f"Rxxx for {fold_sleeve_step}; Claude was not called; "
                    f"rejection_counts={reason_counts}"
                )
            allowed_text = ", ".join(executable_reference_ids)
            fold_reference_instruction = (
                "\nFor this sleeve step, image-left/image-right refer only to the "
                "clockwise-90 canonical upright Camera-A RGB and matching upright Rxxx "
                "overlay: image-left is the viewer's left side (smaller upright x), "
                "image-right is the viewer's right side (larger upright x); never use "
                "wearer-left/wearer-right or mirror the view. Select Camera A only. "
                "The cyan Rxxx markers are the original "
                "uniform calibrated references spread across the entire visible garment; "
                "they are not pre-ranked contact candidates. Select exactly "
                "one Rxxx that is visibly shown on the requested sleeve region and "
                "never name a hidden or unshown ID. "
                "A magenta MOLMO marker may be present as a fallible semantic sleeve-region "
                "hypothesis. Never grasp the magenta point directly. When it agrees with "
                "the visible sleeve, use it only to focus attention and compare nearby cyan "
                "Rxxx; when it visibly disagrees, trust the RGB garment topology instead. "
                "The chosen marker must visibly lie on the requested outer sleeve region. "
                "Torso/print interior markers and opposite-side markers will be rejected "
                "before Stage 2. Compare the full visible grid yourself and use the saved "
                "physical outcomes to decide whether the next hypothesis should alter contact "
                "location, jaw alignment, entry path, or height. The host deliberately does "
                "not reveal a preferred edge/interior/seam/wrinkle answer. "
                "The host has already applied the current workspace, outer-side, "
                "sleeve-height, and region-membership gates to every visible "
                "uniform marker. Workspace candidate gating uses the maximum possible "
                "Y extension at |yaw|=90 because Stage 1 has not chosen a yaw yet; "
                "Stage 2 and the controller revalidate the actual yaw at every waypoint. "
                "The full cyan grid remains visible for context, but the "
                f"only references executable for this step are: {allowed_text}. Select "
                "exactly one ID from that list; every other visible Rxxx is deterministically "
                "invalid and must not be selected."
            )
        visual_prompt = (
            "Observe only the current garment shown in the supplied Camera A/B "
            "images. "
            f"{objective_instruction} Use the visible RGB, garment boundary, "
            "height-above-table, height-gradient/occlusion, and Rxxx overlay evidence "
            "without assuming a garment category beyond the named observations. "
            f"{reference_instruction} and state the intended transport direction and "
            "approximate useful "
            "distance. Apply the following mode exactly:\n"
            f"{planning_mode_instruction}\n"
            "This zero-shot visual stage has no prior coordinates or action values. Do not "
            "invent them; use prior evaluation only to decide whether this is a probe or an "
            f"expansion.{fold_reference_instruction}\n\n"
            "Approved procedural skill library:\n"
            f"{self.skill_guidance or 'No dynamic skill updates are active.'}"
        )
        visual_prompt += (
            "\n\nYAW-DEPENDENT Y WORKSPACE CONTEXT (host-enforced): action yaw is a "
            "relative delta from the calibrated Home TCP orientation. At yaw=0 "
            "the configured Y safe zone is unchanged. The TCP-center allowance "
            "is `0.5 * gripper_width_mm * abs(sin(radians(yaw)))`; with the current "
            f"configured width {session.robot_config.gripper_width_mm:g} mm, the "
            f"maximum at +/-90 degrees is {session.robot_config.y_workspace_extension_mm(90.0):g} mm. "
            "This can make an edge reference potentially executable, but the final "
            "waypoint's actual yaw is revalidated by the host and controller IK; it "
            "does not relax X/Z or authorize moving the whole gripper through the boundary. "
            "When extra Y travel is needed, rotate while still inside the original yaw=0 "
            "Y envelope, then move outward; crossing that envelope before rotating is invalid."
        )
        support_context = _support_layer_context(session)
        visual_prompt += (
            "\n\nPHYSICAL SUPPORT-LAYER CONTEXT (configuration, not an RGB inference):\n"
            f"{json.dumps(support_context, ensure_ascii=False, indent=2)}\n"
            "The garment is placed over the configured support layer when the context "
            "marks it active/confirmed. A deeper compressive bite is then physically "
            "permitted up to press_mm (and never beyond max_compression_mm), while the "
            "host still computes and validates the final TCP Z. Do not assume a hard "
            "table merely because the RGB view shows a tabletop; do not invent a deeper "
            "Z value yourself. Use this context when describing whether a thin sleeve "
            "needs a real compressive engagement, but keep the target choice grounded in RGB."
        )
        if workspace_recovery is not None and workspace_recovery.required:
            recovery_payload = workspace_recovery.as_dict()
            visual_prompt += (
                "\n\nWORKSPACE RECOVERY OVERRIDE. The fused robust garment center is "
                "outside its configured operating rectangle. This iteration must move "
                "the garment back inward; do not choose an opening/probing action. Select "
                "one visible, executable current Rxxx cloth reference that can support "
                "the inward transport. Stage 2 will receive and enforce the exact "
                "Base-frame recovery vector. Recovery request:\n"
                f"{json.dumps(recovery_payload, ensure_ascii=False, indent=2)}"
            )
        rejected_keys: set[tuple[str, str]] = set()
        if feedback and previous_visual_result is not None:
            previous = previous_visual_result.decision.selected_reference
            previous_key = (previous["camera"], previous["reference_id"])
            rejected_keys.add(previous_key)
            self.last_rejected_visual_references.append(
                {
                    "attempt": 0,
                    "camera": previous_key[0],
                    "reference_id": previous_key[1],
                    "reason": "previous grounded proposal failed pre-execution validation",
                    "failure_feedback": feedback,
                    "visual_plan": previous_visual_result.decision.as_dict(),
                }
            )

        visual_started = time.monotonic()
        visual_result: ClaudeVisualPlanResult | None = None
        max_visual_attempts = self.max_reference_reselections + 1
        if fold_sleeve_step is not None:
            # A sleeve candidate has already passed deterministic host
            # prevalidation.  If Claude still names a forbidden ID, allow one
            # compact correction rather than spending three long visual calls.
            max_visual_attempts = min(max_visual_attempts, 2)
        for visual_attempt in range(1, max_visual_attempts + 1):
            attempt_prompt = visual_prompt
            if rejected_keys:
                excluded = ", ".join(
                    f"{camera}/{reference_id}"
                    for camera, reference_id in sorted(rejected_keys)
                )
                attempt_prompt += (
                    "\n\nSTAGE 1 RESELECTION. Deterministic robot validation rejected "
                    f"these references as unexecutable: {excluded}. Do not select any "
                    "of them again. Re-inspect the same current images and choose one "
                    "different visually justified Camera/Rxxx. No rejected-point "
                    "coordinates or robot action details are supplied to this visual stage."
                )
            if phase_callback is not None:
                phase_callback("visual_planning", "started", float(self.timeout_s))
            attempt_started = time.monotonic()
            try:
                candidate = self._visual_plan(
                    image_paths,
                    attempt_prompt,
                    session.run_dir,
                )
            except BaseException:
                visual_failed_duration = time.monotonic() - visual_started
                self.last_plan_timing["visual_planning_s"] = visual_failed_duration
                self.last_plan_timing["visual_planning_attempts"] = float(
                    visual_attempt
                )
                self.last_plan_timing["visual_reselection_count"] = float(
                    len(self.last_rejected_visual_references)
                )
                self.last_plan_timing["total_planning_s"] = visual_failed_duration
                if phase_callback is not None:
                    phase_callback(
                        "visual_planning",
                        "failed",
                        visual_failed_duration,
                    )
                raise

            self.last_visual_plan_result = candidate
            selected = candidate.decision.selected_reference
            selected_key = (selected["camera"], selected["reference_id"])
            rejection: SelectedReferenceNotExecutableError | None = None
            if selected_key in rejected_keys:
                rejection = SelectedReferenceNotExecutableError(
                    selected_key[0],
                    selected_key[1],
                    "Stage 1 selected a reference that was already rejected",
                )
            else:
                try:
                    self.last_reference_validation = self._validate_reference_for_stage2(
                        candidate.decision,
                        session,
                        objective,
                    )
                except SelectedReferenceNotExecutableError as exc:
                    rejection = exc

            if rejection is None:
                visual_result = candidate
                break

            rejected_keys.add(selected_key)
            rejection_record = {
                "attempt": visual_attempt,
                "camera": selected_key[0],
                "reference_id": selected_key[1],
                "reason": rejection.reason,
                "measurement": rejection.measurement,
                "visual_plan": candidate.decision.as_dict(),
            }
            self.last_rejected_visual_references.append(rejection_record)
            self._save_visual_log(
                session.run_dir.resolve(),
                {
                    "stage": "reference_executability_validation",
                    "created_at": _now(),
                    **rejection_record,
                },
                failed=True,
            )
            attempt_duration = time.monotonic() - attempt_started
            if visual_attempt >= max_visual_attempts:
                visual_failed_duration = time.monotonic() - visual_started
                self.last_plan_timing["visual_planning_s"] = visual_failed_duration
                self.last_plan_timing["visual_planning_attempts"] = float(
                    visual_attempt
                )
                self.last_plan_timing["visual_reselection_count"] = float(
                    len(self.last_rejected_visual_references)
                )
                self.last_plan_timing["total_planning_s"] = visual_failed_duration
                if phase_callback is not None:
                    phase_callback(
                        "visual_planning",
                        "failed",
                        visual_failed_duration,
                    )
                rejected = ", ".join(
                    f"{camera}/{reference_id}"
                    for camera, reference_id in sorted(rejected_keys)
                )
                raise ReferenceReselectionExhaustedError(
                    "Stage 1 could not select an executable reference after "
                    f"{visual_attempt} attempt(s); rejected={rejected}"
                ) from rejection
            if phase_callback is not None:
                phase_callback(
                    "visual_planning",
                    "reselecting",
                    attempt_duration,
                )

        if visual_result is None:
            raise ReferenceReselectionExhaustedError(
                "Stage 1 ended without an executable selected reference"
            )
        visual_duration = time.monotonic() - visual_started
        self.last_plan_timing["visual_planning_s"] = visual_duration
        self.last_plan_timing["visual_planning_attempts"] = float(visual_attempt)
        self.last_plan_timing["visual_reselection_count"] = float(
            len(self.last_rejected_visual_references)
        )
        if phase_callback is not None:
            phase_callback("visual_planning", "completed", visual_duration)

        grounding_started = time.monotonic()
        if phase_callback is not None:
            phase_callback(
                "final_grounding",
                "started",
                float(self.grounding_timeout_s),
            )
        try:
            ground_kwargs: dict[str, Any] = {}
            if workspace_recovery is not None:
                ground_kwargs["workspace_recovery"] = workspace_recovery
            response = self._ground_final_plan(
                visual_result.decision,
                session,
                prompt_objective,
                history_items,
                **ground_kwargs,
            )
        except BaseException:
            grounding_failed_duration = time.monotonic() - grounding_started
            self.last_plan_timing["final_grounding_s"] = grounding_failed_duration
            self.last_plan_timing["total_planning_s"] = (
                visual_duration + grounding_failed_duration
            )
            if phase_callback is not None:
                phase_callback(
                    "final_grounding",
                    "failed",
                    grounding_failed_duration,
                )
            raise
        grounding_duration = time.monotonic() - grounding_started
        self.last_plan_timing["final_grounding_s"] = grounding_duration
        self.last_plan_timing["total_planning_s"] = (
            visual_duration + grounding_duration
        )
        if phase_callback is not None:
            phase_callback("final_grounding", "completed", grounding_duration)
        self.last_plan_result = response
        return response.proposal

    def evaluate(
        self,
        before_images: list[Path],
        after_images: list[Path],
        *,
        proposal: ExplorationProposal,
        objective: str | None = None,
        run_dir: Path,
        rollout_recording_dir: Path | None = None,
        skill_guidance: str | None = None,
        workspace_recovery: GarmentWorkspaceRecovery | None = None,
        hold_checkpoint: dict[str, Any] | None = None,
        gripper_telemetry: Mapping[str, Any] | None = None,
        observer_images: Sequence[Path] = (),
    ) -> ExplorationEvaluation:
        self.last_evaluation_result = None
        root = run_dir.resolve()
        video_evidence_images: list[Path] = []
        video_references: list[Path] = []
        video_evidence_errors: list[str] = []
        if rollout_recording_dir is not None:
            try:
                (
                    video_evidence_images,
                    video_references,
                    video_evidence_errors,
                ) = prepare_rollout_video_evidence(rollout_recording_dir)
            except Exception as exc:
                video_evidence_errors.append(f"{type(exc).__name__}: {exc}")
        if is_default_exploration_objective(objective):
            task_evaluation_instruction = (
                "Judge task progress toward making the garment open and spread: more visible "
                "area, less overlap, lower relief, and a useful tabletop laydown."
            )
        else:
            task_evaluation_instruction = (
                f"Judge progress toward the exact user objective: {objective}. "
                "Treat that objective as authoritative. Do not declare progress merely because "
                "the garment became more open or spread unless that change directly advances the "
                "named task. Use the target-selection, target-structure, transport, and laydown "
                "fields to record whether the named target was actually manipulated and reached "
                "the requested state."
            )
        image_lines = [
            "The visual evidence is intentionally restricted to the following labelled files.",
            "Read only these files. Do not search for or read any other JSON, NumPy, image, or video file.",
            "Before RGB/depth images:",
        ]
        image_lines.extend(f"- {path.resolve()}" for path in before_images)
        image_lines.append("After RGB/depth images:")
        image_lines.extend(f"- {path.resolve()}" for path in after_images)
        if video_evidence_images:
            image_lines.append(
                "Rollout video contact sheets (chronological left-to-right, top-to-bottom; these are the only video evidence):"
            )
            image_lines.extend(f"- {path.resolve()}" for path in video_evidence_images)
        elif rollout_recording_dir is not None:
            image_lines.append(
                "Rollout video contact sheets: unavailable; mark temporal acquisition/transport/laydown UNKNOWN."
            )
        if video_evidence_errors:
            image_lines.append("Video evidence extraction caveats:")
            image_lines.extend(f"- {message}" for message in video_evidence_errors)
        if isinstance(gripper_telemetry, Mapping):
            image_lines.append("Host xArm gripper telemetry (action-boundary samples):")
            image_lines.append(
                json.dumps(dict(gripper_telemetry), ensure_ascii=False, indent=2)
            )
        observer_paths = [
            Path(path).resolve()
            for path in observer_images
            if Path(path).expanduser().is_file()
        ]
        if observer_paths:
            image_lines.append(
                "Uncalibrated Camera C observer RGB images (visual evidence only; no geometry):"
            )
            hold_paths = [path for path in observer_paths if "hold_check" in path.name.lower()]
            other_paths = [path for path in observer_paths if path not in hold_paths]
            if hold_paths:
                image_lines.append(
                    "Camera C lift hold-check still(s), captured immediately after the first "
                    "post-close lift action (primary evidence for short-term acquisition):"
                )
                image_lines.extend(f"- {path}" for path in hold_paths)
            if other_paths:
                image_lines.append("Other Camera C observer stills:")
                image_lines.extend(f"- {path}" for path in other_paths)
        evaluation_mode_instruction = (
            ""
            if workspace_recovery is None or not workspace_recovery.required
            else (
                "This rollout was a safety-priority WORKSPACE_RECOVERY action, not a "
                "garment-opening experiment. Judge transport primarily by whether the "
                "visible garment moved along the requested inward Base-frame direction; "
                "do not penalize reduced opening progress when it was necessary to "
                "recenter the garment. A fresh fused perception is required to confirm "
                "re-entry, so do not recommend stopping merely because no further "
                "opening action is obvious in these after images. Recovery request: "
                f"{json.dumps(workspace_recovery.as_dict(), ensure_ascii=False)}\n\n"
            )
        )
        checkpoint_instruction = ""
        if isinstance(hold_checkpoint, dict):
            checkpoint_summary = {
                key: hold_checkpoint.get(key)
                for key in (
                    "status",
                    "classification",
                    "confidence",
                    "evidence",
                    "reason",
                    "continue_transport",
                    "runtime_decision",
                    "executed_branch",
                )
                if key in hold_checkpoint
            }
            checkpoint_instruction = (
                "The runtime used an online post-grasp hold checkpoint. This record is "
                "authoritative for which action branch was actually executed: "
                f"{json.dumps(checkpoint_summary, ensure_ascii=False)}\n"
            )
            if hold_checkpoint.get("executed_branch") == "ABORT_RELEASE" or not bool(
                hold_checkpoint.get("continue_transport")
            ):
                checkpoint_instruction += (
                    "The planned continuation transport was NOT executed. Do not label its "
                    "direction or distance as failed and do not learn transport changes from it. "
                    "Judge acquisition/target support from the hold and mark unexecuted transport "
                    "UNKNOWN; evaluate only the local descent/release that actually occurred.\n\n"
                )
            else:
                checkpoint_instruction += (
                    "The checkpoint authorized the continuation from fresh full-resolution Camera "
                    "A/B evidence. Treat grasp acquisition and target-structure support at the hold "
                    "as established; do not later rewrite that successful hold as acquisition failure "
                    "because a uniformly sampled contact sheet is smaller or ambiguous. Only record "
                    "later slippage as a subsequent transport failure. Evaluate the executed transport "
                    "and laydown normally from the visual evidence.\n\n"
                )
        prompt = (
            evaluation_mode_instruction
            + checkpoint_instruction
            + "Compare the before and after garment images after one robot handling action. "
            "Evaluate the action stage by stage instead of collapsing it into one useful flag. "
            "Use only directly visible evidence from the labelled RGB/depth files and the "
            "rollout video contact sheets listed at the end. Do not use or search for any "
            "other files, including perception JSON, metrics JSON, NumPy arrays, heatmaps, "
            "overlays, or MP4 files. The before/after views are static. When "
            "chronological rollout contact sheets are supplied, use them to assess acquisition, "
            "transport, and laydown over time; otherwise mark those stages UNKNOWN when jaw "
            "motion or layer identity cannot actually be established. Never infer success "
            "only because the commanded action should have produced it. "
            "When a height ridge or sharp relief is present, distinguish two hypotheses: (a) a "
            "separable overlapping layer, which should form a visible tent or hanging patch while "
            "far garment landmarks remain on the table, versus (b) a rolled wrinkle or local curl, "
            "which may only become taller, narrower, or more tightly curled when pulled. Height or "
            "gradient alone does not support target_structure_acquired. If the hold frame shows no "
            "independent hanging material, or shows the ridge tightening without footprint "
            "extension, mark the target structure contradicted/unknown and treat the lateral pull "
            "as untested. When a compression probe is visible, compare the selected peak with "
            "the immediately adjacent cloth during closure: if the peak-to-neighbour height "
            "difference clearly decreases, but no independent hanging patch appears and distant "
            "landmarks stay put, classify the event as `COMPRESSIBLE_SINGLE_PEAK`/rolled wrinkle, "
            "lower graspability confidence, and do not credit target-layer acquisition. If far "
            "landmarks move with the gripper, classify it as whole-garment drag rather than "
            "successful ply isolation. "
            "The host xArm gripper telemetry is an independent mechanical signal. A sample "
            "with state=grasp (status low bits=2) after close and during/after lift is positive "
            "evidence that the gripper controller detected contact. It is not proof that the "
            "intended sleeve or garment ply was acquired, and it can be triggered by the table "
            "or an obstacle. If the gripper is visually occluded but telemetry reports grasp, "
            "do not call acquisition a visual failure solely because no hanging cloth is visible; "
            "use SUCCESS when the mechanical signal is consistent with the rollout, or UNKNOWN "
            "when visual and mechanical evidence conflict. Conversely, unavailable telemetry or "
            "state=stop must not be treated as proof of an empty grasp. "
            "Camera C is an uncalibrated side observer: use its RGB image/video only to resolve "
            "gripper occlusion and temporal cloth motion. Never derive pixels, depth, or robot "
            "coordinates from Camera C. A still labelled as a Camera C lift hold-check was "
            "captured immediately after the first post-close lift; inspect it as the primary "
            "acquisition witness. If that still shows an independent sleeve patch rising with "
            "the gripper, mark grasp_acquisition SUCCESS even when the later release leaves the "
            "final after image unchanged. Do not use an unchanged after image to negate a "
            "reversible short hold. "
            f"{task_evaluation_instruction} Do not invent numeric measurements. For "
            "visible_area_delta, overlap_delta, and relief_delta, use INCREASED, DECREASED, "
            "UNCHANGED, or UNKNOWN unless an exact numeric measurement is explicitly supplied.\n\n"
            "The keep/change decision is mandatory and causal. Put every parameter or strategy "
            "that the evidence supports under keep. Put only the earliest failing/unsupported "
            "choice and necessary downstream choices under change. Do not change a supported "
            "grasp anchor or grasp depth merely because transport failed. For example, when "
            "acquisition and target-layer motion are supported but transport is insufficient, "
            "keep grasp_anchor and grasp_depth, and change pull_direction and/or pull_distance. "
            "When acquisition fails, treat contact location, jaw alignment, pre-close entry "
            "path, closure geometry, and Z as competing causal hypotheses. Do not recommend "
            "another height-only retry merely because the cloth is thin unless the visible "
            "evidence specifically isolates height as the cause. After repeated empty closes "
            "with the same contact geometry, next_experiment.change must include at least one "
            "non-height contact dimension. Do not name a privileged solution in advance; infer "
            "the next hypothesis from the observed failures. "
            "The change list must be non-empty when another safe grounded experiment exists. "
            "Use an empty change list only when the garment is already as open as this setup can "
            "reasonably achieve, or continuing is unsafe, visually ungrounded, or blocked by a "
            "hard physical/infrastructure condition.\n\n"
            "Optionally propose one skill_update only when the completed before/after evidence "
            "supports reusable procedural knowledge. Use operation=create for a genuinely new "
            "pattern, or operation=modify for a real change to an existing skill. The guidance "
            "must remain high-level and must not contain coordinates, SDK calls, joint angles, "
            "or executable code. The skill name must be lowercase kebab-case with hyphens "
            "only, for example `grasp-acquisition-check`; never use underscores, spaces, "
            "or CamelCase. A proposal is not activated automatically: an independent "
            "reviewer checks safety, evidence, and duplicate skills before activation. Omit "
            "skill_update when the result only changes this one experiment.\n\n"
            "Approved procedural skill library for duplicate checking:\n"
            f"{skill_guidance or 'No dynamic skill updates are active.'}\n\n"
            "Return exactly one JSON object with exactly this structure and no markdown:\n"
            "{\n"
            '  "target_selection": {"status": "SUPPORTED|CONTRADICTED|UNKNOWN", '
            '"confidence": 0.0, "evidence": ["direct visible fact"]},\n'
            '  "grasp_acquisition": {"status": "SUCCESS|FAILURE|UNKNOWN", '
            '"confidence": 0.0, "evidence": ["direct visible fact or why unknown"]},\n'
            '  "target_structure_acquired": {"status": "SUPPORTED|CONTRADICTED|UNKNOWN", '
            '"confidence": 0.0, "evidence": ["direct visible fact or why unknown"]},\n'
            '  "transport": {"status": "GOOD|BAD_DIRECTION|INSUFFICIENT|OVERPULL|UNKNOWN", '
            '"confidence": 0.0, "evidence": ["direct visible fact or why unknown"]},\n'
            '  "laydown": {"status": "SUCCESS|FAILURE|NOT_REACHED|UNKNOWN", '
            '"confidence": 0.0, "evidence": ["direct visible fact or why unknown"]},\n'
            '  "task_progress": {"status": "IMPROVED|NEUTRAL|REGRESSED", '
            '"confidence": 0.0, "metrics": {"visible_area_delta": "INCREASED|DECREASED|UNCHANGED|UNKNOWN", '
            '"overlap_delta": "INCREASED|DECREASED|UNCHANGED|UNKNOWN", '
            '"relief_delta": "INCREASED|DECREASED|UNCHANGED|UNKNOWN", '
            '"boundary_change": "direct concise observation or UNKNOWN"}},\n'
            '  "earliest_failure_stage": "ACQUISITION|TARGET|TRANSPORT|LAYDOWN|NONE|UNKNOWN",\n'
            '  "next_experiment": {"keep": ["validated choice"], '
            '"change": ["choice to revise"], "reason": "causal evidence-based explanation"}\n'
            "}\n\n"
            f"Previous selected grasp: {json.dumps(proposal.selected_grasp, ensure_ascii=False)}\n"
            f"Previous proposal strategy: {proposal.reveal_strategy}\n"
            f"Previous invoked skills: {json.dumps(list(proposal.skill_invocations), ensure_ascii=False)}\n"
            f"Previous expected observation: {proposal.expected_observation}\n"
            f"Previous action program: {json.dumps(proposal.actions, ensure_ascii=False)}\n\n"
            + "\n".join(image_lines)
        )
        binary = self.planner.binary
        if Path(binary).name == binary:
            import shutil

            binary = shutil.which(binary)
        if binary is None:
            raise AutoExplorationError(f"Claude CLI not found: {self.planner.binary}")
        command = [
            binary,
            "--print",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(AUTO_EVALUATION_JSON_SCHEMA, separators=(",", ":")),
            "--permission-mode",
            "plan",
            "--allowedTools",
            "Read",
            "--tools",
            "Read",
            "--add-dir",
            str(root),
            "--safe-mode",
            "--system-prompt",
            (
                "You are a cautious visual evaluator for a robotics garment run. "
                "Read only the supplied images. Your final response is machine-validated "
                "against the supplied JSON Schema; return only the structured evaluation. "
                "Do not edit files, execute commands, or control a robot."
            ),
        ]
        command = self._prepare_command(command, stage="evaluation")
        import subprocess

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
            if self.persistent_session is not None:
                self.persistent_session.rollover(
                    reason="claude_timeout", stage="evaluation"
                )
            self._save_evaluation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": None,
                    "stdout": getattr(exc, "stdout", "") or "",
                    "stderr": getattr(exc, "stderr", "") or "",
                    "error": (
                        f"ExplorationTimeoutError: Claude evaluation timed out "
                        f"after {self.timeout_s} seconds"
                    ),
                    "created_at": _now(),
                },
                failed=True,
            )
            raise ExplorationTimeoutError(
                f"Claude evaluation timed out after {self.timeout_s} seconds"
            ) from exc
        except OSError as exc:
            self._save_evaluation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise AutoExplorationError(
                f"Claude evaluation invocation failed: {exc}"
            ) from exc
        if completed.returncode != 0:
            if (
                self.persistent_session is not None
                and self.persistent_session.is_session_conflict_error(
                    completed.stdout, completed.stderr
                )
            ):
                self.persistent_session.rollover(
                    reason="claude_session_conflict", stage="evaluation"
                )
            self._save_evaluation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "error": "non-zero Claude return code",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise AutoExplorationError(
                f"Claude evaluation exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        self._record_successful_turn(stage="evaluation", stdout=completed.stdout)
        try:
            evaluation = validate_evaluation_payload(_json_from_claude_text(completed.stdout))
        except BaseException as exc:
            self._save_evaluation_log(
                root,
                {
                    "prompt": prompt,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": _now(),
                },
                failed=True,
            )
            raise
        result = ClaudeEvaluationResult(
            prompt=prompt,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            created_at=_now(),
            evaluation=evaluation,
            evidence_images=tuple(
                str(path.resolve())
                for path in [
                    *before_images,
                    *after_images,
                    *video_evidence_images,
                    *observer_paths,
                ]
            ),
            video_references=tuple(str(path.resolve()) for path in video_references),
            video_evidence_errors=tuple(video_evidence_errors),
        )
        self.last_evaluation_result = result
        self._save_evaluation_log(root, result.as_dict())
        return evaluation

    def evaluate_acquisition_probe(
        self,
        before_images: Sequence[Path],
        after_images: Sequence[Path],
        *,
        proposal: ExplorationProposal,
        run_dir: Path,
        rollout_recording_dir: Path | None = None,
        rollout_evidence_images: Sequence[Path] = (),
        rollout_video_references: Sequence[Path] = (),
        rollout_evidence_errors: Sequence[str] = (),
        gripper_telemetry: Mapping[str, Any] | None = None,
        observer_images: Sequence[Path] = (),
    ) -> ExplorationEvaluation:
        """Judge only grasp acquisition from compact RGB/video evidence.

        Acquisition probes intentionally reverse and release at the original
        contact, so transport, laydown, and fold-state supervision are not
        useful questions.  This compact contract keeps the expensive model
        turn focused on the physical uncertainty that the probe was designed
        to resolve.
        """

        self.last_evaluation_result = None
        root = Path(run_dir).resolve()

        def select_rgb(paths: Sequence[Path]) -> list[Path]:
            selected: list[Path] = []
            for camera_name in ("camera_0_A.png", "camera_1_B.png"):
                matches = [
                    Path(path).resolve()
                    for path in paths
                    if Path(path).name == camera_name and Path(path).is_file()
                ]
                if matches:
                    # Iteration-local before_raw/after_raw images are the
                    # direct captures; prefer them over duplicate workspace
                    # copies when both are present.
                    matches.sort(
                        key=lambda path: (
                            0
                            if path.parent.name in {"before_raw", "after_raw"}
                            else 1,
                            str(path),
                        )
                    )
                    selected.append(matches[0])
            return selected

        before_rgb = select_rgb(before_images)
        after_rgb = select_rgb(after_images)
        if not before_rgb or not after_rgb:
            raise AutoExplorationError(
                "compact acquisition evaluation requires Camera-A/B RGB evidence"
            )
        video_images = [Path(path).resolve() for path in rollout_evidence_images]
        video_references = [
            Path(path).resolve() for path in rollout_video_references
        ]
        video_errors = [str(item) for item in rollout_evidence_errors]
        if not video_images and rollout_recording_dir is not None:
            try:
                video_images, video_references, video_errors = (
                    prepare_rollout_video_evidence(rollout_recording_dir)
                )
            except Exception as exc:
                video_errors.append(f"{type(exc).__name__}: {exc}")

        evidence_lines = ["Before RGB:"]
        evidence_lines.extend(f"- {path}" for path in before_rgb)
        evidence_lines.append("After RGB:")
        evidence_lines.extend(f"- {path}" for path in after_rgb)
        if video_images:
            evidence_lines.append(
                "Chronological rollout contact sheets (left-to-right, top-to-bottom):"
            )
            evidence_lines.extend(f"- {path}" for path in video_images)
        else:
            evidence_lines.append(
                "Rollout contact sheet unavailable; use UNKNOWN where the static RGB cannot decide."
            )
        if video_errors:
            evidence_lines.append("Video caveats:")
            evidence_lines.extend(f"- {item}" for item in video_errors)
        if isinstance(gripper_telemetry, Mapping):
            evidence_lines.append("Host xArm gripper telemetry (action-boundary samples):")
            evidence_lines.append(
                json.dumps(dict(gripper_telemetry), ensure_ascii=False, indent=2)
            )
        observer_paths = [
            Path(path).resolve()
            for path in observer_images
            if Path(path).expanduser().is_file()
        ]
        if observer_paths:
            evidence_lines.append(
                "Uncalibrated Camera C observer RGB images (visual evidence only; no geometry):"
            )
            hold_paths = [path for path in observer_paths if "hold_check" in path.name.lower()]
            other_paths = [path for path in observer_paths if path not in hold_paths]
            if hold_paths:
                evidence_lines.append(
                    "Camera C lift hold-check still(s), captured immediately after the first "
                    "post-close lift action (primary acquisition evidence):"
                )
                evidence_lines.extend(f"- {path}" for path in hold_paths)
            if other_paths:
                evidence_lines.append("Other Camera C observer stills:")
                evidence_lines.extend(f"- {path}" for path in other_paths)

        prompt = (
            "ACQUISITION-PROBE EVALUATION ONLY. This robot action closed the gripper, "
            "made two short lift checkpoints, then reversed to the original contact and "
            "released. Inspect only the labelled RGB files and chronological rollout "
            "contact sheets below. Decide whether cloth visibly followed and remained "
            "supported by the gripper during the lift, and whether the intended local "
            "garment structure rather than empty space or whole-garment drag was acquired. "
            "Do not evaluate fold transport, laydown, final garment shape, or overall task "
            "progress. Never infer success from the command. The next_experiment decision "
            "must be causal: after an empty close, consider contact XY, jaw alignment, "
            "pre-close entry path, closure geometry, and Z as competing hypotheses; after "
            "repeated similar failures, include a non-height contact change. Do not name a "
            "privileged grasp solution without visible evidence. Also report whether the "
            "garment's persistent after-state visibly changed; use UNKNOWN when the RGB is "
            "ambiguous. Return exactly the compact "
            "JSON schema.\n\n"
            "The host xArm telemetry is an independent mechanical signal. A sample with "
            "state=grasp (status low bits=2) after close and during/after lift is positive "
            "evidence that the gripper controller detected an object. It is not proof that "
            "the object is the intended sleeve, and it can be triggered by a table/obstacle. "
            "If telemetry says grasp while the target is fully occluded in the video, do not "
            "call it a visual empty grasp: use SUCCESS when the mechanical evidence is "
            "consistent, or UNKNOWN when it conflicts with visible contact. Do not use an "
            "unchanged after image to negate a reversible probe, because the probe intentionally "
            "returns and releases at the original pose.\n\n"
            "Camera C is an uncalibrated side observer. Use its RGB/video only for visual "
            "occlusion and cloth-motion evidence; never infer depth, pixels, or robot coordinates "
            "from it. A still labelled as a Camera C lift hold-check was captured immediately "
            "after the first post-close lift action and is the primary acquisition witness. If it "
            "shows an independent sleeve patch elevated with the gripper, classify grasp_acquisition "
            "as SUCCESS even if the reversible probe later releases and the final after image is "
            "unchanged.\n\n"
            f"Selected grasp: {json.dumps(proposal.selected_grasp, ensure_ascii=False)}\n"
            f"Action program: {json.dumps(proposal.actions, ensure_ascii=False)}\n"
            f"Expected observation: {proposal.expected_observation}\n\n"
            + "\n".join(evidence_lines)
        )
        command = [
            self._binary(),
            "--print",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(ACQUISITION_EVALUATION_JSON_SCHEMA, separators=(",", ":")),
            "--permission-mode",
            "plan",
            "--allowedTools",
            "Read",
            "--tools",
            "Read",
            "--add-dir",
            str(root),
            "--safe-mode",
            "--effort",
            "low",
            "--system-prompt",
            (
                "You are the acquisition-inspection turn of one persistent garment "
                "robotics agent. Read only the supplied visual evidence and return the "
                "compact machine-validated judgement. Do not edit files or control a robot."
            ),
        ]
        command = self._prepare_command(command, stage="acquisition_evaluation")
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
            if self.persistent_session is not None:
                self.persistent_session.rollover(
                    reason="claude_timeout", stage="acquisition_evaluation"
                )
            raise ExplorationTimeoutError(
                f"Claude acquisition evaluation timed out after {self.timeout_s} seconds"
            ) from exc
        except OSError as exc:
            raise AutoExplorationError(
                f"Claude acquisition evaluation invocation failed: {exc}"
            ) from exc
        if completed.returncode != 0:
            if (
                self.persistent_session is not None
                and self.persistent_session.is_session_conflict_error(
                    completed.stdout, completed.stderr
                )
            ):
                self.persistent_session.rollover(
                    reason="claude_session_conflict", stage="acquisition_evaluation"
                )
            raise AutoExplorationError(
                f"Claude acquisition evaluation exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        self._record_successful_turn(
            stage="acquisition_evaluation",
            stdout=completed.stdout,
        )
        compact = _json_from_claude_text(completed.stdout)
        if not isinstance(compact, Mapping):
            raise AutoExplorationError(
                "Claude acquisition evaluation must return a JSON object"
            )
        acquisition = compact.get("grasp_acquisition")
        target = compact.get("target_structure_acquired")
        garment_state_change = compact.get("garment_state_change")
        next_experiment = compact.get("next_experiment")
        if garment_state_change not in {"CHANGED", "UNCHANGED", "UNKNOWN"}:
            raise AutoExplorationError(
                "compact acquisition evaluation garment_state_change must be "
                "CHANGED, UNCHANGED, or UNKNOWN"
            )
        acquisition_status = (
            acquisition.get("status") if isinstance(acquisition, Mapping) else None
        )
        target_status = target.get("status") if isinstance(target, Mapping) else None
        if acquisition_status == "FAILURE":
            failure_stage = "ACQUISITION"
        elif acquisition_status == "SUCCESS" and target_status == "CONTRADICTED":
            failure_stage = "TARGET"
        elif acquisition_status == "SUCCESS" and target_status == "SUPPORTED":
            failure_stage = "NONE"
        else:
            failure_stage = "UNKNOWN"
        state_unchanged = garment_state_change == "UNCHANGED"
        synthesized = {
            "target_selection": {
                "status": "UNKNOWN",
                "confidence": 0.0,
                "evidence": [
                    "Acquisition-only evaluation did not reassess semantic target selection."
                ],
            },
            "grasp_acquisition": acquisition,
            "target_structure_acquired": target,
            "transport": {
                "status": "UNKNOWN",
                "confidence": 0.0,
                "evidence": [
                    "Transport was intentionally not executed in the reversible acquisition probe."
                ],
            },
            "laydown": {
                "status": "NOT_REACHED",
                "confidence": 1.0,
                "evidence": [
                    "The probe reversed and released at the original contact instead of laying down a fold."
                ],
            },
            "task_progress": {
                "status": "NEUTRAL",
                "confidence": 1.0,
                "metrics": {
                    "visible_area_delta": "UNCHANGED" if state_unchanged else "UNKNOWN",
                    "overlap_delta": "UNCHANGED" if state_unchanged else "UNKNOWN",
                    "relief_delta": "UNCHANGED" if state_unchanged else "UNKNOWN",
                    "boundary_change": (
                        "The compact evaluator directly reported no persistent garment-state change."
                        if state_unchanged
                        else f"Compact acquisition evaluator reported garment_state_change={garment_state_change}."
                    ),
                },
            },
            "earliest_failure_stage": failure_stage,
            "next_experiment": next_experiment,
        }
        evaluation = validate_evaluation_payload(synthesized)
        result = ClaudeEvaluationResult(
            prompt=prompt,
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            created_at=_now(),
            evaluation=evaluation,
            evidence_images=tuple(
                str(path)
                for path in [*before_rgb, *after_rgb, *video_images, *observer_paths]
            ),
            video_references=tuple(str(path) for path in video_references),
            video_evidence_errors=tuple(video_errors),
        )
        self.last_evaluation_result = result
        self._save_evaluation_log(
            root,
            {
                **result.as_dict(),
                "stage": "acquisition_evaluation",
                "duration_s": time.monotonic() - started,
                "compact_payload": compact,
            },
        )
        return evaluation


def _depth_preview(depth_m: np.ndarray, *, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    """Convert metric depth into a browser-friendly grayscale RGB image."""

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > min_depth_m) & (depth < max_depth_m)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (max_depth_m - depth[valid]) / (max_depth_m - min_depth_m), 0.0, 1.0
    )
    image = np.rint(normalized * 255.0).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


class CameraAWebMonitor:
    """Persistent CamA RGB-D preview rendered directly inside Viser.

    The monitor owns the CamA RealSense pipeline only while the automatic loop
    is idle. The loop stops it before synchronized A/B capture, eliminating the
    device contention caused by a second OpenCV process.
    """

    def __init__(
        self,
        project_root: Path,
        perception_config_path: Path,
        spec: CameraSpec,
        config: PerceptionConfig,
        on_close: Callable[[], None],
        on_frame: Callable[[RGBDFrame], None] | None = None,
    ):
        del project_root, perception_config_path
        self.spec = spec
        self.config = config
        self.on_close = on_close
        # Kept as a compatibility/debug label for callers that used the old
        # native-window monitor. No native window is created anymore.
        self.window_name = f"CamA live monitor ({spec.serial})"
        self.on_frame = on_frame or (lambda frame: None)
        self.X_base_camera: np.ndarray | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.camera: RealSenseRGBD | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._camera_lifecycle_lock = threading.Lock()

    def _start_camera(self, camera: RealSenseRGBD) -> bool:
        """Start once unless a stop was requested before the worker acquired it."""

        with self._camera_lifecycle_lock:
            if self.stop_event.is_set():
                return False
            camera.start()
            return True

    def _stop_camera(self, camera: RealSenseRGBD) -> None:
        """Serialize all pipeline stops so RealSense never receives a duplicate."""

        with self._camera_lifecycle_lock:
            camera.stop()

    def start(self) -> None:
        with self._lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.error = None
            self.camera = RealSenseRGBD(
                self.spec, self.config.width, self.config.height, self.config.fps
            )
            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="claude-auto-camera-a-web-monitor",
            )
            self.thread.start()

    def _run(self) -> None:
        camera = self.camera
        if camera is None:
            return
        try:
            if self.stop_event.is_set():
                return
            if not self._start_camera(camera):
                return
            for _ in range(min(self.config.warmup_frames, 5)):
                if self.stop_event.is_set():
                    return
                camera.read()
            while not self.stop_event.is_set():
                rgb, depth_m = camera.read()
                self.X_base_camera = load_extrinsics(self.spec.extrinsics_file)
                if camera.intrinsics is None:
                    raise AutoExplorationError("CamA intrinsics are unavailable")
                self.on_frame(
                    RGBDFrame(
                        label=self.spec.label,
                        serial=self.spec.serial,
                        rgb=rgb,
                        depth_m=depth_m,
                        intrinsics=camera.intrinsics.copy(),
                        X_base_camera=self.X_base_camera.copy(),
                    )
                )
                self.stop_event.wait(0.15)
        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc
                self.on_close()
        finally:
            self._stop_camera(camera)
            self.camera = None

    def stop(self) -> None:
        with self._lock:
            self.stop_event.set()
            # Release the RealSense pipeline before joining the reader. A
            # blocked wait_for_frames() must be interrupted when capture_two_view
            # is about to claim the same device.
            camera = self.camera
            if camera is not None:
                self._stop_camera(camera)
            if self.thread is not None and self.thread.is_alive():
                self.thread.join(timeout=3.0)
            self.thread = None
            self.camera = None


def _save_frame_images(
    frames: list[RGBDFrame],
    output_dir: Path,
) -> list[Path]:
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for index, frame in enumerate(frames):
        path = output_dir / f"camera_{index}_{frame.label}.png"
        Image.fromarray(frame.rgb.astype(np.uint8)).save(path)
        np.save(output_dir / f"camera_{index}_{frame.label}_depth_m.npy", frame.depth_m)
        paths.append(path)
    return paths


def _video_contact_sheet(
    video_path: Path,
    output_path: Path,
    *,
    sample_count: int = 16,
    columns: int = 4,
) -> dict[str, Any]:
    """Extract a chronological contact sheet that Claude can inspect as images."""

    import cv2
    from PIL import Image, ImageDraw, ImageFont

    video_path = video_path.resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise AutoExplorationError(f"cannot decode rollout video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise AutoExplorationError(f"rollout video has no frames: {video_path}")
        sample_count = max(2, min(int(sample_count), frame_count))
        indices = np.rint(np.linspace(0, frame_count - 1, sample_count)).astype(int)
        frames: list[tuple[int, np.ndarray]] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if not ok or bgr is None:
                continue
            frames.append((int(index), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if len(frames) < 2:
        raise AutoExplorationError(
            f"rollout video yielded fewer than two sampled frames: {video_path}"
        )

    tile_width = 480
    tile_height = 360
    label_height = 38
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        (24, 27, 33),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for sample_index, (frame_index, rgb) in enumerate(frames):
        column = sample_index % columns
        row = sample_index // columns
        x = column * tile_width
        y = row * (tile_height + label_height)
        frame = Image.fromarray(rgb).resize(
            (tile_width, tile_height), Image.Resampling.LANCZOS
        )
        sheet.paste(frame, (x, y + label_height))
        timestamp_s = frame_index / fps if math.isfinite(fps) and fps > 0 else float("nan")
        timestamp = f"{timestamp_s:06.2f}s" if math.isfinite(timestamp_s) else "time UNKNOWN"
        draw.text(
            (x + 10, y + 7),
            f"#{sample_index + 1:02d}  frame {frame_index:05d}  {timestamp}",
            fill=(245, 247, 250),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG", optimize=True)
    return {
        "video": str(video_path),
        "contact_sheet": str(output_path.resolve()),
        "source_frame_count": frame_count,
        "source_fps": fps,
        "sampled_frame_indices": [index for index, _ in frames],
    }


def prepare_rollout_video_evidence(
    recording_dir: Path,
) -> tuple[list[Path], list[Path], list[str]]:
    """Build temporal evidence for calibrated A/B and optional RGB observers."""

    recording_dir = recording_dir.resolve()
    manifest_path = recording_dir / "recording_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[tuple[str, Path]] = []
    composite_relative = manifest.get("composite_video")
    if isinstance(composite_relative, str) and composite_relative.strip():
        candidates.append(("AB_DEPTH", recording_dir / composite_relative))
    elif (recording_dir / "composite_AB_depth.mp4").is_file():
        candidates.append(("AB_DEPTH", recording_dir / "composite_AB_depth.mp4"))
    if not candidates:
        for camera in manifest.get("cameras", []):
            if not isinstance(camera, dict):
                continue
            label = str(camera.get("label", "")).upper()
            relative = camera.get("rgb_video")
            if label and isinstance(relative, str) and relative.strip():
                candidates.append((label, recording_dir / relative))
    if not candidates:
        candidates = [
            (label, recording_dir / f"camera_{label}_rgb.mp4") for label in ("A", "B")
        ]
    observer_manifest_path = recording_dir / "observer_recording_manifest.json"
    if observer_manifest_path.is_file():
        try:
            observer_manifest = json.loads(
                observer_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            observer_manifest = {}
        if isinstance(observer_manifest, Mapping):
            observer_label = str(observer_manifest.get("label", "C")).upper() or "C"
            observer_video = observer_manifest.get("rgb_video")
            if isinstance(observer_video, str) and observer_video.strip():
                candidate = (observer_label, recording_dir / observer_video)
                if candidate not in candidates:
                    candidates.append(candidate)
    else:
        for observer_video in sorted(recording_dir.glob("camera_*_observer_rgb.mp4")):
            stem = observer_video.stem
            observer_label = stem.removeprefix("camera_").removesuffix("_observer_rgb")
            candidates.append((observer_label.upper() or "C", observer_video))

    output_dir = recording_dir / "evaluator_video_evidence"
    contact_sheets: list[Path] = []
    references: list[Path] = []
    errors: list[str] = []
    manifest_items: list[dict[str, Any]] = []
    for label, video_path in candidates:
        if not video_path.is_file():
            errors.append(f"Camera {label} RGB video is missing: {video_path}")
            continue
        references.append(video_path.resolve())
        output_path = output_dir / f"camera_{label}_rgb_contact_sheet.png"
        try:
            item = _video_contact_sheet(video_path, output_path)
        except Exception as exc:
            errors.append(f"Camera {label}: {type(exc).__name__}: {exc}")
            continue
        item["camera"] = label
        manifest_items.append(item)
        contact_sheets.append(output_path.resolve())
    if manifest_items or errors:
        _write_json(
            output_dir / "manifest.json",
            {
                "created_at": _now(),
                "sampling": "16 uniform chronological frames per selected rollout video",
                "items": manifest_items,
                "errors": errors,
            },
        )
    return contact_sheets, references, errors


@dataclass
class _AutoState:
    running: bool = False
    stop_requested: bool = False
    iteration: int = 0
    objective: str = DEFAULT_AUTO_OBJECTIVE
    proposal: ExplorationProposal | None = None
    evaluation: ExplorationEvaluation | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def run_auto_exploration_viewer(
    session: AgentSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8082,
    max_iterations: int | None = None,
    settle_s: float = 2.0,
    enable_real: bool = False,
    perception_config_path: Path | None = None,
    claude_binary: str = "claude",
    claude_timeout_s: int = 900,
    claude_grounding_timeout_s: int = 120,
    max_replans: int = 2,
    record_rollouts: bool = True,
    recording_native: bool = True,
    recording_codec: str = "mp4v",
    recording_warmup_frames: int | None = None,
    molmo_keypoints: bool = False,
    molmo_keypoint_confidence_threshold: float = (
        DEFAULT_MOLMO_KEYPOINT_CONFIDENCE_THRESHOLD
    ),
    molmo_python: Path | None = None,
    molmo_model: str = "allenai/MolmoPoint-8B",
    molmo_keypoints_path: Path | None = None,
    molmo_keypoint_cameras: Sequence[str] = ("A", "B"),
    molmo_keypoint_timeout_s: int = 900,
    molmo_allow_download: bool = False,
    continue_on_recoverable_errors: bool = False,
    max_consecutive_recoverable_failures: int = 3,
    recovery_backoff_s: float = 2.0,
) -> int:
    """Run the continuous automatic real-agent loop with a Viser preview.

    ``max_iterations=None`` (the default) keeps iterating until Claude returns
    ``stop=true``, the operator requests a stop, or a hard validation/runtime
    failure occurs. A positive value provides an optional iteration cap.
    """

    if not enable_real:
        raise PermissionError(
            "automatic exploration is real-execution only; pass --enable-real explicitly"
        )
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise PermissionError("physical execution is allowed only on a loopback-only Viser server")
    if max_iterations == 0:
        max_iterations = None
    if max_iterations is not None and (max_iterations < 0 or max_iterations > 20):
        raise ValueError("max_iterations must be between 1 and 20 when a cap is supplied")
    if settle_s < 0 or settle_s > 60:
        raise ValueError("settle_s must be between 0 and 60 seconds")
    if max_replans < 0 or max_replans > 5:
        raise ValueError("max_replans must be between 0 and 5")
    if not 30 <= claude_timeout_s <= 1200:
        raise ValueError("claude_timeout_s must be between 30 and 1200 seconds")
    if not 15 <= claude_grounding_timeout_s <= 400:
        raise ValueError(
            "claude_grounding_timeout_s must be between 15 and 400 seconds"
        )
    if recording_warmup_frames is not None and not 0 <= recording_warmup_frames <= 300:
        raise ValueError("recording_warmup_frames must be between 0 and 300")
    if len(recording_codec) != 4:
        raise ValueError("recording_codec must be a four-character code")
    keypoint_threshold = validate_confidence_threshold(
        molmo_keypoint_confidence_threshold
    )
    keypoint_cameras = tuple(
        str(camera).strip().upper() for camera in molmo_keypoint_cameras
    )
    if not keypoint_cameras or any(
        camera not in {"A", "B"} for camera in keypoint_cameras
    ):
        raise ValueError("molmo_keypoint_cameras must contain A and/or B")
    if len(set(keypoint_cameras)) != len(keypoint_cameras):
        raise ValueError("molmo_keypoint_cameras must be unique")
    if not 30 <= molmo_keypoint_timeout_s <= 3600:
        raise ValueError("molmo_keypoint_timeout_s must be between 30 and 3600")
    if not 1 <= max_consecutive_recoverable_failures <= 20:
        raise ValueError(
            "max_consecutive_recoverable_failures must be between 1 and 20"
        )
    if not 0 <= recovery_backoff_s <= 300:
        raise ValueError("recovery_backoff_s must be between 0 and 300 seconds")
    keypoint_specs = load_keypoint_specs(molmo_keypoints_path) if molmo_keypoints else ()
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise RuntimeError(
            "Viser with URDF support is required; install it with: "
            "python -m pip install 'viser[urdf]>=1.0,<2'"
        ) from exc

    root = session.project_root
    robot = session.robot_config
    robot_urdf_path = (
        root / "assets" / "robots" / "xarm6" / "xarm6_wo_ee.urdf"
    ).resolve()
    perception_path = (
        perception_config_path
        or root / "config" / "perception.free_exploration.json"
    ).expanduser().resolve()
    config = PerceptionConfig.load(root, perception_path)
    garment_workspace = config.garment_center_workspace
    if garment_workspace is not None:
        robot_margin = float(robot.workspace_margin_mm)
        for axis in ("x", "y"):
            allowed_low = float(getattr(garment_workspace, f"{axis}_min"))
            allowed_high = float(getattr(garment_workspace, f"{axis}_max"))
            robot_low = getattr(robot.boundaries, f"{axis}_min")
            robot_high = getattr(robot.boundaries, f"{axis}_max")
            if robot_low is not None and allowed_low < float(robot_low) + robot_margin:
                raise ValueError(
                    f"garment center {axis}_min={allowed_low:g} is below the robot "
                    f"safe lower bound {float(robot_low) + robot_margin:g}"
                )
            if robot_high is not None and allowed_high > float(robot_high) - robot_margin:
                raise ValueError(
                    f"garment center {axis}_max={allowed_high:g} is above the robot "
                    f"safe upper bound {float(robot_high) - robot_margin:g}"
                )
    camera_a_spec = next(
        camera for camera in config.cameras if camera.label == config.active_camera_labels[0]
    )
    if camera_a_spec.label != "A":
        raise AutoExplorationError(
            "automatic web preview requires camera A as the first active camera"
        )
    server = viser.ViserServer(host=host, port=port, label="Claude automatic garment exploration")
    state = _AutoState()
    state_lock = threading.Lock()
    auto_thread: threading.Thread | None = None
    auto_run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    auto_results_dir = session.results / "auto_exploration" / auto_run_stamp

    server.scene.set_up_direction("+z")
    bounds = robot.boundaries
    grid_x = ((bounds.x_min or 0.0) + (bounds.x_max or 900.0)) / 2000.0
    grid_y = ((bounds.y_min or -400.0) + (bounds.y_max or 400.0)) / 2000.0
    server.scene.add_grid(
        "/workspace/table",
        width=1.2,
        height=0.8,
        cell_size=0.05,
        section_size=0.25,
        position=(grid_x, grid_y, 0.0),
    )
    if garment_workspace is not None:
        x0 = garment_workspace.x_min / 1000.0
        x1 = garment_workspace.x_max / 1000.0
        y0 = garment_workspace.y_min / 1000.0
        y1 = garment_workspace.y_max / 1000.0
        workspace_edges = np.asarray(
            [
                [[x0, y0, 0.005], [x1, y0, 0.005]],
                [[x1, y0, 0.005], [x1, y1, 0.005]],
                [[x1, y1, 0.005], [x0, y1, 0.005]],
                [[x0, y1, 0.005], [x0, y0, 0.005]],
            ],
            dtype=np.float32,
        )
        server.scene.add_line_segments(
            "/workspace/garment_center_bounds",
            points=workspace_edges,
            colors=np.tile(
                np.asarray([255, 145, 20], dtype=np.uint8), (4, 2, 1)
            ),
            line_width=5.0,
        )
    server.scene.add_frame("/robot_base", axes_length=0.15, axes_radius=0.006)
    server.scene.add_frame("/xarm", show_axes=False)
    robot_model = ViserUrdf(
        server,
        robot_urdf_path,
        root_node_name="/xarm",
        load_meshes=True,
        load_collision_meshes=False,
    )
    kinematics = XArm6Kinematics(robot_urdf_path)
    home_cfg = np.concatenate(
        [np.radians(np.asarray(robot.init_joints_deg, dtype=np.float64)), [0.0]]
    )
    robot_model.update_cfg(home_cfg)
    robot_animation_lock = threading.Lock()
    robot_animation_stop = threading.Event()
    robot_animation_thread: threading.Thread | None = None

    status = server.gui.add_markdown(
        "### Automatic exploration ready\n\n"
        "The web page owns the CamA RGB-D preview. The preview pauses during "
        "synchronized A/B capture so both cameras are never opened twice."
    )
    reference_policy_line = (
        "- grasp-reference policy: `Molmo keypoints only`; each iteration reruns "
        f"Molmo, requires confidence `> {keypoint_threshold:.3f}`, and stops before "
        "Claude/robot if none pass"
        if molmo_keypoints
        else "- grasp-reference policy: `uniform calibrated Rxxx` "
        "(Molmo keypoint mode disabled)"
    )
    garment_workspace_line = (
        "- garment-center workspace: disabled"
        if garment_workspace is None
        else (
            "- garment-center workspace: "
            f"`x=[{garment_workspace.x_min:g}, {garment_workspace.x_max:g}] mm`, "
            f"`y=[{garment_workspace.y_min:g}, {garment_workspace.y_max:g}] mm`; "
            "out-of-range perception forces the next rollout into WORKSPACE_RECOVERY"
        )
    )
    yaw_workspace_line = (
        "- TCP Y workspace allowance: "
        f"`0.5 * {robot.gripper_width_mm:g} mm * |sin(yaw)|`; "
        f"0 mm at yaw=0, {robot.y_workspace_extension_mm(90.0):g} mm at +/-90"
    )
    controls = server.gui.add_markdown(
        f"### Loop contract\n\n- max iterations: `{'continuous' if max_iterations is None else max_iterations}`\n"
        f"- settle time after motion: `{settle_s:.1f}s`\n"
        "- default: continuous iterations until Claude/user stop or hard failure\n"
        f"- pre-execution Claude replans on validation failure: `{max_replans}`\n"
        f"- Claude visual-planning timeout: `{claude_timeout_s}s`; final Rxx grounding timeout: `{claude_grounding_timeout_s}s`\n"
        "- perception diagnostics: A/B garment height-above-table heatmaps and fused garment boundary are shown and sent to Claude\n"
        "- Claude grounding: choose one Rxxx visually, then perform exactly one final exact-coordinate lookup\n"
        f"{reference_policy_line}\n"
        "- grasp target visualization: Base XYZ/yaw, Viser 3-D marker, and Camera A/B projection overlays\n"
        f"{garment_workspace_line}\n"
        f"{yaw_workspace_line}\n"
        f"- rollout A/B RGB-D recording: `{'enabled' if record_rollouts else 'disabled'}`\n"
        "- stop takes effect between phases; it cannot interrupt a command already sent"
    )
    run_log_panel = server.gui.add_markdown(
        f"### Agent log\n\nRun artifacts will be saved under `{_run_relative(auto_results_dir, session.run_dir)}`."
    )
    start_button = server.gui.add_button("Restart automatic exploration", color="red")
    stop_button = server.gui.add_button("Stop after current phase", disabled=True, color="orange")
    iteration_slider = server.gui.add_slider(
        "Maximum iterations (0 = continuous)",
        min=0,
        max=20,
        step=1,
        initial_value=0 if max_iterations is None else max_iterations,
    )
    history_panel = server.gui.add_markdown("### Agent history\n\nNo iteration has run.")
    proposal_panel = server.gui.add_markdown("### Current proposal\n\nNone.")
    target_panel = server.gui.add_markdown(
        "### Planned grasp target\n\n`unknown` — no proposal has been generated."
    )
    planning_timer_panel = server.gui.add_markdown(
        "### Claude stage timer\n\nNo Claude stage is active."
    )
    planning_timer_lock = threading.Lock()
    planning_timer_stop = threading.Event()
    planning_timer_state: dict[str, Any] = {
        "phase": None,
        "status": "idle",
        "started_monotonic": None,
        "limit_s": None,
        "visual_attempt": 0,
        "durations": {},
    }

    def planning_phase_callback(phase: str, event: str, value: float) -> None:
        with planning_timer_lock:
            if event == "started":
                if phase == "visual_planning":
                    if (
                        planning_timer_state["phase"] == "visual_planning"
                        and planning_timer_state["status"] == "reselecting"
                    ):
                        planning_timer_state["visual_attempt"] += 1
                    else:
                        planning_timer_state["visual_attempt"] = 1
                    planning_timer_state["durations"] = {}
                planning_timer_state["phase"] = phase
                planning_timer_state["status"] = "running"
                planning_timer_state["started_monotonic"] = time.monotonic()
                planning_timer_state["limit_s"] = float(value)
            else:
                planning_timer_state["phase"] = phase
                planning_timer_state["status"] = event
                planning_timer_state["started_monotonic"] = None
                planning_timer_state["durations"][phase] = float(value)

    def render_planning_timer() -> None:
        labels = {
            "visual_planning": "Stage 1 — visual planning",
            "final_grounding": "Stage 2 — final Rxx grounding/run generation",
        }
        while not planning_timer_stop.wait(0.5):
            with planning_timer_lock:
                snapshot = {
                    "phase": planning_timer_state["phase"],
                    "status": planning_timer_state["status"],
                    "started_monotonic": planning_timer_state["started_monotonic"],
                    "limit_s": planning_timer_state["limit_s"],
                    "visual_attempt": planning_timer_state["visual_attempt"],
                    "durations": dict(planning_timer_state["durations"]),
                }
            phase = snapshot["phase"]
            status_value = snapshot["status"]
            lines = ["### Claude stage timer", ""]
            if phase is None:
                lines.append("No Claude stage is active.")
            else:
                lines.append(f"- current: `{labels.get(phase, phase)}`")
                lines.append(f"- status: `{status_value}`")
                if phase == "visual_planning":
                    lines.append(
                        "- Stage 1 reference attempt: "
                        f"`{int(snapshot['visual_attempt'])}/{max_replans + 1}`"
                    )
                started_monotonic = snapshot["started_monotonic"]
                if started_monotonic is not None:
                    elapsed = time.monotonic() - float(started_monotonic)
                    limit_s = float(snapshot["limit_s"] or 0.0)
                    lines.append(f"- elapsed: `{elapsed:.1f}s / {limit_s:.0f}s`")
            durations = snapshot["durations"]
            if durations:
                lines.extend(["", "Completed stage durations:"])
                for name in ("visual_planning", "final_grounding"):
                    if name in durations:
                        lines.append(
                            f"- {labels[name]}: `{float(durations[name]):.1f}s`"
                        )
                if {"visual_planning", "final_grounding"}.issubset(durations):
                    total = float(durations["visual_planning"]) + float(
                        durations["final_grounding"]
                    )
                    lines.append(f"- total two-stage planning: `{total:.1f}s`")
            planning_timer_panel.content = "\n".join(lines)

    planning_timer_thread = threading.Thread(
        target=render_planning_timer,
        daemon=True,
        name="claude-stage-timer",
    )
    planning_timer_thread.start()
    evaluation_panel = server.gui.add_markdown("### Before/after judgement\n\nNone.")
    preview_panel = server.gui.add_markdown(
        "### Live CamA RGB-D\n\nWaiting for the camera preview to start."
    )
    preview_rgb_handle = server.gui.add_image(
        np.zeros((240, 320, 3), dtype=np.uint8), label="CamA live RGB"
    )
    preview_depth_handle = server.gui.add_image(
        np.zeros((240, 320, 3), dtype=np.uint8), label="CamA height-above-table heatmap"
    )
    capture_rgb_handles: dict[str, Any] = {
        "A": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture RGB A"
        ),
        "B": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture RGB B"
        ),
    }
    capture_depth_handles: dict[str, Any] = {
        "A": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture height-above-table heatmap A"
        ),
        "B": server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8), label="Latest capture height-above-table heatmap B"
        ),
    }
    diagnostic_image_handles: list[Any] = []

    def clear_diagnostic_images() -> None:
        while diagnostic_image_handles:
            diagnostic_image_handles.pop().remove()

    def render_perception_diagnostics(
        result: dict[str, Any],
        result_path: Path,
    ) -> None:
        """Show the same heatmaps/boundaries that are supplied to Claude."""

        from PIL import Image

        clear_diagnostic_images()
        for view in result.get("views", []):
            if not isinstance(view, dict):
                continue
            image_path = result_path.parent / str(
                view.get(
                    "height_map_boundary",
                    view.get(
                        "height_map",
                        view.get("depth_heatmap_boundary", view.get("depth_heatmap", "")),
                    ),
                )
            )
            if not image_path.is_file():
                continue
            label = str(view.get("label", "")).upper()
            focused_heatmap = result_path.parent / str(
                view.get("height_map", view.get("depth_heatmap", ""))
            )
            if label in capture_depth_handles and focused_heatmap.is_file():
                from PIL import Image
                capture_depth_handles[label].image = np.asarray(
                    Image.open(focused_heatmap).convert("RGB")
                )
            diagnostic_image_handles.append(
                server.gui.add_image(
                    np.asarray(Image.open(image_path).convert("RGB")),
                    label=f"Camera {view.get('label', '?')} garment height-above-table heatmap + boundary",
                )
            )
            fold_edges = result_path.parent / str(
                view.get(
                    "height_gradient_overlay",
                    view.get("fold_edge_overlay", ""),
                )
            )
            if fold_edges.is_file():
                diagnostic_image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(fold_edges).convert("RGB")),
                        label=f"Camera {view.get('label', '?')} internal height-gradient/occlusion edges",
                    )
                )
            coordinate_overlay = result_path.parent / str(
                view.get("coordinate_overlay", "")
            )
            if coordinate_overlay.is_file():
                diagnostic_image_handles.append(
                    server.gui.add_image(
                        np.asarray(Image.open(coordinate_overlay).convert("RGB")),
                        label=(
                            f"Camera {view.get('label', '?')} unranked robot-base "
                            "coordinate references"
                        ),
                    )
                )
        artifacts = result.get("depth_fusion", {}).get("artifacts", {})
        for key, label in (
            ("heatmap", "Fused garment height-above-table heatmap"),
            ("boundary_overlay", "Fused height map + garment boundary"),
            ("fold_edge_overlay", "Fused height-gradient/occlusion edges"),
        ):
            image_path = result_path.parent / str(artifacts.get(key, ""))
            if not image_path.is_file():
                continue
            diagnostic_image_handles.append(
                server.gui.add_image(
                    np.asarray(Image.open(image_path).convert("RGB")),
                    label=label,
                )
            )
    live_cloud_handle = server.scene.add_point_cloud(
        "/live_preview/CamA",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.004,
        point_shape="circle",
    )
    capture_cloud_handles: dict[str, Any] = {
        label: server.scene.add_point_cloud(
            f"/live_preview/capture_{label}",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.uint8),
            point_size=0.003,
            point_shape="circle",
            visible=False,
        )
        for label in ("A", "B")
    }
    fused_cloud_handle = server.scene.add_point_cloud(
        "/live_preview/fused_AB",
        points=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.uint8),
        point_size=0.004,
        point_shape="circle",
        visible=False,
    )

    robot_panel = server.gui.add_markdown(
        "### xArm7 mesh\n\n"
        "The xArm7 + gripper URDF is loaded at Home. It follows the validated "
        "automatic action sequence during each physical rollout."
    )

    def set_status(message: str) -> None:
        status.content = message

    def apply_robot_frame(frame: AnimationFrame) -> None:
        """Apply one URDF configuration from the automatic rollout preview."""

        with robot_animation_lock:
            robot_model.update_cfg(frame.configuration_rad)
        robot_panel.content = (
            "### xArm7 mesh\n\n"
            f"- phase: `{frame.label}`\n"
            f"- action: `{frame.action_index + 1}`\n"
            f"- gripper drive: `{frame.configuration_rad[-1]:.3f} rad`"
        )

    def stop_robot_animation(*, reset_to_home: bool = False) -> None:
        """Stop the preview animation without sending any robot command."""

        nonlocal robot_animation_thread
        robot_animation_stop.set()
        thread = robot_animation_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        robot_animation_thread = None
        if reset_to_home:
            apply_robot_frame(AnimationFrame(home_cfg.copy(), -1, "home"))

    def animate_robot_frames(frames: list[AnimationFrame]) -> None:
        """Replay the validated arm/gripper trajectory in the Viser scene."""

        nonlocal robot_animation_thread
        robot_animation_stop.clear()

        def run() -> None:
            nonlocal robot_animation_thread
            try:
                for frame in frames:
                    if robot_animation_stop.is_set():
                        return
                    apply_robot_frame(frame)
                    robot_animation_stop.wait(1.0 / 12.0)
            finally:
                with robot_animation_lock:
                    robot_animation_thread = None

        robot_animation_thread = threading.Thread(
            target=run,
            daemon=True,
            name="xarm-auto-exploration-animation",
        )
        robot_animation_thread.start()

    @server.on_client_connect
    def _(client: Any) -> None:
        client.camera.position = (1.3, -0.9, 0.9)
        client.camera.look_at = (0.62, -0.07, 0.1)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    def render_live_frame(frame: RGBDFrame) -> None:
        """Update the browser preview from the CamA reader thread."""

        try:
            preview_rgb_handle.image = np.asarray(frame.rgb, dtype=np.uint8)
            height_map, height_valid, _ = camera_height_map_mm(frame, config)
            preview_depth_handle.image = _scalar_heatmap_rgb(
                height_map,
                height_valid,
                higher_is_bright=True,
            )
            points, colors = _frame_point_cloud(
                frame,
                stride=4,
                height_above_table_mm=height_map,
            )
            points, colors = _voxel_balance_cloud(
                points, colors, voxel_size_mm=5.0, max_points=25000
            )
            if len(points):
                live_cloud_handle.points = points
                live_cloud_handle.colors = colors
                live_cloud_handle.visible = True
            else:
                live_cloud_handle.visible = False
            depth = np.asarray(frame.depth_m)
            valid = np.isfinite(depth) & (depth > config.min_depth_m) & (depth < config.max_depth_m)
            preview_panel.content = (
                "### Live CamA RGB-D\n\n"
                f"- resolution: `{frame.rgb.shape[1]} × {frame.rgb.shape[0]}`\n"
                f"- valid depth: `{float(valid.mean()) * 100:.1f}%`\n"
                f"- point cloud: `{len(points):,}` points in base frame\n"
                f"- serial: `{frame.serial}`"
            )
        except Exception as exc:
            live_cloud_handle.visible = False
            preview_depth_handle.image = _depth_heatmap_preview(
                frame.depth_m,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
            )
            preview_panel.content = (
                "### Live CamA RGB-D\n\n"
                f"Table validation failed; point cloud hidden: `{exc}`"
            )

    def render_capture_frames(frames: list[RGBDFrame]) -> None:
        for frame in frames:
            label = frame.label.upper()
            if label not in capture_rgb_handles:
                continue
            capture_rgb_handles[label].image = np.asarray(frame.rgb, dtype=np.uint8)
            handle = capture_cloud_handles[label]
            try:
                height_map, height_valid, _ = camera_height_map_mm(frame, config)
            except Exception:
                capture_depth_handles[label].image = _depth_heatmap_preview(
                    frame.depth_m,
                    min_depth_m=config.min_depth_m,
                    max_depth_m=config.max_depth_m,
                )
                handle.visible = False
                continue
            capture_depth_handles[label].image = _scalar_heatmap_rgb(
                height_map,
                height_valid,
                higher_is_bright=True,
            )
            points, colors = _frame_point_cloud(
                frame,
                stride=4,
                height_above_table_mm=height_map,
            )
            points, colors = _voxel_balance_cloud(
                points, colors, voxel_size_mm=5.0, max_points=25000
            )
            if len(points):
                handle.points = points
                handle.colors = colors
                handle.visible = True
            else:
                handle.visible = False

    def render_grasp_target_visualization(
        actions: Sequence[dict[str, Any]],
        frames: list[RGBDFrame],
        *,
        plan_status: str,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Show and optionally persist the grasp targets for one plan."""

        from PIL import Image

        server.scene.remove_by_name("/agent_targets")
        targets = grasp_targets_from_actions(actions)
        payload: dict[str, Any] = {
            "definition": "last finite move immediately preceding each close_gripper",
            "status": plan_status,
            "targets": targets,
            "camera_overlays": {},
        }
        if not targets:
            target_panel.content = (
                "### Planned grasp target\n\n"
                f"- plan status: `{plan_status}`\n"
                "- target: `unknown`\n"
                "- reason: no `close_gripper()` has a grounded preceding `move()`"
            )
        else:
            lines = [
                "### Planned grasp target",
                "",
                f"- plan status: `{plan_status}`",
                "- frame: `robot base`",
            ]
            for target in targets:
                index = int(target["target_index"])
                position = np.asarray(
                    [target["x"], target["y"], target["z"]], dtype=np.float64
                ) / 1000.0
                yaw_rad = math.radians(float(target["yaw"]))
                yaw_axis = np.asarray(
                    [math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float64
                )
                server.scene.add_icosphere(
                    f"/agent_targets/T{index}/position",
                    radius=0.018,
                    color=(255, 35, 35),
                    position=tuple(position),
                )
                server.scene.add_label(
                    f"/agent_targets/T{index}/label",
                    (
                        f"T{index} grasp: ({target['x']:.1f}, {target['y']:.1f}, "
                        f"{target['z']:.1f}) mm, yaw={target['yaw']:.1f} deg"
                    ),
                    position=tuple(position + np.asarray([0.0, 0.0, 0.035])),
                )
                guide_points = np.asarray(
                    [
                        [position - yaw_axis * 0.045, position + yaw_axis * 0.045],
                        [np.asarray([position[0], position[1], 0.0]), position],
                    ],
                    dtype=np.float32,
                )
                guide_colors = np.asarray(
                    [
                        [(255, 220, 30), (255, 220, 30)],
                        [(255, 70, 70), (255, 70, 70)],
                    ],
                    dtype=np.uint8,
                )
                server.scene.add_line_segments(
                    f"/agent_targets/T{index}/guides",
                    points=guide_points,
                    colors=guide_colors,
                    line_width=5.0,
                )
                lines.append(
                    f"- T{index}: `({target['x']:.1f}, {target['y']:.1f}, "
                    f"{target['z']:.1f}) mm`, yaw `{target['yaw']:.1f}°` "
                    f"(move {target['move_action_index']} → close {target['close_action_index']})"
                )
            lines.extend(
                [
                    "",
                    "Red sphere = TCP grasp position; yellow line = planned yaw axis.",
                ]
            )
            target_panel.content = "\n".join(lines)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            label = frame.label.upper()
            overlay, projections = target_overlay_image(frame, targets)
            if label in capture_rgb_handles:
                capture_rgb_handles[label].image = overlay
            overlay_payload: dict[str, Any] = {"projections": projections}
            if output_dir is not None:
                overlay_path = output_dir / f"grasp_target_camera_{label}.png"
                Image.fromarray(overlay).save(overlay_path)
                overlay_payload["image"] = _run_relative(
                    overlay_path, session.run_dir
                )
            payload["camera_overlays"][label] = overlay_payload
        return payload

    def camera_window_closed() -> None:
        with state_lock:
            state.stop_requested = True
        status.content = (
            "### CamA preview stopped\n\n"
            "The CamA reader stopped unexpectedly. Automatic exploration will stop "
            "at the next safe phase boundary."
        )

    monitor = CameraAWebMonitor(
        root,
        perception_path,
        camera_a_spec,
        config,
        camera_window_closed,
        render_live_frame,
    )

    def render_history() -> None:
        with state_lock:
            entries = list(state.history)
        if not entries:
            history_panel.content = "### Agent history\n\nNo iteration has run."
            return
        lines = [
            "### Agent history",
            "",
            "| iteration | plan | progress | confidence | earliest failure | keep | change |",
            "|---:|---|---|---:|---|---|---|",
        ]
        for entry in entries:
            evaluation = entry.get("evaluation") or {}
            progress = evaluation.get("task_progress") or {}
            next_experiment = evaluation.get("next_experiment") or {}
            keep = ", ".join(next_experiment.get("keep") or []) or "-"
            change = ", ".join(next_experiment.get("change") or []) or "STOP"
            lines.append(
                f"| {entry['iteration']} | `{entry.get('plan_status', 'unknown')}` | "
                f"`{progress.get('status', '-')}` | `{progress.get('confidence', '-')}` | "
                f"`{evaluation.get('earliest_failure_stage', '-')}` | {keep} | {change} |"
            )
        history_panel.content = "\n".join(lines)

    def request_stop(_: Any) -> None:
        with state_lock:
            state.stop_requested = True
        set_status(
            "### Stop requested\n\nThe loop will stop after the current safe phase. "
            "An in-progress robot command is not interrupted automatically."
        )

    @stop_button.on_click
    def _(event: Any) -> None:
        request_stop(event)

    def stopped() -> bool:
        with state_lock:
            return state.stop_requested

    def set_running(value: bool) -> None:
        with state_lock:
            state.running = value
        start_button.disabled = value
        stop_button.disabled = not value
        iteration_slider.disabled = value

    def save_auto_record(name: str, payload: dict[str, Any]) -> None:
        directory = auto_results_dir
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / name, payload)

    def save_agent_artifact(
        iteration: int,
        record: dict[str, Any],
        *,
        phase: str,
        payload: Any,
        suffix: str = ".json",
    ) -> str:
        """Persist one agent phase in a stable per-iteration artifact file."""

        directory = auto_results_dir / f"iteration_{iteration:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{phase}{suffix}"
        if suffix == ".json":
            _write_json(path, payload)
        else:
            path.write_text(str(payload), encoding="utf-8")
        relative = _run_relative(path, session.run_dir)
        record.setdefault("artifacts", {})[phase] = relative
        return relative

    def update_iteration_camera_report(
        iteration: int,
        record: dict[str, Any],
        saved_perception: dict[str, Any],
        saved_result_path: Path,
        *,
        target_visualization: dict[str, Any] | None = None,
        visual_plan_result: ClaudeVisualPlanResult | None = None,
    ) -> dict[str, Any] | None:
        """Create or refresh the Camera-A report sheet for one iteration."""

        output_dir = auto_results_dir / f"iteration_{iteration:03d}"
        output_path = output_dir / "camera_A_perception_report.png"
        target_overlay_path = output_dir / "grasp_target_camera_A.png"
        selected_reference = None
        if visual_plan_result is not None:
            selected_reference = visual_plan_result.decision.selected_reference
        target = None
        if target_visualization:
            targets = target_visualization.get("targets", [])
            if targets:
                target = targets[0]
        try:
            manifest = compose_camera_perception_report(
                saved_perception,
                saved_result_path,
                output_path,
                camera="A",
                run_name=session.run_dir.name,
                iteration=iteration,
                target_overlay_path=(
                    target_overlay_path if target_overlay_path.is_file() else None
                ),
                selected_reference=selected_reference,
                target=target,
            )
        except Exception as exc:
            record["camera_A_perception_report_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return None
        gallery_dir = session.results / "report_figures" / auto_results_dir.name
        gallery_dir.mkdir(parents=True, exist_ok=True)
        gallery_path = gallery_dir / f"iteration_{iteration:03d}_camera_A.png"
        shutil.copy2(output_path, gallery_path)
        manifest["image"] = _run_relative(output_path, session.run_dir)
        manifest["gallery_image"] = _run_relative(gallery_path, session.run_dir)
        manifest_path = output_dir / "camera_A_perception_report.json"
        _write_json(manifest_path, manifest)
        record["camera_A_perception_report"] = manifest
        record.setdefault("artifacts", {})["camera_A_perception_report"] = manifest[
            "image"
        ]
        record["artifacts"]["camera_A_perception_report_gallery"] = manifest[
            "gallery_image"
        ]
        record["artifacts"]["camera_A_perception_report_manifest"] = _run_relative(
            manifest_path, session.run_dir
        )
        return manifest

    def _replan_feedback(exc: BaseException, *, proposal: ExplorationProposal | None) -> str:
        details = [f"Error type: {type(exc).__name__}", f"Error: {exc}"]
        if proposal is not None:
            details.append("Rejected proposal actions:")
            details.extend(json.dumps(action, ensure_ascii=False) for action in proposal.actions)
        details.append(
            "Generate a new proposal that addresses this exact failure. Keep the "
            "waypoint count compact, but preserve a meaningful net transport distance; "
            "stay inside the safe workspace margin rather than retreating to a tiny move."
        )
        return "\n".join(details)

    def run_loop(iterations: int | None) -> None:
        skill_store = SkillStore(session.project_root / "data" / "skills")
        run_skill_ledger = RunSkillLedger(session.workspace)

        def run_skill_prompt() -> str:
            appendix = run_skill_ledger.prompt_appendix()
            return skill_store.prompt() + (("\n\n" + appendix) if appendix else "")

        client = ClaudeAutoClient(
            binary=claude_binary,
            timeout_s=claude_timeout_s,
            grounding_timeout_s=claude_grounding_timeout_s,
            max_reference_reselections=max_replans,
            skill_guidance=run_skill_prompt(),
            skill_names=tuple(skill.name for skill in skill_store.approved()),
        )
        try:
            objective = state.objective
            iteration = 0
            consecutive_recoverable_failures = 0
            while iterations is None or iteration < iterations:
                iteration += 1
                client.skill_guidance = run_skill_prompt()
                client.skill_names = tuple(
                    skill.name for skill in skill_store.approved()
                )
                if stopped():
                    break
                with state_lock:
                    state.iteration = iteration
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "started_at": _now(),
                    "objective": objective,
                }
                workspace_recovery: GarmentWorkspaceRecovery | None = None
                planning_objective = objective
                record_saved = False
                try:
                    limit_label = "continuous" if iterations is None else str(iterations)
                    set_status(
                        f"### Iteration {iteration}/{limit_label}: perceiving\n\n"
                        "Pausing the CamA monitor and capturing synchronized A/B RGB-D."
                    )
                    monitor.stop()
                    frames = capture_two_view_rgbd(config)
                    render_capture_frames(frames)
                    perception = session.locate_cloth_center(config, frames=frames)
                    saved, saved_path = _load_latest_perception(session)
                    if saved is None or saved_path is None:
                        raise AutoExplorationError("perception completed without saved result")
                    if garment_workspace is not None:
                        workspace_recovery = assess_garment_workspace(
                            saved.get("center_base_mm", []), garment_workspace
                        )
                        record["garment_workspace"] = workspace_recovery.as_dict()
                        if workspace_recovery.required:
                            dx_mm, dy_mm = (
                                workspace_recovery.requested_translation_xy_mm
                            )
                            planning_objective = (
                                "Safety-priority workspace recovery: move the garment "
                                "inward before any further opening action. Use the "
                                "configured recovery request and produce a safe "
                                f"grasp-to-release translation of approximately "
                                f"dx={dx_mm:.1f} mm, dy={dy_mm:.1f} mm."
                            )
                            record["iteration_mode"] = "workspace_recovery"
                        else:
                            record["iteration_mode"] = "garment_opening"
                    else:
                        record["iteration_mode"] = "garment_opening"
                    record["planner_objective"] = planning_objective
                    render_perception_diagnostics(saved, saved_path)
                    fused_points, fused_colors = _load_fused_point_cloud(
                        saved, saved_path.parent
                    )
                    if len(fused_points):
                        fused_cloud_handle.points = fused_points
                        fused_cloud_handle.colors = fused_colors
                        fused_cloud_handle.visible = True
                    else:
                        fused_cloud_handle.visible = False
                    record["perception"] = perception
                    before_images = perception_image_paths(saved, saved_path)
                    flat_reference_image = (
                        root
                        / "data"
                        / "reference"
                        / "flat_garment_reference"
                        / "camera_A_flat_reference_anchors.png"
                    ).resolve()
                    if flat_reference_image.is_file():
                        # This is a semantic/layout reference only. Folded
                        # observations must re-detect their own axis and
                        # anchors; reference pixels are never current targets.
                        before_images.append(flat_reference_image)
                        record.setdefault("artifacts", {})[
                            "flat_garment_reference"
                        ] = _run_relative(flat_reference_image, session.run_dir)
                    reference_policy = "uniform"
                    if molmo_keypoints:
                        set_status(
                            f"### Iteration {iteration}/{limit_label}: Molmo keypoints\n\n"
                            f"Axis-first labeling, then querying {len(keypoint_specs)} keypoints on "
                            f"Camera {','.join(keypoint_cameras)} and keeping only "
                            f"confidence > {keypoint_threshold:.3f}."
                        )
                        keypoint_artifact_dir = (
                            auto_results_dir
                            / f"iteration_{iteration:03d}"
                            / "molmo_keypoints"
                        )
                        keypoint_manifest = run_molmo_keypoint_pipeline(
                            project_root=root,
                            perception_dir=session.workspace / "perception_views",
                            artifact_dir=keypoint_artifact_dir,
                            confidence_threshold=keypoint_threshold,
                            molmo_python=molmo_python,
                            model=molmo_model,
                            timeout_s=molmo_keypoint_timeout_s,
                            local_files_only=not molmo_allow_download,
                            keypoint_specs=keypoint_specs,
                            cameras=keypoint_cameras,
                            install=True,
                        )
                        record["molmo_keypoints"] = keypoint_manifest
                        record.setdefault("artifacts", {})[
                            "molmo_keypoints"
                        ] = _run_relative(
                            keypoint_artifact_dir
                            / "molmo_keypoint_grasp_references.json",
                            session.run_dir,
                        )
                        uniform_overlay_names = {
                            "camera_A_coordinate_overlay.png",
                            "camera_B_coordinate_overlay.png",
                        }
                        before_images = [
                            path
                            for path in before_images
                            if path.name not in uniform_overlay_names
                        ]
                        for view in keypoint_manifest["views"]:
                            overlay = Path(str(view["accepted_overlay"])).resolve()
                            if overlay.is_file() and overlay not in before_images:
                                before_images.append(overlay)
                        if keypoint_manifest["status"] != "READY":
                            raise AutoExplorationError(
                                "Molmo keypoint pipeline produced no grasp reference "
                                f"with confidence > {keypoint_threshold:.3f} and valid "
                                "calibrated geometry; Claude planning and robot execution "
                                "were not started"
                            )
                        reference_policy = "molmo_confidence_filtered_keypoints"
                    record["before_images"] = [
                        _run_relative(path, session.run_dir) for path in before_images
                    ]
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="perception",
                        payload={
                            "result": perception,
                            "saved_result": saved,
                            "saved_result_path": _run_relative(saved_path, session.run_dir),
                            "before_images": record["before_images"],
                        },
                    )
                    update_iteration_camera_report(
                        iteration,
                        record,
                        saved,
                        saved_path,
                    )
                    if stopped():
                        break

                    if workspace_recovery is not None and workspace_recovery.required:
                        dx_mm, dy_mm = workspace_recovery.requested_translation_xy_mm
                        set_status(
                            f"### Iteration {iteration}/{limit_label}: workspace recovery\n\n"
                            "The garment center is outside the configured range. "
                            f"Planning the mandatory inward move `dx={dx_mm:.1f} mm`, "
                            f"`dy={dy_mm:.1f} mm` before any opening action."
                        )
                    else:
                        set_status(
                            f"### Iteration {iteration}/{limit_label}: Claude thinking\n\n"
                            "Planning one restricted action to open and spread the current garment."
                        )
                    proposal_feedback: str | None = None
                    proposal: ExplorationProposal | None = None
                    max_plan_attempts = max_replans + 1
                    for plan_attempt in range(1, max_plan_attempts + 1):
                        record["plan_attempt"] = plan_attempt
                        try:
                            proposal = client.plan(
                                before_images,
                                session,
                                planning_objective,
                                feedback=proposal_feedback,
                                history=state.history,
                                phase_callback=planning_phase_callback,
                                reference_policy=reference_policy,
                                workspace_recovery=workspace_recovery,
                            )
                            break
                        except (ExplorationTimeoutError, ReferenceReselectionExhaustedError):
                            raise
                        except Exception as exc:
                            if plan_attempt >= max_plan_attempts:
                                raise
                            proposal_feedback = _replan_feedback(exc, proposal=proposal)
                            save_agent_artifact(
                                iteration,
                                record,
                                phase=f"replan_{plan_attempt:02d}_feedback",
                                payload={
                                    "attempt": plan_attempt,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "feedback": proposal_feedback,
                                },
                            )
                    if proposal is None:
                        raise AutoExplorationError("Claude did not return an exploration proposal")
                    plan_result = client.last_plan_result
                    visual_plan_result = client.last_visual_plan_result
                    if plan_result is None:
                        raise AutoExplorationError("Claude plan completed without a raw result")
                    if visual_plan_result is None:
                        raise AutoExplorationError(
                            "Claude visual planning completed without a raw result"
                        )
                    with state_lock:
                        state.proposal = proposal
                    proposal_panel.content = _proposal_markdown(proposal, exploration_source(proposal))
                    render_grasp_target_visualization(
                        proposal.actions,
                        frames,
                        plan_status="candidate",
                    )
                    source = exploration_source(proposal)
                    record["proposal"] = proposal.as_dict()
                    record["proposal_source"] = source
                    record["planning_timing"] = dict(client.last_plan_timing)
                    if client.last_rejected_visual_references:
                        record["rejected_visual_references"] = list(
                            client.last_rejected_visual_references
                        )
                        save_agent_artifact(
                            iteration,
                            record,
                            phase="rejected_visual_references",
                            payload=client.last_rejected_visual_references,
                        )
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="claude_visual_plan",
                        payload=visual_plan_result,
                    )
                    save_agent_artifact(iteration, record, phase="claude_plan", payload=plan_result)
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="proposal",
                        payload=proposal.as_dict(),
                    )
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="proposal_source",
                        payload=source,
                        suffix=".py",
                    )
                    source_path = session.workspace / "_auto_exploration.py"
                    source_path.write_text(source, encoding="utf-8")
                    try:
                        validation_feedback: str | None = None
                        controller = None
                        for validation_attempt in range(1, max_replans + 2):
                            try:
                                preflight = session.runner.preflight(source_path.name)
                                if preflight.error:
                                    raise ExperimentValidationError(preflight.error)
                                set_status(
                                    f"### Iteration {iteration}/{limit_label}: controller IK\n\n"
                                    "Validating every target without motion."
                                )
                                controller = validate_controller_trajectory(
                                    session.robot_config, preflight.actions
                                )
                                break
                            except Exception as exc:
                                if (
                                    validation_attempt >= max_replans + 1
                                    or not _is_preexecution_replan_error(exc)
                                ):
                                    raise
                                validation_feedback = _replan_feedback(exc, proposal=proposal)
                                set_status(
                                    f"### Iteration {iteration}/{limit_label}: Claude replanning "
                                    f"({validation_attempt}/{max_replans})\n\n"
                                    f"The candidate was rejected before execution: `{exc}`"
                                )
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase=f"replan_{validation_attempt:02d}_ik_feedback",
                                    payload={
                                        "attempt": validation_attempt,
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "feedback": validation_feedback,
                                    },
                                )
                                proposal = client.plan(
                                    before_images,
                                    session,
                                    planning_objective,
                                    feedback=validation_feedback,
                                    history=state.history,
                                    phase_callback=planning_phase_callback,
                                    reference_policy=reference_policy,
                                    workspace_recovery=workspace_recovery,
                                )
                                replanned_result = client.last_plan_result
                                replanned_visual_result = client.last_visual_plan_result
                                source = exploration_source(proposal)
                                record["proposal"] = proposal.as_dict()
                                record["proposal_source"] = source
                                record["planning_timing"] = dict(
                                    client.last_plan_timing
                                )
                                if client.last_rejected_visual_references:
                                    record.setdefault(
                                        "rejected_visual_references", []
                                    ).extend(client.last_rejected_visual_references)
                                source_path.write_text(source, encoding="utf-8")
                                if replanned_visual_result is not None:
                                    save_agent_artifact(
                                        iteration,
                                        record,
                                        phase=(
                                            f"replan_{validation_attempt:02d}_visual_plan"
                                        ),
                                        payload=replanned_visual_result,
                                    )
                                if replanned_result is not None:
                                    save_agent_artifact(
                                        iteration,
                                        record,
                                        phase=f"replan_{validation_attempt:02d}_claude_plan",
                                        payload=replanned_result,
                                    )
                                with state_lock:
                                    state.proposal = proposal
                                proposal_panel.content = _proposal_markdown(
                                    proposal, source
                                )
                                render_grasp_target_visualization(
                                    proposal.actions,
                                    frames,
                                    plan_status="replanned candidate",
                                )
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase=f"replan_{validation_attempt:02d}_proposal",
                                    payload=proposal.as_dict(),
                                )
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase=f"replan_{validation_attempt:02d}_source",
                                    payload=source,
                                    suffix=".py",
                                )
                        if controller is None:
                            raise AutoExplorationError("controller validation returned no result")
                        if workspace_recovery is not None and workspace_recovery.required:
                            recovery_validation = validate_garment_recovery_actions(
                                preflight.actions, workspace_recovery
                            )
                            record["garment_workspace_recovery_validation"] = (
                                recovery_validation
                            )
                            save_agent_artifact(
                                iteration,
                                record,
                                phase="garment_workspace_recovery_validation",
                                payload=recovery_validation,
                            )
                        animation_frames = kinematics.build_animation(
                            preflight.actions,
                            robot.init_joints_deg,
                            robot.orientation_roll_deg,
                            robot.orientation_pitch_deg,
                            yaw_offset_deg=robot.init_pose_mm_deg[5],
                            joint_targets_rad=controller.joint_targets_rad,
                        )
                        if not animation_frames:
                            raise AutoExplorationError(
                                "xArm URDF animation returned no frames"
                            )
                        apply_robot_frame(animation_frames[0])
                        record["requested_actions"] = preflight.actions
                        record["controller_warning_code"] = controller.controller_warning_code
                        record["robot_animation_frames"] = len(animation_frames)
                        target_visualization = render_grasp_target_visualization(
                            preflight.actions,
                            frames,
                            plan_status="preflight + controller IK validated",
                            output_dir=(
                                auto_results_dir / f"iteration_{iteration:03d}"
                            ),
                        )
                        record["grasp_target_visualization"] = target_visualization
                        update_iteration_camera_report(
                            iteration,
                            record,
                            saved,
                            saved_path,
                            target_visualization=target_visualization,
                            visual_plan_result=client.last_visual_plan_result,
                        )
                        save_agent_artifact(
                            iteration,
                            record,
                            phase="grasp_target_visualization",
                            payload=target_visualization,
                        )
                        save_agent_artifact(
                            iteration,
                            record,
                            phase="preflight",
                            payload={
                                "source": preflight.source,
                                "actions": preflight.actions,
                                "stdout": preflight.stdout,
                                "error": preflight.error,
                            },
                        )
                        save_agent_artifact(
                            iteration,
                            record,
                            phase="controller_ik",
                            payload=controller,
                        )
                        if stopped():
                            break

                        rollout_recorder: DualRealSenseRolloutRecorder | None = None
                        recording_thread: threading.Thread | None = None
                        recording_result: dict[str, Any] = {}
                        recording_errors: list[str] = []
                        recording_dir = (
                            auto_results_dir
                            / f"iteration_{iteration:03d}"
                            / (
                                "rollout_recording_"
                                + datetime.now(timezone.utc).strftime(
                                    "%Y%m%dT%H%M%S%fZ"
                                )
                            )
                        )
                        if record_rollouts:
                            set_status(
                                f"### Iteration {iteration}/{limit_label}: starting A/B recording\n\n"
                                "Opening both RealSense cameras after perception/IK and before "
                                "physical execution. A recording failure blocks execution."
                            )
                            rollout_recorder = DualRealSenseRolloutRecorder(
                                config,
                                recording_dir,
                                record_bag=recording_native,
                                record_depth_video=True,
                                record_composite=True,
                                codec=recording_codec,
                                warmup_frames=recording_warmup_frames,
                            )
                            try:
                                rollout_recorder.start()
                            except BaseException as exc:
                                raise AutoExplorationError(
                                    "rollout recording failed before physical execution; "
                                    f"no robot command was sent: {type(exc).__name__}: {exc}"
                                ) from exc

                            def record_rollout_video() -> None:
                                try:
                                    recording_result["manifest"] = rollout_recorder.record()
                                except BaseException as exc:
                                    recording_errors.append(
                                        f"{type(exc).__name__}: {exc}"
                                    )

                            recording_thread = threading.Thread(
                                target=record_rollout_video,
                                daemon=True,
                                name=f"rollout-recorder-iteration-{iteration}",
                            )
                            recording_thread.start()
                            time.sleep(0.25)
                            if recording_errors:
                                rollout_recorder.request_stop(
                                    "recording_failed_before_execution"
                                )
                                recording_thread.join(timeout=5.0)
                                raise AutoExplorationError(
                                    "rollout recording failed before physical execution; "
                                    f"no robot command was sent: {recording_errors[-1]}"
                                )
                            record["rollout_recording"] = {
                                "status": "recording",
                                "directory": _run_relative(
                                    recording_dir, session.run_dir
                                ),
                            }

                        set_status(
                            f"### Iteration {iteration}/{limit_label}: executing\n\n"
                            "Executing exactly one validated physical rollout; "
                            "the Viser xArm mesh is following the gripper trajectory."
                            + (
                                " Camera A/B RGB-D recording is active."
                                if record_rollouts
                                else ""
                            )
                        )
                        result: dict[str, Any] | None = None
                        try:
                            animate_robot_frames(animation_frames)
                            result = session.run_experiment(
                                source_path.name,
                                real=True,
                                confirmed=True,
                                single_view_confirmed=(
                                    json.loads(
                                        (session.run_dir / "run_metadata.json").read_text(
                                            encoding="utf-8"
                                        )
                                    ).get("last_perception_mode")
                                    == "single_camera_rgbd"
                                ),
                                notes=f"Automatic Claude exploration iteration {iteration}.",
                            )
                        finally:
                            home_outcome = session.last_return_home_outcome
                            home_completed = bool(
                                home_outcome and home_outcome.get("completed")
                            )
                            stop_robot_animation(reset_to_home=home_completed)
                            if home_outcome is not None:
                                record["mandatory_return_home"] = home_outcome
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase="mandatory_return_home",
                                    payload=home_outcome,
                                )
                            if rollout_recorder is not None:
                                rollout_recorder.request_stop(
                                    "rollout_and_return_home_completed"
                                )
                            if recording_thread is not None:
                                # Camera pipelines are released before H.264 finalization,
                                # but encoding the five output videos can take longer than
                                # the old 10-second recorder shutdown budget.
                                recording_thread.join(timeout=300.0)
                                if recording_thread.is_alive():
                                    recording_errors.append(
                                        "recording thread did not stop within 300 seconds"
                                    )
                                    if rollout_recorder is not None:
                                        rollout_recorder.close()
                                    recording_thread.join(timeout=3.0)
                            if rollout_recorder is not None:
                                manifest = recording_result.get("manifest")
                                manifest_path = recording_dir / "recording_manifest.json"
                                if manifest is None and manifest_path.is_file():
                                    manifest = json.loads(
                                        manifest_path.read_text(encoding="utf-8")
                                    )
                                recording_payload = {
                                    "status": (
                                        "failed" if recording_errors else "completed"
                                    ),
                                    "directory": _run_relative(
                                        recording_dir, session.run_dir
                                    ),
                                    "manifest": manifest,
                                    "errors": list(recording_errors),
                                }
                                record["rollout_recording"] = recording_payload
                                save_agent_artifact(
                                    iteration,
                                    record,
                                    phase="rollout_recording",
                                    payload=recording_payload,
                                )
                        if result is None:
                            raise AutoExplorationError(
                                "physical rollout returned no result"
                            )
                        record["execution"] = result
                        save_agent_artifact(iteration, record, phase="execution", payload=result)
                        if not result.get("execution_completed"):
                            errors = result.get("robot_errors") or []
                            raise AutoExplorationError(
                                "physical rollout did not complete; automatic loop stopped: "
                                f"{errors}"
                            )
                    finally:
                        if source_path.is_file():
                            source_path.unlink()
                    if stopped():
                        break

                    set_status(
                        f"### Iteration {iteration}/{limit_label}: settling\n\n"
                        f"Waiting `{settle_s:.1f}s`, then resuming CamA monitoring."
                    )
                    time.sleep(settle_s)
                    monitor.start()
                    time.sleep(0.5)
                    set_status(
                        f"### Iteration {iteration}/{limit_label}: evaluating\n\n"
                        "Comparing before/after garment views with Claude."
                    )
                    monitor.stop()
                    after_frames = capture_two_view_rgbd(config)
                    render_capture_frames(after_frames)
                    after_dir = (
                        auto_results_dir
                        / f"iteration_{iteration:03d}_after"
                    )
                    after_images = _save_frame_images(after_frames, after_dir)
                    after_perception = session.locate_cloth_center(
                        config, frames=after_frames
                    )
                    after_saved, after_saved_path = _load_latest_perception(session)
                    if after_saved is None or after_saved_path is None:
                        raise AutoExplorationError(
                            "post-action perception completed without saved result"
                        )
                    evaluation_depth_scale = evaluation_depth_ranges(
                        (saved, saved_path),
                        (after_saved, after_saved_path),
                    )
                    evaluation_before_images = evaluation_perception_image_paths(
                        saved,
                        saved_path,
                        stage="BEFORE",
                        depth_ranges=evaluation_depth_scale,
                    )
                    evaluation_after_images = evaluation_perception_image_paths(
                        after_saved,
                        after_saved_path,
                        stage="AFTER",
                        depth_ranges=evaluation_depth_scale,
                    )
                    record["after_images"] = [
                        _run_relative(path, session.run_dir) for path in after_images
                    ]
                    record["evaluation_before_images"] = [
                        _run_relative(path, session.run_dir)
                        for path in evaluation_before_images
                    ]
                    record["evaluation_after_images"] = [
                        _run_relative(path, session.run_dir)
                        for path in evaluation_after_images
                    ]
                    record["after_perception"] = after_perception
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="after_capture",
                        payload={
                            "images": record["after_images"],
                            "frame_labels": [frame.label for frame in after_frames],
                        },
                    )
                    evaluation = client.evaluate(
                        evaluation_before_images,
                        evaluation_after_images,
                        proposal=proposal,
                        objective=objective,
                        run_dir=session.run_dir,
                        rollout_recording_dir=(
                            recording_dir
                            if record_rollouts
                            and record.get("rollout_recording", {}).get("status")
                            == "completed"
                            else None
                        ),
                        skill_guidance=run_skill_prompt(),
                        workspace_recovery=workspace_recovery,
                    )
                    evaluation_result = client.last_evaluation_result
                    if evaluation_result is None:
                        raise AutoExplorationError(
                            "Claude evaluation completed without a raw result"
                        )
                    save_agent_artifact(
                        iteration,
                        record,
                        phase="claude_evaluation",
                        payload=evaluation_result,
                    )
                    # Older evaluator adapters may return an evaluation-shaped
                    # object without the optional skill_update field.
                    skill_update = getattr(evaluation, "skill_update", None)
                    skill_review = run_skill_ledger.stage_skill_update(
                        skill_update,
                        iteration=iteration,
                        source="evaluation",
                    )
                    if skill_review is not None:
                        record["skill_review"] = skill_review.as_dict()
                        save_agent_artifact(
                            iteration,
                            record,
                            phase="skill_review",
                            payload=skill_review.as_dict(),
                        )
                        client.skill_guidance = run_skill_prompt()
                        client.skill_names = tuple(
                            skill.name for skill in skill_store.approved()
                        )
                    evidence = build_evidence_record(
                        record,
                        iteration=iteration,
                        run_dir=session.run_dir,
                    )
                    evidence_paths = persist_evidence_record(
                        session.run_dir,
                        evidence,
                        iteration_dir=auto_results_dir / f"iteration_{iteration:03d}",
                    )
                    record["evidence"] = evidence
                    record["evidence_artifacts"] = evidence_paths
                    run_skill_ledger.append_experience(
                        {
                            "created_at": _now(),
                            "iteration": iteration,
                            "objective": objective,
                            "proposal": proposal.as_dict(),
                            "evaluation": evaluation.as_dict(),
                            "skill_review": (
                                skill_review.as_dict()
                                if skill_review is not None
                                else None
                            ),
                            "evidence": evidence,
                        }
                    )
                    with state_lock:
                        state.evaluation = evaluation
                        state.history.append(
                            {
                                "iteration": iteration,
                                "plan_status": "executed",
                                "iteration_mode": record.get(
                                    "iteration_mode", "garment_opening"
                                ),
                                "proposal": proposal.as_dict(),
                                "execution_completed": bool(
                                    record.get("execution", {}).get("execution_completed")
                                ),
                                "before_images": list(record.get("before_images", [])),
                                "after_images": list(record.get("after_images", [])),
                                "evaluation": evaluation.as_dict(),
                            }
                        )
                    stage_rows = []
                    stage_evidence = []
                    for label, stage in (
                        ("target selection", evaluation.target_selection),
                        ("grasp acquisition", evaluation.grasp_acquisition),
                        ("target structure", evaluation.target_structure_acquired),
                        ("transport", evaluation.transport),
                        ("laydown", evaluation.laydown),
                    ):
                        stage_rows.append(
                            f"| {label} | `{stage.status}` | `{stage.confidence:.2f}` |"
                        )
                        stage_evidence.append(
                            f"- **{label}**: " + "; ".join(stage.evidence)
                        )
                    metrics = evaluation.task_progress.metrics
                    keep = ", ".join(evaluation.next_experiment.keep) or "none"
                    change = ", ".join(evaluation.next_experiment.change) or "STOP"
                    evaluation_panel.content = "\n".join(
                        [
                            "### Stage-wise evaluator",
                            "",
                            "| stage | status | confidence |",
                            "|---|---|---:|",
                            *stage_rows,
                            "",
                            f"**Task progress:** `{evaluation.task_progress.status}` "
                            f"(`{evaluation.task_progress.confidence:.2f}`)",
                            "",
                            f"- visible area delta: `{metrics.visible_area_delta}`",
                            f"- overlap delta: `{metrics.overlap_delta}`",
                            f"- relief delta: `{metrics.relief_delta}`",
                            f"- boundary change: {metrics.boundary_change}",
                            f"- earliest failure: `{evaluation.earliest_failure_stage}`",
                            "",
                            "**Evidence**",
                            "",
                            *stage_evidence,
                            "",
                            "**Next experiment**",
                            "",
                            f"- keep: `{keep}`",
                            f"- change: `{change}`",
                            f"- reason: {evaluation.next_experiment.reason}",
                        ]
                    )
                    render_history()
                    save_auto_record(
                        f"iteration_{iteration:03d}.json",
                        {**record, "evaluation": evaluation.as_dict(), "completed_at": _now()},
                    )
                    run_log_panel.content = (
                        f"### Agent log\n\n"
                        f"Saved iteration `{iteration}` under `{_run_relative(auto_results_dir, session.run_dir)}`.\n\n"
                        f"Claude artifacts: `{len(record.get('artifacts', {}))}`"
                    )
                    record_saved = True
                    consecutive_recoverable_failures = 0
                    if evaluation.stop and not (
                        workspace_recovery is not None
                        and workspace_recovery.required
                    ):
                        set_status(
                            f"### Automatic exploration stopped after iteration {iteration}\n\n"
                            "Claude judged that safe grounded continuation is not currently "
                            f"possible: {evaluation.reason}"
                        )
                        break
                    objective = (
                        DEFAULT_AUTO_OBJECTIVE
                        if workspace_recovery is not None
                        and workspace_recovery.required
                        else evaluation.next_objective
                    )
                    with state_lock:
                        state.objective = objective
                except ExplorationTimeoutError as exc:
                    if client.last_plan_timing:
                        record["planning_timing"] = dict(client.last_plan_timing)
                    if client.last_rejected_visual_references:
                        record["rejected_visual_references"] = list(
                            client.last_rejected_visual_references
                        )
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if (
                        continue_on_recoverable_errors
                        and _is_recoverable_viewer_error(exc, record)
                        and consecutive_recoverable_failures
                        < max_consecutive_recoverable_failures
                    ):
                        consecutive_recoverable_failures += 1
                        record["status"] = "RECOVERABLE_ERROR"
                        record["traceback"] = traceback.format_exc()
                        record["recovery"] = {
                            "enabled": True,
                            "consecutive_failure": consecutive_recoverable_failures,
                            "max_consecutive_failures": max_consecutive_recoverable_failures,
                            "next_step": "fresh perception and new planning iteration",
                        }
                        record["completed_at"] = _now()
                        save_auto_record(
                            f"iteration_{iteration:03d}_recoverable.json",
                            record,
                        )
                        record_saved = True
                        with state_lock:
                            state.history.append(
                                {
                                    "iteration": iteration,
                                    "plan_status": "recoverable_error",
                                    "error": record["error"],
                                    "evaluation": {},
                                }
                            )
                        render_history()
                        set_status(
                            f"### Recoverable Claude timeout at iteration {iteration}\n\n"
                            f"`{exc}`. Continuing with a fresh perception/planning iteration "
                            f"({consecutive_recoverable_failures}/"
                            f"{max_consecutive_recoverable_failures})."
                        )
                        if recovery_backoff_s > 0:
                            time.sleep(recovery_backoff_s)
                        continue
                    save_auto_record(
                        f"iteration_{iteration:03d}_failed.json",
                        {**record, "completed_at": _now()},
                    )
                    record_saved = True
                    with state_lock:
                        state.history.append(
                            {
                                "iteration": iteration,
                                "plan_status": "claude_timeout",
                                "evaluation": {},
                            }
                        )
                    render_history()
                    execution_note = (
                        "The completed rollout is preserved, and no additional Claude "
                        "replan or robot action will run."
                        if "execution" in record
                        else "No Claude replan or robot execution was attempted."
                    )
                    set_status(
                        f"### Automatic exploration stopped: Claude timeout\n\n"
                        f"`{exc}` during iteration `{iteration}`. The loop stopped "
                        f"immediately. {execution_note}"
                    )
                    break
                except Exception as exc:
                    if client.last_plan_timing:
                        record["planning_timing"] = dict(client.last_plan_timing)
                    if client.last_rejected_visual_references:
                        record["rejected_visual_references"] = list(
                            client.last_rejected_visual_references
                        )
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if record.get("execution") is not None and "evidence_artifacts" not in record:
                        try:
                            evidence = build_evidence_record(
                                record,
                                iteration=iteration,
                                run_dir=session.run_dir,
                            )
                            record["evidence"] = evidence
                            record["evidence_artifacts"] = persist_evidence_record(
                                session.run_dir,
                                evidence,
                                iteration_dir=auto_results_dir / f"iteration_{iteration:03d}",
                            )
                        except BaseException as evidence_exc:
                            record["evidence_persistence_error"] = str(evidence_exc)
                    if (
                        continue_on_recoverable_errors
                        and _is_recoverable_viewer_error(exc, record)
                        and consecutive_recoverable_failures
                        < max_consecutive_recoverable_failures
                    ):
                        consecutive_recoverable_failures += 1
                        record["status"] = "RECOVERABLE_ERROR"
                        record["traceback"] = traceback.format_exc()
                        record["recovery"] = {
                            "enabled": True,
                            "consecutive_failure": consecutive_recoverable_failures,
                            "max_consecutive_failures": max_consecutive_recoverable_failures,
                            "next_step": "fresh perception and new planning iteration",
                        }
                        record["completed_at"] = _now()
                        save_auto_record(
                            f"iteration_{iteration:03d}_recoverable.json",
                            record,
                        )
                        record_saved = True
                        with state_lock:
                            state.history.append(
                                {
                                    "iteration": iteration,
                                    "plan_status": "recoverable_error",
                                    "error": record["error"],
                                    "evaluation": {},
                                }
                            )
                        render_history()
                        set_status(
                            f"### Recoverable pre-execution failure at iteration {iteration}\n\n"
                            f"`{exc}`. Continuing with a fresh perception/planning iteration "
                            f"({consecutive_recoverable_failures}/"
                            f"{max_consecutive_recoverable_failures})."
                        )
                        if recovery_backoff_s > 0:
                            time.sleep(recovery_backoff_s)
                        continue
                    save_auto_record(
                        f"iteration_{iteration:03d}_failed.json",
                        {**record, "completed_at": _now()},
                    )
                    record_saved = True
                    with state_lock:
                        state.history.append(
                            {
                                "iteration": iteration,
                                "plan_status": "hard_failed",
                                "evaluation": {},
                            }
                        )
                    render_history()
                    failure_note = (
                        "The completed physical rollout and recordings were preserved. "
                        "This failure occurred during post-rollout evaluation; no robot retry "
                        "or pre-execution replan was attempted."
                        if record.get("execution", {}).get("execution_completed")
                        else (
                            "No physical retry was attempted. Pre-execution replanning was "
                            "bounded by `--max-replans`."
                        )
                    )
                    set_status(
                        f"### Automatic exploration hard-stopped at iteration {iteration}\n\n"
                        f"`{type(exc).__name__}: {exc}`\n\n"
                        f"{failure_note} Inspect the saved iteration record."
                    )
                    break
                finally:
                    if not record_saved:
                        save_auto_record(
                            f"iteration_{iteration:03d}_stopped.json",
                            {
                                **record,
                                "stop_requested": stopped(),
                                "completed_at": _now(),
                            },
                        )
                    if not stopped():
                        monitor.start()
            if iterations is not None and not stopped() and iteration >= iterations:
                set_status(
                    f"### Automatic exploration reached its limit ({iterations} iterations)\n\n"
                    "Review the CamA stream and saved before/after records."
                )
        finally:
            try:
                run_skill_synthesis = run_skill_ledger.finalize(skill_store)
                save_agent_artifact(
                    iteration,
                    {"run_skill_synthesis": run_skill_synthesis},
                    phase="run_skill_synthesis",
                    payload=run_skill_synthesis,
                )
                set_status(
                    "### Run skill synthesis completed\n\n"
                    f"Synthesized {run_skill_synthesis['skill_group_count']} "
                    "run-local skill group(s); global persistence was deferred "
                    "until this run ended."
                )
            except Exception as exc:
                set_status(
                    "### Run skill synthesis failed\n\n"
                    f"`{type(exc).__name__}: {exc}`"
                )
            stop_robot_animation()
            set_running(False)

    @start_button.on_click
    def _(event: Any) -> None:
        nonlocal auto_thread
        with state_lock:
            if state.running:
                return
            state.stop_requested = False
            state.history = []
        state.objective = DEFAULT_AUTO_OBJECTIVE
        selected_iterations = int(iteration_slider.value)
        iterations = None if selected_iterations == 0 else selected_iterations
        set_running(True)
        auto_thread = threading.Thread(
            target=run_loop,
            args=(iterations,),
            daemon=True,
            name="claude-auto-exploration-loop",
        )
        auto_thread.start()

    try:
        monitor.start()
        # Automatic mode starts immediately when the module is launched. The
        # restart button remains available for a fresh run after a stop.
        with state_lock:
            state.stop_requested = False
            state.history = []
            state.objective = DEFAULT_AUTO_OBJECTIVE
        selected_iterations = int(iteration_slider.value)
        initial_iterations = None if selected_iterations == 0 else selected_iterations
        set_running(True)
        auto_thread = threading.Thread(
            target=run_loop,
            args=(initial_iterations,),
            daemon=True,
            name="claude-auto-exploration-loop",
        )
        auto_thread.start()
        print(f"Viser Claude automatic exploration console: http://{host}:{port}")
        print(f"Run workspace: {session.workspace}")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        request_stop(None)
    finally:
        monitor.stop()
        planning_timer_stop.set()
        planning_timer_thread.join(timeout=1.0)
        if auto_thread is not None and auto_thread.is_alive():
            auto_thread.join(timeout=2.0)
        server.stop()
    return 0


def _load_session(
    root: Path,
    run_dir: Path | None,
    run_id: str | None,
    robot_config: Path | None,
) -> AgentSession:
    from .free_exploration import _load_or_create_session

    return _load_or_create_session(root, run_dir, run_id, robot_config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--robot-config",
        default="config/robot.example.json",
        help=(
            "robot configuration JSON (default: config/robot.example.json; "
            "uses absolute camera depth without live tabletop Z flooring)"
        ),
    )
    parser.add_argument("--perception-config")
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=900)
    parser.add_argument("--claude-grounding-timeout-s", type=int, default=120)
    parser.add_argument(
        "--max-replans",
        type=int,
        default=2,
        help="maximum Claude replans after pre-execution validation failure",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="number of automatic iterations; 0 means continuous until stop/hard failure",
    )
    parser.add_argument("--settle-s", type=float, default=2.0)
    recording_group = parser.add_mutually_exclusive_group()
    recording_group.add_argument(
        "--record-rollouts",
        dest="record_rollouts",
        action="store_true",
        default=True,
        help="record Camera A/B RGB, depth, composite video, timestamps, and native data during each physical rollout (default)",
    )
    recording_group.add_argument(
        "--no-record-rollouts",
        dest="record_rollouts",
        action="store_false",
        help="disable Camera A/B rollout recording",
    )
    parser.add_argument(
        "--recording-no-native",
        action="store_true",
        help="disable the SDK-native Camera A/B .db3 recordings while keeping MP4 and timestamps",
    )
    parser.add_argument("--recording-codec", default="mp4v")
    parser.add_argument("--recording-warmup-frames", type=int)
    parser.add_argument(
        "--molmo-keypoints",
        action="store_true",
        help=(
            "rerun Molmo keypoints every iteration and expose only confidence-filtered "
            "points as task grasp references"
        ),
    )
    parser.add_argument(
        "--molmo-keypoint-confidence-threshold",
        type=float,
        default=DEFAULT_MOLMO_KEYPOINT_CONFIDENCE_THRESHOLD,
        help=(
            "accept a Molmo keypoint only when confidence is strictly greater "
            "than this value"
        ),
    )
    parser.add_argument("--molmo-python", type=Path)
    parser.add_argument("--molmo-model", default="allenai/MolmoPoint-8B")
    parser.add_argument(
        "--molmo-keypoints-json",
        type=Path,
        help="optional JSON list of {name, description, color}; defaults to garment landmarks",
    )
    parser.add_argument(
        "--molmo-keypoint-camera",
        action="append",
        choices=["A", "B"],
        help="camera to query; repeat for both (default: A and B)",
    )
    parser.add_argument("--molmo-keypoint-timeout-s", type=int, default=900)
    parser.add_argument(
        "--molmo-allow-download",
        action="store_true",
        help="allow Molmo/Hugging Face files not already present locally",
    )
    parser.add_argument(
        "--continue-on-recoverable-errors",
        action="store_true",
        help=(
            "continue with a fresh perception/planning iteration after bounded "
            "pre-execution or post-home Claude failures"
        ),
    )
    parser.add_argument(
        "--max-consecutive-recoverable-failures",
        type=int,
        default=3,
        help="hard-stop after this many consecutive recoverable failures",
    )
    parser.add_argument(
        "--recovery-backoff-s",
        type=float,
        default=2.0,
        help="seconds to wait before a recoverable retry",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--enable-real",
        action="store_true",
        help="required: automatic mode sends physical xArm commands",
    )
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    robot_config = Path(args.robot_config).resolve() if args.robot_config else None
    perception_config = Path(args.perception_config).resolve() if args.perception_config else None
    session = _load_session(root, run_dir, args.run_id, robot_config)
    return run_auto_exploration_viewer(
        session,
        host=args.host,
        port=args.port,
        max_iterations=args.max_iterations,
        settle_s=args.settle_s,
        enable_real=args.enable_real,
        perception_config_path=perception_config,
        claude_binary=args.claude_binary,
        claude_timeout_s=args.claude_timeout_s,
        claude_grounding_timeout_s=args.claude_grounding_timeout_s,
        max_replans=args.max_replans,
        record_rollouts=args.record_rollouts,
        recording_native=not args.recording_no_native,
        recording_codec=args.recording_codec,
        recording_warmup_frames=args.recording_warmup_frames,
        molmo_keypoints=args.molmo_keypoints,
        molmo_keypoint_confidence_threshold=args.molmo_keypoint_confidence_threshold,
        molmo_python=args.molmo_python.resolve() if args.molmo_python else None,
        molmo_model=args.molmo_model,
        molmo_keypoints_path=(
            args.molmo_keypoints_json.resolve()
            if args.molmo_keypoints_json
            else None
        ),
        molmo_keypoint_cameras=tuple(
            args.molmo_keypoint_camera or ("A", "B")
        ),
        molmo_keypoint_timeout_s=args.molmo_keypoint_timeout_s,
        molmo_allow_download=args.molmo_allow_download,
        continue_on_recoverable_errors=args.continue_on_recoverable_errors,
        max_consecutive_recoverable_failures=args.max_consecutive_recoverable_failures,
        recovery_backoff_s=args.recovery_backoff_s,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
