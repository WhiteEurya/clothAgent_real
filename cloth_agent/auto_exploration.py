"""Standalone automatic Claude/RealSense/xArm exploration loop.

The existing manual free-exploration console remains unchanged.  This module
adds an opt-in state machine for one cautious real rollout at a time:

``Viser RGB-D preview -> RGB-D perception -> Claude plan -> preflight/IK -> execute ->
Viser RGB-D preview -> before/after Claude evaluation -> next iteration``.

Pre-execution validation failures are returned to Claude for a bounded
replanning attempt. The loop never retries after physical motion has started or
interrupts an in-progress physical command.
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
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import SafetyError
from .experiment import ExperimentValidationError
from .free_exploration import (
    ClaudeExplorationClient,
    ExplorationPlanningError,
    ExplorationTimeoutError,
    ClaudeExplorationResult,
    ExplorationProposal,
    _json_from_claude_text,
    _proposal_markdown,
    _voxel_balance_cloud,
    exploration_source,
    grounding_mcp_config,
    GROUNDING_MCP_TOOLS,
    perception_image_paths,
    validate_exploration_payload,
)
from .kinematics import AnimationFrame, XArm7Kinematics
from .garment_grounding_mcp import GarmentGrounding, GroundingToolError
from .perception import (
    CameraSpec,
    PerceptionConfig,
    RGBDFrame,
    RealSenseRGBD,
    _scalar_heatmap_rgb,
    camera_height_map_mm,
    capture_two_view_rgbd,
    load_extrinsics,
)
from .robot_api import RobotExecutionError, validate_controller_trajectory
from .rollout_recorder import DualRealSenseRolloutRecorder
from .report_figure import compose_camera_perception_report
from .session import AgentSession
from .viewer import _frame_point_cloud, _load_latest_perception


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
    }
)
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


def _planning_mode_from_history(
    history: Sequence[dict[str, Any]] | None,
) -> tuple[str, str]:
    """Choose probe-versus-expansion behavior from the last evaluator result."""

    last_evaluation: dict[str, Any] | None = None
    for item in reversed(list(history or [])):
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
        return {
            "target_selection": self.target_selection.as_dict(),
            "grasp_acquisition": self.grasp_acquisition.as_dict(),
            "target_structure_acquired": self.target_structure_acquired.as_dict(),
            "transport": self.transport.as_dict(),
            "laydown": self.laydown.as_dict(),
            "task_progress": self.task_progress.as_dict(),
            "earliest_failure_stage": self.earliest_failure_stage,
            "next_experiment": self.next_experiment.as_dict(),
        }


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
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AutoExplorationError(f"{context} must be a JSON object")
    missing = fields.difference(payload)
    unknown = set(payload).difference(fields)
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
    )


def validate_visual_plan_payload(payload: Any) -> VisualPlanDecision:
    """Validate stage-one output before exact Rxx grounding is permitted."""

    if not isinstance(payload, dict):
        raise ExplorationPlanningError("visual plan must be a JSON object")
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
    if not isinstance(raw_skills, list):
        raise ExplorationPlanningError("visual plan skill_invocations must be a list")
    skills: list[dict[str, str]] = []
    for item in raw_skills:
        if not isinstance(item, dict) or set(item) != {"name", "reason"}:
            raise ExplorationPlanningError(
                "each visual skill invocation needs exactly name and reason"
            )
        name, skill_reason = item["name"], item["reason"]
        if name != "laydown" or not isinstance(skill_reason, str) or not skill_reason.strip():
            raise ExplorationPlanningError(
                "visual skill invocation must use laydown with a non-empty reason"
            )
        skills.append({"name": name, "reason": skill_reason.strip()})
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


class ClaudeAutoClient:
    """Claude adapter that plans actions and judges before/after images."""

    def __init__(
        self,
        binary: str = "claude",
        timeout_s: int = 400,
        grounding_timeout_s: int = 120,
        max_reference_reselections: int = 2,
    ):
        if max_reference_reselections < 0 or max_reference_reselections > 10:
            raise ValueError("max_reference_reselections must be between 0 and 10")
        self.binary = binary
        self.timeout_s = timeout_s
        self.grounding_timeout_s = grounding_timeout_s
        self.max_reference_reselections = max_reference_reselections
        self.planner = ClaudeExplorationClient(binary=binary, timeout_s=timeout_s)
        self.last_plan_result: ClaudeExplorationResult | None = None
        self.last_visual_plan_result: ClaudeVisualPlanResult | None = None
        self.last_plan_timing: dict[str, float] = {}
        self.last_rejected_visual_references: list[dict[str, Any]] = []
        self.last_evaluation_result: ClaudeEvaluationResult | None = None

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
        image_text = "\n".join(f"- {path}" for path in safe_images)
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
            "Return exactly one JSON object with these fields and no others: "
            "garment_observation (string), opening_strategy (string), confidence "
            "(number 0..1), selected_reference ({camera: A|B, reference_id: Rxxx, "
            "reason: string}), motion_intent (string), expected_observation (string), "
            "safety_notes (non-empty list of strings), and optional skill_invocations "
            "(list containing only {name: laydown, reason: string}). Do not return "
            "actions, XYZ coordinates, Python, or a run function in this stage."
        )
        command = [
            self._binary(),
            "--print",
            prompt,
            "--output-format",
            "json",
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
                "You are the visual-planning stage of a cautious garment-opening "
                "robotics agent. Read only the supplied run images. Select one final "
                "Camera/Rxxx reference and return the requested JSON decision. Do not "
                "write files, execute commands, call MCP tools, or control a robot."
            ),
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
            duration_s = time.monotonic() - started
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
        try:
            decision = validate_visual_plan_payload(
                _json_from_claude_text(completed.stdout)
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
    def _validate_reference_for_stage2(
        visual: VisualPlanDecision,
        session: AgentSession,
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

        xyz = np.asarray(measurement.get("base_xyz_mm", []), dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise SelectedReferenceNotExecutableError(
                camera,
                reference_id,
                "saved calibrated Base XYZ is missing or non-finite",
                measurement=measurement,
            )

        bounds = session.robot_config.boundaries
        margin = float(session.robot_config.workspace_margin_mm)
        violations: list[str] = []
        for axis, value in (("x", float(xyz[0])), ("y", float(xyz[1]))):
            low = getattr(bounds, f"{axis}_min")
            high = getattr(bounds, f"{axis}_max")
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
        return measurement

    def _ground_final_plan(
        self,
        visual: VisualPlanDecision,
        session: AgentSession,
        objective: str,
        history: Sequence[dict[str, Any]] | None = None,
    ) -> ClaudeExplorationResult:
        root = session.run_dir.resolve()
        selected = visual.selected_reference
        planning_mode, planning_mode_instruction = _planning_mode_from_history(history)
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
            "fixed_orientation_deg": {
                "roll": session.robot_config.orientation_roll_deg,
                "pitch": session.robot_config.orientation_pitch_deg,
            },
        }
        prompt = (
            "STAGE 2 — FINAL RXX GROUNDING AND RUN GENERATION. The visual-planning "
            "stage below has already selected the final grasp reference. Do not revisit "
            "the images, compare alternatives, or change the selected reference. Call "
            "`lookup_reference` exactly once with camera="
            f"{selected['camera']} and reference_id={selected['reference_id']}. Then use "
            "that returned measurement to ground the grasp location and immediately "
            "compose the final numeric RobotAPI proposal. Claude chooses all remaining "
            "approach, grasp TCP height, lift, retreat, laydown, release, and yaw values "
            "from the visual motion intent, returned measurement, robot bounds, and "
            "safety margin. Do not call another tool. Always release before the action "
            "list ends and keep at most 12 actions.\n\n"
            f"Two-stage planning context:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            "Return exactly these fields and no others: garment_observation (string), "
            "reveal_strategy (string), confidence (number 0..1), actions (non-empty "
            "list of {name,args}), expected_observation (string), safety_notes "
            "(non-empty list of strings), and optional skill_invocations (list of "
            "{name,reason}; only laydown). For move, args must contain exactly numeric "
            "x,y,z,yaw in millimetres/degrees. The action contract permits only move, "
            "open_gripper, close_gripper, and home. Follow the planning mode in the context: "
            "in EXPLORATION, use a small reversible probe sufficient to distinguish the layer "
            "response; in VALIDATED_EXPANSION, preserve the validated grasp anchor/depth and "
            "complete a meaningful outward transport, normally covering most of the visible "
            "safe distance and at least about 40 mm when scale and workspace permit; in "
            "VALIDATED_TRANSPORT_CORRECTION, preserve acquisition and change only the proven "
            "transport direction/profile with a deliberate correction. Do not turn an "
            "exploration probe into a long pull, and do not turn a validated hypothesis into "
            "a few-millimetre re-probe. Use multiple waypoints to shape the path, not to "
            "silently change the intended net displacement."
        )
        mcp_config = grounding_mcp_config(root)
        command = [
            self._binary(),
            "--print",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            ",".join(GROUNDING_MCP_TOOLS),
            "--tools",
            "",
            "--mcp-config",
            json.dumps(mcp_config, ensure_ascii=False, separators=(",", ":")),
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--add-dir",
            str(root),
            "--system-prompt",
            (
                "You are the final grounding/compiler stage of a garment-opening "
                "robotics agent. The visual decision is fixed. Call the single exact "
                "Rxxx lookup once, then return only the final JSON proposal. Do not "
                "read images, write files, execute commands, or control a robot."
            ),
        ]
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
                },
                failed=True,
            )
            raise ExplorationTimeoutError(
                f"Claude final grounding timed out after "
                f"{self.grounding_timeout_s} seconds"
            ) from exc
        duration_s = time.monotonic() - started
        if completed.returncode != 0:
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
                },
                failed=True,
            )
            raise ExplorationPlanningError(
                f"Claude final grounding exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            proposal = validate_exploration_payload(
                _json_from_claude_text(completed.stdout)
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
            if grounding_error_mm > 2.0:
                raise ExplorationPlanningError(
                    "final grasp XY does not use the visually selected Rxxx measurement: "
                    f"selected={selected['camera']}/{selected['reference_id']} "
                    f"expected_xy={expected_xy.tolist()} actual_xy={actual_xy.tolist()} "
                    f"error={grounding_error_mm:.1f} mm"
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
            "grounding_verification": {
                "measurement": measurement,
                "grasp_xy_error_mm": grounding_error_mm,
            },
            "proposal": proposal.as_dict(),
        }
        self.planner._save_invocation_log(root, payload)
        return result

    def plan(
        self,
        image_paths: list[Path],
        session: AgentSession,
        objective: str,
        feedback: str | None = None,
        history: list[dict[str, Any]] | None = None,
        phase_callback: Callable[[str, str, float], None] | None = None,
    ) -> ExplorationProposal:
        previous_visual_result = self.last_visual_plan_result
        self.last_plan_result = None
        self.last_visual_plan_result = None
        self.last_plan_timing = {}
        self.last_rejected_visual_references = []
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
        visual_prompt = (
            "Observe only the current garment shown in the supplied Camera A/B "
            "images. The goal is to make this garment as open and spread on the "
            "table as safely possible. Use the visible RGB, garment boundary, "
            "height-above-table, height-gradient/occlusion, and Rxxx overlay evidence "
            "without assuming a garment category or semantic part labels. Rxxx "
            "markers are uniform coordinate references, not ranked candidates. "
            "Choose the single visible reference that best supports the next opening "
            "motion and state the intended transport direction and approximate useful "
            "distance. Apply the following mode exactly:\n"
            f"{planning_mode_instruction}\n"
            "This zero-shot visual stage has no prior coordinates or action values. Do not "
            "invent them; use prior evaluation only to decide whether this is a probe or an "
            "expansion."
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
                    self._validate_reference_for_stage2(candidate.decision, session)
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
            response = self._ground_final_plan(
                visual_result.decision,
                session,
                prompt_objective,
                history_items,
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
        run_dir: Path,
        rollout_recording_dir: Path | None = None,
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
        image_lines = ["Before images:"]
        image_lines.extend(f"- {path.resolve()}" for path in before_images)
        image_lines.append("After images:")
        image_lines.extend(f"- {path.resolve()}" for path in after_images)
        if video_evidence_images:
            image_lines.append(
                "Rollout temporal contact sheets (chronological left-to-right, top-to-bottom):"
            )
            image_lines.extend(f"- {path.resolve()}" for path in video_evidence_images)
        if video_references:
            image_lines.append(
                "Original rollout MP4 references (provenance only; use the extracted contact "
                "sheets as visual evidence):"
            )
            image_lines.extend(f"- {path.resolve()}" for path in video_references)
        if video_evidence_errors:
            image_lines.append("Video evidence extraction caveats:")
            image_lines.extend(f"- {message}" for message in video_evidence_errors)
        prompt = (
            "Compare the before and after garment images after one robot opening action. "
            "Evaluate the action stage by stage instead of collapsing it into one useful flag. "
            "Use only directly visible evidence. The before/after views are static. When "
            "chronological rollout contact sheets are supplied, use them to assess acquisition, "
            "transport, and laydown over time; otherwise mark those stages UNKNOWN when jaw "
            "motion or layer identity cannot actually be established. Never infer success "
            "only because the commanded action should have produced it. Judge task progress "
            "toward making the garment open and spread: more visible area, less overlap, lower "
            "relief, and a useful tabletop laydown. Do not invent numeric measurements. For "
            "visible_area_delta, overlap_delta, and relief_delta, use INCREASED, DECREASED, "
            "UNCHANGED, or UNKNOWN unless an exact numeric measurement is explicitly supplied.\n\n"
            "The keep/change decision is mandatory and causal. Put every parameter or strategy "
            "that the evidence supports under keep. Put only the earliest failing/unsupported "
            "choice and necessary downstream choices under change. Do not change a supported "
            "grasp anchor or grasp depth merely because transport failed. For example, when "
            "acquisition and target-layer motion are supported but transport is insufficient, "
            "keep grasp_anchor and grasp_depth, and change pull_direction and/or pull_distance. "
            "The change list must be non-empty when another safe grounded experiment exists. "
            "Use an empty change list only when the garment is already as open as this setup can "
            "reasonably achieve, or continuing is unsafe, visually ungrounded, or blocked by a "
            "hard physical/infrastructure condition.\n\n"
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
                for path in [*before_images, *after_images, *video_evidence_images]
            ),
            video_references=tuple(str(path.resolve()) for path in video_references),
            video_evidence_errors=tuple(video_evidence_errors),
        )
        self.last_evaluation_result = result
        self._save_evaluation_log(root, result.as_dict())
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
            self.X_base_camera = load_extrinsics(self.spec.extrinsics_file)
            if not self._start_camera(camera):
                return
            for _ in range(min(self.config.warmup_frames, 5)):
                if self.stop_event.is_set():
                    return
                camera.read()
            while not self.stop_event.is_set():
                rgb, depth_m = camera.read()
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
    """Build Camera A/B temporal evidence while preserving original MP4 references."""

    recording_dir = recording_dir.resolve()
    manifest_path = recording_dir / "recording_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[tuple[str, Path]] = []
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
    composite_relative = manifest.get("composite_video")
    if isinstance(composite_relative, str) and composite_relative.strip():
        candidates.append(("AB_DEPTH", recording_dir / composite_relative))
    elif (recording_dir / "composite_AB_depth.mp4").is_file():
        candidates.append(("AB_DEPTH", recording_dir / "composite_AB_depth.mp4"))

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
    objective: str = (
        "Take one planning-mode-appropriate action that makes the current garment as open and spread as safely possible."
    )
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
    claude_timeout_s: int = 400,
    claude_grounding_timeout_s: int = 120,
    max_replans: int = 2,
    record_rollouts: bool = False,
    recording_native: bool = True,
    recording_codec: str = "mp4v",
    recording_warmup_frames: int | None = None,
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
        root / "assets" / "robots" / "xarm7" / "xarm7.urdf"
    ).resolve()
    perception_path = (
        perception_config_path
        or root / "config" / "perception.free_exploration.json"
    ).expanduser().resolve()
    config = PerceptionConfig.load(root, perception_path)
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
    server.scene.add_frame("/robot_base", axes_length=0.15, axes_radius=0.006)
    server.scene.add_frame("/xarm", show_axes=False)
    robot_model = ViserUrdf(
        server,
        robot_urdf_path,
        root_node_name="/xarm",
        load_meshes=True,
        load_collision_meshes=False,
    )
    kinematics = XArm7Kinematics(robot_urdf_path)
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
    controls = server.gui.add_markdown(
        f"### Loop contract\n\n- max iterations: `{'continuous' if max_iterations is None else max_iterations}`\n"
        f"- settle time after motion: `{settle_s:.1f}s`\n"
        "- default: continuous iterations until Claude/user stop or hard failure\n"
        f"- pre-execution Claude replans on validation failure: `{max_replans}`\n"
        f"- Claude visual-planning timeout: `{claude_timeout_s}s`; final Rxx grounding timeout: `{claude_grounding_timeout_s}s`\n"
        "- perception diagnostics: A/B garment height-above-table heatmaps and fused garment boundary are shown and sent to Claude\n"
        "- Claude grounding: choose one Rxxx visually, then perform exactly one final exact-coordinate lookup\n"
        "- grasp target visualization: Base XYZ/yaw, Viser 3-D marker, and Camera A/B projection overlays\n"
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
            preview_depth_handle.image = _height_map_heatmap_preview(frame, config)
            points, colors = _frame_point_cloud(frame, stride=4)
            points, colors = _voxel_balance_cloud(
                points, colors, voxel_size_mm=5.0, max_points=25000
            )
            if len(points):
                live_cloud_handle.points = points
                live_cloud_handle.colors = colors
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
            preview_panel.content = f"### Live CamA RGB-D\n\nPreview update failed: `{exc}`"

    def render_capture_frames(frames: list[RGBDFrame]) -> None:
        for frame in frames:
            label = frame.label.upper()
            if label not in capture_rgb_handles:
                continue
            capture_rgb_handles[label].image = np.asarray(frame.rgb, dtype=np.uint8)
            capture_depth_handles[label].image = _height_map_heatmap_preview(frame, config)
            points, colors = _frame_point_cloud(frame, stride=4)
            points, colors = _voxel_balance_cloud(
                points, colors, voxel_size_mm=5.0, max_points=25000
            )
            handle = capture_cloud_handles[label]
            if len(points):
                handle.points = points
                handle.colors = colors
                handle.visible = True

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
        client = ClaudeAutoClient(
            binary=claude_binary,
            timeout_s=claude_timeout_s,
            grounding_timeout_s=claude_grounding_timeout_s,
            max_reference_reselections=max_replans,
        )
        try:
            objective = state.objective
            iteration = 0
            while iterations is None or iteration < iterations:
                iteration += 1
                if stopped():
                    break
                with state_lock:
                    state.iteration = iteration
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "started_at": _now(),
                    "objective": objective,
                }
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
                    render_perception_diagnostics(saved, saved_path)
                    fusion_artifacts = saved.get("depth_fusion", {}).get("artifacts", {})
                    fused_points_path = saved_path.parent / str(
                        fusion_artifacts.get("fused_points_base_mm", "")
                    )
                    fused_colors_path = saved_path.parent / str(
                        fusion_artifacts.get("fused_colors_rgb", "")
                    )
                    if fused_points_path.is_file() and fused_colors_path.is_file():
                        fused_cloud_handle.points = (
                            np.load(fused_points_path).astype(np.float32) / 1000.0
                        )
                        fused_cloud_handle.colors = np.load(fused_colors_path).astype(np.uint8)
                        fused_cloud_handle.visible = True
                    before_images = perception_image_paths(saved, saved_path)
                    record["perception"] = perception
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

                    set_status(
                        f"### Iteration {iteration}/{iterations}: Claude thinking\n\n"
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
                                objective,
                                feedback=proposal_feedback,
                                history=state.history,
                                phase_callback=planning_phase_callback,
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
                                    objective,
                                    feedback=validation_feedback,
                                    history=state.history,
                                    phase_callback=planning_phase_callback,
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
                        animation_frames = kinematics.build_animation(
                            preflight.actions,
                            robot.init_joints_deg,
                            robot.orientation_roll_deg,
                            robot.orientation_pitch_deg,
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
                    record["after_images"] = [
                        _run_relative(path, session.run_dir) for path in after_images
                    ]
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
                        before_images,
                        after_images,
                        proposal=proposal,
                        run_dir=session.run_dir,
                        rollout_recording_dir=(
                            recording_dir
                            if record_rollouts
                            and record.get("rollout_recording", {}).get("status")
                            == "completed"
                            else None
                        ),
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
                    with state_lock:
                        state.evaluation = evaluation
                        state.history.append(
                            {
                                "iteration": iteration,
                                "plan_status": "executed",
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
                    if evaluation.stop:
                        set_status(
                            f"### Automatic exploration stopped after iteration {iteration}\n\n"
                            "Claude judged that safe grounded continuation is not currently "
                            f"possible: {evaluation.reason}"
                        )
                        break
                    objective = evaluation.next_objective
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
        state.objective = (
                "Take one planning-mode-appropriate action that makes the current garment as open and spread as safely possible."
            )
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
            state.objective = (
                "Take one planning-mode-appropriate action that makes the current garment as open and spread as safely possible."
            )
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
    parser.add_argument("--robot-config")
    parser.add_argument("--perception-config")
    parser.add_argument("--claude-binary", default="claude")
    parser.add_argument("--claude-timeout-s", type=int, default=400)
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
    parser.add_argument(
        "--record-rollouts",
        action="store_true",
        help="record Camera A/B RGB, depth, composite video, timestamps, and native data during each physical rollout",
    )
    parser.add_argument(
        "--recording-no-native",
        action="store_true",
        help="disable the SDK-native Camera A/B .db3 recordings while keeping MP4 and timestamps",
    )
    parser.add_argument("--recording-codec", default="mp4v")
    parser.add_argument("--recording-warmup-frames", type=int)
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
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
