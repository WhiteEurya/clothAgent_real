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

With ``--unattended`` (or ``--continue-on-error``), failures before physical
execution are kept as unchanged-state iteration experiences and the same loop
continues to the next iteration.  Failures in perception/evaluation and other
non-physical stages may restart a fresh timestamped attempt; explicit user
interrupts and physical execution failures remain stop conditions.

An optional Camera-C observer is enabled by default through
``--observer-camera-serial``.  It records RGB-only before/after frames and a
rollout video for grasp and occlusion evaluation.  After the first move above
the selected grasp, the host also saves a dedicated ``hold_check`` still from
the recorder's live frame stream; this is the primary visual witness for
short-term acquisition.  Camera C is deliberately uncalibrated and is never
used for depth fusion, point grounding, workspace checks, or robot commands.

Claude remains the strategy authority.  In acquisition-learning iterations it
must return the reversible lift probe itself; the host validates and compiles
that proposal without silently replacing it.  The legacy host-side probe
compiler is available only with ``--host-compile-acquisition-probe``.
"""

from __future__ import annotations

import argparse
import errno
import itertools
import json
import math
import re
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
    _first_grasp_move,
    _load_latest_perception,
    _load_or_create_session,
    exploration_source,
    global_perception_image_paths,
    split_global_lift_checkpoint_plan,
    validate_exploration_payload,
)
from .garment_grounding_mcp import GarmentGrounding, GroundingToolError
from .grasp_height import GraspHeightError, resolve_grasp_height
from .perception import PerceptionConfig, RGBDFrame, capture_two_view_rgbd
from .persistent_claude import PersistentClaudeSession
from .molmo_keypoint_pipeline import (
    KeypointSpec,
    MolmoKeypointPipelineError,
    run_molmo_keypoint_pipeline,
)
from .rollout_recorder import (
    DualRealSenseRolloutRecorder,
    ObserverRGBRolloutRecorder,
    capture_observer_rgb,
)
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
MAX_ACQUISITION_PROBES_PER_STEP = 3
_BUNCHED_TERMS = (
    "bunched",
    "gathered",
    "gathering",
    "rolled",
    "rolled tube",
    "rolled onto itself",
    "ruck",
    "pucker",
    "plough",
    "plowed",
    "scrubbed",
    "knotted",
    "tangled",
    "wrinkled mass",
)


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


def _compact_supervisor_state(value: Any) -> dict[str, Any] | None:
    """Remove raw Claude invocation data from a supervisor decision."""

    if not isinstance(value, Mapping):
        return None
    keys = (
        "status",
        "current_step",
        "completed_steps",
        "garment_visibility",
        "trajectory_decision",
        "confidence",
        "reason",
        "fallback",
        "local_deterministic",
        "completion_ledger_source",
        "confirmed_completed_steps",
    )
    result = {key: value.get(key) for key in keys if key in value}
    # A fallback completion is a loop-avoidance hint, not visual proof.  Do
    # not expose its fabricated completion list to the next Claude context.
    if value.get("fallback"):
        result["completed_steps"] = []
        result["completion_ledger_source"] = "fallback_bookkeeping"
        result.pop("confirmed_completed_steps", None)
    if "reason" in result:
        result["reason"] = str(result["reason"])[:800]
    evidence = value.get("evidence")
    if isinstance(evidence, list):
        result["evidence"] = [str(item)[:300] for item in evidence[:4]]
    return result


def _compact_history(history: Sequence[Mapping[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Keep planning/supervision context bounded and focused on decisions."""

    def compact_supervisor(value: Any) -> dict[str, Any] | None:
        return _compact_supervisor_state(value)

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
            "action_mode": row.get("action_mode") or row.get("mode"),
            "garment_condition_before": (
                dict(row.get("garment_condition_before"))
                if isinstance(row.get("garment_condition_before"), Mapping)
                else None
            ),
            "garment_condition_after": (
                dict(row.get("garment_condition_after"))
                if isinstance(row.get("garment_condition_after"), Mapping)
                else None
            ),
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
            "plan_authority": (
                {
                    "host_compilation": dict(row.get("host_compilation")),
                    "model_selected_grasp": dict(
                        ((row.get("proposal") or {}).get("selected_grasp") or {})
                    )
                    if isinstance(row.get("proposal"), Mapping)
                    and isinstance((row.get("proposal") or {}).get("selected_grasp"), Mapping)
                    else None,
                    "model_grasp_strategy": _grasp_strategy_signature(
                        {"proposal": row.get("proposal")}
                    ),
                }
                if isinstance(row.get("host_compilation"), Mapping)
                else None
            ),
            "grasp_strategy": _grasp_strategy_signature(row),
            "acquisition_learning": (
                dict(row.get("acquisition_learning"))
                if isinstance(row.get("acquisition_learning"), Mapping)
                else None
            ),
        }
        # Planning/preflight failures are real run-local evidence even though
        # no robot command was sent.  Keep the compact failure lesson in the
        # next Claude context so unattended iterations do not rediscover the
        # same invalid trajectory without knowing why it was rejected.
        failure = row.get("planning_failure")
        if isinstance(failure, Mapping):
            next_experiment = failure.get("next_experiment")
            rejected_plan = row.get("execution_proposal") or row.get("proposal")
            rejected_plan_summary: dict[str, Any] | None = None
            if isinstance(rejected_plan, Mapping):
                actions = rejected_plan.get("actions")
                rejected_plan_summary = {
                    "selected_grasp": (
                        dict(rejected_plan.get("selected_grasp"))
                        if isinstance(rejected_plan.get("selected_grasp"), Mapping)
                        else None
                    ),
                    "action_count": len(actions) if isinstance(actions, list) else 0,
                    "action_names": [
                        str(action.get("name"))
                        for action in (actions or [])[:12]
                        if isinstance(action, Mapping)
                    ],
                }
            item["planning_failure"] = {
                "stage": str(failure.get("stage", "planning")),
                "failure_kind": str(failure.get("failure_kind", "")),
                "exception_type": str(failure.get("exception_type", "")),
                "error": str(failure.get("error", ""))[:800],
                "physical_command_sent": bool(
                    failure.get("physical_command_sent", False)
                ),
                "fold_command_sent": bool(failure.get("fold_command_sent", False)),
                "failed_pose": failure.get("failed_pose"),
                "rejected_plan": rejected_plan_summary,
                "next_experiment": (
                    {
                        "keep": [str(v)[:240] for v in list(next_experiment.get("keep") or [])[:6]],
                        "change": [str(v)[:240] for v in list(next_experiment.get("change") or [])[:6]],
                        "reason": str(next_experiment.get("reason", ""))[:600],
                    }
                    if isinstance(next_experiment, Mapping)
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
        execution_proposal = value.get("execution_proposal")
        if isinstance(execution_proposal, Mapping) and isinstance(
            execution_proposal.get("actions"), list
        ):
            return [
                item
                for item in execution_proposal["actions"]
                if isinstance(item, Mapping)
            ]
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


def _first_post_close_move_index(value: Any) -> int | None:
    """Return the zero-based first lift move after ``close_gripper``.

    The fold executor uses this action boundary to save a Camera-C still while
    the gripper is already above the grasp.  It is an observation hook only:
    it does not alter the validated trajectory or gate robot motion.
    """

    actions = _proposal_actions(value)
    close_index = next(
        (index for index, action in enumerate(actions) if action.get("name") == "close_gripper"),
        None,
    )
    if close_index is None:
        return None
    grasp_move = next(
        (
            action
            for action in reversed(actions[:close_index])
            if action.get("name") == "move" and isinstance(action.get("args"), Mapping)
        ),
        None,
    )
    grasp_z: float | None = None
    if grasp_move is not None:
        try:
            grasp_z = float(grasp_move["args"]["z"])
        except (KeyError, TypeError, ValueError):
            grasp_z = None
    for index, action in enumerate(
        actions[close_index + 1 :], start=close_index + 1
    ):
        if action.get("name") != "move":
            continue
        if grasp_z is None:
            return index
        args = action.get("args")
        if not isinstance(args, Mapping):
            continue
        try:
            post_z = float(args["z"])
        except (KeyError, TypeError, ValueError):
            continue
        if post_z > grasp_z:
            return index
    return None


def _is_reference_grounding_mismatch(error: BaseException) -> bool:
    """Identify a Stage-2 error that invalidates the Stage-1 Rxxx choice.

    These failures are different from malformed actions or IK failures: the
    visual reference itself and the final grounded grasp disagree.  Retrying
    Stage 2 with the same selected Rxxx cannot repair that contradiction, so
    the caller should restart the visual-selection stage.
    """

    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "final grasp xy does not use the visually selected rxxx measurement",
            "final fold grasp offset from the selected rxxx anchor is invalid",
            "selected reference is inconsistent with the final grasp",
        )
    )


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


def _validate_model_acquisition_probe(
    proposal: ExplorationProposal,
) -> dict[str, Any]:
    """Validate that Claude itself returned a reversible acquisition probe.

    Acquisition learning is a strategy selected by the agent.  The host may
    enforce the safety contract, but it must not silently replace a full fold
    plan with a different probe.  This check therefore accepts only a plan
    whose post-close path is a near-vertical lift, reversal, release, and
    optional home.  Lateral entry before closing is allowed because it is part
    of Claude's grasp hypothesis.
    """

    if not proposal.requires_lift_checkpoint:
        raise ExplorationPlanningError(
            "acquisition probe must set requires_lift_checkpoint=true in Claude's plan"
        )
    actions = list(proposal.actions)
    close_index = next(
        (index for index, action in enumerate(actions) if action.get("name") == "close_gripper"),
        None,
    )
    if close_index is None:
        raise ExplorationPlanningError(
            "acquisition probe must contain close_gripper in Claude's returned plan"
        )
    release_index = next(
        (
            index
            for index, action in enumerate(actions[close_index + 1 :], start=close_index + 1)
            if action.get("name") == "open_gripper"
        ),
        None,
    )
    if release_index is None:
        raise ExplorationPlanningError(
            "acquisition probe must release with open_gripper after the hold check"
        )
    grasp_move = next(
        (
            action
            for action in reversed(actions[:close_index])
            if action.get("name") == "move" and isinstance(action.get("args"), Mapping)
        ),
        None,
    )
    if grasp_move is None:
        raise ExplorationPlanningError(
            "acquisition probe must contain a move immediately before close_gripper"
        )
    grasp_args = grasp_move["args"]
    try:
        grasp_xy = (float(grasp_args["x"]), float(grasp_args["y"]))
        grasp_z = float(grasp_args["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExplorationPlanningError(
            "acquisition probe grasp move has invalid numeric coordinates"
        ) from exc

    post_moves: list[Mapping[str, Any]] = []
    for action in actions[close_index + 1 : release_index]:
        if action.get("name") != "move":
            continue
        args = action.get("args")
        if not isinstance(args, Mapping):
            raise ExplorationPlanningError(
                "acquisition probe move args must be an object"
            )
        post_moves.append(args)
    if not post_moves:
        raise ExplorationPlanningError(
            "acquisition probe must include at least one post-close lift move"
        )
    lifts: list[float] = []
    lateral_offsets: list[float] = []
    for args in post_moves:
        try:
            x = float(args["x"])
            y = float(args["y"])
            z = float(args["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExplorationPlanningError(
                "acquisition probe post-close move has invalid numeric coordinates"
            ) from exc
        lifts.append(z - grasp_z)
        lateral_offsets.append(math.hypot(x - grasp_xy[0], y - grasp_xy[1]))
    max_lateral = max(lateral_offsets)
    max_lift = max(lifts)
    if lifts[0] <= 0.0:
        raise ExplorationPlanningError(
            "acquisition probe first post-close move must lift above the grasp height"
        )
    if max_lateral > 5.0:
        raise ExplorationPlanningError(
            "Claude returned lateral transport in acquisition-probe mode "
            f"(maximum post-close offset {max_lateral:.1f} mm); keep the probe reversible"
        )
    return {
        "status": "VALID",
        "authority": "Claude",
        "close_action_index": close_index,
        "release_action_index": release_index,
        "post_close_move_count": len(post_moves),
        "max_lift_mm": max_lift,
        "max_post_close_lateral_mm": max_lateral,
        "host_rewrite": False,
    }


def _extract_gripper_telemetry(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize host-side gripper feedback recorded at action boundaries."""

    raw_actions = execution.get("requested_robot_actions")
    if not isinstance(raw_actions, list):
        raw_actions = execution.get("actual_robot_actions")
    samples: list[dict[str, Any]] = []
    for index, action in enumerate(raw_actions or [], start=1):
        if not isinstance(action, Mapping):
            continue
        state = action.get("robot_state")
        feedback = (
            action.get("gripper_result", {}).get("feedback")
            if isinstance(action.get("gripper_result"), Mapping)
            else None
        )
        if not isinstance(feedback, Mapping) and isinstance(state, Mapping):
            feedback = state.get("gripper_feedback")
        if not isinstance(feedback, Mapping):
            continue
        samples.append(
            {
                "action_index": index,
                "action": action.get("name"),
                "feedback": dict(feedback),
            }
        )
    post_close = False
    grasp_samples: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("action") == "close_gripper":
            post_close = True
        elif sample.get("action") == "open_gripper":
            post_close = False
        if post_close:
            feedback = sample.get("feedback", {})
            if feedback.get("mechanical_grasp_detected") is True:
                grasp_samples.append(sample)
    return {
        "schema_version": 1,
        "source": "xarm_sdk_gripper_feedback_at_action_boundaries",
        "available": bool(samples),
        "samples": samples,
        "mechanical_grasp_detected_after_close": bool(grasp_samples),
        "grasp_state_samples": grasp_samples,
        "note": (
            "status=grasp is controller contact evidence, not visual proof of a held garment; "
            "combine with position, safe pre-close height, and lift behavior."
        ),
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
        # Repair/fold records are not acquisition probes.  Counting their
        # evaluator output here would inflate the probe budget and could trap
        # the state machine in another probe cycle.
        row_mode = row.get("mode") or row.get("status")
        if row_mode is not None and row_mode not in {"ACQUISITION_PROBE", "FOLD"}:
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
    if bool(learning.get("use_lift_only_probe")) and proposal.requires_lift_checkpoint:
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


def _proposal_moves_between(
    actions: Sequence[Mapping[str, Any]],
    *,
    start: int = 0,
    end: int | None = None,
) -> list[tuple[int, Mapping[str, Any]]]:
    result: list[tuple[int, Mapping[str, Any]]] = []
    stop = len(actions) if end is None else min(len(actions), end)
    for index in range(max(0, start), stop):
        action = actions[index]
        if action.get("name") != "move" or not isinstance(action.get("args"), Mapping):
            continue
        result.append((index, action["args"]))
    return result


def _validate_action_mode_contract(
    proposal: ExplorationProposal,
    *,
    mode: str,
) -> dict[str, Any]:
    """Validate whether a proposal matches PROBE/FOLD/REPAIR semantics."""

    actions = _proposal_actions(proposal)
    close_index = next(
        (index for index, action in enumerate(actions) if action.get("name") == "close_gripper"),
        None,
    )
    if close_index is None:
        raise ExplorationPlanningError(f"{mode} plan must contain close_gripper")
    release_index = next(
        (
            index
            for index, action in enumerate(actions[close_index + 1 :], start=close_index + 1)
            if action.get("name") == "open_gripper"
        ),
        None,
    )
    if release_index is None:
        raise ExplorationPlanningError(f"{mode} plan must release with open_gripper")
    before_close = _proposal_moves_between(actions, end=close_index)
    # Consecutive low-Z XY motion is a cloth-pushing sweep.  A probe must not
    # use it, and repair/fold plans must declare a different, deliberate
    # structure rather than accidentally dragging the sleeve into a knot.
    low_z_sweeps: list[dict[str, Any]] = []
    for (first_index, first), (second_index, second) in zip(before_close, before_close[1:]):
        try:
            z1, z2 = float(first["z"]), float(second["z"])
            xy = math.hypot(
                float(second["x"]) - float(first["x"]),
                float(second["y"]) - float(first["y"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if max(z1, z2) <= 70.0 and xy > 5.0:
            low_z_sweeps.append(
                {
                    "from_action": first_index,
                    "to_action": second_index,
                    "distance_mm": xy,
                    "z_mm": min(z1, z2),
                }
            )
    if low_z_sweeps:
        raise ExplorationPlanningError(
            f"{mode} plan contains a low-Z pre-close lateral sweep that can bunch cloth: "
            f"{low_z_sweeps[0]}"
        )

    grasp_move = next(
        (
            (index, args)
            for index, args in reversed(before_close)
            if isinstance(args, Mapping)
        ),
        None,
    )
    if grasp_move is None:
        raise ExplorationPlanningError(f"{mode} plan must move to a grasp pose before close_gripper")
    _, grasp_args = grasp_move
    try:
        grasp_xy = (float(grasp_args["x"]), float(grasp_args["y"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ExplorationPlanningError(f"{mode} grasp pose has invalid XY") from exc
    post_close = _proposal_moves_between(actions, start=close_index + 1, end=release_index)
    if mode in {"FOLD", "REPAIR_SLEEVE"} and post_close:
        try:
            first_post_z = float(post_close[0][1]["z"])
            grasp_z = float(grasp_args["z"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExplorationPlanningError(
                f"{mode} first post-grasp move has invalid height"
            ) from exc
        if first_post_z <= grasp_z:
            raise ExplorationPlanningError(
                f"{mode} first post-grasp move must lift above the grasp height"
            )
    post_offsets: list[float] = []
    for _, args in post_close:
        try:
            post_offsets.append(
                math.hypot(float(args["x"]) - grasp_xy[0], float(args["y"]) - grasp_xy[1])
            )
        except (KeyError, TypeError, ValueError):
            continue
    max_post_offset = max(post_offsets, default=0.0)
    if mode == "ACQUISITION_PROBE":
        if max_post_offset > 5.0:
            raise ExplorationPlanningError(
                "ACQUISITION_PROBE must remain reversible and cannot transport cloth after close"
            )
    elif mode == "FOLD":
        if max_post_offset < 20.0:
            raise ExplorationPlanningError(
                "FOLD plan must include a post-grasp inward transport of at least 20 mm"
            )
    elif mode == "REPAIR_SLEEVE":
        if max_post_offset < 10.0:
            raise ExplorationPlanningError(
                "REPAIR_SLEEVE plan must move the gathered sleeve outward by at least 10 mm"
            )
    return {
        "status": "VALID",
        "mode": mode,
        "close_action_index": close_index,
        "release_action_index": release_index,
        "max_post_close_xy_offset_mm": max_post_offset,
        "low_z_sweep_count": len(low_z_sweeps),
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
        # The upright Camera-A RGB is the canonical semantic view: collar at
        # the top, image-left/image-right sleeves unambiguous.  Keep it ahead
        # of raw landscape duplicates so the supervisor can judge the sleeve
        # order from the same orientation used by the planner.
        if name == "camera_a_rgb_upright.png":
            rank = -2
        elif name == "camera_a_rxxx_overlay_upright.png":
            rank = -1
        elif name in {"camera_0_a.png", "camera_1_b.png"}:
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


def _select_observer_images(images: Sequence[Path]) -> list[Path]:
    """Return optional uncalibrated observer RGB images from an image bundle."""

    selected: list[Path] = []
    seen: set[Path] = set()
    for raw in images:
        path = Path(raw).resolve()
        name = path.name.lower()
        if "observer" not in name or not name.endswith((".png", ".jpg", ".jpeg")):
            continue
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        selected.append(path)
    return selected


def _evaluation_payload(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    return value if isinstance(value, Mapping) else {}


def _is_acquisition_failure(value: Any) -> bool:
    evaluation = _evaluation_payload(value)
    acquisition = evaluation.get("grasp_acquisition")
    return bool(
        isinstance(acquisition, Mapping)
        and acquisition.get("status") == "FAILURE"
        and evaluation.get("earliest_failure_stage") == "ACQUISITION"
    )


def _acquisition_supervisor_reuse_step(
    history: Sequence[Mapping[str, Any]],
) -> str | None:
    """Return the unchanged fold step after a failed/reversible grasp test."""

    if not history:
        return None
    latest = history[-1]
    if not isinstance(latest, Mapping):
        return None
    step = latest.get("planned_step")
    if step not in FOLD_STEP_IDS:
        return None
    mode = latest.get("mode") or latest.get("status")
    if mode == "REPAIR_SLEEVE":
        return None
    if mode == "ACQUISITION_PROBE" or _is_acquisition_failure(
        latest.get("evaluation")
    ):
        return str(step)
    return None


def _evaluation_reports_unchanged(value: Any) -> bool:
    evaluation = _evaluation_payload(value)
    progress = evaluation.get("task_progress")
    metrics = progress.get("metrics") if isinstance(progress, Mapping) else None
    if not isinstance(metrics, Mapping):
        return False
    return all(
        metrics.get(key) == "UNCHANGED"
        for key in (
            "visible_area_delta",
            "overlap_delta",
            "relief_delta",
        )
    )


def _garment_condition_from_history(
    history: Sequence[Mapping[str, Any]],
    *,
    step: str,
) -> dict[str, Any]:
    """Infer a conservative cloth condition from saved visual evidence.

    This is intentionally a gate hint, not a replacement for RGB inspection.
    Explicit records written by newer runs win; older runs are supported by a
    bounded search through Claude's visual observations/evidence.  A positive
    bunching signal is sticky until a later record explicitly reports a flat
    sleeve, because a reversible gripper path does not guarantee reversible
    cloth motion.
    """

    latest_condition: str | None = None
    source_iteration: int | None = None
    matched_terms: list[str] = []
    for row in reversed(list(history)):
        if not isinstance(row, Mapping) or row.get("planned_step") != step:
            continue
        explicit = row.get("garment_condition")
        if not isinstance(explicit, str):
            after_condition = row.get("garment_condition_after")
            if isinstance(after_condition, Mapping):
                explicit = after_condition.get("condition")
        if not isinstance(explicit, str):
            before_condition = row.get("garment_condition_before")
            if isinstance(before_condition, Mapping):
                explicit = before_condition.get("condition")
        if isinstance(explicit, str) and explicit.upper() in {
            "FLAT",
            "BUNCHED",
            "UNKNOWN",
        }:
            latest_condition = explicit.upper()
            source_iteration = row.get("iteration")
            if latest_condition == "BUNCHED":
                return {
                    "condition": latest_condition,
                    "source": "explicit_record",
                    "source_iteration": source_iteration,
                    "matched_terms": [],
                }
            if latest_condition == "FLAT":
                return {
                    "condition": latest_condition,
                    "source": "explicit_record",
                    "source_iteration": source_iteration,
                    "matched_terms": [],
                }
        texts: list[str] = []
        for key in (
            "proposal",
            "execution_proposal",
            "supervisor_before",
            "supervisor_after",
            "evaluation",
            "planning_failure",
        ):
            value = row.get(key)
            if isinstance(value, Mapping):
                try:
                    texts.append(json.dumps(value, ensure_ascii=False, default=str))
                except (TypeError, ValueError):
                    texts.append(str(value))
            elif value is not None:
                texts.append(str(value))
        text_value = " ".join(texts).lower()
        # Ignore common negated phrases such as "no bunching" and "not rolled".
        positive: list[str] = []
        for term in _BUNCHED_TERMS:
            escaped = re.escape(term)
            if re.search(rf"\b(?:no|not|without|never)\s+(?:\w+\s+){{0,2}}{escaped}\b", text_value):
                continue
            if term in text_value:
                positive.append(term)
        if positive:
            return {
                "condition": "BUNCHED",
                "source": "saved_visual_evidence",
                "source_iteration": row.get("iteration"),
                "matched_terms": sorted(set(positive))[:8],
            }
        if latest_condition is None and any(
            token in text_value
            for token in ("flat and smooth", "lying flat", "flat single ply", "no wrinkles")
        ):
            latest_condition = "FLAT"
            source_iteration = row.get("iteration")
    return {
        "condition": latest_condition or "UNKNOWN",
        "source": "saved_visual_evidence" if latest_condition else "no_prior_condition",
        "source_iteration": source_iteration,
        "matched_terms": matched_terms,
    }


def _proposal_action_mode(
    learning: Mapping[str, Any],
    garment_condition: Mapping[str, Any],
    *,
    step: str,
) -> tuple[str, dict[str, Any]]:
    """Choose the host-permitted mode before asking Claude for a trajectory."""

    condition = str(garment_condition.get("condition", "UNKNOWN")).upper()
    probe_count = int(learning.get("attempt_count", 0) or 0)
    if condition == "BUNCHED" and step in {"left_sleeve", "right_sleeve"}:
        return "REPAIR_SLEEVE", {
            "reason": "saved RGB/evaluation evidence reports a gathered or rolled sleeve",
            "probe_budget_remaining": 0,
        }
    if bool(learning.get("use_lift_only_probe")) and probe_count < MAX_ACQUISITION_PROBES_PER_STEP:
        return "ACQUISITION_PROBE", {
            "reason": "acquisition evidence is still being gathered",
            "probe_budget_remaining": max(0, MAX_ACQUISITION_PROBES_PER_STEP - probe_count),
        }
    if bool(learning.get("use_lift_only_probe")) and probe_count >= MAX_ACQUISITION_PROBES_PER_STEP:
        return "FOLD", {
            "reason": "acquisition probe budget exhausted; do not repeat another lift-only probe",
            "probe_budget_remaining": 0,
        }
    return "FOLD", {
        "reason": "no acquisition probe is required",
        "probe_budget_remaining": max(0, MAX_ACQUISITION_PROBES_PER_STEP - probe_count),
    }


def _condition_after_action(
    before: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Update the cloth-condition ledger without treating a plan as proof."""

    current = dict(before)
    if mode != "REPAIR_SLEEVE":
        return current
    try:
        text_value = json.dumps(evaluation, ensure_ascii=False, default=str).lower()
    except (TypeError, ValueError):
        text_value = str(evaluation).lower()
    flat_signals = (
        "became flatter",
        "is flatter",
        "flattened",
        "less gathered",
        "unbunched",
        "no longer bunched",
        "spread flatter",
        "sleeve is flat",
        "flat again",
        "distal edge became visible",
        "distal edge is visible",
        "free edge became visible",
        "free edge is visible",
        "bunching reduced",
    )
    if any(signal in text_value for signal in flat_signals):
        return {
            "condition": "FLAT",
            "source": "repair_evaluation",
            "source_iteration": current.get("source_iteration"),
            "matched_terms": [signal for signal in flat_signals if signal in text_value][:8],
        }
    # A repair that is not positively confirmed must remain conservative.  In
    # particular, replacing UNKNOWN with UNKNOWN would immediately re-enable
    # another acquisition probe and lose the reason repair was scheduled.
    if str(current.get("condition", "UNKNOWN")).upper() != "FLAT":
        current["condition"] = "BUNCHED"
    current["source"] = "repair_evaluation_not_confirmed"
    return current


def _write_fold_evidence_package(
    iteration_dir: Path,
    *,
    stage: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Write compact, staged evidence files and a manifest for Claude/Viser."""

    evidence_dir = iteration_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    filename_by_stage = {
        "rgb_selection": "01_rgb_selection.json",
        "metric_grounding": "02_metric_grounding.json",
        "execution_gate": "03_execution_gate.json",
        "post_grasp": "04_post_grasp.json",
        "experience_update": "05_experience_update.json",
    }
    if stage not in filename_by_stage:
        raise ValueError(f"unknown fold evidence package stage: {stage}")
    path = evidence_dir / filename_by_stage[stage]
    _write_json(path, dict(payload))
    files = {
        name: str((evidence_dir / name).resolve())
        for name in filename_by_stage.values()
        if (evidence_dir / name).is_file()
    }
    manifest = {
        "schema_version": 1,
        "iteration": iteration_dir.name,
        "updated_at": _now(),
        "current_stage": stage,
        "files": files,
        "read_order": [files[name] for name in filename_by_stage.values() if name in files],
        "instruction": (
            "Use RGB selection evidence for semantic target choice. Use metric grounding "
            "only after an Rxxx is selected. The execution gate is host-authoritative."
        ),
    }
    manifest_path = evidence_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "directory": str(evidence_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "files": files,
        "current_stage": stage,
    }


def _validate_fold_evidence_package(
    iteration_dir: Path,
    *,
    required_stages: Sequence[str] = ("rgb_selection", "metric_grounding", "execution_gate"),
) -> dict[str, Any]:
    """Require the minimum host evidence bundle before physical execution.

    The package is deliberately small and local.  Claude may choose the
    semantic target and strategy, but it cannot bypass this deterministic gate:
    the RGB-selection record, the metric-grounding record, and the execution
    gate must all be present and internally marked as passed.  This function
    does not inspect RGB pixels or solve IK; those checks are recorded by their
    respective producers and remain host-authoritative.
    """

    evidence_dir = (Path(iteration_dir) / "evidence").resolve()
    manifest_path = evidence_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ExplorationPlanningError(
            f"evidence package manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExplorationPlanningError(
            f"evidence package manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ExplorationPlanningError("evidence package manifest must be a JSON object")

    filename_by_stage = {
        "rgb_selection": "01_rgb_selection.json",
        "metric_grounding": "02_metric_grounding.json",
        "execution_gate": "03_execution_gate.json",
        "post_grasp": "04_post_grasp.json",
        "experience_update": "05_experience_update.json",
    }
    checked: dict[str, Any] = {}
    for stage in required_stages:
        filename = filename_by_stage.get(str(stage))
        if filename is None:
            raise ValueError(f"unknown evidence package stage: {stage}")
        path = evidence_dir / filename
        if not path.is_file():
            raise ExplorationPlanningError(
                f"evidence package is incomplete: missing {filename}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExplorationPlanningError(
                f"evidence package stage is unreadable: {filename}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ExplorationPlanningError(
                f"evidence package stage must be an object: {filename}"
            )
        checked[str(stage)] = dict(payload)

    rgb = checked.get("rgb_selection", {})
    images = rgb.get("images")
    if not isinstance(images, list) or not images:
        raise ExplorationPlanningError(
            "evidence package RGB selection must list at least one image"
        )
    if not rgb.get("step") or not rgb.get("action_mode"):
        raise ExplorationPlanningError(
            "evidence package RGB selection must identify step and action_mode"
        )

    grounding = checked.get("metric_grounding", {})
    selected_grasp = grounding.get("selected_grasp")
    selected_reference = grounding.get("selected_reference")
    if not (
        isinstance(selected_grasp, Mapping)
        and bool(selected_grasp)
    ) and not (
        isinstance(selected_reference, Mapping)
        and bool(selected_reference)
    ):
        raise ExplorationPlanningError(
            "evidence package metric grounding is missing the selected reference/grasp"
        )
    grounding_checks = grounding.get("checks")
    if not isinstance(grounding_checks, Mapping):
        raise ExplorationPlanningError(
            "evidence package metric grounding is missing checks"
        )
    grounding_ik = grounding_checks.get("controller_ik")
    if not isinstance(grounding_ik, str) or grounding_ik not in {"PASS", "SKIPPED"}:
        raise ExplorationPlanningError(
            "evidence package metric grounding did not pass controller IK"
        )

    gate = checked.get("execution_gate", {})
    if gate.get("decision") != "ALLOW":
        raise ExplorationPlanningError(
            "evidence package execution gate is not ALLOW"
        )
    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        raise ExplorationPlanningError(
            "evidence package execution gate is missing checks"
        )
    failed_checks = {
        str(name): value
        for name, value in checks.items()
        if name in {
            "semantic_target",
            "metric_grounding",
            "controller_ik",
            "action_mode_contract",
            "low_z_preclose_sweep",
        }
        and (not isinstance(value, str) or value not in {"PASS", "CLAUDE_SELECTED"})
    }
    if failed_checks:
        raise ExplorationPlanningError(
            "evidence package execution gate contains failed checks: "
            + json.dumps(failed_checks, ensure_ascii=False, sort_keys=True)
        )
    return {
        "status": "VALID",
        "manifest": str(manifest_path),
        "required_stages": [str(stage) for stage in required_stages],
        "checked_stages": sorted(checked),
        "execution_decision": gate.get("decision"),
    }


def _safe_claude(binary: str) -> str:
    resolved = shutil.which(binary) if Path(binary).name == binary else binary
    if resolved is None:
        raise RuntimeError(f"Claude CLI not found: {binary}")
    return str(resolved)


def _extract_failure_pose(exc: BaseException) -> list[float] | None:
    """Extract a controller-rejected Cartesian pose for diagnostics.

    ``validate_controller_trajectory`` intentionally raises a plain
    ``SafetyError`` whose message contains ``pose=[...]``.  Preserve that
    useful datum in the iteration record without making the controller error
    type part of the persistence contract.  The immediate exception and its
    cause/context are inspected because IK errors are often wrapped once by
    the fold planner.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        match = re.search(r"pose=\[([^\]]+)\]", str(current))
        if match:
            values: list[float] = []
            for token in match.group(1).split(","):
                try:
                    values.append(float(token.strip()))
                except (TypeError, ValueError):
                    values = []
                    break
            if values:
                return values
        current = current.__cause__ or current.__context__
    return None


def _planning_failure_next_experiment(exc: BaseException) -> dict[str, Any]:
    """Turn a deterministic pre-execution rejection into a small lesson."""

    error_text = f"{type(exc).__name__}: {exc}"
    lowered = error_text.lower()
    changes: list[str] = []
    if "ik" in lowered or "controller" in lowered:
        changes.extend(
            [
                "keep the same fold step but shorten the transport/laydown path",
                "use a less extreme wrist yaw and re-check every interpolated waypoint",
                "prefer a conservative approach and lift pose that the controller can solve",
            ]
        )
    elif "workspace" in lowered or "safe lower bound" in lowered or "safe upper bound" in lowered:
        changes.extend(
            [
                "choose a different visually valid reference inside the calibrated workspace",
                "keep the target on the garment while moving the laydown inward",
            ]
        )
    elif "reference" in lowered or "ground" in lowered or "pixel" in lowered:
        changes.extend(
            [
                "re-run visual reference selection instead of reusing the rejected Rxxx",
                "keep the selected point anchored, then apply only a small validated offset",
            ]
        )
    else:
        changes.append("return a materially different schema-valid plan for the same current step")
    return {
        "keep": [
            "the ordered supervisor current_step",
            "the RGB interpretation of the garment",
            "the fact that no fold trajectory command was sent",
        ],
        "change": changes[:6],
        "reason": (
            "This iteration failed before fold execution, so the garment state is unchanged. "
            "Use the deterministic rejection below as a constraint for the next plan: "
            + error_text[:900]
        ),
    }


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


def _earliest_incomplete_fold_step(completed_steps: Sequence[Any]) -> str:
    """Return the host-owned action implied by the ordered fold ledger."""

    completed = {str(step) for step in completed_steps if str(step) in FOLD_STEP_IDS}
    return next(
        (step for step in FOLD_STEP_IDS if step not in completed),
        "COMPLETE",
    )


def _confirmed_completion_ledger(
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return the monotonic ledger of visually confirmed fold steps.

    ``fallback`` supervisor states are deliberately excluded.  They are only
    bookkeeping used to keep an unattended run moving after Claude is
    unavailable, and must never become visual evidence that a sleeve was
    folded.  Non-fallback ``supervisor_before``/``supervisor_after`` states
    are observations (or deterministic states that merely carry an earlier
    observation), so their completion sets can safely be unioned and kept
    monotonic across later ambiguous frames.
    """

    confirmed: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            continue
        candidates = (
            row.get("supervisor_before"),
            row.get("supervisor_after"),
        )
        for state in candidates:
            if not isinstance(state, Mapping) or state.get("fallback"):
                continue
            for step in state.get("completed_steps", ()):
                if step in FOLD_STEP_IDS:
                    confirmed.add(str(step))
    return [step for step in FOLD_STEP_IDS if step in confirmed]


def _merge_supervisor_completion_ledger(
    payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge current visual evidence with the host's monotonic ledger.

    A later supervisor call may be uncertain because a sleeve is bunched or
    partly occluded.  That uncertainty must not undo a step that an earlier
    non-fallback visual supervisor already confirmed.  Conversely, a fallback
    result is not allowed to add a completion on its own.
    """

    result = dict(payload)
    historical = _confirmed_completion_ledger(history)
    current = (
        []
        if result.get("fallback")
        else [step for step in result.get("completed_steps", ()) if step in FOLD_STEP_IDS]
    )
    merged = set(historical)
    merged.update(current)
    result["completed_steps"] = [
        step for step in FOLD_STEP_IDS if step in merged
    ]
    if historical and result["completed_steps"] != current:
        result["completion_ledger_source"] = "historical_visual_union"
        result["confirmed_completed_steps"] = list(result["completed_steps"])
    elif not result.get("fallback"):
        result["completion_ledger_source"] = "current_visual_supervisor"
    else:
        result["completion_ledger_source"] = "fallback_bookkeeping"
    # A fallback must never terminate the fold merely because its
    # bookkeeping list reached five planned actions.  Only the merged
    # non-fallback visual ledger can authorize COMPLETE.
    if result.get("fallback") and len(result["completed_steps"]) < len(FOLD_STEP_IDS):
        result["status"] = "READY"
    elif result.get("status") == "COMPLETE" and len(result["completed_steps"]) < len(FOLD_STEP_IDS):
        # A model can occasionally emit COMPLETE while its own completion
        # list is partial.  The ordered ledger is authoritative, so treat
        # that response as a normal READY state instead of ending the run.
        result["status"] = "READY"
    return _normalize_supervisor_current_step(result)


def _normalize_supervisor_current_step(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Make ``current_step`` mean the one action the host executes now.

    Claude no longer returns a separate ``next_step`` field. The ordered
    completion ledger is authoritative, so READY responses are corrected to
    the earliest incomplete action even if the model's prose is inconsistent.
    """

    result = dict(payload)
    if result.get("status") == "COMPLETE" or len(result.get("completed_steps", [])) == len(FOLD_STEP_IDS):
        result["current_step"] = "COMPLETE"
    elif result.get("status") == "BLOCKED":
        result["current_step"] = "BLOCKED"
    else:
        result["current_step"] = _earliest_incomplete_fold_step(
            result.get("completed_steps", [])
        )
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
        self.condition_path = self.root / "garment_condition.json"
        # Older runs may have a summary generated before fallback states were
        # separated from visual evidence.  Recompute it on load so the file
        # shown in Viser/reporting agrees with the host ledger immediately.
        if self.path.is_file():
            self.refresh_summary()

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
        condition_after = experience.get("garment_condition_after")
        step = experience.get("planned_step")
        if isinstance(step, str) and isinstance(condition_after, Mapping):
            condition = str(condition_after.get("condition", "UNKNOWN")).upper()
            if condition in {"FLAT", "BUNCHED", "UNKNOWN"}:
                ledger: dict[str, Any] = {}
                if self.condition_path.is_file():
                    try:
                        loaded = json.loads(self.condition_path.read_text(encoding="utf-8"))
                        if isinstance(loaded, Mapping):
                            ledger = dict(loaded)
                    except (OSError, json.JSONDecodeError):
                        ledger = {}
                entries = ledger.setdefault("steps", {})
                if not isinstance(entries, dict):
                    entries = {}
                    ledger["steps"] = entries
                # UNKNOWN must not erase a previously confirmed condition;
                # otherwise a failed evaluator could re-enable probes after a
                # sleeve was already recognized as bunched.
                previous = entries.get(step)
                previous_condition = (
                    str(previous.get("condition", "UNKNOWN")).upper()
                    if isinstance(previous, Mapping)
                    else "UNKNOWN"
                )
                if condition != "UNKNOWN" or previous_condition == "UNKNOWN":
                    entries[step] = dict(condition_after)
                    entries[step]["condition"] = condition
                    entries[step]["source_iteration"] = experience.get("iteration")
                    ledger["updated_at"] = _now()
                    ledger["schema_version"] = 1
                    _write_json(self.condition_path, ledger)
        return self.refresh_summary()

    def refresh_summary(self) -> dict[str, Any]:
        """Rebuild the compact summary from the append-only experience log."""

        rows = self._read()
        # Only non-fallback supervisor observations may advance the durable
        # fold ledger.  Fallback rows intentionally remain in the experience
        # log, but their completion list is bookkeeping rather than proof.
        completed = _confirmed_completion_ledger(rows)
        summary = {
            "schema_version": 1,
            "updated_at": _now(),
            "experience_count": len(rows),
            "completed_steps_in_order": completed,
            "next_step": next((step for step in FOLD_STEP_IDS if step not in completed), "COMPLETE"),
            "status_counts": {
                status: sum(1 for row in rows if row.get("status") == status)
                for status in (
                "FOLD",
                "ACQUISITION_PROBE",
                "REPAIR_SLEEVE",
                "RECOVERY",
                "PLANNING_FAILURE",
                "FAILED",
                )
            },
            "last_experiences": rows[-8:],
        }
        _write_json(self.summary_path, summary)
        return summary

    def history(self, limit: int | None = 8) -> list[dict[str, Any]]:
        rows = self._read()
        # ``None`` is used by the host state machine to recover the complete
        # monotonic completion ledger after a watchdog restart.  The default
        # remains bounded for callers that only need recent planning context.
        if limit is None:
            return rows
        return rows[-max(1, int(limit)) :]

    def condition(self, step: str) -> dict[str, Any] | None:
        """Return the durable condition ledger entry for one fold step."""

        if not self.condition_path.is_file():
            return None
        try:
            value = json.loads(self.condition_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        steps = value.get("steps") if isinstance(value, Mapping) else None
        entry = steps.get(step) if isinstance(steps, Mapping) else None
        return dict(entry) if isinstance(entry, Mapping) else None


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
    recovery_high = min(
        float(bounds.z_max - robot_config.workspace_margin_mm) if bounds.z_max is not None else math.inf,
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

    def __init__(
        self,
        binary: str = "claude",
        timeout_s: int = 900,
        persistent_session: PersistentClaudeSession | None = None,
    ):
        self.binary = binary
        self.timeout_s = int(timeout_s)
        self.persistent_session = persistent_session

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
                    "Use the upright Camera-A RGB view as the semantic reference whenever it is listed: the collar/neck must be at the TOP, and image-left/image-right refer to that displayed upright image (not the robot's or raw camera's left/right).",
                    "Sleeve completion rubric: mark an inward sleeve complete when its distal lobe no longer protrudes outward from its torso side and the fabric lies over/inboard on the torso. A wrinkled, curled, or bunched cuff is still complete if it is visibly deposited inboard; do not require a perfectly flat cuff or a strong height-map ridge. Do not mark it complete when the sleeve remains extended outside the torso silhouette or when only the torso/print changed.",
                    "If both image-left and image-right sleeves satisfy that rubric, completed_steps MUST contain both left_sleeve and right_sleeve and current_step MUST be left_side. Never leave current_step at right_sleeve merely because the second cuff is bunched.",
                    "A later ambiguous/occluded frame must not undo a sleeve completion that is supported by an earlier non-fallback supervisor observation in the recent history. Fallback/bookkeeping entries are not visual evidence.",
                    "current_step means the single fold action the host should execute next, not the step after it. Do not return a next_step field.",
                    "Set current_step to the earliest incomplete action in the required order, based on completed_steps. With completed_steps empty, current_step must be left_sleeve.",
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
        if self.persistent_session is not None:
            command = self.persistent_session.prepare_command(
                command,
                stage="fold_supervisor",
            )
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
                    reason="claude_timeout", stage="fold_supervisor"
                )
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
            if (
                self.persistent_session is not None
                and self.persistent_session.is_session_conflict_error(
                    completed.stdout, completed.stderr
                )
            ):
                self.persistent_session.rollover(
                    reason="claude_session_conflict", stage="fold_supervisor"
                )
            raise RuntimeError(
                f"fold supervisor exited with {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        if self.persistent_session is not None:
            self.persistent_session.record_success(
                stage="fold_supervisor",
                stdout=completed.stdout,
            )
        result = _normalize_supervisor_current_step(
            validate_supervisor_payload(_json_from_claude_text(completed.stdout))
        )
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
        grounding_timeout_s: int | None = None,
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
        max_stage_retries: int = 1,
        retry_backoff_s: float = 5.0,
        unattended: bool = False,
        host_compile_acquisition_probe: bool = False,
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
        observer_camera_serial: str | None = None,
        observer_camera_label: str = "C",
        observer_camera_width: int = 1280,
        observer_camera_height: int = 720,
        observer_camera_fps: int = 15,
        observer_camera_exposure: float | None = 700.0,
        observer_camera_white_balance: float | None = 3800.0,
    ):
        self.session = session
        self.project_root = session.project_root
        self.perception_config = Path(perception_config).resolve()
        self.claude_binary = claude_binary
        self.claude_timeout_s = int(claude_timeout_s)
        # Final grounding is a separate Claude turn, but it must not have a
        # smaller hidden ceiling than the user-configured Claude timeout.  A
        # separate override is available for boundary experiments; when it is
        # omitted, use the full Claude timeout with no historical 400 s cap.
        self.grounding_timeout_s = (
            self.claude_timeout_s
            if grounding_timeout_s is None
            else int(grounding_timeout_s)
        )
        if self.grounding_timeout_s <= 0:
            raise ValueError("grounding_timeout_s must be positive")
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
        self.unattended = bool(unattended)
        # Claude remains the strategy authority by default.  The legacy host
        # compiler can be enabled explicitly for compatibility, but when it is
        # disabled a full fold plan is rejected and fed back to Claude instead
        # of being silently rewritten into an acquisition probe.
        self.host_compile_acquisition_probe = bool(host_compile_acquisition_probe)
        self._unattended_restart_count = 0
        self._last_operational_stage: str | None = None
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
        self.observer_camera_serial = (
            str(observer_camera_serial).strip() if observer_camera_serial else None
        )
        self.observer_camera_label = str(observer_camera_label).strip() or "C"
        self.observer_camera_width = int(observer_camera_width)
        self.observer_camera_height = int(observer_camera_height)
        self.observer_camera_fps = int(observer_camera_fps)
        self.observer_camera_exposure = observer_camera_exposure
        self.observer_camera_white_balance = observer_camera_white_balance
        if min(
            self.observer_camera_width,
            self.observer_camera_height,
            self.observer_camera_fps,
        ) <= 0:
            raise ValueError("observer camera width/height/fps must be positive")
        if self.viser_host not in {"127.0.0.1", "localhost", "::1"}:
            raise PermissionError("fold exploration Viser must bind to loopback")
        if not 0.1 <= self.viser_refresh_s <= 30.0:
            raise ValueError("viser_refresh_s must be between 0.1 and 30 seconds")
        self.persistent_claude = PersistentClaudeSession(session.run_dir)
        # Load the reviewed skill library before constructing the Claude client.
        # The client owns the allow-list used by the visual/final-grounding
        # validators, so leaving it at ``available_skill_names()`` would make
        # dynamically approved skills visible in the run's experience store
        # but still rejectable at validation time.
        self.skill_store = SkillStore(self.project_root / "data" / "skills")
        approved_skills = self.skill_store.approved()
        self.client = ClaudeAutoClient(
            binary=claude_binary,
            timeout_s=self.claude_timeout_s,
            grounding_timeout_s=self.grounding_timeout_s,
            skill_guidance=self.skill_store.prompt(),
            skill_names=tuple(
                sorted({str(skill.name).strip().lower() for skill in approved_skills})
            ),
            persistent_session=self.persistent_claude,
        )
        self.supervisor = FoldSupervisor(
            claude_binary,
            supervisor_timeout_s,
            persistent_session=self.persistent_claude,
        )
        self.experiences = FoldExperienceStore(session.run_dir)
        self.skill_ledger = RunSkillLedger(session.workspace)
        self._debug_logger: FoldDebugLogger | None = None
        self._viser_process: subprocess.Popen[Any] | None = None

    def _refresh_client_skills(self) -> None:
        """Synchronize the Claude prompt and validator allow-list.

        Skill approvals are persisted on disk and may be updated between
        unattended iterations.  Keep both the guidance shown to Claude and
        the names accepted by the deterministic payload validator derived from
        the same ``SkillStore`` snapshot.
        """

        approved_skills = self.skill_store.approved()
        self.client.skill_guidance = self.skill_store.prompt()
        self.client.skill_names = tuple(
            sorted({str(skill.name).strip().lower() for skill in approved_skills})
        )

    def _debug(self, stage: str, message: str, **fields: Any) -> None:
        if stage != "run":
            self._last_operational_stage = str(stage)
        logger = self._debug_logger
        if logger is None:
            print(f"[fold-debug] {stage}: {message}", flush=True)
            return
        logger.log(stage, message, **fields)

    def _debug_exception(self, stage: str, exc: BaseException, **fields: Any) -> None:
        if stage != "run":
            self._last_operational_stage = str(stage)
        logger = self._debug_logger
        if logger is None:
            print(f"[fold-debug] {stage}: {type(exc).__name__}: {exc}", flush=True)
            return
        logger.exception(stage, exc, **fields)

    def _stop_viser_for_restart(self) -> None:
        """Stop the previous attempt's viewer before an unattended restart."""

        process = self._viser_process
        self._viser_process = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    @staticmethod
    def _unattended_error_is_retriable(
        exc: BaseException,
        operational_stage: str | None,
    ) -> bool:
        """Return whether an error can safely start a fresh run attempt.

        This is intentionally not a blanket ``except: continue``.  Failures
        raised while communicating with the real robot are not retried because
        the physical state may be unknown.  Most planning/preflight failures
        are handled inside the iteration loop and persisted there; this method
        covers the remaining safe non-physical failures (perception,
        supervisor, evaluation, and process-level faults).
        """

        if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            return False
        if operational_stage == "execution":
            return False
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(
            token in text
            for token in (
                "no space left",
                "disk quota",
                "read-only file system",
                "permission denied",
            )
        ):
            return False
        if operational_stage in {
            "perception",
            "supervisor",
            "molmo",
            "planning",
            "trajectory",
            "evaluation",
            "recording",
            "gripper",
            "screen",
            "evidence",
            "skills",
            "experience",
            "cleanup",
            "iteration",
            "viser",
        }:
            return True
        return any(
            token in text
            for token in (
                "claude",
                "planning",
                "grounding",
                "preflight",
                "reference",
                "perception",
                "camera",
                "supervisor",
                "evaluation",
                "schema",
                "molmo",
                "argument list too long",
            )
        )

    def _record_unattended_restart(
        self,
        exc: BaseException,
        *,
        operational_stage: str | None,
    ) -> None:
        """Persist a restart event without allowing logging to stop recovery."""

        self._unattended_restart_count += 1
        event = {
            "schema_version": 1,
            "created_at": _now(),
            "restart_index": self._unattended_restart_count,
            "error": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
            "operational_stage": operational_stage,
            "policy": "unattended_retry_after_nonphysical_failure",
        }
        path = self.session.run_dir / "unattended_restarts.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass
        self._debug(
            "run",
            "unattended restart after recoverable failure",
            restart_index=self._unattended_restart_count,
            operational_stage=operational_stage,
            error=event["error"],
        )

    def _persist_preexecution_planning_failure(
        self,
        *,
        output: Path,
        iteration_dir: Path,
        iteration: int,
        current_step: str,
        supervisor_before: Mapping[str, Any],
        screen_before: Mapping[str, Any],
        before_images: Sequence[Path],
        observer_before_images: Sequence[Path],
        acquisition_learning: Mapping[str, Any],
        molmo_hint: Mapping[str, Any] | None,
        planning_attempts: Sequence[Mapping[str, Any]],
        exc: BaseException,
        summary: dict[str, Any],
        history: list[dict[str, Any]],
        source_path: Path | None = None,
        model_proposal: Any = None,
        execution_proposal: Any = None,
        acquisition_strategy_validation: Mapping[str, Any] | None = None,
        acquisition_probe_plan: Mapping[str, Any] | None = None,
        host_compilation: Mapping[str, Any] | None = None,
        evidence_package: Mapping[str, Any] | None = None,
        action_mode: str | None = None,
        garment_condition: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a plan/preflight/IK rejection as a usable next-iteration lesson.

        No fold trajectory command has been sent when this method is called.
        (A real run may already have moved to the calibrated perception pose.)
        The garment therefore remains in the exact state represented by the
        ``before`` evidence, and the next iteration may safely capture it
        again.  Keeping this record in the normal experience store is what
        lets Claude see the rejection and change more than just the grasp
        height on the next unattended iteration.
        """

        error_text = f"{type(exc).__name__}: {exc}"
        failure = {
            "schema_version": 1,
            "status": "PLANNING_FAILURE",
            "stage": "planning",
            "failure_kind": (
                "CONTROLLER_IK"
                if "ik" in error_text.lower() or "controller" in error_text.lower()
                else "PREFLIGHT_OR_GROUNDING"
            ),
            "exception_type": type(exc).__name__,
            "error": error_text,
            "failed_pose": _extract_failure_pose(exc),
            "physical_command_sent": False,
            "fold_command_sent": False,
            "perception_pose_command_sent": bool(getattr(self, "real", False)),
            "garment_state_changed": False,
            "next_experiment": _planning_failure_next_experiment(exc),
            "action_mode": action_mode or "UNKNOWN",
            "garment_condition_before": (
                dict(garment_condition)
                if isinstance(garment_condition, Mapping)
                else {"condition": "UNKNOWN"}
            ),
            "garment_condition_after": (
                dict(garment_condition)
                if isinstance(garment_condition, Mapping)
                else {"condition": "UNKNOWN"}
            ),
            "created_at": _now(),
        }
        _write_json(iteration_dir / "planning_failure.json", failure)

        def payload(value: Any) -> dict[str, Any] | None:
            if value is None:
                return None
            if hasattr(value, "as_dict"):
                value = value.as_dict()
            if isinstance(value, Mapping):
                return dict(value)
            return None

        # A failed preflight cannot have changed the ordered fold ledger.
        # Provide an explicit after-state so a later supervisor fallback does
        # not mistake this planned step for a completed physical fold.
        supervisor_after = dict(supervisor_before)
        supervisor_after["status"] = "READY"
        supervisor_after["current_step"] = current_step
        supervisor_after["trajectory_decision"] = "CONTINUE"
        supervisor_after["confidence"] = min(
            float(supervisor_after.get("confidence", 0.0) or 0.0), 0.5
        )
        supervisor_after["reason"] = (
            "Planning/preflight/IK failed before any fold trajectory command was sent; "
            "the garment state and ordered fold ledger are unchanged. "
            + error_text[:900]
        )
        evidence = list(supervisor_after.get("evidence") or [])
        evidence.append("No fold trajectory command was sent; repeat the same current step with a revised plan.")
        supervisor_after["evidence"] = evidence[:8]
        _write_json(iteration_dir / "supervisor_after.json", supervisor_after)

        planning_diagnostics: dict[str, Any] = {
            "mode": "PLANNING_FAILURE",
            "attempts": [dict(item) for item in planning_attempts],
            "error": failure,
            "client_timing": getattr(self.client, "last_plan_timing", {}),
            "rejected_visual_references": getattr(
                self.client, "last_rejected_visual_references", []
            ),
            "selected_reference_validation": getattr(
                self.client, "last_reference_validation", None
            ),
            "molmo_sleeve_hint": molmo_hint,
            "acquisition_learning": dict(acquisition_learning),
            "acquisition_strategy_validation": (
                dict(acquisition_strategy_validation)
                if isinstance(acquisition_strategy_validation, Mapping)
                else acquisition_strategy_validation
            ),
            "acquisition_probe_plan": (
                dict(acquisition_probe_plan)
                if isinstance(acquisition_probe_plan, Mapping)
                else acquisition_probe_plan
            ),
            "host_compilation": (
                dict(host_compilation)
                if isinstance(host_compilation, Mapping)
                else host_compilation
            ),
            "model_proposal": payload(model_proposal),
            "execution_proposal": payload(execution_proposal),
        }
        visual_result = getattr(self.client, "last_visual_plan_result", None)
        plan_result = getattr(self.client, "last_plan_result", None)
        if visual_result is not None and hasattr(visual_result, "as_dict"):
            planning_diagnostics["visual_plan_result"] = visual_result.as_dict()
        if plan_result is not None and hasattr(plan_result, "as_dict"):
            planning_diagnostics["plan_result"] = plan_result.as_dict()
        _write_json(iteration_dir / "planning_diagnostics.json", planning_diagnostics)
        execution_gate = {
            "schema_version": 1,
            "iteration": iteration,
            "step": current_step,
            "mode": "PLANNING_FAILURE",
            "decision": "REJECT",
            "required_checks": {
                "semantic_target": "UNKNOWN",
                "metric_grounding": "FAILED",
                "workspace": "UNKNOWN",
                "legal_grasp_z": "UNKNOWN",
                "controller_ik": "FAILED" if failure.get("failure_kind") == "CONTROLLER_IK" else "UNKNOWN",
                "physical_command_sent": "PASS",
                "reason": failure["error"],
            },
        }
        # Keep the staged package structurally complete even when planning
        # stopped before metric grounding.  The explicit REJECT state prevents
        # a later reader from mistaking an absent file for an unrecorded pass.
        _write_fold_evidence_package(
            iteration_dir,
            stage="metric_grounding",
            payload={
                "schema_version": 1,
                "iteration": iteration,
                "step": current_step,
                "action_mode": action_mode or "UNKNOWN",
                "selected_reference": None,
                "selected_grasp": (
                    payload(model_proposal).get("selected_grasp")
                    if payload(model_proposal) is not None
                    else None
                ),
                "checks": {
                    "reference_validation": "FAILED",
                    "controller_ik": "UNKNOWN",
                },
                "error": failure["error"],
            },
        )
        _write_fold_evidence_package(
            iteration_dir,
            stage="execution_gate",
            payload=execution_gate,
        )
        model_payload = payload(model_proposal)
        execution_payload = payload(execution_proposal)
        if model_payload is not None:
            _write_json(iteration_dir / "claude_plan.json", model_payload)
        if execution_payload is not None:
            _write_json(iteration_dir / "execution_plan.json", execution_payload)
        if isinstance(host_compilation, Mapping):
            _write_json(iteration_dir / "host_compilation.json", dict(host_compilation))

        _write_fold_evidence_package(
            iteration_dir,
            stage="post_grasp",
            payload={
                "schema_version": 1,
                "iteration": iteration,
                "step": current_step,
                "mode": action_mode or "UNKNOWN",
                "status": "SKIPPED_BEFORE_EXECUTION",
                "before_images": [str(path) for path in before_images],
                "after_images": [],
                "video_evidence": [],
                "gripper_telemetry": None,
                "evaluation": None,
                "reason": "No physical fold command was sent.",
            },
        )

        record: dict[str, Any] = {
            "iteration": iteration,
            "planned_step": current_step,
            "mode": "PLANNING_FAILURE",
            "status": "PLANNING_FAILURE",
            "action_mode": action_mode or "UNKNOWN",
            "garment_condition_before": (
                dict(garment_condition)
                if isinstance(garment_condition, Mapping)
                else {"condition": "UNKNOWN"}
            ),
            "garment_condition_after": (
                dict(garment_condition)
                if isinstance(garment_condition, Mapping)
                else {"condition": "UNKNOWN"}
            ),
            "failure_stage": "planning",
            "physical_command_sent": False,
            "fold_command_sent": False,
            "perception_pose_command_sent": bool(getattr(self, "real", False)),
            "proposal": model_payload,
            "execution_proposal": execution_payload,
            "host_compilation": dict(host_compilation) if isinstance(host_compilation, Mapping) else None,
            "trajectory": None,
            "preflight": None,
            "controller_ik": None,
            "planning_attempts": [dict(item) for item in planning_attempts],
            "planning_failure": failure,
            "execution": None,
            "gripper_telemetry": None,
            "recording": None,
            "before_images": [str(path) for path in before_images],
            "after_images": [],
            "observer_images_before": [str(path) for path in observer_before_images],
            "observer_images_after": [],
            "observer_images_hold_check": [],
            "video_evidence": [],
            "video_references": [],
            "video_errors": [],
            "screen_before": dict(screen_before),
            "screen_after": None,
            "supervisor_before": dict(supervisor_before),
            "supervisor_after": supervisor_after,
            "molmo_sleeve_hint": molmo_hint,
            "acquisition_learning": dict(acquisition_learning),
            "acquisition_strategy_validation": (
                dict(acquisition_strategy_validation)
                if isinstance(acquisition_strategy_validation, Mapping)
                else acquisition_strategy_validation
            ),
            "acquisition_probe_plan": (
                dict(acquisition_probe_plan)
                if isinstance(acquisition_probe_plan, Mapping)
                else acquisition_probe_plan
            ),
            "planning_diagnostics": planning_diagnostics,
            "evidence_package": (
                dict(evidence_package)
                if isinstance(evidence_package, Mapping)
                else {
                    "directory": str((iteration_dir / "evidence").resolve()),
                    "manifest": str((iteration_dir / "evidence" / "manifest.json").resolve()),
                }
            ),
            "evaluation": None,
            "evaluation_raw": None,
            "completed_at": _now(),
        }
        _write_json(iteration_dir / "record.json", record)
        try:
            self.skill_ledger.append_experience(
                {
                    "created_at": _now(),
                    "iteration": iteration,
                    "mode": "PLANNING_FAILURE",
                    "supervisor_before": dict(supervisor_before),
                    "supervisor_after": supervisor_after,
                    "evaluation": None,
                    "planning_failure": failure,
                }
            )
        except Exception as skill_exc:
            self._debug_exception(
                "skills", skill_exc, iteration=iteration, nonfatal=True
            )
            _write_json(
                iteration_dir / "skill_append_error.json",
                {"error": f"{type(skill_exc).__name__}: {skill_exc}"},
            )
        experience_summary = self.experiences.append(record)
        record["experience_summary"] = experience_summary
        evidence_package = _write_fold_evidence_package(
            iteration_dir,
            stage="experience_update",
            payload={
                "schema_version": 1,
                "iteration": iteration,
                "step": current_step,
                "mode": action_mode or "UNKNOWN",
                "status": "PLANNING_FAILURE",
                "garment_condition_before": (
                    dict(garment_condition)
                    if isinstance(garment_condition, Mapping)
                    else {"condition": "UNKNOWN"}
                ),
                "supervisor_before": _compact_supervisor_state(supervisor_before),
                "supervisor_after": _compact_supervisor_state(supervisor_after),
                "planning_failure": failure,
                "next_step": current_step,
            },
        )
        record["evidence_package"] = evidence_package
        _write_json(iteration_dir / "record.json", record)
        history.append(record)

        summary["iterations"].append(
            {
                "iteration": iteration,
                "mode": "PLANNING_FAILURE",
                "action_mode": action_mode or "UNKNOWN",
                "garment_condition_before": (
                    (garment_condition or {}).get("condition", "UNKNOWN")
                    if isinstance(garment_condition, Mapping)
                    else "UNKNOWN"
                ),
                "next_step": current_step,
                "visibility": supervisor_after.get("garment_visibility", "UNKNOWN"),
                "evaluation_status": None,
                "planning_status": "FAILED_BEFORE_EXECUTION",
                "physical_command_sent": False,
                "failed_pose": failure.get("failed_pose"),
                "error": error_text,
                "evidence_package": (
                    dict(evidence_package)
                    if isinstance(evidence_package, Mapping)
                    else {
                        "manifest": str(
                            (iteration_dir / "evidence" / "manifest.json").resolve()
                        )
                    }
                ),
            }
        )
        failure_counts = summary.setdefault("failure_counts", {})
        if isinstance(failure_counts, dict):
            failure_counts["planning_before_execution"] = int(
                failure_counts.get("planning_before_execution", 0) or 0
            ) + 1
        persistent = getattr(self, "persistent_claude", None)
        if persistent is not None and hasattr(persistent, "as_dict"):
            summary["persistent_claude_session"] = persistent.as_dict()
        _write_json(output / "summary.json", summary)
        self._debug(
            "iteration",
            "recorded pre-execution planning failure; continuing to next iteration",
            iteration=iteration,
            current_step=current_step,
            error=error_text,
            failed_pose=failure.get("failed_pose"),
            experience_count=experience_summary.get("experience_count"),
        )
        if source_path is not None and source_path.is_file():
            try:
                shutil.copy2(source_path, iteration_dir / "failed_plan_source.py")
                source_path.unlink()
            except OSError as cleanup_exc:
                self._debug_exception(
                    "cleanup", cleanup_exc, path=str(source_path), nonfatal=True
                )
        return record

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

    def _capture_observer_rgb(self, output_dir: Path) -> Path | None:
        """Capture one optional uncalibrated observer RGB frame.

        Camera C is intentionally diagnostic-only.  A missing observer or an
        unsupported profile must never block calibrated A/B perception or
        robot execution; the evaluator will mark the observer evidence as
        unavailable for that iteration.
        """

        serial = self.observer_camera_serial
        if not serial:
            return None
        try:
            manifest = capture_observer_rgb(
                serial,
                output_dir,
                label=self.observer_camera_label,
                width=self.observer_camera_width,
                height=self.observer_camera_height,
                fps=self.observer_camera_fps,
                color_exposure=self.observer_camera_exposure,
                color_white_balance=self.observer_camera_white_balance,
            )
        except Exception as exc:
            self._debug_exception(
                "observer-camera",
                exc,
                serial=serial,
                output_dir=str(output_dir),
                nonfatal=True,
            )
            return None
        image = Path(str(manifest.get("rgb_image", ""))).resolve()
        if not image.is_file():
            self._debug(
                "observer-camera",
                "observer capture returned no RGB image",
                serial=serial,
                output_dir=str(output_dir),
            )
            return None
        self._debug(
            "observer-camera",
            "captured uncalibrated RGB observer frame",
            label=self.observer_camera_label,
            serial=serial,
            image=str(image),
            resolution=manifest.get("resolution"),
            fps=manifest.get("fps"),
        )
        return image

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
            observer_image = self._capture_observer_rgb(output_dir)
            images = [*upright, *global_perception_image_paths(saved, saved_path)]
            if observer_image is not None:
                images.append(observer_image)
            staged = _stage_image_artifacts(images, output_dir)
            self._debug(
                "perception",
                "reused latest perception",
                result=str(saved_path),
                images=len(images),
                observer_images=1 if observer_image is not None else 0,
                upright_planning_images=[path.name for path in upright],
                staged_images=len(staged),
                duration_s=round(time.monotonic() - started, 3),
            )
            return saved, saved_path, images
        if self.real:
            self._debug("perception", "moving robot to calibrated perception pose")
            move_robot_to_perception_position(self.session.robot_config)
            self._debug("perception", "capturing synchronized configured RGB-D cameras")
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
        observer_image = self._capture_observer_rgb(output_dir)
        images = [*upright, *raw, *global_perception_image_paths(saved, saved_path)]
        if observer_image is not None:
            images.append(observer_image)
        staged = _stage_image_artifacts(images, output_dir)
        self._debug(
            "perception",
            "capture completed",
            result=str(saved_path),
            raw_images=len(raw),
            images=len(images),
            observer_images=1 if observer_image is not None else 0,
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
                    capture_stage=stage,
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
        hold_action_index: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started = time.monotonic()
        self._debug("execution", "starting trajectory execution", label=label, source=str(source_path))
        recording_dir = iteration_dir / "rollout_recording"
        recorder: DualRealSenseRolloutRecorder | None = None
        thread: threading.Thread | None = None
        observer_recorder: ObserverRGBRolloutRecorder | None = None
        observer_thread: threading.Thread | None = None
        recording_result: dict[str, Any] = {}
        observer_recording_result: dict[str, Any] = {}
        recording_errors: list[str] = []
        hold_snapshot: dict[str, Any] | None = None
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

            if recorder is not None:
                def record() -> None:
                    try:
                        recording_result["manifest"] = recorder.record()
                    except BaseException as exc:
                        recording_errors.append(f"{type(exc).__name__}: {exc}")

                thread = threading.Thread(target=record, daemon=True, name=f"fold-record-{label}")
                thread.start()

            if self.observer_camera_serial:
                self._debug(
                    "recording",
                    "starting uncalibrated observer RGB recorder",
                    directory=str(recording_dir),
                    label=self.observer_camera_label,
                    serial=self.observer_camera_serial,
                )
                try:
                    observer_recorder = ObserverRGBRolloutRecorder(
                        self.observer_camera_serial,
                        recording_dir,
                        label=self.observer_camera_label,
                        width=self.observer_camera_width,
                        height=self.observer_camera_height,
                        fps=self.observer_camera_fps,
                        color_exposure=self.observer_camera_exposure,
                        color_white_balance=self.observer_camera_white_balance,
                        codec=self.recording_codec,
                    )
                    observer_recorder.start()
                except Exception as exc:
                    recording_errors.append(
                        f"observer recorder start: {type(exc).__name__}: {exc}"
                    )
                    self._debug_exception(
                        "recording",
                        exc,
                        observer=True,
                        nonfatal=True,
                    )
                    if observer_recorder is not None:
                        try:
                            observer_recorder.close()
                        except Exception as close_exc:
                            recording_errors.append(
                                "observer recorder cleanup: "
                                f"{type(close_exc).__name__}: {close_exc}"
                            )
                    observer_recorder = None
                if observer_recorder is not None:
                    def record_observer() -> None:
                        try:
                            observer_recording_result["manifest"] = observer_recorder.record()
                        except BaseException as exc:
                            recording_errors.append(
                                f"observer recorder: {type(exc).__name__}: {exc}"
                            )

                    observer_thread = threading.Thread(
                        target=record_observer,
                        daemon=True,
                        name=f"fold-observer-record-{label}",
                    )
                    observer_thread.start()
            if thread is not None or observer_thread is not None:
                time.sleep(0.25)
        def on_robot_action(action_index: int, action: Mapping[str, Any]) -> None:
            """Capture one Camera-C still immediately after the first lift move."""

            nonlocal hold_snapshot
            if hold_action_index is None or action_index != hold_action_index:
                return
            if hold_snapshot is not None:
                return
            snapshot_started_ns = time.monotonic_ns()
            hold_dir = iteration_dir / "hold_check"
            hold_path = hold_dir / f"camera_{self.observer_camera_label}_observer_rgb_hold_check.png"
            try:
                if observer_recorder is not None:
                    hold_snapshot = observer_recorder.save_snapshot(
                        hold_path,
                        # Accept a frame captured just before the action callback
                        # (the recorder and robot run in parallel), but never use
                        # a stale frame from before the lift began.
                        after_monotonic_ns=snapshot_started_ns - 250_000_000,
                        timeout_s=3.0,
                    )
                elif self.real and self.observer_camera_serial:
                    # If video recording was disabled or failed to start, Camera C
                    # is still useful as a one-shot diagnostic observer.  This
                    # path opens Camera C only when no recorder owns it.
                    image = self._capture_observer_rgb(hold_dir)
                    if image is None:
                        raise RuntimeError("observer capture returned no hold-check image")
                    image = image.resolve()
                    if image != hold_path.resolve():
                        hold_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(image, hold_path)
                    hold_snapshot = {
                        "status": "CAPTURED",
                        "label": self.observer_camera_label,
                        "serial": self.observer_camera_serial,
                        "image": str(hold_path.resolve()),
                        "source": "observer_single_frame_capture",
                    }
                else:
                    hold_snapshot = {
                        "status": "UNAVAILABLE",
                        "reason": "observer camera serial is not configured",
                    }
                hold_snapshot.update(
                    {
                        "action_index": int(action_index),
                        "action": dict(action),
                        "captured_after": "first_post_close_lift",
                    }
                )
                self._debug(
                    "observer-camera",
                    "captured Camera-C lift hold-check still",
                    iteration=iteration_dir.name,
                    action_index=action_index,
                    image=hold_snapshot.get("image"),
                    status=hold_snapshot.get("status"),
                )
            except Exception as exc:
                hold_snapshot = {
                    "status": "FAILED",
                    "action_index": int(action_index),
                    "action": dict(action),
                    "captured_after": "first_post_close_lift",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                self._debug_exception(
                    "observer-camera",
                    exc,
                    iteration=iteration_dir.name,
                    action_index=action_index,
                    nonfatal=True,
                )

        execution_interrupted = False
        try:
            self._debug("execution", "sending validated trajectory to session runner", real=self.real)
            session_kwargs: dict[str, Any] = {
                "real": self.real,
                "confirmed": self.confirm_real,
                "notes": f"Closed-loop five-step folding {label}.",
            }
            if hold_action_index is not None:
                session_kwargs["action_callback"] = on_robot_action
            execution = self.session.run_experiment(source_path.name, **session_kwargs)
        except KeyboardInterrupt:
            # Do not turn an operator stop into an ordinary failed rollout.
            # The session's finally block still attempts mandatory Home; use a
            # short recorder shutdown timeout so Ctrl-C is responsive even if
            # a camera thread is blocked in SDK I/O.
            execution_interrupted = True
            raise
        finally:
            recorder_join_timeout = 5.0 if execution_interrupted else 300.0
            if recorder is not None:
                recorder.request_stop("fold_action_completed")
            if thread is not None:
                thread.join(timeout=recorder_join_timeout)
            if recorder is not None and thread is not None and thread.is_alive():
                recording_errors.append("recording thread did not stop within 300 seconds")
                recorder.close()
                thread.join(timeout=3.0)
            if observer_recorder is not None:
                observer_recorder.request_stop("fold_action_completed")
            if observer_thread is not None:
                observer_thread.join(timeout=recorder_join_timeout)
            if (
                observer_recorder is not None
                and observer_thread is not None
                and observer_thread.is_alive()
            ):
                recording_errors.append(
                    "observer recording thread did not stop within 300 seconds"
                )
                observer_recorder.close()
                observer_thread.join(timeout=3.0)
        recording = {
            "status": "failed"
            if recording_errors
            else ("completed" if recorder is not None or observer_recorder is not None else "disabled"),
            "directory": (
                str(recording_dir.resolve())
                if recorder is not None or observer_recorder is not None
                else None
            ),
            "manifest": recording_result.get("manifest"),
            "observer": {
                "status": (
                    "failed"
                    if any(item.startswith("observer") for item in recording_errors)
                    else ("completed" if observer_recorder is not None else "disabled")
                ),
                "label": self.observer_camera_label if self.observer_camera_serial else None,
                "serial": self.observer_camera_serial,
                "manifest": observer_recording_result.get("manifest")
                or (
                    observer_recorder.manifest()
                    if observer_recorder is not None
                    else None
                ),
                "video": (
                    str(observer_recorder.video_path.resolve())
                    if observer_recorder is not None
                    and observer_recorder.video_path.is_file()
                    else None
                ),
                "hold_check": hold_snapshot,
            },
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
        history: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any] | None:
        """Return one fallible Molmo sleeve-region hint before Claude selects Rxxx."""

        if not self.molmo_sleeve_grounding or step not in {
            "left_sleeve",
            "right_sleeve",
        }:
            return None
        artifact_dir = iteration_dir / "molmo_sleeve_locator"
        if history and _evaluation_reports_unchanged(history[-1].get("evaluation")):
            for row in reversed(history):
                if not isinstance(row, Mapping) or row.get("planned_step") != step:
                    continue
                previous = row.get("molmo_sleeve_hint")
                if not isinstance(previous, Mapping):
                    continue
                if (
                    previous.get("status") != "MOLMO_POINT_AVAILABLE"
                    or not isinstance(previous.get("upright_pixel_xy"), list)
                    or len(previous["upright_pixel_xy"]) != 2
                ):
                    continue
                hint = dict(previous)
                hint.update(
                    {
                        "status": "MOLMO_POINT_AVAILABLE",
                        "reused": True,
                        "reused_from_iteration": row.get("iteration"),
                        "reuse_reason": (
                            "latest evaluation reported unchanged visible area, overlap, "
                            "and relief; Molmo remains a hint only"
                        ),
                        "duration_s": 0.0,
                    }
                )
                _write_json(iteration_dir / "molmo_sleeve_hint.json", hint)
                self._debug(
                    "molmo",
                    "reused prior sleeve-region hint because garment state is unchanged",
                    iteration=iteration,
                    step=step,
                    reused_from_iteration=row.get("iteration"),
                    raw_pixel=hint.get("raw_pixel_xy"),
                    upright_pixel=hint.get("upright_pixel_xy"),
                )
                return hint
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
            if hasattr(self, "skill_store") and hasattr(self, "client"):
                self._refresh_client_skills()
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

    def _resolve_fold_grasp_height(
        self,
        proposal: ExplorationProposal,
    ) -> tuple[ExplorationProposal, dict[str, Any] | None]:
        """Apply the shared measured-surface grasp-Z policy before execution.

        Claude remains responsible for the target, entry, transport, and
        laydown strategy. The engaged grasp height is safety-critical and must
        be derived from the selected calibrated pixel, including the configured
        sponge allowance, rather than copied from Claude's guessed trajectory.
        """

        reference = getattr(self.client, "last_reference_validation", None)
        if not isinstance(reference, Mapping):
            raise ExplorationPlanningError(
                "cannot resolve fold grasp height without selected-reference measurement"
            )
        camera = str(reference.get("camera", "A"))
        pixel = reference.get("pixel_xy")
        if (
            not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or any(isinstance(value, bool) for value in pixel)
        ):
            raise ExplorationPlanningError(
                "selected-reference measurement has no valid pixel_xy for grasp-height resolution"
            )
        try:
            x_px, y_px = int(pixel[0]), int(pixel[1])
        except (TypeError, ValueError) as exc:
            raise ExplorationPlanningError(
                "selected-reference measurement has non-integer pixel_xy"
            ) from exc
        perception_dir = self.session.run_dir.resolve() / "workspace" / "perception_views"
        try:
            measurement = GarmentGrounding(perception_dir).sample_local_surface(
                camera,
                x_px,
                y_px,
                radius_px=3,
                include_nearest_reference=False,
            )
        except GroundingToolError as exc:
            raise ExplorationPlanningError(
                f"selected fold pixel has no usable calibrated surface: {exc}"
            ) from exc
        if measurement.get("valid") is not True:
            raise ExplorationPlanningError(
                "selected fold pixel has no usable calibrated surface: "
                f"{measurement.get('reason', 'unknown measurement failure')}"
            )
        try:
            resolution = resolve_grasp_height(
                measurement=measurement,
                table_plane_abc=None,
                robot_config=self.session.robot_config,
            )
        except GraspHeightError as exc:
            raise ExplorationPlanningError(
                f"shared grasp-height policy rejected the selected fold pixel: {exc}"
            ) from exc
        grasp_index, grasp_move = _first_grasp_move(proposal)
        requested_z = float(grasp_move["z"])
        target_z = float(resolution.target_xyz_mm[2])
        actions = [
            {"name": action["name"], "args": dict(action["args"])}
            for action in proposal.actions
        ]
        actions[grasp_index]["args"]["z"] = target_z
        grounded = replace(proposal, actions=tuple(actions))
        audit = {
            "camera": camera,
            "pixel_xy": [x_px, y_px],
            "selected_reference_id": reference.get("reference_id"),
            "requested_grasp_z_mm": requested_z,
            "resolved_grasp_z_mm": target_z,
            "z_rewritten": bool(abs(requested_z - target_z) > 1e-6),
            "measurement": measurement,
            "resolution": resolution.as_dict(),
        }
        self._debug(
            "grasp-height",
            "resolved fold grasp height from calibrated surface",
            camera=camera,
            pixel_xy=[x_px, y_px],
            selected_reference_id=reference.get("reference_id"),
            requested_z_mm=round(requested_z, 3),
            resolved_z_mm=round(target_z, 3),
            z_rewritten=audit["z_rewritten"],
            support_layer_active=resolution.support_layer_active,
            support_layer_confirmed=resolution.support_layer_confirmed,
            support_layer_activation_source=resolution.support_layer_activation_source,
            desired_compression_mm=resolution.desired_compression_mm,
            achieved_compression_mm=resolution.achieved_compression_mm,
        )
        return grounded, audit

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

        # Skill approvals are persisted on disk and may change while an
        # unattended run is alive. Keep Claude's prompt and the deterministic
        # validator allow-list derived from the same current snapshot.
        if hasattr(self, "skill_store") and hasattr(self, "client"):
            self._refresh_client_skills()

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
                if _is_reference_grounding_mismatch(exc) and attempt < attempts:
                    # The selected Rxxx survived the initial deterministic
                    # semantic gate, but Stage 2 proved that Claude's final
                    # grasp XY does not actually use that reference.  A
                    # grounding-only repair would keep the same bad ID, so
                    # explicitly restart Stage 1 on the next attempt.  The
                    # Claude client sees ``feedback`` together with its
                    # previous visual result and excludes that Rxxx from the
                    # fresh visual-selection turn.
                    feedback = (
                        "Stage 2 found a contradiction between the selected Rxxx and "
                        "the final grasp coordinates. Restart STAGE 1 visual planning "
                        "and select a different Rxxx that is visibly on the requested "
                        "garment region; do not retain the previous reference ID and "
                        "do not substitute a Molmo or historical coordinate while "
                        "leaving selected_reference unchanged. The previous Stage-2 "
                        f"validation error was: {exc}"
                    )
                    self._debug(
                        "planning",
                        "forcing Stage-1 reselection after Stage-2 Rxxx mismatch",
                        iteration=iteration,
                        attempt=attempt,
                        next_attempt=attempt + 1,
                        attempt_kind=attempt_kind,
                    )
                    delay = self.retry_backoff_s * attempt
                    if delay:
                        self._debug(
                            "planning",
                            "waiting before Stage-1 reselection retry",
                            delay_s=delay,
                        )
                        time.sleep(delay)
                    continue
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
            and result.get("current_step") in FOLD_STEP_IDS
            and result.get("trajectory_decision") == "CONTINUE"
        ):
            # Keep the run moving when Claude emits an internally inconsistent
            # BLOCKED/CONTINUE pair.  The fold planner still has to pass all
            # static and controller checks before anything physical happens.
            result["status"] = "READY"
            self._debug(
                "supervisor",
                "normalized inconsistent BLOCKED/CONTINUE response",
                current_step=result.get("current_step"),
            )
        # Recovery/legacy normalization above may change BLOCKED to READY.
        # Re-derive the executable step once more so a stale terminal marker
        # can never survive into the fold planner.
        if result.get("status") == "READY":
            reported_step = result.get("current_step")
            result = _normalize_supervisor_current_step(result)
            if reported_step != result.get("current_step"):
                self._debug(
                    "supervisor",
                    "normalized current step from ordered completion ledger",
                    reported_current_step=reported_step,
                    current_step=result.get("current_step"),
                )
        # Keep completion monotonic across supervisor calls.  This is applied
        # after legacy recovery/status normalization so a transient Claude
        # failure or an ambiguous/bunched sleeve cannot roll a confirmed
        # right_sleeve back to right_sleeve after the task has reached
        # left_side.  Fallback completions are discarded by the merge.
        reported_completed = list(result.get("completed_steps", []))
        reported_step = result.get("current_step")
        result = _merge_supervisor_completion_ledger(result, history)
        if (
            reported_completed != result.get("completed_steps")
            or reported_step != result.get("current_step")
        ):
            self._debug(
                "supervisor",
                "merged monotonic visual completion ledger",
                reported_completed_steps=reported_completed,
                confirmed_completed_steps=result.get("completed_steps"),
                reported_current_step=reported_step,
                current_step=result.get("current_step"),
                completion_ledger_source=result.get("completion_ledger_source"),
            )
        return result

    def _local_acquisition_supervisor(
        self,
        screen: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        *,
        step: str,
        base: Mapping[str, Any] | None = None,
        reason: str,
    ) -> dict[str, Any]:
        """Reuse the fold ledger when an action cannot have completed a fold."""
        # Reuse only the host's non-fallback visual ledger.  In particular, a
        # failed acquisition probe must not inherit a fallback entry that
        # advanced the planned step merely to avoid an unattended loop.
        ledger_records: list[Mapping[str, Any]] = list(history)
        if isinstance(base, Mapping):
            ledger_records.append({"supervisor_before": base})
        completed = _confirmed_completion_ledger(ledger_records)
        if step in FOLD_STEP_IDS and step not in completed:
            current_step = step
        else:
            current_step = _earliest_incomplete_fold_step(completed)
        visibility = str(screen.get("visibility", "UNKNOWN"))
        if visibility not in {"FULL", "PARTIAL", "UNKNOWN"}:
            visibility = "UNKNOWN"
        return {
            "status": "READY",
            "current_step": current_step,
            "completed_steps": completed,
            "garment_visibility": visibility,
            "trajectory_decision": "CONTINUE",
            "confidence": 1.0,
            "evidence": [
                "The latest action was acquisition-only or visibly closed empty.",
                "No successful fold transport/laydown occurred, so the ordered fold ledger is unchanged.",
            ],
            "reason": reason,
            "fallback": False,
            "local_deterministic": True,
            "completion_ledger_source": "historical_visual_union",
        }

    def _fallback_supervisor(
        self,
        screen: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        error: BaseException | None,
    ) -> dict[str, Any]:
        """Produce a bookkeeping state when the read-only supervisor is down.

        The optional planned-step advancement below exists only to prevent an
        unattended process from repeating an action whose after-state was not
        written.  The returned ``fallback=true`` marker makes this state
        ineligible for the visual completion ledger; it is never treated as
        proof that a sleeve was folded.
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
        current_step = _earliest_incomplete_fold_step(completed)
        visibility = str(screen.get("visibility", "UNKNOWN"))
        if visibility not in {"FULL", "PARTIAL", "UNKNOWN"}:
            visibility = "UNKNOWN"
        error_text = f"{type(error).__name__}: {error}" if error else "unknown supervisor failure"
        return {
            "status": "COMPLETE" if current_step == "COMPLETE" else "READY",
            "current_step": current_step,
            "completed_steps": completed,
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
            "completion_ledger_source": "fallback_bookkeeping",
        }

    def _evaluate_with_retries(
        self,
        *args: Any,
        iteration: int,
        acquisition_probe: bool = False,
        compact_video_images: Sequence[Path] = (),
        compact_video_references: Sequence[Path] = (),
        compact_video_errors: Sequence[str] = (),
        **kwargs: Any,
    ) -> tuple[Any, ClaudeEvaluationResult | None]:
        attempts = self.max_stage_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if acquisition_probe:
                    compact_kwargs = {
                        key: kwargs[key]
                        for key in (
                            "proposal",
                            "run_dir",
                            "rollout_recording_dir",
                            "gripper_telemetry",
                            "observer_images",
                        )
                        if key in kwargs
                    }
                    evaluation = self.client.evaluate_acquisition_probe(
                        *args,
                        **compact_kwargs,
                        rollout_evidence_images=compact_video_images,
                        rollout_video_references=compact_video_references,
                        rollout_evidence_errors=compact_video_errors,
                    )
                else:
                    full_kwargs = dict(kwargs)
                    evaluation = self.client.evaluate(*args, **full_kwargs)
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
        """Run until completion, optionally restarting after safe failures.

        Normal runs preserve the original one-shot behavior.  In unattended
        mode pre-execution planning failures are recorded inside the current
        run and the next iteration starts with that lesson in its history;
        failures in other safe non-physical stages may start a fresh attempt
        with the same persistent experience store and Claude session.
        KeyboardInterrupt and errors from the physical execution stage are
        never swallowed.
        """

        if not self.unattended:
            return self._run_once()

        while True:
            try:
                return self._run_once()
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as exc:
                if not self._unattended_error_is_retriable(
                    exc,
                    self._last_operational_stage,
                ):
                    raise
                self._record_unattended_restart(
                    exc,
                    operational_stage=self._last_operational_stage,
                )
                self._stop_viser_for_restart()
                # A failed attempt may have left a partially written or stale
                # perception bundle.  The next attempt must acquire fresh A/B
                # RGB-D rather than reusing it.
                self.reuse_latest_perception = False
                delay = min(60.0, self.retry_backoff_s * max(1, self._unattended_restart_count))
                if delay > 0:
                    time.sleep(delay)

    def _run_once(self) -> dict[str, Any]:
        self._last_operational_stage = None
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
            persistent_claude_session=self.persistent_claude.session_id,
        )
        self._start_viser(output)
        try:
            config = PerceptionConfig.load(self.project_root, self.perception_config)
        except BaseException as exc:
            self._debug_exception("run", exc, stage_detail="loading perception configuration")
            raise
        self._debug("run", "loaded perception configuration", path=str(self.perception_config))
        # Keep the full local experience list for host-owned completion
        # bookkeeping; Claude still receives only the compact recent window
        # through ``_compact_history``.
        history: list[dict[str, Any]] = self.experiences.history(limit=None)
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
            "unattended": self.unattended,
            "unattended_attempt": self._unattended_restart_count + 1,
            "unattended_restart_log": (
                str((self.session.run_dir / "unattended_restarts.jsonl").resolve())
                if self.unattended
                else None
            ),
            "timeouts": {
                "claude_s": self.claude_timeout_s,
                "grounding_s": self.grounding_timeout_s,
                "supervisor_s": self.supervisor.timeout_s,
                "recording_join_s": 300,
            },
            "retry_policy": {
                "max_stage_retries": self.max_stage_retries,
                "retry_backoff_s": self.retry_backoff_s,
                "max_acquisition_probes_per_step": MAX_ACQUISITION_PROBES_PER_STEP,
                "low_z_preclose_sweep": "rejected",
                "bunched_sleeve_policy": "REPAIR_SLEEVE_before_probe_or_fold",
                "preexecution_planning_failure": (
                    "persist_experience_and_continue_next_iteration"
                    if self.unattended
                    else "fail_fast"
                ),
                "supervisor_fallback": True,
                "evaluation_fallback": True,
            },
            "plan_authority": {
                "strategy": "Claude",
                "host_role": "schema, coordinate, safety, preflight, IK, execution",
                "host_compile_acquisition_probe": self.host_compile_acquisition_probe,
                "silent_plan_rewrite": False,
            },
            "persistent_claude_session": self.persistent_claude.as_dict(),
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
            "observer_camera": {
                "enabled": self.observer_camera_serial is not None,
                "label": self.observer_camera_label if self.observer_camera_serial else None,
                "serial": self.observer_camera_serial,
                "calibrated": False,
                "geometry_used": False,
                "stream": "RGB-only observer for grasp/occlusion evaluation",
                "resolution": [
                    self.observer_camera_width,
                    self.observer_camera_height,
                ],
                "fps": self.observer_camera_fps,
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
            "failure_counts": {
                "planning_before_execution": 0,
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
                observer_before_images = _select_observer_images(before_images)
                self._debug(
                    "observer-camera",
                    "before observer evidence selected",
                    iteration=iteration,
                    images=[str(path) for path in observer_before_images],
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
                reuse_step = _acquisition_supervisor_reuse_step(history)
                if reuse_step is not None:
                    supervisor_before = self._local_acquisition_supervisor(
                        screen_before,
                        history,
                        step=reuse_step,
                        reason=(
                            "Reused the previous ordered fold state because the latest "
                            "iteration was an acquisition failure/reversible probe."
                        ),
                    )
                    self._debug(
                        "supervisor",
                        "skipped redundant before-supervisor during acquisition learning",
                        iteration=iteration,
                        current_step=reuse_step,
                    )
                else:
                    supervisor_before = self._supervisor(
                        before_images,
                        screen_before,
                        history,
                    )
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
                if supervisor_before["status"] == "COMPLETE" or supervisor_before["current_step"] == "COMPLETE":
                    summary["status"] = "COMPLETE"
                    summary["completed_at"] = _now()
                    _write_json(output / "summary.json", summary)
                    return summary
                if supervisor_before["status"] == "BLOCKED" or supervisor_before["current_step"] == "BLOCKED":
                    summary["status"] = "BLOCKED"
                    summary["blocked_reason"] = supervisor_before["reason"]
                    summary["completed_at"] = _now()
                    _write_json(output / "summary.json", summary)
                    return summary
                # This experiment intentionally has no recovery branch.  The
                # screen result is retained in the record/debug stream, while
                # folding proceeds from the host-normalized current step.
                mode = "FOLD"
                proposal: ExplorationProposal | None = None
                current_step = supervisor_before["current_step"]
                acquisition_learning = _fold_acquisition_learning_state(
                    history,
                    current_step,
                )
                garment_condition = _garment_condition_from_history(
                    history,
                    step=current_step,
                )
                durable_condition = self.experiences.condition(current_step)
                if isinstance(durable_condition, Mapping):
                    durable_name = str(
                        durable_condition.get("condition", "UNKNOWN")
                    ).upper()
                    if durable_name in {"FLAT", "BUNCHED", "UNKNOWN"}:
                        garment_condition = dict(durable_condition)
                        garment_condition["condition"] = durable_name
                        garment_condition.setdefault("source", "durable_condition_ledger")
                action_mode, mode_policy = _proposal_action_mode(
                    acquisition_learning,
                    garment_condition,
                    step=current_step,
                )
                mode = action_mode
                _write_json(
                    iteration_dir / "acquisition_learning_before.json",
                    acquisition_learning,
                )
                _write_json(
                    iteration_dir / "garment_condition_before.json",
                    garment_condition,
                )
                self._debug(
                    "learning",
                    "built acquisition learning state",
                    iteration=iteration,
                    step=current_step,
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
                    garment_condition=garment_condition.get("condition"),
                    action_mode=action_mode,
                    probe_budget_remaining=mode_policy.get("probe_budget_remaining"),
                )
                molmo_hint = self._locate_sleeve_with_molmo(
                    step=current_step,
                    iteration=iteration,
                    iteration_dir=iteration_dir,
                    history=history,
                )
                evidence_package = _write_fold_evidence_package(
                    iteration_dir,
                    stage="rgb_selection",
                    payload={
                        "schema_version": 1,
                        "iteration": iteration,
                        "step": current_step,
                        "action_mode": action_mode,
                        "garment_condition": garment_condition,
                        "supervisor": _compact_supervisor_state(supervisor_before),
                        "images": [str(path) for path in before_images],
                        "observer_images": [str(path) for path in observer_before_images],
                        "molmo_hint": molmo_hint,
                        "history": _compact_history(history, 6),
                    },
                )
                step_label = next((item["label"] for item in FOLD_STEPS if item["id"] == current_step), "the current incomplete fold step")
                objective = (
                    "Fold this shirt using exactly five ordered steps: left sleeve inward, right sleeve inward, "
                    "first torso side inward, second torso side inward, bottom hem upward. "
                    "For this task, left/right always mean the viewer's left/right in the displayed clockwise-90 "
                    "upright Camera-A image (smaller/larger upright x); never use the wearer's anatomical sides "
                    "and never mirror the image. "
                    f"The supervisor says current_step is {current_step} ({step_label}). "
                    "Plan exactly one action for that current step; do not skip ahead. "
                    "Use current RGB as the primary evidence and the calibrated references only for grounding. "
                    "The host does not reveal a privileged grasp structure; infer the contact hypothesis from "
                    "the current RGB and saved physical outcomes."
                )
                if action_mode == "REPAIR_SLEEVE":
                    objective += (
                        "\n\nACTION MODE — REPAIR_SLEEVE: the saved visual evidence says the target sleeve is "
                        "gathered, rolled, or bunched. Do not perform another acquisition probe and do not "
                        "fold the sleeve inward yet. Plan one controlled repair that grasps a visible outer "
                        "sleeve/cuff region, lifts first, moves outward away from the shoulder/torso, and lays "
                        "the sleeve flatter so the distal edge becomes visible again. Do not use a low-Z "
                        "pre-close lateral sweep or scrub; the first post-close move must be upward. Keep "
                        f"current_step={current_step} and explain the expected unbunching evidence."
                    )
                elif action_mode == "FOLD":
                    objective += (
                        "\n\nACTION MODE — FOLD: do not return another lift-only acquisition probe. Plan the actual "
                        "inward fold for the current step: grasp the visible sleeve/free edge, lift above the "
                        "cloth, transport it toward the garment center, lay it on the torso, and release. "
                        "Do not push or scrub the fabric before closing."
                    )
                objective += "\nSupervisor state:\n" + json.dumps(
                    _compact_supervisor_state(supervisor_before),
                    ensure_ascii=False,
                    indent=2,
                )
                objective += (
                    "\nAcquisition learning state (physical evidence, not a grasp answer):\n"
                    + json.dumps(acquisition_learning, ensure_ascii=False, indent=2)
                )
                objective += (
                    "\nPlanning history rule: entries marked PLANNING_FAILURE were rejected "
                    "before any robot command and therefore did not change the garment. "
                    "Use their deterministic error and next_experiment lesson to avoid "
                    "repeating the same invalid trajectory, while keeping current_step "
                    "unchanged.\n"
                )
                if action_mode == "ACQUISITION_PROBE":
                    objective += (
                        "\n\nEXECUTION CONTRACT — ACQUISITION PROBE: this iteration is a "
                        "reversible acquisition experiment chosen by Claude's learning "
                        "strategy. Your returned action list must itself contain the "
                        "probe: approach/open/close, one or more near-vertical lift "
                        "checkpoints, reversal, release, and home. Do not include any "
                        "post-close lateral transport or fold laydown. Set "
                        "requires_lift_checkpoint=true. The host will validate this "
                        "contract and will not silently replace a full fold plan with a "
                        "different experiment. If the probe is not appropriate, explain "
                        "why in reveal_strategy, but still return a schema-valid plan."
                    )
                objective += (
                    "\nEvidence package (host gate is authoritative): the RGB selection, metric grounding, "
                    "execution gate, and prior outcome files will be written under this iteration's "
                    "evidence/ directory. Select the target from RGB; do not invent coordinates."
                )
                if isinstance(molmo_hint, Mapping):
                    objective += (
                        "\nMolmo sleeve-region hypothesis (fallible topology guide, "
                        "never a direct grasp point):\n"
                        + json.dumps(molmo_hint, ensure_ascii=False, indent=2)
                    )
                self._debug("planning", "asking Claude for fold proposal", iteration=iteration, current_step=current_step)
                planning_attempts: list[dict[str, Any]] = []
                source_path = self.session.workspace / f"_fold_experiment_{iteration:03d}.py"
                preflight = None
                controller = None
                acquisition_strategy_validation: dict[str, Any] | None = None
                acquisition_probe_plan: dict[str, Any] | None = None
                host_compilation: dict[str, Any] = {
                    "authority": "Claude",
                    "rewritten": False,
                    "reason": "The validated trajectory is compiled directly from Claude's proposal.",
                }
                # Visual planning itself can fail after several expensive
                # Claude retries.  In unattended mode this is still a
                # pre-execution failure: persist it as an unchanged-state
                # experience and move to the next iteration instead of
                # restarting the whole run and losing the current evidence.
                try:
                    proposal = self._plan_fold_with_retries(
                        before_images,
                        objective,
                        history,
                        iteration=iteration,
                        molmo_hint=molmo_hint,
                    )
                except Exception as exc:
                    planning_attempts.append(
                        {
                            "attempt": "visual_planning",
                            "status": "REJECTED_BEFORE_EXECUTION",
                            "generation_mode": "VISUAL_PLANNING",
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_pose": _extract_failure_pose(exc),
                            "client_timing": getattr(self.client, "last_plan_timing", {}),
                        }
                    )
                    self._debug_exception(
                        "planning",
                        exc,
                        iteration=iteration,
                        attempt_kind="visual_planning",
                        nonphysical=True,
                    )
                    if not self.unattended:
                        raise
                    self._persist_preexecution_planning_failure(
                        output=output,
                        iteration_dir=iteration_dir,
                        iteration=iteration,
                        current_step=current_step,
                        supervisor_before=supervisor_before,
                        screen_before=screen_before,
                        before_images=before_images,
                        observer_before_images=observer_before_images,
                        acquisition_learning=acquisition_learning,
                        molmo_hint=molmo_hint,
                        planning_attempts=planning_attempts,
                        exc=exc,
                        summary=summary,
                        history=history,
                        source_path=source_path,
                        evidence_package=evidence_package,
                        action_mode=action_mode,
                        garment_condition=garment_condition,
                    )
                    self.reuse_latest_perception = False
                    continue
                if proposal is None:
                    exc = RuntimeError("Claude visual planner returned no proposal")
                    planning_attempts.append(
                        {
                            "attempt": "visual_planning",
                            "status": "REJECTED_BEFORE_EXECUTION",
                            "generation_mode": "VISUAL_PLANNING",
                            "error": str(exc),
                        }
                    )
                    if not self.unattended:
                        raise exc
                    self._persist_preexecution_planning_failure(
                        output=output,
                        iteration_dir=iteration_dir,
                        iteration=iteration,
                        current_step=current_step,
                        supervisor_before=supervisor_before,
                        screen_before=screen_before,
                        before_images=before_images,
                        observer_before_images=observer_before_images,
                        acquisition_learning=acquisition_learning,
                        molmo_hint=molmo_hint,
                        planning_attempts=planning_attempts,
                        exc=exc,
                        summary=summary,
                        history=history,
                        source_path=source_path,
                        evidence_package=evidence_package,
                        action_mode=action_mode,
                        garment_condition=garment_condition,
                    )
                    self.reuse_latest_perception = False
                    continue
                model_proposal: ExplorationProposal = proposal
                execution_proposal: ExplorationProposal = proposal
                grasp_height_resolution: dict[str, Any] | None = None
                # Deterministic failures (bad schema, grounding, workspace,
                # preflight, IK) are fed back to Claude immediately.  No
                # physical command is sent until one attempt passes all gates.
                plan_feedback: str | None = None
                planning_failure: Exception | None = None
                for plan_attempt in range(1, self.max_replans + 2):
                    attempt_started = time.monotonic()
                    generation_mode = "INITIAL_PLAN"
                    self._debug(
                        "planning",
                        "validating Claude proposal",
                        iteration=iteration,
                        attempt=plan_attempt,
                        mode=mode,
                    )
                    try:
                        if plan_attempt > 1:
                            if plan_attempt % 2 == 0:
                                generation_mode = "GROUNDING_REPAIR"
                                self._debug(
                                    "planning",
                                    "repairing grounded trajectory without repeating visual planning",
                                    iteration=iteration,
                                    attempt=plan_attempt,
                                )
                                proposal = self.client.repair_last_grounding_plan(
                                    self.session,
                                    objective,
                                    feedback=plan_feedback or "unknown host rejection",
                                    history=_compact_history(history),
                                )
                            else:
                                generation_mode = "FULL_REPLAN"
                                self._debug(
                                    "planning",
                                    "requesting fresh visual plan after compiler repair failed",
                                    iteration=iteration,
                                    attempt=plan_attempt,
                                )
                                proposal = self._plan_fold_with_retries(
                                    before_images,
                                    objective,
                                    history,
                                    feedback=plan_feedback,
                                    iteration=iteration,
                                    attempt_kind=f"fold-replan-{plan_attempt}",
                                    molmo_hint=molmo_hint,
                                )
                        # Keep the model proposal separate from the program that
                        # will be sent to the robot.  This makes any host-side
                        # transformation explicit and auditable.
                        model_proposal = proposal
                        execution_proposal, grasp_height_resolution = (
                            self._resolve_fold_grasp_height(proposal)
                        )
                        contract_learning = (
                            acquisition_learning
                            if action_mode == "ACQUISITION_PROBE"
                            else {
                                **acquisition_learning,
                                "use_lift_only_probe": False,
                                "require_non_height_change": False,
                            }
                        )
                        acquisition_strategy_validation = (
                            _validate_acquisition_strategy_change(
                                execution_proposal,
                                contract_learning,
                            )
                        )
                        acquisition_probe_plan = None
                        host_compilation = {
                            "authority": "Claude+host_grasp_height",
                            "rewritten": bool(
                                grasp_height_resolution
                                and grasp_height_resolution.get("z_rewritten")
                            ),
                            "reason": (
                                "Claude supplied the target and trajectory strategy; the host "
                                "resolved the engaged grasp Z from the selected calibrated "
                                "surface and support-layer policy."
                            ),
                            "model_action_count": len(proposal.actions),
                            "execution_action_count": len(execution_proposal.actions),
                            "grasp_height_resolution": grasp_height_resolution,
                        }
                        if action_mode == "ACQUISITION_PROBE":
                            if self.host_compile_acquisition_probe:
                                # Legacy compatibility switch.  This path is
                                # deliberately opt-in because it changes the
                                # experiment Claude requested.
                                lift_plan = split_global_lift_checkpoint_plan(
                                    execution_proposal,
                                    checkpoint_count=2,
                                )
                                acquisition_probe_plan = lift_plan.as_dict()
                                execution_proposal = ExplorationProposal(
                                    garment_observation=proposal.garment_observation,
                                    reveal_strategy=proposal.reveal_strategy,
                                    confidence=proposal.confidence,
                                    actions=tuple(lift_plan.actions),
                                    expected_observation=(
                                        "Acquisition-only experiment compiled by the host "
                                        "from Claude's requested lift checkpoints; no fold "
                                        "transport is executed."
                                    ),
                                    safety_notes=proposal.safety_notes,
                                    skill_invocations=proposal.skill_invocations,
                                    selected_grasp=proposal.selected_grasp,
                                    requires_lift_checkpoint=False,
                                )
                                host_compilation = {
                                    "authority": "host_safety_compiler",
                                    "rewritten": True,
                                    "reason": (
                                        "Legacy host compilation converted Claude's full plan "
                                        "to a reversible lift-only probe. Enable only when this "
                                        "change is intentional."
                                    ),
                                    "model_action_count": len(proposal.actions),
                                    "execution_action_count": len(execution_proposal.actions),
                                    "grasp_height_resolution": grasp_height_resolution,
                                }
                            else:
                                acquisition_probe_plan = _validate_model_acquisition_probe(
                                    execution_proposal
                                )
                                host_compilation = {
                                    "authority": "Claude",
                                    "rewritten": False,
                                    "reason": (
                                        "Claude returned the acquisition probe directly; the "
                                        "host only performed contract, preflight, and IK checks."
                                    ),
                                    "model_action_count": len(proposal.actions),
                                    "execution_action_count": len(proposal.actions),
                                    "probe_validation": acquisition_probe_plan,
                                    "grasp_height_resolution": grasp_height_resolution,
                                }
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
                        mode = action_mode
                        mode_contract = _validate_action_mode_contract(
                            execution_proposal,
                            mode=mode,
                        )
                        acquisition_strategy_validation = {
                            **(acquisition_strategy_validation or {}),
                            "mode_contract": mode_contract,
                        }
                        grounding_payload = getattr(
                            self.client, "last_reference_validation", None
                        )
                        evidence_package = _write_fold_evidence_package(
                            iteration_dir,
                            stage="metric_grounding",
                            payload={
                                "schema_version": 1,
                                "iteration": iteration,
                                "step": current_step,
                                "action_mode": mode,
                                "selected_reference": grounding_payload,
                                "selected_grasp": (
                                    dict(proposal.selected_grasp)
                                    if isinstance(proposal.selected_grasp, Mapping)
                                    else None
                                ),
                                "grasp_height_resolution": grasp_height_resolution,
                                "checks": {
                                    "reference_validation": "PASS"
                                    if isinstance(grounding_payload, Mapping)
                                    else "UNKNOWN",
                                    "controller_ik": "PASS",
                                },
                            },
                        )
                        evidence_package = _write_fold_evidence_package(
                            iteration_dir,
                            stage="execution_gate",
                            payload={
                                "schema_version": 1,
                                "iteration": iteration,
                                "step": current_step,
                                "mode": mode,
                                "decision": "ALLOW",
                                "checks": {
                                    "semantic_target": "CLAUDE_SELECTED",
                                    "metric_grounding": "PASS",
                                    "controller_ik": "PASS",
                                    "action_mode_contract": "PASS",
                                    "low_z_preclose_sweep": "PASS",
                                },
                                "grasp_height_resolution": grasp_height_resolution,
                                "mode_contract": mode_contract,
                                "probe_budget_remaining": mode_policy.get(
                                    "probe_budget_remaining"
                                ),
                            },
                        )
                        evidence_gate = _validate_fold_evidence_package(
                            iteration_dir,
                            required_stages=(
                                "rgb_selection",
                                "metric_grounding",
                                "execution_gate",
                            ),
                        )
                        evidence_package["execution_gate_validation"] = evidence_gate
                        self._debug(
                            "planning",
                            "compiled Claude proposal for execution",
                            iteration=iteration,
                            attempt=plan_attempt,
                            mode=mode,
                            authority=host_compilation.get("authority"),
                            host_rewrite=host_compilation.get("rewritten"),
                            model_actions=len(model_proposal.actions),
                            execution_actions=len(execution_proposal.actions),
                            evidence_gate=evidence_gate.get("status"),
                        )
                        planning_attempts.append(
                            {
                                "attempt": plan_attempt,
                                "status": "ACCEPTED",
                                "mode": mode,
                                "generation_mode": generation_mode,
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
                            "generation_mode": generation_mode,
                            "error": f"{type(exc).__name__}: {exc}",
                            "failed_pose": _extract_failure_pose(exc),
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
                            # The proposal was rejected before any robot
                            # command was sent.  Unattended mode records this
                            # unchanged-state iteration below and continues;
                            # interactive mode keeps the historical fail-fast
                            # behavior.
                            planning_failure = exc
                            break
                        plan_feedback = (
                            f"{type(exc).__name__}: {exc}\n"
                            "No robot command was sent. Correct this deterministic failure "
                            "and return a materially different, controller-valid fold proposal."
                        )
                if planning_failure is not None:
                    if not self.unattended:
                        raise planning_failure
                    self._persist_preexecution_planning_failure(
                        output=output,
                        iteration_dir=iteration_dir,
                        iteration=iteration,
                        current_step=current_step,
                        supervisor_before=supervisor_before,
                        screen_before=screen_before,
                        before_images=before_images,
                        observer_before_images=observer_before_images,
                        acquisition_learning=acquisition_learning,
                        molmo_hint=molmo_hint,
                        planning_attempts=planning_attempts,
                        exc=planning_failure,
                        summary=summary,
                        history=history,
                        source_path=source_path,
                        model_proposal=model_proposal,
                        execution_proposal=execution_proposal,
                        acquisition_strategy_validation=acquisition_strategy_validation,
                        acquisition_probe_plan=acquisition_probe_plan,
                        host_compilation=host_compilation,
                        evidence_package=evidence_package,
                        action_mode=action_mode,
                        garment_condition=garment_condition,
                    )
                    self.reuse_latest_perception = False
                    continue
                if preflight is None or controller is None:
                    planning_failure = RuntimeError(
                        "fold planning ended without a validated proposal"
                    )
                    if not self.unattended:
                        raise planning_failure
                    self._persist_preexecution_planning_failure(
                        output=output,
                        iteration_dir=iteration_dir,
                        iteration=iteration,
                        current_step=current_step,
                        supervisor_before=supervisor_before,
                        screen_before=screen_before,
                        before_images=before_images,
                        observer_before_images=observer_before_images,
                        acquisition_learning=acquisition_learning,
                        molmo_hint=molmo_hint,
                        planning_attempts=planning_attempts,
                        exc=planning_failure,
                        summary=summary,
                        history=history,
                        source_path=source_path,
                        model_proposal=model_proposal,
                        execution_proposal=execution_proposal,
                        acquisition_strategy_validation=acquisition_strategy_validation,
                        acquisition_probe_plan=acquisition_probe_plan,
                        host_compilation=host_compilation,
                        evidence_package=evidence_package,
                        action_mode=action_mode,
                        garment_condition=garment_condition,
                    )
                    self.reuse_latest_perception = False
                    continue
                trajectory = {
                    "mode": mode,
                    "actions": preflight.actions,
                    "source": str(source_path),
                    "authority": host_compilation.get("authority", "Claude"),
                    "host_compilation": host_compilation,
                    "created_at": _now(),
                }
                _write_json(iteration_dir / "claude_plan.json", model_proposal.as_dict())
                _write_json(iteration_dir / "execution_plan.json", execution_proposal.as_dict())
                _write_json(iteration_dir / "host_compilation.json", host_compilation)
                _write_json(iteration_dir / "trajectory.json", trajectory)
                planning_diagnostics: dict[str, Any] = {
                    "mode": mode,
                    "action_mode": action_mode,
                    "garment_condition": garment_condition,
                    "mode_policy": mode_policy,
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
                    "host_compilation": host_compilation,
                    "grasp_height_resolution": grasp_height_resolution,
                    "model_proposal": model_proposal.as_dict(),
                    "execution_proposal": execution_proposal.as_dict(),
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
                    claude_plan=str(iteration_dir / "claude_plan.json"),
                    execution_plan=str(iteration_dir / "execution_plan.json"),
                    host_compilation=str(iteration_dir / "host_compilation.json"),
                )
                execution, recording = self._execute(
                    source_path,
                    config,
                    iteration_dir,
                    label=f"iteration_{iteration:03d}_{mode.lower()}",
                    hold_action_index=_first_post_close_move_index(execution_proposal),
                )
                _write_json(iteration_dir / "execution.json", execution)
                _write_json(iteration_dir / "recording.json", recording)
                gripper_telemetry = _extract_gripper_telemetry(execution)
                _write_json(iteration_dir / "gripper_telemetry.json", gripper_telemetry)
                self._debug(
                    "gripper",
                    "captured SDK gripper telemetry",
                    iteration=iteration,
                    available=gripper_telemetry.get("available"),
                    samples=len(gripper_telemetry.get("samples", [])),
                    mechanical_grasp_detected=gripper_telemetry.get(
                        "mechanical_grasp_detected_after_close"
                    ),
                )
                after, after_path, after_images = self._capture_with_retries(
                    config,
                    iteration_dir / "after_raw",
                    reuse=False,
                    stage="after perception",
                )
                observer_after_images = _select_observer_images(after_images)
                hold_check_images: list[Path] = []
                observer_recording = recording.get("observer")
                if isinstance(observer_recording, Mapping):
                    hold_check = observer_recording.get("hold_check")
                    if isinstance(hold_check, Mapping):
                        raw_hold_image = hold_check.get("image")
                        if raw_hold_image:
                            hold_path = Path(str(raw_hold_image)).expanduser().resolve()
                            if hold_path.is_file():
                                hold_check_images.append(hold_path)
                self._debug(
                    "observer-camera",
                    "after observer evidence selected",
                    iteration=iteration,
                    images=[str(path) for path in observer_after_images],
                    hold_check_images=[str(path) for path in hold_check_images],
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
                if recording.get("directory"):
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
                    else (
                        "This iteration repaired a gathered or rolled sleeve. Judge whether the "
                        "sleeve became flatter and its distal/free edge became more visible. "
                        "Do not credit completion of the left-sleeve inward fold; the ordered "
                        "fold ledger remains on the same current step until a later inward-fold "
                        "attempt is executed."
                        if mode == "REPAIR_SLEEVE"
                        else "Fold the shirt into the five ordered steps and a neat compact stack."
                    )
                )
                evaluation, evaluation_result = self._evaluate_with_retries(
                    list(before_images),
                    list(after_images),
                    iteration=iteration,
                    proposal=execution_proposal,
                    objective=evaluation_objective,
                    run_dir=self.session.run_dir,
                    rollout_recording_dir=(Path(recording["directory"]) if recording.get("status") == "completed" else None),
                    skill_guidance=self.skill_store.prompt(),
                    acquisition_probe=(mode == "ACQUISITION_PROBE"),
                    compact_video_images=video_images,
                    compact_video_references=video_refs,
                    compact_video_errors=video_errors,
                    gripper_telemetry=gripper_telemetry,
                    observer_images=[
                        *observer_before_images,
                        *hold_check_images,
                        *observer_after_images,
                    ],
                )
                self._debug(
                    "evaluation",
                    "Claude evaluation returned",
                    iteration=iteration,
                    task_progress=(evaluation.as_dict().get("task_progress") if hasattr(evaluation, "as_dict") else evaluation.get("task_progress") if isinstance(evaluation, Mapping) else None),
                )
                evaluation_payload = _evaluation_payload(evaluation)
                supervisor_history = [
                    *history,
                    {
                        "iteration": iteration,
                        "mode": mode,
                        "planned_step": current_step,
                        "evaluation": evaluation_payload,
                    },
                ]
                if mode == "REPAIR_SLEEVE":
                    supervisor_after = self._local_acquisition_supervisor(
                        screen_after,
                        supervisor_history,
                        step=current_step,
                        base=supervisor_before,
                        reason=(
                            "The sleeve repair was executed without attempting the ordered inward "
                            "fold; keep the same fold step and reassess the repaired RGB state."
                        ),
                    )
                    self._debug(
                        "supervisor",
                        "kept current fold step after sleeve repair",
                        iteration=iteration,
                        current_step=current_step,
                    )
                elif mode == "ACQUISITION_PROBE" or _is_acquisition_failure(
                    evaluation_payload
                ):
                    supervisor_after = self._local_acquisition_supervisor(
                        screen_after,
                        supervisor_history,
                        step=current_step,
                        base=supervisor_before,
                        reason=(
                            "The acquisition probe reversed without fold transport, or "
                            "the evaluator observed an empty grasp; keep the same fold step."
                        ),
                    )
                    self._debug(
                        "supervisor",
                        "skipped redundant after-supervisor during acquisition learning",
                        iteration=iteration,
                        current_step=current_step,
                        acquisition_status=(
                            evaluation_payload.get("grasp_acquisition", {}) or {}
                        ).get("status"),
                    )
                else:
                    supervisor_after = self._supervisor(
                        after_images,
                        screen_after,
                        supervisor_history,
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
                post_history_row = {
                    "iteration": iteration,
                    "planned_step": current_step,
                    "mode": mode,
                    "proposal": execution_proposal.as_dict(),
                    "evaluation": evaluation_payload,
                    "supervisor_after": supervisor_after,
                }
                garment_condition_after = _garment_condition_from_history(
                    [*history, post_history_row],
                    step=current_step,
                )
                garment_condition_after = _condition_after_action(
                    garment_condition_after,
                    evaluation_payload,
                    mode=mode,
                )
                _write_json(
                    iteration_dir / "garment_condition_after.json",
                    garment_condition_after,
                )
                evidence_package = _write_fold_evidence_package(
                    iteration_dir,
                    stage="post_grasp",
                    payload={
                        "schema_version": 1,
                        "iteration": iteration,
                        "step": current_step,
                        "mode": mode,
                        "before_images": [str(path) for path in before_images],
                        "after_images": [str(path) for path in after_images],
                        "observer_images_before": [
                            str(path) for path in observer_before_images
                        ],
                        "observer_images_hold_check": [
                            str(path) for path in hold_check_images
                        ],
                        "observer_images_after": [
                            str(path) for path in observer_after_images
                        ],
                        "video_evidence": [str(path) for path in video_images],
                        "gripper_telemetry": gripper_telemetry,
                        "evaluation": evaluation_payload,
                        "screen_after": screen_after,
                        "garment_condition_after": garment_condition_after,
                    },
                )
                record: dict[str, Any] = {
                    "iteration": iteration,
                    "planned_step": current_step,
                    "status": mode,
                    "mode": mode,
                    "action_mode": action_mode,
                    "garment_condition_before": garment_condition,
                    "garment_condition_after": garment_condition_after,
                    # ``proposal`` is always Claude's original decision.  The
                    # executable version is stored separately so a later
                    # evaluator can distinguish model reasoning from host
                    # compilation.
                    "proposal": model_proposal.as_dict(),
                    "execution_proposal": execution_proposal.as_dict(),
                    "host_compilation": host_compilation,
                    "trajectory": trajectory,
                    "preflight": asdict(preflight),
                    "controller_ik": asdict(controller),
                    "planning_attempts": planning_attempts,
                    "execution": execution,
                    "gripper_telemetry": gripper_telemetry,
                    "recording": recording,
                    "before_images": [str(path) for path in before_images],
                    "after_images": [str(path) for path in after_images],
                    "observer_images_before": [
                        str(path) for path in observer_before_images
                    ],
                    "observer_images_after": [
                        str(path) for path in observer_after_images
                    ],
                    "observer_images_hold_check": [
                        str(path) for path in hold_check_images
                    ],
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
                    "evidence_package": evidence_package,
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
                evidence_package = _write_fold_evidence_package(
                    iteration_dir,
                    stage="experience_update",
                    payload={
                        "schema_version": 1,
                        "iteration": iteration,
                        "step": current_step,
                        "mode": mode,
                        "garment_condition_before": garment_condition,
                        "garment_condition_after": garment_condition_after,
                        "supervisor_before": _compact_supervisor_state(
                            supervisor_before
                        ),
                        "supervisor_after": _compact_supervisor_state(
                            supervisor_after
                        ),
                        "evaluation": evaluation_payload,
                        "planning_attempts": planning_attempts,
                        "next_step": supervisor_after.get("current_step"),
                    },
                )
                record["evidence_package"] = evidence_package
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
                summary["persistent_claude_session"] = (
                    self.persistent_claude.as_dict()
                )
                summary["iterations"].append(
                    {
                        "iteration": iteration,
                        "mode": mode,
                        "action_mode": action_mode,
                        "garment_condition_before": garment_condition.get("condition"),
                        "garment_condition_after": garment_condition_after.get("condition"),
                        "next_step": supervisor_after["current_step"],
                        "visibility": supervisor_after["garment_visibility"],
                        "evaluation_status": (record["evaluation"].get("task_progress", {}) or {}).get("status") if isinstance(record["evaluation"], Mapping) else None,
                        "evidence_package": dict(evidence_package),
                    }
                )
                _write_json(output / "summary.json", summary)
                self._debug(
                    "iteration",
                    "iteration completed",
                    iteration=iteration,
                    mode=mode,
                    next_step=supervisor_after.get("current_step"),
                    duration_s=round(time.monotonic() - iteration_started, 3),
                )
                if supervisor_after["status"] == "COMPLETE" or supervisor_after["current_step"] == "COMPLETE":
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
            # Viser is a diagnostic child process, not part of the robot
            # safety path.  Reap it on every pipeline exit (including
            # KeyboardInterrupt) so a finished run cannot leave an orphaned
            # viewer occupying port 8765 indefinitely.
            self._stop_viser_for_restart()
            try:
                synthesis = self.skill_ledger.finalize(self.skill_store)
                _write_json(output / "skill_synthesis.json", synthesis)
            except Exception as exc:
                _write_json(output / "skill_synthesis_error.json", {"error": f"{type(exc).__name__}: {exc}"})
                self._debug_exception("skills", exc)
            summary["persistent_claude_session"] = self.persistent_claude.as_dict()
            _write_json(output / "summary.json", summary)
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
    parser.add_argument(
        "--grounding-timeout-s",
        type=int,
        default=None,
        help=(
            "final Rxxx grounding timeout in seconds; defaults to "
            "--claude-timeout-s with no additional cap"
        ),
    )
    parser.add_argument("--supervisor-timeout-s", type=int, default=900)
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means continuous until supervisor COMPLETE or a hard failure")
    parser.add_argument("--max-replans", type=int, default=4)
    parser.add_argument("--max-stage-retries", type=int, default=1, help="extra retries for capture, supervisor, and evaluation failures")
    parser.add_argument("--retry-backoff-s", type=float, default=5.0)
    parser.add_argument(
        "--host-compile-acquisition-probe",
        action="store_true",
        help=(
            "legacy compatibility: let the host convert Claude's full plan into a "
            "lift-only acquisition probe; disabled by default so Claude remains the "
            "strategy authority"
        ),
    )
    parser.add_argument(
        "--unattended",
        "--continue-on-error",
        dest="unattended",
        action="store_true",
        help=(
            "after a pre-execution planning failure, persist the unchanged-state "
            "experience and continue to the next iteration; other safe "
            "non-physical failures may start a fresh attempt; never swallow "
            "Ctrl-C or real robot execution errors"
        ),
    )
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
    parser.add_argument(
        "--experience-dir",
        type=Path,
        help="seed the new run with experiences from another fold_experience directory",
    )
    parser.add_argument("--screen-margin-px", type=int, default=8)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--recording-no-native", action="store_true")
    parser.add_argument("--recording-codec", default="mp4v")
    parser.add_argument("--viser", action="store_true", help="start a read-only Viser viewer for every run artifact")
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8765)
    parser.add_argument("--viser-refresh-s", type=float, default=0.5)
    parser.add_argument(
        "--observer-camera-serial",
        default=None,
        help="optional separate RGB-only observer serial (disabled by default); never used for geometry",
    )
    parser.add_argument(
        "--no-observer-camera",
        action="store_true",
        help="disable the optional RGB-only observer camera",
    )
    parser.add_argument("--observer-camera-label", default="C")
    parser.add_argument("--observer-camera-width", type=int, default=1280)
    parser.add_argument("--observer-camera-height", type=int, default=720)
    parser.add_argument("--observer-camera-fps", type=int, default=15)
    parser.add_argument("--observer-camera-exposure", type=float, default=700.0)
    parser.add_argument("--observer-camera-white-balance", type=float, default=3800.0)
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
    if args.experience_dir:
        source = args.experience_dir.expanduser()
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        destination = session.workspace / "fold_experience"
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "experiences.jsonl",
            "experience_summary.json",
            "garment_condition.json",
        ):
            source_file = source / name
            if source_file.is_file():
                shutil.copy2(source_file, destination / name)
    summary = FoldExplorationPipeline(
        session,
        perception_config=perception.resolve(),
        claude_binary=args.claude_binary,
        claude_timeout_s=args.claude_timeout_s,
        grounding_timeout_s=args.grounding_timeout_s,
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
        unattended=args.unattended,
        host_compile_acquisition_probe=args.host_compile_acquisition_probe,
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
        observer_camera_serial=(
            None if args.no_observer_camera else args.observer_camera_serial
        ),
        observer_camera_label=args.observer_camera_label,
        observer_camera_width=args.observer_camera_width,
        observer_camera_height=args.observer_camera_height,
        observer_camera_fps=args.observer_camera_fps,
        observer_camera_exposure=args.observer_camera_exposure,
        observer_camera_white_balance=args.observer_camera_white_balance,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") in {"COMPLETE", "MAX_ITERATIONS_REACHED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
