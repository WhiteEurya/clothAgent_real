"""Evidence-based recovery and learning for interrupted fold iterations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .skill_lifecycle import SkillProposal
from .rollout_recorder import (append_mp4_to_cumulative, build_rollout_phase_timeline,
                              label_iteration_mp4, prune_rollout_video_files, speed_up_mp4)


class RecoveryExhausted(RuntimeError):
    """Repeated failures provided no usable new evidence; retain the run for inspection."""


def released_and_homed(execution: Mapping[str, Any]) -> bool:
    if (execution.get("execution_completed") is not True
            or execution.get("operator_interrupted")
            or execution.get("gripper_completion_failed")
            or execution.get("robot_errors")):
        return False
    actions = execution.get("actual_robot_actions") or []
    if not actions or actions[-1].get("name") != "home":
        return False
    gripper_actions = [action for action in actions
                       if action.get("name") in {"open_gripper", "close_gripper"}]
    if not gripper_actions or gripper_actions[-1].get("name") != "open_gripper":
        return False
    if any(action.get("error") for action in actions):
        return False
    home = execution.get("mandatory_return_home")
    return not isinstance(home, Mapping) or home.get("completed") is True


def checkpoint_evaluation(decision: Mapping[str, Any]) -> dict[str, Any]:
    empty = (decision.get("status") == "ASSESSED"
             and decision.get("classification") == "EMPTY"
             and float(decision.get("confidence", 0)) >= 0.8)
    reason = str(decision.get("reason") or decision.get("error") or "Grasp evidence is inconclusive")
    evidence = list(decision.get("evidence") or [reason])
    unknown = {"status": "UNKNOWN", "confidence": 0.0, "evidence": evidence}
    return {
        "target_selection": dict(unknown),
        "grasp_acquisition": {"status": "FAILURE" if empty else "UNKNOWN",
                              "confidence": float(decision.get("confidence", 0)) if empty else 0.0,
                              "evidence": evidence},
        "target_structure_acquired": dict(unknown),
        "transport": dict(unknown),
        "laydown": {**unknown, "status": "NOT_REACHED"},
        "task_progress": {"status": "NEUTRAL", "confidence": 0.0,
                          "metrics": {name: "UNKNOWN" for name in
                                      ("visible_area_delta", "overlap_delta", "relief_delta", "boundary_change")}},
        "earliest_failure_stage": "ACQUISITION" if empty else "UNKNOWN",
        "next_experiment": {
            "keep": ["same ordered fold step", "workspace, IK and measured gripper completion checks"],
            "change": (["reassess contact location, jaw alignment and entry; do not assume height is the cause"]
                       if empty else ["restore usable grasp evidence before repeating the contact hypothesis"]),
            "reason": reason,
        },
    }


def failure_detection(record: Mapping[str, Any]) -> dict[str, Any]:
    execution = record.get("execution") or {}
    checkpoint = execution.get("checkpoint") or {}
    evaluation = record.get("evaluation") or {}
    acquisition = evaluation.get("grasp_acquisition") or {}
    if record.get("status") == "INTERRUPTED" or execution.get("operator_interrupted"):
        category = "OPERATOR_INTERRUPTED"
    elif execution and not released_and_homed(execution):
        category = "EXECUTION_UNCONFIRMED"
    elif checkpoint.get("executed_branch") == "ABORT_RELEASE":
        category = "EMPTY_GRASP" if acquisition.get("status") == "FAILURE" else "GRASP_UNOBSERVABLE"
        if checkpoint.get("status") == "FAILED_CLOSED":
            category = "GRASP_INSPECTION_ERROR"
    elif acquisition.get("status") == "FAILURE":
        category = "EMPTY_GRASP"
    elif record.get("error") or record.get("planning_failure"):
        category = "POST_EXECUTION_ERROR" if execution else "PREEXECUTION_ERROR"
    elif execution and acquisition.get("status") == "UNKNOWN" and checkpoint.get("classification") != "GRASP_CONFIRMED":
        category = "GRASP_UNOBSERVABLE"
    else:
        category = "NONE"
    return {"schema_version": 1, "category": category,
            "physical_execution": execution.get("physical_execution", False),
            "safe_return_confirmed": released_and_homed(execution),
            "classification": checkpoint.get("classification"),
            "reason": record.get("error") or (record.get("planning_failure") or {}).get("error") or checkpoint.get("reason") or checkpoint.get("error"),
            "earliest_failure_stage": evaluation.get("earliest_failure_stage", "UNKNOWN"),
            "next_experiment": evaluation.get("next_experiment"),
            "evidence": acquisition.get("evidence", [])}


def failure_skill(record: Mapping[str, Any]) -> SkillProposal | None:
    failure = failure_detection(record)
    if failure["category"] != "EMPTY_GRASP":
        return None
    acquisition = record["evaluation"]["grasp_acquisition"]
    if float(acquisition.get("confidence", 0)) < 0.8:
        return None
    return SkillProposal(
        operation="create", name="fold-empty-grasp-detection",
        purpose="Distinguish visibly empty acquisition from unobservable grasp evidence during folding.",
        guidance=("Treat clear empty jaws after closure and a short lift as failed acquisition, not failed transport. "
                  "Do not advance the fold step. Black, occluded or missing images mean UNKNOWN, not EMPTY. "
                  "Compare contact location, jaw alignment, entry path and height as hypotheses; do not infer "
                  "which parameter caused failure from an empty grasp alone. Preserve workspace and IK checks, "
                  "measured gripper completion, and a validated release and Home before another attempt."),
        rationale="The online checkpoint observed empty acquisition before fold transport; retain the failure detector, not an unproven fix.",
        evidence=tuple(str(item) for item in acquisition["evidence"][:4]),
        confidence=float(acquisition["confidence"]),
    )


def inherit_fold_lessons(project_root: Path, run_dir: Path) -> dict[str, Any]:
    destination = run_dir / "workspace" / "fold_experience" / "experiences.jsonl"
    if destination.is_file():
        return {"status": "EXISTING_RUN", "count": 0}
    candidates = [path for path in (project_root / "runs").glob("*/workspace/fold_experience/experiences.jsonl")
                  if run_dir.resolve() not in path.resolve().parents]
    if not candidates:
        return {"status": "NO_PREVIOUS_EXPERIENCE", "count": 0}
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    lessons = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        lesson = copy.deepcopy(row)
        lesson["inherited_lesson"] = True
        lesson["source_experience"] = str(source)
        for key in ("supervisor_before", "supervisor_after", "garment_condition_before", "garment_condition_after"):
            lesson.pop(key, None)
        lessons.append(lesson)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lessons[-64:]), encoding="utf-8")
    return {"status": "LESSONS_ONLY", "source": str(source), "count": min(64, len(lessons)),
            "completion_inherited": False}


def archive_iteration_video(iteration_dir: Path, record: dict[str, Any], *, prune: bool = False) -> dict[str, Any]:
    """Retain a labelled cumulative video before optionally pruning evaluated segments."""
    directory = (record.get("recording") or {}).get("directory")
    if not directory:
        return {"status": "NO_RECORDING"}
    root = Path(directory).resolve()
    if iteration_dir.resolve() not in root.parents:
        raise ValueError("rollout directory must belong to this iteration")
    receipt = iteration_dir / "video_archive.json"
    if receipt.is_file():
        return json.loads(receipt.read_text(encoding="utf-8"))
    sources = [path for path in (root / 'composite_AB_depth.mp4', root / 'camera_A_rgb.mp4')
               if path.is_file()]
    if not sources:
        return {"status": "NO_VIDEO"}
    labelled = root / ".fold_archive_labelled.mp4"
    accelerated = root / ".fold_archive_32x.mp4"
    cumulative = iteration_dir.parent / "combined_rollout.mp4"
    timeline = build_rollout_phase_timeline(record.get("execution"), (record.get("recording") or {}).get("manifest"))
    try:
        source_errors = []
        for source in sources:
            try:
                label_iteration_mp4(source, labelled, iteration=record["iteration"], phase_timeline=timeline)
                break
            except Exception as exc:
                source_errors.append({'source': str(source), 'error': f'{type(exc).__name__}: {exc}'})
                labelled.unlink(missing_ok=True)
        else:
            raise RuntimeError(f'No usable archive video: {source_errors}')
        speed_up_mp4(labelled, accelerated, speed=32.0)
        info = append_mp4_to_cumulative(accelerated, cumulative)
        result = {"status": "ARCHIVED", "video": str(cumulative), "speed": 32.0, "append": info,
                  "source": str(source), "source_errors": source_errors}
        # Commit receipt before cleanup; an interrupted retry must not append twice.
        receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if (prune and record.get("status") in {"FOLD", "ACQUISITION_PROBE", "REPAIR_SLEEVE"}
                and (record.get("failure_detection") or {}).get("category", "NONE") == "NONE"):
            result["pruned_files"] = prune_rollout_video_files(root)
            receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        labelled.unlink(missing_ok=True)
        accelerated.unlink(missing_ok=True)
