"""Compact per-operation evidence aggregation for unattended garment runs.

The raw iteration records remain available for audit.  This module writes a
small ledger entry after each completed physical operation so the next Claude
iteration can read a stable summary without receiving the entire run history in
its prompt.  Skill evidence is indexed here, but skill activation remains the
responsibility of :mod:`cloth_agent.skill_lifecycle`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clip(value: Any, limit: int = 1800) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 32)] + " …[truncated]"


def _stage_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for name in (
        "target_selection",
        "grasp_acquisition",
        "target_structure_acquired",
        "transport",
        "laydown",
        "task_progress",
    ):
        value = evaluation.get(name)
        if not isinstance(value, Mapping):
            continue
        stages[name] = {
            "status": value.get("status"),
            "confidence": value.get("confidence"),
            "metrics": value.get("metrics") if name == "task_progress" else None,
            "evidence": [_clip(item, 900) for item in value.get("evidence", [])[:4]]
            if isinstance(value.get("evidence"), list)
            else [],
        }
    return stages


def _skill_names(record: Mapping[str, Any], evaluation: Mapping[str, Any]) -> list[str]:
    names: set[str] = set()
    proposal = record.get("proposal")
    if isinstance(proposal, Mapping):
        invocations = proposal.get("skill_invocations", [])
        if isinstance(invocations, list):
            for item in invocations:
                if isinstance(item, Mapping) and item.get("name"):
                    names.add(str(item["name"]))
    update = evaluation.get("skill_update")
    if isinstance(update, Mapping) and update.get("name"):
        names.add(str(update["name"]))
    review = record.get("skill_review")
    if isinstance(review, Mapping):
        activated = review.get("activated_skill")
        if isinstance(activated, Mapping) and activated.get("name"):
            names.add(str(activated["name"]))
    return sorted(names)


def build_evidence_record(
    record: Mapping[str, Any],
    *,
    iteration: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Build a compact, JSON-serializable evidence record from one operation."""

    evaluation = record.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    proposal = record.get("proposal")
    proposal = proposal if isinstance(proposal, Mapping) else {}
    grounding = record.get("global_grounding") or record.get("grounding")
    grounding = grounding if isinstance(grounding, Mapping) else {}
    measurement = grounding.get("measurement")
    measurement = measurement if isinstance(measurement, Mapping) else {}
    execution = record.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    return_home = record.get("mandatory_return_home")
    return_home = return_home if isinstance(return_home, Mapping) else {}
    task_progress = evaluation.get("task_progress")
    task_status = task_progress.get("status") if isinstance(task_progress, Mapping) else None
    acquisition = evaluation.get("grasp_acquisition")
    acquisition_status = acquisition.get("status") if isinstance(acquisition, Mapping) else None
    if task_status == "IMPROVED" or acquisition_status == "SUCCESS":
        outcome = "supported"
    elif task_status in {"CONTRADICTED", "DEGRADED"} or acquisition_status in {"FAILURE", "CONTRADICTED"}:
        outcome = "contradicted"
    else:
        outcome = "uncertain"

    selected = proposal.get("selected_grasp")
    selected = dict(selected) if isinstance(selected, Mapping) else None
    return {
        "schema_version": 1,
        "created_at": _now(),
        "iteration": int(iteration),
        "outcome": outcome,
        "objective": _clip(record.get("objective"), 1000),
        "hypothesis": _clip(
            proposal.get("reveal_strategy") or proposal.get("garment_observation"),
            2200,
        ),
        "selected_grasp": selected,
        "measurement": {
            key: measurement.get(key)
            for key in (
                "camera",
                "query_pixel_xy",
                "base_xyz_median_mm",
                "height_above_table_median_mm",
                "base_z_p90_minus_p10_mm",
                "valid",
            )
            if key in measurement
        },
        "execution": {
            "physical_execution": execution.get("physical_execution"),
            "execution_completed": execution.get("execution_completed"),
            "robot_errors": execution.get("robot_errors", []),
            "release_completed": bool(
                any(
                    item.get("name") == "open_gripper"
                    for item in execution.get("actual_robot_actions", [])
                    if isinstance(item, Mapping)
                )
            ),
            "return_home_completed": return_home.get("completed"),
        },
        "evaluation": {
            "earliest_failure_stage": evaluation.get("earliest_failure_stage"),
            "reason": _clip(evaluation.get("reason"), 1800),
            "stages": _stage_summary(evaluation),
            "next_experiment": evaluation.get("next_experiment"),
        },
        "artifacts": {
            "before_images": list(record.get("before_images", []))[-16:],
            "after_images": list(record.get("after_images", []))[-16:],
        },
        "skills": {
            "names": _skill_names(record, evaluation),
            "review": record.get("skill_review"),
        },
        "source_iteration_record": str(
            (run_dir / "results").resolve()
        ),
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def persist_evidence_record(
    run_dir: Path,
    evidence: Mapping[str, Any],
    *,
    iteration_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist one ledger entry and update the per-skill evidence index."""

    run_dir = Path(run_dir).resolve()
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    ledger_path = workspace / "evidence_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(evidence), ensure_ascii=False) + "\n")

    if iteration_dir is not None:
        _atomic_write(Path(iteration_dir) / "evidence.json", dict(evidence))

    index_path = workspace / "skill_evidence_index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"schema_version": 1, "updated_at": None, "skills": {}}
    skills = index.setdefault("skills", {})
    outcome = str(evidence.get("outcome", "uncertain"))
    for name in evidence.get("skills", {}).get("names", []):
        entry = skills.setdefault(
            str(name),
            {
                "supporting_iterations": [],
                "contradicted_iterations": [],
                "uncertain_iterations": [],
                "evidence_count": 0,
                "review_statuses": [],
            },
        )
        key = {
            "supported": "supporting_iterations",
            "contradicted": "contradicted_iterations",
        }.get(outcome, "uncertain_iterations")
        iteration = int(evidence["iteration"])
        if iteration not in entry[key]:
            entry[key].append(iteration)
        entry["evidence_count"] = int(entry.get("evidence_count", 0)) + 1
        review = evidence.get("skills", {}).get("review")
        if isinstance(review, Mapping) and review.get("status"):
            status = str(review["status"])
            if status not in entry["review_statuses"]:
                entry["review_statuses"].append(status)
        entry["last_iteration"] = iteration
    index["updated_at"] = _now()
    _atomic_write(index_path, index)
    return {
        "ledger_path": str(ledger_path),
        "skill_index_path": str(index_path),
        "iteration_evidence_path": (
            str(Path(iteration_dir) / "evidence.json")
            if iteration_dir is not None
            else None
        ),
    }
