from __future__ import annotations

import json
from pathlib import Path

from cloth_agent.evidence_ledger import (
    build_evidence_record,
    persist_evidence_record,
)


def test_evidence_is_compacted_and_indexed_after_operation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    iteration_dir = run_dir / "results" / "iteration_001"
    record = {
        "objective": "open garment",
        "before_images": ["before.png"],
        "after_images": ["after.png"],
        "proposal": {
            "selected_grasp": {"camera": "A", "pixel_xy": [10, 20]},
            "reveal_strategy": "lift the free edge",
            "skill_invocations": [{"name": "flatten-garment", "reason": "open"}],
        },
        "global_grounding": {
            "measurement": {
                "camera": "A",
                "query_pixel_xy": [10, 20],
                "base_xyz_median_mm": [500, -20, 25],
                "height_above_table_median_mm": 15,
                "base_z_p90_minus_p10_mm": 4,
                "valid": True,
            }
        },
        "execution": {
            "physical_execution": True,
            "execution_completed": True,
            "robot_errors": [],
            "actual_robot_actions": [{"name": "open_gripper"}],
        },
        "mandatory_return_home": {"completed": True},
        "evaluation": {
            "earliest_failure_stage": "NONE",
            "reason": "improved",
            "grasp_acquisition": {"status": "SUCCESS", "confidence": 0.9, "evidence": ["moved"]},
            "task_progress": {
                "status": "IMPROVED",
                "confidence": 0.9,
                "metrics": {"visible_area_delta": "INCREASED"},
                "evidence": ["coverage increased"],
            },
            "next_experiment": {"keep": ["edge"], "change": [], "reason": "continue"},
        },
        "skill_review": {"status": "APPROVED", "activated_skill": {"name": "flatten-garment"}},
    }

    evidence = build_evidence_record(record, iteration=1, run_dir=run_dir)
    paths = persist_evidence_record(run_dir, evidence, iteration_dir=iteration_dir)

    assert evidence["outcome"] == "supported"
    assert Path(paths["iteration_evidence_path"]).is_file()
    ledger = run_dir / "workspace" / "evidence_ledger.jsonl"
    assert ledger.is_file()
    assert json.loads(ledger.read_text(encoding="utf-8"))[
        "skills"
    ]["names"] == ["flatten-garment"]
    index = json.loads(
        (run_dir / "workspace" / "skill_evidence_index.json").read_text(encoding="utf-8")
    )
    assert index["skills"]["flatten-garment"]["supporting_iterations"] == [1]
