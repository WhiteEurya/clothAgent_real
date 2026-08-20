from __future__ import annotations

import json
from pathlib import Path

from cloth_agent.molmo_artifact_viser import (
    _claude_markdown,
    _claude_output_markdown,
    _skill_markdown,
    discover_perception_result,
    _sample_cloud,
)


def test_discovers_saved_perception_result_from_latest_iteration(tmp_path: Path) -> None:
    output = tmp_path / "runs" / "run" / "results" / "molmo_keypoint_cli" / "stamp"
    iteration = output / "iteration_001"
    iteration.mkdir(parents=True)
    perception = tmp_path / "runs" / "run" / "results" / "perception" / "result.json"
    perception.parent.mkdir(parents=True)
    perception.write_text(json.dumps({"status": "VALIDATED_DENSE_AB_FUSION"}), encoding="utf-8")
    (iteration / "result.json").write_text(
        json.dumps({"saved_perception_result": str(perception)}),
        encoding="utf-8",
    )

    assert discover_perception_result(output) == perception.resolve()


def test_cloud_sampling_preserves_aligned_arrays() -> None:
    import numpy as np

    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    colors = np.arange(30, dtype=np.uint8).reshape(10, 3)
    sampled_points, sampled_colors = _sample_cloud(points, colors, max_points=4)

    assert len(sampled_points) <= 4
    assert sampled_points.shape[1:] == (3,)
    assert sampled_colors.shape == sampled_points.shape
    assert (sampled_points[:, 0] * 0 + sampled_colors[:, 0] >= 0).all()


def test_claude_panel_uses_structured_outputs_and_grounding(tmp_path: Path) -> None:
    claude, evaluation, runtime, overlay = _claude_markdown(
        {
            "iteration": 2,
            "status": "GLOBAL_PREEXECUTION_VALIDATED",
            "proposal": {
                "garment_observation": "raised left ridge",
                "reveal_strategy": "lift and lay down",
                "expected_observation": "footprint grows",
                "confidence": 0.7,
                "selected_grasp": {"camera": "A", "pixel_xy": [10, 20]},
                "skill_invocations": [{"name": "laydown", "reason": "controlled release"}],
            },
            "global_grounding": {
                "measurement": {
                    "base_xyz_median_mm": [1, 2, 3],
                    "height_above_table_median_mm": 4,
                },
                "xy_correction_mm": 0.5,
            },
        },
        tmp_path,
    )

    assert "raised left ridge" in claude
    assert "[10, 20]" in claude
    assert "laydown" in claude
    assert "GLOBAL_PREEXECUTION_VALIDATED" in runtime
    assert "before/after" not in evaluation.lower()
    assert overlay is None


def test_skill_and_claude_output_panels_include_audit_and_intermediates(tmp_path: Path) -> None:
    iteration = tmp_path / "iteration_001"
    iteration.mkdir()
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "reviews.jsonl").write_text(
        json.dumps(
            {
                "created_at": "2026-08-20T00:00:00Z",
                "proposal": {"operation": "create", "name": "ridge-release"},
                "review": {"status": "APPROVED", "reason": "passed review"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (skills / "approved.json").write_text(
        json.dumps({"skills": [{"name": "ridge-release", "version": 1}]}),
        encoding="utf-8",
    )
    record = {
        "objective": "open garment",
        "candidate_policy": "NONE",
        "proposal": {
            "skill_invocations": [{"name": "ridge-release", "reason": "ridge"}],
        },
        "evaluation": {
            "skill_update": {
                "operation": "create",
                "name": "ridge-release",
                "purpose": "release ridge",
                "guidance": "check workspace and release safely",
                "rationale": "repeated evidence",
                "evidence": ["iteration 1"],
                "confidence": 0.8,
            }
        },
        "claude_global_attempts": [
            {
                "returncode": 0,
                "command": ["claude", "--print"],
                "prompt": "inspect scene",
                "stdout": "{\"proposal\":{}}",
                "stderr": "",
                "proposal": {"selected_grasp": {"pixel_xy": [1, 2]}},
            }
        ],
        "global_grounding": {"measurement": {"valid": True}},
        "preflight": {"actions": [{"name": "move"}]},
        "stage_timestamps": {"planning": "now"},
    }
    skill_text = _skill_markdown(record, iteration, skills)
    output_text = _claude_output_markdown(record, iteration, tmp_path)

    assert "ridge-release" in skill_text
    assert "APPROVED" in skill_text
    assert "stdout" in output_text
    assert "inspect scene" in output_text
    assert "grounding" in output_text
