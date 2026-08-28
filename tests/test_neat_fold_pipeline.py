from __future__ import annotations

import pytest

from cloth_agent.neat_fold_pipeline import (
    FOLD_EVALUATION_SCHEMA,
    NEAT_FOLD_INSTRUCTION,
    NeatFoldError,
    _validate_fold_evaluation,
    _validate_grounded_workspace_xy,
    _claude_runtime_metrics,
)
from cloth_agent.config import RobotConfig, WorkspaceBounds
from cloth_agent.free_exploration import ExplorationPlanningError


def _evaluation(**overrides):
    payload = {
        "status": "CONTINUE",
        "confidence": 0.8,
        "stack_alignment": "MISALIGNED",
        "flatness": "FLAT",
        "protruding_parts": "PRESENT",
        "evidence": ["The body is flatter, but one sleeve still extends beyond the stack."],
        "reason": "The fold improved compactness but is not complete.",
        "next_fold_target": "Fold the visible sleeve inward over the body.",
    }
    payload.update(overrides)
    return payload


def test_standalone_fold_contract_is_specific() -> None:
    assert "compact" in NEAT_FOLD_INSTRUCTION
    assert "major edges" in NEAT_FOLD_INSTRUCTION
    assert FOLD_EVALUATION_SCHEMA["additionalProperties"] is False


def test_fold_evaluation_accepts_continue() -> None:
    result = _validate_fold_evaluation(_evaluation())
    assert result["status"] == "CONTINUE"
    assert result["confidence"] == pytest.approx(0.8)


def test_fold_evaluation_rejects_unknown_fields() -> None:
    payload = _evaluation(unexpected="must be rejected")
    with pytest.raises(NeatFoldError, match="fields mismatch"):
        _validate_fold_evaluation(payload)


def test_fold_evaluation_complete_requires_explicit_visual_state_fields() -> None:
    result = _validate_fold_evaluation(
        _evaluation(
            status="COMPLETE",
            stack_alignment="ALIGNED",
            flatness="FLAT",
            protruding_parts="NONE",
            evidence=["Edges overlap, no sleeve protrudes, and the stack rests flat."],
            reason="All visible completion conditions are supported.",
            next_fold_target="none; stop",
        )
    )
    assert result["status"] == "COMPLETE"


def test_grounded_workspace_gate_rejects_base_y_outside_robot_bounds() -> None:
    robot = RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(x_min=360, x_max=800, y_min=-300, y_max=171.087, z_min=-5, z_max=500),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
    )
    with pytest.raises(ExplorationPlanningError, match="base_y=271.179"):
        _validate_grounded_workspace_xy(
            {"measurement": {"base_xyz_median_mm": [395.395, 271.179, 114.634]}},
            robot,
        )


def test_claude_runtime_metrics_extracts_turns_model_and_duration() -> None:
    import json

    metrics = _claude_runtime_metrics(
        json.dumps(
            {
                "duration_ms": 745464,
                "duration_api_ms": 747851,
                "num_turns": 16,
                "fast_mode_state": "off",
                "modelUsage": {
                    "claude-opus-5": {
                        "inputTokens": 20736,
                        "outputTokens": 17505,
                        "costUSD": 0.72,
                        "canonicalModel": "claude-opus-5",
                    }
                },
                "usage": {"input_tokens": 20736, "output_tokens": 17505},
            }
        )
    )
    assert metrics["duration_ms"] == 745464
    assert metrics["num_turns"] == 16
    assert "claude-opus-5" in metrics["models"]
