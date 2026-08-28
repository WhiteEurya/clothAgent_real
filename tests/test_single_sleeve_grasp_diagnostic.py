from __future__ import annotations

import numpy as np

from scripts.test_claude_single_sleeve_grasp import (
    LIFT_DELTAS_MM,
    _abort_actions,
    _experiment_source,
    _mask_diagnostic,
    _raw_to_upright,
    _task_instruction,
    _upright_to_raw,
)


def test_clockwise_rotation_pixel_mapping_round_trips() -> None:
    raw = [803, 412]
    upright = _raw_to_upright(
        raw,
        raw_width=1280,
        raw_height=720,
        rotation="clockwise90",
    )
    assert _upright_to_raw(
        upright,
        raw_width=1280,
        raw_height=720,
        rotation="clockwise90",
    ) == raw


def test_sleeve_prompt_excludes_shoulder_body_and_table() -> None:
    prompt = _task_instruction("either")
    assert "袖口" in prompt
    assert "不要选择袖根、肩膀、衣身内部" in prompt
    assert "桌面" in prompt


def test_mask_diagnostic_separates_exact_pixel_and_local_support() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 10:20] = True
    inside = _mask_diagnostic(mask, 15, 10, radius_px=2)
    edge_outside = _mask_diagnostic(mask, 9, 10, radius_px=2)
    assert inside["exact_pixel_is_garment"] is True
    assert inside["local_mask_fraction"] == 1.0
    assert edge_outside["exact_pixel_is_garment"] is False
    assert edge_outside["local_mask_fraction"] > 0.0


def test_experiment_contains_only_vertical_lifts_and_abort_releases() -> None:
    grasp_z = 5.0
    lifts = [grasp_z + delta for delta in LIFT_DELTAS_MM]
    source = _experiment_source(
        x_mm=500.0,
        y_mm=-100.0,
        grasp_z_mm=grasp_z,
        approach_z_mm=75.0,
        lift_zs_mm=lifts,
        yaw_deg=0.0,
    )
    assert "close_gripper()" in source
    assert source.count("move(500.0, -100.0") == 5
    assert "open_gripper()" not in source.split("close_gripper()", 1)[1]
    abort = _abort_actions(
        x_mm=500.0,
        y_mm=-100.0,
        grasp_z_mm=grasp_z,
        approach_z_mm=75.0,
        yaw_deg=0.0,
    )
    assert [action["name"] for action in abort] == ["move", "open_gripper", "move", "home"]
