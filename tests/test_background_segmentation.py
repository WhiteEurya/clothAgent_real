from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cloth_agent.config import ExperimentConfig, RobotConfig, WorkspaceBounds
from cloth_agent.perception import (
    CameraSpec, ClothCenterPerception, PerceptionConfig, PerceptionError, RGBDFrame,
    _estimate_camera_table_appearance, _camera_table_appearance_mask,
    _fused_source_appearance_mask,
)


@pytest.mark.parametrize("background,cloth", [(30, 245), (240, 25), (100, 235)])
def test_light_and_dark_cloth_use_same_camera_and_fusion_contract(background, cloth):
    rgb = np.full((100, 100, 3), background, dtype=np.uint8)
    rgb[25:75, 25:75] = cloth
    heights = np.zeros((100, 100))
    valid = np.ones((100, 100), dtype=bool)
    appearance = _estimate_camera_table_appearance(
        rgb, heights, valid, minimum_color_distance=24., mode="border_background")
    assert appearance["confident"]
    assert appearance["table_rgb_median"] == [background] * 3
    mask, _ = _camera_table_appearance_mask(
        rgb, heights, valid, minimum_color_distance=24., table_appearance=appearance)
    expected = np.zeros((100, 100), dtype=bool)
    expected[25:75, 25:75] = True
    np.testing.assert_array_equal(mask, expected)
    fused, *_ = _fused_source_appearance_mask(
        rgb.reshape(-1, 3), np.ones(10000, dtype=np.uint8), ("A",), {"A": appearance})
    np.testing.assert_array_equal(fused.reshape(100, 100), mask)


def test_background_requires_depth_support_and_rejects_ambiguous_colors():
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    for valid in (np.ones((100, 100), bool), np.zeros((100, 100), bool)):
        estimate = _estimate_camera_table_appearance(
            rgb, np.zeros((100, 100)), valid, minimum_color_distance=24., mode="border_background")
        assert not estimate["confident"]
        with pytest.raises(PerceptionError, match="unvalidated table estimate"):
            _fused_source_appearance_mask(np.array([[255, 255, 255]], dtype=np.uint8),
                                         np.array([1], dtype=np.uint8), ("A",), {"A": estimate})


@pytest.mark.parametrize("plane_mode", ["reference_fit", "camera_parallel"])
def test_complete_single_camera_perception_keeps_white_cloth_and_excludes_rails(tmp_path, plane_mode):
    root = Path(__file__).resolve().parents[1]
    robot = RobotConfig.load(root, root / "config/robot.example.json")
    robot = replace(robot, boundaries=WorkspaceBounds(x_min=0, x_max=800, y_min=-400,
                                                       y_max=400, z_min=-10, z_max=800))
    config = PerceptionConfig(
        cameras=(CameraSpec("A", "offline", tmp_path / "unused.yaml"),),
        width=200, height=200, active_camera_labels=("A",), table_appearance_mode="border_background",
        table_roi_xyxy=(0.15, 0.1, 0.85, 0.9),
        table_plane_mode=plane_mode,
    )
    rgb = np.full((200, 200, 3), 30, dtype=np.uint8)
    rgb[:, :15] = 255  # bright rail outside the work surface
    rgb[60:140, 60:140] = 245
    depth = np.full((200, 200), 0.6, dtype=np.float32)
    depth[60:140, 60:140] = 0.595
    depth[95:100, 95:100] = np.nan
    transform = np.diag([1., -1., -1., 1.])
    transform[:3, 3] = [0.4, 0., 0.6]
    frame = RGBDFrame("A", "offline", rgb, depth,
                      np.array([[200., 0., 100.], [0., 200., 100.], [0., 0., 1.]]), transform)
    output = tmp_path / "perception"
    result, _ = ClothCenterPerception(root, robot, config).locate(output, ExperimentConfig(), frames=[frame])
    mask = np.load(output / "camera_A_garment_mask.npy")
    assert mask[65:135, 65:135].mean() > .98
    assert not mask[:, :30].any()
    assert not mask[:40].any()
    assert abs(result["center_base_mm"][0]-400) < 10
    assert abs(result["center_base_mm"][1]) < 10
    assert (output / "camera_A_appearance_overlay.png").is_file()
    assert (output / "camera_A_background_diagnostics.json").is_file()
    reasons = np.load(output / "camera_A_mask_rejections.npz")
    assert reasons["missing_or_out_of_range_depth"][95:100, 95:100].all()
    assert not mask[95:100, 95:100].any()
    height_map = np.load(output / "camera_A_height_above_table_mm.npy")
    assert np.isnan(height_map[95:100, 95:100]).all()
    if plane_mode == "camera_parallel":
        assert np.nanmedian(height_map[65:135, 65:135]) == pytest.approx(5., abs=.01)
