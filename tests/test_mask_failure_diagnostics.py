"""Distinguish downstream mask gates without cameras or robot connections."""

import json

import numpy as np
import pytest
from PIL import Image

from cloth_agent.perception import RGBDFrame, _occlusion_aware_garment_mask
from scripts.test_height_map_pipeline import _load_raw_capture, _save_raw_capture


@pytest.mark.parametrize("height_mm,gray,failed_gate", [
    (200.0, 20, "height"), (0.0, 240, "appearance"), (0.0, 20, None),
])
def test_depth_match_does_not_imply_appearance_failure(height_mm, gray, failed_gate):
    shape = (40, 40)
    frame = RGBDFrame(
        "A", "wrist", np.full((*shape, 3), gray, dtype=np.uint8),
        np.full(shape, 0.4, dtype=np.float32),
        np.array([[100., 0., 20.], [0., 100., 20.], [0., 0., 1.]]), np.eye(4),
    )
    points = np.array([[(x-20)*4., (y-20)*4., 400.]
                       for y in range(5, 35) for x in range(5, 35)])
    mask, _, diag = _occlusion_aware_garment_mask(
        points, frame, np.full(shape, height_mm), np.ones(shape, dtype=bool),
        minimum_table_color_distance=24.,
        table_appearance={"confident": True, "table_rgb_median": [240., 240., 240.],
                          "applied_color_distance": 24.},
    )
    assert diag["depth_consistent_pixels"] > 100
    if failed_gate == "height":
        assert diag["after_height_pixels"] == 0
    else:
        assert diag["after_height_pixels"] > 100
    assert bool(mask.any()) == (failed_gate is None)
    if failed_gate == "appearance":
        assert diag["pre_component_garment_mask_pixels"] == 0


def test_single_wrist_capture_replays_saved_transform(tmp_path):
    transform = np.eye(4)
    transform[0, 3] = 0.42
    original = RGBDFrame("A", "wrist", np.zeros((4, 4, 3), dtype=np.uint8),
                         np.ones((4, 4), dtype=np.float32), np.eye(3), transform)
    capture = tmp_path / "capture"
    _save_raw_capture([original], capture)
    frames = _load_raw_capture(capture)
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0].X_base_camera, transform)
    np.testing.assert_array_equal(frames[0].depth_m, original.depth_m)
