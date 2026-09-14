from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cloth_agent.perception import (
    CameraSpec, PerceptionConfig, PerceptionError, RGBDFrame,
    _fit_table_plane_from_references, camera_height_map_mm,
)


def scene(tmp_path, angle=180.):
    rgb = np.full((100, 100, 3), 35, dtype=np.uint8)
    rgb[30:70, 30:70] = 245
    rgb[:, :10] = 255
    depth = np.full((100, 100), .5)
    depth[30:70, 30:70] = .49
    depth[:, :10] = .8  # non-table rail is outside ROI
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("x", angle, degrees=True).as_matrix()
    transform[:3, 3] = [.4, 0, .6]
    frame = RGBDFrame("A", "offline", rgb, depth,
                      np.array([[120., 0., 50.], [0., 120., 50.], [0., 0., 1.]]), transform)
    config = PerceptionConfig(cameras=(CameraSpec("A", "offline", tmp_path / "unused"),),
                              table_plane_mode="camera_parallel", table_appearance_mode="border_background",
                              table_reference_clearance_px=3,
                              table_roi_xyxy=(.15, .1, .85, .9))
    return frame, config


@pytest.mark.parametrize("angle", [180., 168.])
def test_camera_plane_is_transformed_to_base_not_forced_horizontal(tmp_path, angle):
    frame, config = scene(tmp_path, angle)
    coefficients, stats = _fit_table_plane_from_references([frame], config, np.zeros(3))
    assert stats["mode"] == "camera_parallel"
    assert stats["camera_table_depth_m"] == pytest.approx(.5)
    heights, valid, inferred = camera_height_map_mm(frame, config)
    np.testing.assert_allclose(coefficients, inferred)
    assert valid.all()
    assert np.median(heights[15:25, 20:80]) == pytest.approx(0., abs=.001)
    # Compatibility maps remain vertical base-Z differences, not camera range.
    assert np.median(heights[35:65, 35:65]) == pytest.approx(
        10. / abs(frame.X_base_camera[2, 2]), abs=.001)
    if angle != 180.:
        assert abs(coefficients[1]) > .1
    for record in stats["cameras"]["A"]:
        x, y = record["pixel_xy"]
        assert 15 <= x < 85 and 10 <= y < 90
        assert frame.rgb[y, x, 0] == 35


def test_nonparallel_depth_or_missing_background_does_not_silently_fit(tmp_path):
    frame, config = scene(tmp_path)
    depth = frame.depth_m.copy()
    depth += np.linspace(-.05, .05, depth.shape[1])[None, :]
    with pytest.raises(PerceptionError, match="table depths disagree"):
        _fit_table_plane_from_references([replace(frame, depth_m=depth)], config, np.zeros(3))
    with pytest.raises(PerceptionError, match="background rejected"):
        _fit_table_plane_from_references(
            [replace(frame, depth_m=np.full(depth.shape, np.nan))], config, np.zeros(3))


def test_grid_patches_avoid_border_cloth_dark_print_and_clearance(tmp_path):
    from scipy.ndimage import binary_dilation

    frame, config = scene(tmp_path)
    rgb, depth = frame.rgb.copy(), frame.depth_m.copy()
    garment = np.zeros(depth.shape, dtype=bool)
    garment[30:95, 30:70] = True  # crosses the ROI's bottom edge
    rgb[garment] = 245
    depth[garment] = .49
    rgb[45:60, 45:60] = 35  # same-color print enclosed by garment
    forbidden = binary_dilation(garment, iterations=config.table_reference_clearance_px)
    _, stats = _fit_table_plane_from_references(
        [replace(frame, rgb=rgb, depth_m=depth)], config, np.zeros(3))
    records = stats["cameras"]["A"]
    cells = [tuple(record["grid_cell"]) for record in records]
    assert len(cells) == len(set(cells))
    assert len({row for row, col in cells}) >= 4
    assert sum(n > 0 for n in stats["quadrant_reference_counts"]) >= 3
    for record in records:
        x, y = record["pixel_xy"]
        assert not forbidden[y-1:y+2, x-1:x+2].any()
        assert record["depth_median_m"] == pytest.approx(.5)
        assert record["used_in_fit"]
    assert stats["camera_table_depth_m"] == np.median([r["depth_median_m"] for r in records])
    assert stats["reference_count"] == len(records)


def test_clearance_does_not_fall_back_to_cloth_when_background_is_insufficient(tmp_path):
    frame, config = scene(tmp_path)
    with pytest.raises(PerceptionError, match="bare table depth samples"):
        _fit_table_plane_from_references(
            [frame], replace(config, table_reference_clearance_px=50), np.zeros(3))
