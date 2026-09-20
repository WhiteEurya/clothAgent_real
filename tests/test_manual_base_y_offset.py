import numpy as np
import pytest

from cloth_agent.perception import (RGBDFrame, PerceptionConfig, PerceptionError,
                                    with_manual_base_y_offset, camera_base_xyz_map_mm,
                                    pixel_to_base_mm, _frame_points_base_mm)


def test_y_correction_consistent_and_not_accumulated():
    frame = RGBDFrame('A', 'test', np.zeros((2, 2, 3), np.uint8),
                      np.full((2, 2), .5), np.eye(3), np.eye(4))
    config = PerceptionConfig(cameras=())
    corrected = with_manual_base_y_offset(frame, -20.)
    raw, valid = camera_base_xyz_map_mm(frame, config)
    shifted, _ = camera_base_xyz_map_mm(corrected, config)
    np.testing.assert_allclose(shifted[valid], raw[valid] + [0, -20, 0], atol=1e-4)
    points, _ = _frame_points_base_mm(corrected, config)
    np.testing.assert_allclose(points, shifted[valid], atol=1e-4)
    np.testing.assert_allclose(pixel_to_base_mm(0, 0, .5, frame.intrinsics,
                                               corrected.X_base_camera), [0, -20, 500])
    np.testing.assert_array_equal(frame.X_base_camera, np.eye(4))
    np.testing.assert_array_equal(with_manual_base_y_offset(corrected, -20).X_base_camera,
                                  corrected.X_base_camera)
    np.testing.assert_allclose(with_manual_base_y_offset(corrected, 0).X_base_camera, np.eye(4))
    with pytest.raises(PerceptionError):
        with_manual_base_y_offset(frame, float('nan'))
