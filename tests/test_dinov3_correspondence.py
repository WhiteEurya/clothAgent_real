from __future__ import annotations

import pytest

from scripts.dinov3_correspondence import FrozenDINOv3


def test_rectangular_patch_coordinate_mapping_preserves_camera_aspect_ratio() -> None:
    extractor = object.__new__(FrozenDINOv3)
    extractor.input_width = 448
    extractor.input_height = 336
    extractor.patch_size = 14
    extractor.grid_w = 32
    extractor.grid_h = 24

    assert extractor.point_to_patch(67, 211, (640, 480)) == (3, 10)
    x, y = extractor.patch_to_point(3, 10, (640, 480))
    assert x == pytest.approx(70.0, abs=0.5)
    assert y == pytest.approx(210.2, abs=0.5)
