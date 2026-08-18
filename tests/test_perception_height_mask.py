from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cloth_agent.perception import (
    CameraSpec,
    PerceptionConfig,
    RGBDFrame,
    _occlusion_aware_garment_mask,
    _outer_mask_boundary,
    _save_camera_height_heatmap,
)


def test_projected_garment_mask_rejects_table_colored_silhouette_pixels() -> None:
    height = width = 20
    intrinsics = np.asarray(
        [[10.0, 0.0, 9.5], [0.0, 10.0, 9.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rgb = np.full((height, width, 3), 240, dtype=np.uint8)
    # The upper portion is dark fabric. The lower portion is white table that
    # falls inside the solidified projected silhouette and has nearly the same
    # camera depth as cloth lying on the table.
    rgb[4:12, 4:16] = 20
    depth = np.ones((height, width), dtype=np.float32)
    frame = RGBDFrame("A", "A1", rgb, depth, intrinsics, np.eye(4))

    boundary_pixels: list[tuple[int, int]] = []
    for x in range(4, 16):
        boundary_pixels.extend(((x, 4), (x, 15)))
    for y in range(5, 15):
        boundary_pixels.extend(((4, y), (15, y)))
    garment_points = np.asarray(
        [
            [
                (x - 9.5) * 100.0,
                (y - 9.5) * 100.0,
                1000.0,
            ]
            for x, y in boundary_pixels
        ],
        dtype=np.float64,
    )
    height_map = np.zeros((height, width), dtype=np.float32)
    mask, sparse, diagnostics = _occlusion_aware_garment_mask(
        garment_points,
        frame,
        height_map,
        np.ones((height, width), dtype=bool),
        minimum_table_color_distance=24.0,
    )

    assert sparse[4, 4]
    assert mask[8, 8]
    assert not mask[14, 8]
    assert diagnostics["appearance_filter"]["applied"] is True
    assert diagnostics["appearance_filter"]["table_rgb_median"] == [240.0] * 3
    assert diagnostics["garment_mask_pixels"] < diagnostics["silhouette_pixels"]


def test_outer_boundary_does_not_outline_internal_mask_holes() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 2:10] = True
    mask[5:7, 5:7] = False

    boundary = _outer_mask_boundary(mask)

    assert boundary[2, 5]
    assert not boundary[4, 5]
    assert not boundary[5, 4]


def test_coordinate_guide_uses_final_garment_mask_not_sparse_projection(
    tmp_path: Path,
) -> None:
    height = width = 100
    intrinsics = np.asarray(
        [[50.0, 0.0, 49.5], [0.0, 50.0, 49.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rgb = np.full((height, width, 3), 240, dtype=np.uint8)
    rgb[20:56, 20:81] = 20
    depth = np.ones((height, width), dtype=np.float32)
    frame = RGBDFrame("A", "A1", rgb, depth, intrinsics, np.eye(4))
    config = PerceptionConfig(
        cameras=(
            CameraSpec("A", "A1", tmp_path / "A.yaml"),
            CameraSpec("B", "B1", tmp_path / "B.yaml"),
        ),
        width=width,
        height=height,
        temporal_median_frames=1,
    )
    boundary_pixels: list[tuple[int, int]] = []
    for x in range(20, 81):
        boundary_pixels.extend(((x, 20), (x, 80)))
    for y in range(21, 80):
        boundary_pixels.extend(((20, y), (80, y)))
    garment_points = np.asarray(
        [
            [
                (x - 49.5) * 20.0,
                (y - 49.5) * 20.0,
                1000.0,
            ]
            for x, y in boundary_pixels
        ],
        dtype=np.float64,
    )

    artifacts = _save_camera_height_heatmap(
        tmp_path,
        frame,
        config,
        garment_points,
        np.asarray([0.0, 0.0, 1000.0]),
        minimum_table_color_distance=24.0,
    )
    guide = json.loads(
        (tmp_path / artifacts["coordinate_guide"]).read_text(encoding="utf-8")
    )

    assert guide["samples"]
    assert all(
        20 <= sample["pixel_xy"][0] <= 80
        and 20 <= sample["pixel_xy"][1] <= 55
        for sample in guide["samples"]
    )
