from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cloth_agent.perception import (
    CameraSpec,
    PerceptionConfig,
    PerceptionError,
    RGBDFrame,
    _camera_height_view_artifacts,
    _estimate_camera_table_appearance,
    _edge_connected_fixture_mask,
    _fused_source_appearance_mask,
    _occlusion_aware_garment_mask,
    _outer_mask_boundary,
    _require_projected_garment_validation,
    _save_camera_height_heatmap,
)


def test_table_appearance_uses_bright_tail_when_flat_black_cloth_dominates() -> None:
    rgb = np.full((100, 100, 3), [18, 29, 48], dtype=np.uint8)
    rgb[:, 92:] = [166, 179, 242]
    height_map = np.zeros((100, 100), dtype=np.float32)

    appearance = _estimate_camera_table_appearance(
        rgb,
        height_map,
        np.ones((100, 100), dtype=bool),
        minimum_color_distance=24.0,
    )

    assert appearance["confident"] is True
    assert appearance["high_luma_percentile"] == pytest.approx(95.0)
    assert appearance["table_rgb_median"] == [166.0, 179.0, 242.0]
    assert appearance["table_luma_median"] > 150.0


def test_dark_table_appearance_is_not_trusted() -> None:
    rgb = np.full((40, 40, 3), [18, 29, 48], dtype=np.uint8)

    appearance = _estimate_camera_table_appearance(
        rgb,
        np.zeros((40, 40), dtype=np.float32),
        np.ones((40, 40), dtype=bool),
        minimum_color_distance=24.0,
    )

    assert appearance["confident"] is False
    assert "table luma" in appearance["reason"]


def test_fused_appearance_uses_each_voxels_camera_sources() -> None:
    appearances = {
        "A": {
            "confident": True,
            "table_rgb_median": [250.0, 255.0, 254.0],
            "applied_color_distance": 24.0,
        },
        "B": {
            "confident": True,
            "table_rgb_median": [166.0, 179.0, 242.0],
            "applied_color_distance": 80.0,
        },
    }
    colors = np.asarray(
        [
            [250, 255, 254],
            [166, 179, 242],
            [208, 217, 248],
            [20, 30, 50],
            [20, 30, 50],
            [20, 30, 50],
        ],
        dtype=np.uint8,
    )
    sources = np.asarray([1, 2, 3, 1, 2, 3], dtype=np.uint8)

    distinct, distances, thresholds, diagnostics = _fused_source_appearance_mask(
        colors,
        sources,
        ("A", "B"),
        appearances,
    )

    assert distinct.tolist() == [False, False, False, True, True, True]
    assert distances[:3].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert thresholds.tolist() == pytest.approx([24.0, 80.0, 80.0, 24.0, 80.0, 80.0])
    assert diagnostics["method"] == "source_specific_camera_table_color"


def test_thin_projected_garment_mask_is_a_hard_failure() -> None:
    camera_artifacts = {
        "A": {
            "projection_diagnostics": {
                "silhouette_pixels": 50_000,
                "garment_mask_pixels": 1_900,
            }
        },
        "B": {
            "projection_diagnostics": {
                "silhouette_pixels": 4_300,
                "garment_mask_pixels": 1_300,
            }
        },
    }

    with pytest.raises(PerceptionError, match="blocked Molmo/Claude/robot"):
        _require_projected_garment_validation(
            camera_artifacts,
            {"A": 3_000, "B": 3_000},
        )


def test_near_camera_voxel_projection_uses_adaptive_support_radius() -> None:
    height = width = 120
    focal_length_px = 600.0
    depth_m = 0.4
    intrinsics = np.asarray(
        [
            [focal_length_px, 0.0, 59.5],
            [0.0, focal_length_px, 59.5],
            [0.0, 0.0, 1.0],
        ]
    )
    rgb = np.full((height, width, 3), 240, dtype=np.uint8)
    rgb[15:106, 15:106] = 20
    frame = RGBDFrame(
        "B",
        "B1",
        rgb,
        np.full((height, width), depth_m, dtype=np.float32),
        intrinsics,
        np.eye(4),
    )
    projected_pixels = range(15, 106, 9)
    garment_points = np.asarray(
        [
            [
                (x_px - 59.5) * depth_m / focal_length_px * 1000.0,
                (y_px - 59.5) * depth_m / focal_length_px * 1000.0,
                depth_m * 1000.0,
            ]
            for y_px in projected_pixels
            for x_px in projected_pixels
        ]
    )

    mask, sparse, diagnostics = _occlusion_aware_garment_mask(
        garment_points,
        frame,
        np.zeros((height, width), dtype=np.float32),
        np.ones((height, width), dtype=bool),
        minimum_table_color_distance=24.0,
    )

    assert diagnostics["projection_support_radius_px"] == 5
    assert sparse.sum() == len(garment_points)
    assert mask[60, 60]
    assert mask.sum() > 7_000


def test_high_edge_fixture_is_removed_without_erasing_garment() -> None:
    garment = np.zeros((120, 160), dtype=bool)
    garment[25:111, 20:140] = True
    garment[45:75, 140:160] = True
    heights = np.zeros(garment.shape, dtype=np.float32)
    # Compact high bracket entering from the right edge, with a lower stem.
    heights[45:75, 148:160] = 55.0
    fixture, diagnostics = _edge_connected_fixture_mask(
        garment,
        heights,
        padding_px=15,
    )

    assert diagnostics["applied"] is True
    assert diagnostics["excluded_component_count"] == 1
    assert fixture[60, 150]
    assert not fixture[60, 80]


def test_high_garment_fold_touching_edge_is_not_treated_as_small_fixture() -> None:
    garment = np.ones((120, 160), dtype=bool)
    heights = np.zeros(garment.shape, dtype=np.float32)
    heights[0:30, :] = 45.0

    fixture, diagnostics = _edge_connected_fixture_mask(garment, heights)

    assert diagnostics["applied"] is False
    assert not fixture.any()


def test_camera_view_exposes_final_garment_mask_to_session_copy() -> None:
    artifacts = {
        "height_map": "heat.png",
        "height_map_global": "heat_global.png",
        "height_map_boundary": "boundary.png",
        "height_map_path": "height.npy",
        "garment_mask": "garment_mask.npy",
        "fold_edge_overlay": "fold.png",
    }

    view = _camera_height_view_artifacts(artifacts)

    assert view["garment_mask"] == "garment_mask.npy"
    assert view["height_map_path"] == "height.npy"


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


def test_appearance_mask_rejects_shadowed_table_with_same_chromaticity() -> None:
    height = width = 30
    intrinsics = np.asarray(
        [[20.0, 0.0, 14.5], [0.0, 20.0, 14.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rgb = np.full((height, width, 3), 240, dtype=np.uint8)
    rgb[5:25, 5:15] = [20, 30, 50]
    # Darkened neutral table: far in RGB magnitude but close in chromaticity.
    rgb[5:25, 15:25] = [120, 122, 121]
    frame = RGBDFrame(
        "A",
        "A1",
        rgb,
        np.ones((height, width), dtype=np.float32),
        intrinsics,
        np.eye(4),
    )
    garment_points = np.asarray(
        [
            [(x - 14.5) * 50.0, (y - 14.5) * 50.0, 1000.0]
            for y in range(5, 25)
            for x in (5, 24)
        ]
        + [
            [(x - 14.5) * 50.0, (y - 14.5) * 50.0, 1000.0]
            for x in range(6, 24)
            for y in (5, 24)
        ]
    )

    mask, _, diagnostics = _occlusion_aware_garment_mask(
        garment_points,
        frame,
        np.zeros((height, width), dtype=np.float32),
        np.ones((height, width), dtype=bool),
        minimum_table_color_distance=24.0,
    )

    assert mask[12, 10]
    assert not mask[12, 20]
    assert diagnostics["appearance_filter"]["applied_chromaticity_distance"] == pytest.approx(0.07)


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
    saved_mask = np.load(tmp_path / artifacts["garment_mask"], allow_pickle=False)

    assert guide["samples"]
    assert saved_mask.dtype == np.bool_
    assert saved_mask.shape == (height, width)
    assert saved_mask[30, 30]
    assert not saved_mask[70, 30]
    assert all(
        20 <= sample["pixel_xy"][0] <= 80
        and 20 <= sample["pixel_xy"][1] <= 55
        for sample in guide["samples"]
    )
