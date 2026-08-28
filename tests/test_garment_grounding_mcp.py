from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cloth_agent.garment_grounding_mcp import GarmentGrounding


def _perception_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "perception_views"
    directory.mkdir()
    guide = {
        "camera_label": "A",
        "coordinate_frame": "robot_base_mm",
        "sample_stride_px": 48,
        "reference_semantics": "Uniform calibrated references; not grasp candidates.",
        "samples": [
            {
                "reference_id": "R001",
                "pixel_xy": [1, 1],
                "base_xyz_mm": [100.0, 200.0, 10.0],
                "height_above_table_mm": 8.0,
            },
            {
                "reference_id": "R026",
                "pixel_xy": [3, 2],
                "base_xyz_mm": [522.1, -197.1, 19.8],
                "height_above_table_mm": 15.1,
            },
        ],
    }
    (directory / "camera_A_coordinate_guide.json").write_text(
        json.dumps(guide), encoding="utf-8"
    )
    xyz = np.zeros((4, 5, 3), dtype=np.float32)
    for y_px in range(4):
        for x_px in range(5):
            xyz[y_px, x_px] = [500.0 + x_px, -200.0 + y_px, 12.0 + x_px]
    height = np.full((4, 5), 7.5, dtype=np.float32)
    table = xyz[:, :, 2] - height
    np.save(directory / "camera_A_base_xyz_mm.npy", xyz)
    np.save(directory / "camera_A_height_above_table_mm.npy", height)
    np.save(directory / "camera_A_table_z_mm.npy", table)
    return directory


def test_lookup_reference_returns_exact_saved_rxx_measurement(tmp_path: Path):
    grounding = GarmentGrounding(_perception_dir(tmp_path))
    result = grounding.lookup_reference("a", "r026")
    assert result["reference_id"] == "R026"
    assert result["pixel_xy"] == [3, 2]
    assert result["base_xyz_mm"] == pytest.approx([522.1, -197.1, 19.8])
    assert result["table_z_mm"] == pytest.approx(4.7)
    assert result["sample_stride_px"] == 48
    assert "not a ranked grasp candidate" in result["warning"]


def test_pixel_and_local_tools_use_full_resolution_maps(tmp_path: Path):
    grounding = GarmentGrounding(_perception_dir(tmp_path))
    pixel = grounding.sample_pixel_xyz("A", 2, 1)
    assert pixel["base_xyz_mm"] == pytest.approx([502.0, -199.0, 14.0])
    assert pixel["height_above_table_mm"] == pytest.approx(7.5)
    assert pixel["nearest_reference"]["reference_id"] == "R001"

    local = grounding.sample_local_surface("A", 2, 1, radius_px=1)
    assert local["valid"] is True
    assert local["sample_count"] == 9
    assert local["base_xyz_median_mm"] == pytest.approx([502.0, -199.0, 14.0])
    assert local["height_above_table_median_mm"] == pytest.approx(7.5)
    assert local["surface_shape_diagnostic"]["surface_shape"] == "LOW_RELIEF"


def test_local_surface_marks_narrow_relief_as_roll_wrinkle_triage(tmp_path: Path):
    directory = _perception_dir(tmp_path)
    height = np.full((20, 20), 7.5, dtype=np.float32)
    height[7:12, 7:12] = 25.0
    yy, xx = np.indices(height.shape)
    xyz = np.stack((500.0 + xx, -200.0 + yy, 12.0 + height), axis=2).astype(
        np.float32
    )
    np.save(directory / "camera_A_base_xyz_mm.npy", xyz)
    np.save(directory / "camera_A_height_above_table_mm.npy", height)
    np.save(directory / "camera_A_table_z_mm.npy", xyz[:, :, 2] - height)

    local = GarmentGrounding(directory).sample_local_surface("A", 9, 9, radius_px=1)
    diagnostic = local["surface_shape_diagnostic"]
    assert diagnostic["surface_shape"] == "NARROW_RIDGE_OR_SPIKE"
    assert diagnostic["requires_structure_hold_check"] is True
    assert diagnostic["compression_probe_recommended"] is True
    assert diagnostic["recommended_press_below_surface_mm"] == pytest.approx(1.0)


def test_local_surface_uses_fused_camera_b_z_support_for_camera_a(tmp_path: Path):
    directory = _perception_dir(tmp_path)
    points = np.asarray(
        [
            [501.0, -199.0, 19.0],
            [502.0, -199.0, 20.0],
            [503.0, -199.0, 21.0],
        ],
        dtype=np.float32,
    )
    np.save(directory / "fused_points_base_mm.npy", points)
    np.save(directory / "fused_height_above_table_mm.npy", np.full(3, 15.0, dtype=np.float32))
    np.save(directory / "fused_source_mask.npy", np.asarray([2, 2, 3], dtype=np.uint8))
    np.save(directory / "fused_garment_mask.npy", np.ones(3, dtype=bool))

    local = GarmentGrounding(directory).sample_local_surface("A", 2, 1, radius_px=1)

    assert local["fused_surface_support_used"] is True
    assert local["surface_measurement_policy"] == (
        "camera_A_xy_with_fused_camera_B_or_AB_surface_z"
    )
    assert local["base_xyz_median_mm"][2] == pytest.approx(20.0)
    assert local["height_above_table_median_mm"] == pytest.approx(15.0)


def test_local_surface_moves_edge_query_to_nearest_stable_garment_interior(tmp_path: Path):
    directory = tmp_path / "perception_views"
    directory.mkdir()
    height, width = 24, 24
    mask = np.zeros((height, width), dtype=bool)
    mask[3:21, 3:21] = True
    xyz = np.zeros((height, width, 3), dtype=np.float32)
    height_map = np.full((height, width), np.nan, dtype=np.float32)
    for y_px in range(height):
        for x_px in range(width):
            xyz[y_px, x_px] = [500.0 + x_px, -200.0 + y_px, 10.0]
            if mask[y_px, x_px]:
                height_map[y_px, x_px] = 2.0
    # The visible edge is semantically valid but its depth is contaminated by
    # the table/background; the interior remains a stable cloth surface.
    height_map[9:12, 3:5] = -6.0
    xyz[9:12, 3:5, 2] = 2.0
    table = xyz[:, :, 2] - np.nan_to_num(height_map, nan=0.0)
    np.save(directory / "camera_A_base_xyz_mm.npy", xyz)
    np.save(directory / "camera_A_height_above_table_mm.npy", height_map)
    np.save(directory / "camera_A_table_z_mm.npy", table)
    np.save(directory / "camera_A_garment_mask.npy", mask)

    local = GarmentGrounding(directory).sample_local_surface(
        "A", 3, 10, radius_px=1, include_nearest_reference=False
    )

    assert local["query_pixel_xy"] == [3, 10]
    assert local["support_pixel_xy"] != [3, 10]
    assert local["support_pixel_diagnostic"]["applied"] is True
    assert local["height_above_table_median_mm"] >= 0.0


def test_stdio_server_lists_and_calls_read_only_tools(tmp_path: Path):
    perception_dir = _perception_dir(tmp_path)
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "lookup_reference",
                "arguments": {"camera": "A", "reference_id": "R026"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "lookup_reference",
                "arguments": {"camera": "A", "reference_id": "R001"},
            },
        },
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloth_agent.garment_grounding_mcp",
            "--perception-dir",
            str(perception_dir),
        ],
        input="\n".join(json.dumps(item) for item in requests) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["id"] for response in responses] == [1, 2, 3, 4]
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert tool_names == {"lookup_reference"}
    payload = json.loads(responses[2]["result"]["content"][0]["text"])
    assert payload["reference_id"] == "R026"
    assert payload["base_xyz_mm"] == pytest.approx([522.1, -197.1, 19.8])
    assert payload["lookup_budget_remaining"] == 0
    assert responses[3]["result"]["isError"] is True
    second_payload = json.loads(responses[3]["result"]["content"][0]["text"])
    assert "already been used" in second_payload["error"]


def test_stdio_pixel_mode_exposes_only_one_arbitrary_pixel_lookup(tmp_path: Path):
    perception_dir = _perception_dir(tmp_path)
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "sample_local_surface",
                "arguments": {"camera": "A", "x_px": 2, "y_px": 1, "radius_px": 1},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "sample_local_surface",
                "arguments": {"camera": "A", "x_px": 3, "y_px": 2},
            },
        },
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cloth_agent.garment_grounding_mcp",
            "--perception-dir",
            str(perception_dir),
            "--mode",
            "pixel",
        ],
        input="\n".join(json.dumps(item) for item in requests) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {
        "sample_local_surface"
    }
    payload = json.loads(responses[2]["result"]["content"][0]["text"])
    assert payload["query_pixel_xy"] == [2, 1]
    assert payload["base_xyz_median_mm"] == pytest.approx([502.0, -199.0, 14.0])
    assert payload["lookup_budget_remaining"] == 0
    assert "nearest_reference" not in payload
    assert responses[3]["result"]["isError"] is True
