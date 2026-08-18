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
