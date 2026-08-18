from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from cloth_agent.molmo_artifact_viewer import (
    build_pages,
    discover_output_dir,
    main,
    render_dashboard,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "runs" / "viewer_test"
    output = run_dir / "results" / "molmo_keypoint_cli" / "20260818T120000Z"
    iteration = output / "iteration_001"
    perception = run_dir / "results" / "perception" / "capture_001"
    molmo = iteration / "molmo_keypoints"
    after = iteration / "after_capture"
    for directory in (perception, molmo, after):
        directory.mkdir(parents=True, exist_ok=True)

    image_names = {
        perception: (
            "camera_0_A.png",
            "camera_1_B.png",
            "camera_A_height_map_heatmap.png",
            "camera_B_height_map_heatmap.png",
            "camera_A_height_map_boundary.png",
            "camera_B_height_map_boundary.png",
            "camera_A_coordinate_overlay.png",
            "camera_B_coordinate_overlay.png",
            "fused_height_map_boundary.png",
        ),
        molmo: (
            "camera_A_molmo_keypoint_candidates.png",
            "camera_A_molmo_keypoint_references.png",
            "camera_B_molmo_keypoint_candidates.png",
            "camera_B_molmo_keypoint_references.png",
        ),
        after: ("camera_0_A.png", "camera_1_B.png"),
    }
    color = 20
    for directory, names in image_names.items():
        for name in names:
            Image.new("RGB", (64, 48), (color, 80, 140)).save(directory / name)
            color += 10
    for camera, index in (("A", 0), ("B", 1)):
        np.save(
            perception / f"camera_{index}_{camera}_depth_m.npy",
            np.linspace(0.4, 0.9, 64 * 48, dtype=np.float32).reshape(48, 64),
        )
        np.save(
            perception / f"camera_{camera}_height_above_table_mm.npy",
            np.linspace(0.0, 30.0, 64 * 48, dtype=np.float32).reshape(48, 64),
        )

    result_path = perception / "result.json"
    _write_json(
        result_path,
        {
            "status": "VALIDATED_DENSE_AB_FUSION",
            "depth_fusion": {"garment_color_distance_threshold": 24.0},
        },
    )
    _write_json(
        iteration / "result.json",
        {
            "iteration": 1,
            "status": "RUNNING",
            "saved_perception_result": str(result_path),
        },
    )
    _write_json(
        output / "summary.json",
        {"status": "RUNNING", "run_dir": str(run_dir)},
    )
    (output / "events.jsonl").write_text(
        json.dumps(
            {
                "local_time": "2026-08-18T20:00:00+08:00",
                "level": "WAIT",
                "iteration": 1,
                "phase": "evaluation",
                "message": "still running: comparing before/after",
                "run_elapsed_s": 42.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, output, perception


def test_discovers_latest_cli_output_under_run_directory(tmp_path: Path) -> None:
    run_dir, output, _ = _make_run(tmp_path)
    older = run_dir / "results" / "molmo_keypoint_cli" / "20260818T110000Z"
    _write_json(older / "summary.json", {"status": "FAILED"})
    os.utime(older / "summary.json", (1, 1))

    assert discover_output_dir(run_dir) == output.resolve()
    assert discover_output_dir(output) == output.resolve()


def test_builds_four_pages_with_current_phase_and_saved_images(tmp_path: Path) -> None:
    _, output, _ = _make_run(tmp_path)
    pages, status = build_pages(output)

    assert [page.name for page in pages] == [
        "Overview",
        "Perception",
        "Molmo keypoints",
        "Before / after",
    ]
    assert all(len(page.tiles) == 6 for page in pages)
    assert all(path is not None for _, path in pages[1].tiles)
    assert all(path is not None for _, path in pages[2].tiles)
    assert status["iteration"] == 1
    assert status["phase"] == "evaluation"
    assert status["level"] == "WAIT"
    assert status["run_elapsed_s"] == 42.5


def test_renders_snapshot_without_opening_gui(tmp_path: Path) -> None:
    run_dir, output, _ = _make_run(tmp_path)
    dashboard, page_index = render_dashboard(
        output,
        1,
        width=900,
        height=600,
    )
    assert dashboard.size == (900, 600)
    assert page_index == 1

    snapshot = tmp_path / "dashboard.png"
    code = main(
        [
            str(run_dir),
            "--page",
            "2",
            "--width",
            "900",
            "--height",
            "600",
            "--snapshot",
            str(snapshot),
        ]
    )
    assert code == 0
    assert snapshot.is_file()
    with Image.open(snapshot) as saved:
        assert saved.size == (900, 600)
