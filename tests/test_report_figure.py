from __future__ import annotations

from pathlib import Path

from PIL import Image

from cloth_agent.report_figure import compose_camera_perception_report


def _save(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (640, 480), color).save(path)


def test_camera_report_is_created_and_preserves_source_pixels(tmp_path: Path):
    result_dir = tmp_path / "perception"
    result_dir.mkdir()
    files = {
        "image": ("camera_A.png", (11, 22, 33)),
        "height_map": ("height.png", (44, 55, 66)),
        "height_map_boundary": ("boundary.png", (77, 88, 99)),
        "height_gradient_overlay": ("edges.png", (100, 110, 120)),
        "coordinate_overlay": ("coordinates.png", (130, 140, 150)),
    }
    for name, color in files.values():
        _save(result_dir / name, color)
    result_path = result_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    perception = {
        "views": [
            {
                "label": "A",
                **{key: value[0] for key, value in files.items()},
                "height_map_min_mm": -4.0,
                "height_map_max_mm": 17.3,
            }
        ]
    }
    output = tmp_path / "iteration_001" / "camera_A_perception_report.png"
    manifest = compose_camera_perception_report(
        perception,
        result_path,
        output,
        camera="A",
        run_name="test_run",
        iteration=1,
    )

    assert output.is_file()
    assert manifest["generation_mode"] == "deterministic_exact_pixel_composite"
    assert manifest["selected_reference"] == {}
    assert manifest["target"] == {}
    report = Image.open(output).convert("RGB")
    assert report.size == tuple(manifest["size_px"])
    # First panel source starts at x=30, y=112+50. Check a central pixel,
    # safely away from panel borders and labels.
    assert report.getpixel((30 + 320, 112 + 50 + 240)) == (11, 22, 33)


def test_camera_report_can_be_refreshed_with_validated_target(tmp_path: Path):
    result_dir = tmp_path / "perception"
    result_dir.mkdir()
    names = {
        "image": "camera_A.png",
        "height_map": "height.png",
        "height_map_boundary": "boundary.png",
        "height_gradient_overlay": "edges.png",
        "coordinate_overlay": "coordinates.png",
    }
    for name in names.values():
        _save(result_dir / name, (25, 35, 45))
    target_overlay = tmp_path / "target.png"
    _save(target_overlay, (200, 10, 20))
    result_path = result_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    perception = {"views": [{"label": "A", **names}]}
    output = tmp_path / "iteration_002" / "camera_A_perception_report.png"

    manifest = compose_camera_perception_report(
        perception,
        result_path,
        output,
        camera="A",
        run_name="test_run",
        iteration=2,
        target_overlay_path=target_overlay,
        selected_reference={"camera": "A", "reference_id": "R011"},
        target={"x": 458.577, "y": 128.984, "z": 14.5, "yaw": 155.0},
    )

    assert manifest["selected_reference"]["reference_id"] == "R011"
    assert manifest["target"]["x"] == 458.577
    report = Image.open(output).convert("RGB")
    # Sixth panel is column 2, row 1. Check its exact copied target pixel.
    x = 30 + 2 * (640 + 22) + 320
    y = 112 + (480 + 50 + 22) + 50 + 240
    assert report.getpixel((x, y)) == (200, 10, 20)


def test_camera_report_adds_molmo_panel_without_modifying_annotation(tmp_path: Path):
    result_dir = tmp_path / "perception"
    result_dir.mkdir()
    names = {
        "image": "camera_A.png",
        "height_map": "height.png",
        "height_map_global": "height_global.png",
        "height_map_boundary": "boundary.png",
        "height_gradient_overlay": "edges.png",
        "coordinate_overlay": "coordinates.png",
    }
    for name in names.values():
        _save(result_dir / name, (25, 35, 45))
    molmo = tmp_path / "molmo.png"
    _save(molmo, (14, 160, 220))
    result_path = result_dir / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "iteration_003" / "camera_A_perception_report.png"

    manifest = compose_camera_perception_report(
        {"views": [{"label": "A", **names}]},
        result_path,
        output,
        camera="A",
        run_name="test_run",
        iteration=3,
        molmo_annotation_path=molmo,
    )

    assert manifest["molmo_annotation"] == str(molmo)
    report = Image.open(output).convert("RGB")
    assert report.size[0] == 30 * 2 + 4 * 640 + 3 * 22
    # Molmo is panel (g): column 2, row 1 in the 4x2 layout.
    x = 30 + 2 * (640 + 22) + 320
    y = 112 + (480 + 50 + 22) + 50 + 240
    assert report.getpixel((x, y)) == (14, 160, 220)
