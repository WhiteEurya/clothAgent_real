from __future__ import annotations

from pathlib import Path

from PIL import Image

from cloth_agent.molmo_report import annotate_molmo_all_parts


def test_molmo_annotation_draws_points_and_preserves_unknown(tmp_path: Path):
    image_path = tmp_path / "camera_A.png"
    Image.new("RGB", (640, 480), (30, 40, 50)).save(image_path)
    records = [
        {
            "name": "garment_center",
            "color": [255, 40, 40],
            "status": "point_returned",
            "selected_pixel_xy": [320.0, 240.0],
        },
        {
            "name": "neckline",
            "color": [255, 170, 0],
            "status": "unknown",
            "selected_pixel_xy": None,
        },
    ]
    overlay = tmp_path / "overlay.png"
    legend = tmp_path / "legend.png"

    manifest = annotate_molmo_all_parts(
        image_path,
        records,
        overlay,
        legend,
        camera_label="A",
    )

    assert overlay.is_file()
    assert legend.is_file()
    assert manifest["point_count"] == 1
    assert manifest["unknown_count"] == 1
    assert Image.open(overlay).size == (640, 480)
    assert Image.open(legend).size == (1080, 480)
    # A remote pixel remains exactly the original RGB value.
    assert Image.open(overlay).convert("RGB").getpixel((500, 400)) == (30, 40, 50)
