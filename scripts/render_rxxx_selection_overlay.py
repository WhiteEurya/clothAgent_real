#!/usr/bin/env python3
"""Highlight Claude's selected Rxxx markers on an upright planning overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (235, 45, 45),
    (35, 115, 255),
    (30, 185, 75),
    (225, 125, 20),
    (170, 65, 220),
    (220, 40, 150),
]


def _clockwise90_pixel(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return float(height - 1 - y), float(x)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _raw_camera_a_size(summary: dict[str, Any]) -> tuple[int, int]:
    result_path = Path(str(summary["perception_result"])).resolve()
    result = _load_json(result_path)
    view = next(
        item
        for item in result.get("views", [])
        if isinstance(item, dict) and str(item.get("label", "")).upper() == "A"
    )
    image_path = result_path.parent / str(view["image"])
    with Image.open(image_path) as image:
        return image.size


def render(preview_output: Path) -> Path:
    preview_output = preview_output.resolve()
    summary = _load_json(preview_output / "summary.json")
    plans_payload = json.loads((preview_output / "plans.json").read_text(encoding="utf-8"))
    if not isinstance(plans_payload, list) or not plans_payload:
        raise ValueError(f"no plans found in {preview_output / 'plans.json'}")
    overlay_path = preview_output / "camera_A_rxxx_overlay_upright.png"
    if not overlay_path.is_file():
        raise FileNotFoundError(overlay_path)
    raw_width, raw_height = _raw_camera_a_size(summary)
    with Image.open(overlay_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    selected: list[dict[str, Any]] = []

    for index, record in enumerate(plans_payload):
        if not isinstance(record, dict):
            continue
        anchor = record.get("selected_reference_anchor", {})
        if not isinstance(anchor, dict):
            continue
        pixel = anchor.get("pixel_xy")
        reference_id = anchor.get("reference_id")
        if (
            not isinstance(reference_id, str)
            or not isinstance(pixel, list)
            or len(pixel) != 2
        ):
            continue
        x, y = _clockwise90_pixel(
            float(pixel[0]), float(pixel[1]), raw_width, raw_height
        )
        color = COLORS[index % len(COLORS)]
        radius = 18
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 255),
            width=10,
        )
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=6,
        )
        draw.line((x - 28, y, x + 28, y), fill=color, width=3)
        draw.line((x, y - 28, x, y + 28), fill=color, width=3)
        label = f"I{record.get('round', index + 1)} {reference_id}"
        draw.text(
            (x + 24, y - 13),
            label,
            fill=color,
            font=font,
            stroke_width=3,
            stroke_fill=(255, 255, 255),
        )
        selected.append(
            {
                "iteration": record.get("round", index + 1),
                "reference_id": reference_id,
                "raw_pixel_xy": [int(round(float(pixel[0]))), int(round(float(pixel[1])))],
                "upright_pixel_xy": [float(x), float(y)],
                "color_rgb": list(color),
            }
        )

    panel_width = 300
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (28, 28, 28))
    canvas.paste(image, (0, 0))
    panel = ImageDraw.Draw(canvas)
    panel_x = image.width + 18
    panel.text((panel_x, 18), "Claude Rxxx selections", fill=(255, 255, 255), font=font)
    panel.text((panel_x, 38), "upright Camera A", fill=(190, 190, 190), font=font)
    for index, item in enumerate(selected):
        y = 78 + index * 46
        color = tuple(item["color_rgb"])
        panel.line((panel_x, y + 7, panel_x + 28, y + 7), fill=color, width=7)
        panel.text(
            (panel_x + 40, y),
            f"Iter {item['iteration']}: {item['reference_id']}",
            fill=color,
            font=font,
        )
        panel.text(
            (panel_x + 40, y + 17),
            f"display {item['upright_pixel_xy'][0]:.0f}, {item['upright_pixel_xy'][1]:.0f}",
            fill=(215, 215, 215),
            font=font,
        )

    output = preview_output / "camera_A_rxxx_selected_points_upright.png"
    canvas.save(output)
    report = {
        "preview_output": str(preview_output),
        "source_overlay": str(overlay_path),
        "operation": "highlight_selected_rxxx_markers",
        "selected": selected,
        "output": str(output),
    }
    (preview_output / "camera_A_rxxx_selected_points_upright.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.preview_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
