#!/usr/bin/env python3
"""Render a deterministic Camera-A grounding query/support debug image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont

from cloth_agent.garment_grounding_mcp import GarmentGrounding


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _latest_result(root: Path) -> Path:
    paths = [
        path
        for path in (root / "runs").glob("*/results/perception/*/result.json")
        if (path.parent / "camera_A_base_xyz_mm.npy").is_file()
    ]
    if not paths:
        raise FileNotFoundError("no saved perception result found")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def render(result_path: Path, query_x: int, query_y: int, output: Path) -> Path:
    result_path = result_path.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    view = next(
        item for item in result["views"] if str(item.get("label", "")).upper() == "A"
    )
    perception_dir = result_path.parent
    grounding = GarmentGrounding(perception_dir)
    measurement = grounding.sample_local_surface(
        "A", query_x, query_y, radius_px=3, include_nearest_reference=False
    )
    support_x, support_y = measurement.get("support_pixel_xy", [query_x, query_y])
    image_path = perception_dir / str(view["image"])
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font()

    # The runtime search is a local square around Claude's selected pixel.
    search_radius = int(
        measurement.get("support_pixel_diagnostic", {}).get("search_radius_px", 24)
    )
    draw.rectangle(
        (
            query_x - search_radius,
            query_y - search_radius,
            query_x + search_radius,
            query_y + search_radius,
        ),
        outline=(40, 160, 255),
        width=3,
    )
    if (support_x, support_y) != (query_x, query_y):
        draw.line((query_x, query_y, support_x, support_y), fill=(255, 220, 0), width=5)
    draw.ellipse(
        (query_x - 14, query_y - 14, query_x + 14, query_y + 14),
        outline=(255, 35, 35),
        width=5,
    )
    draw.ellipse(
        (support_x - 14, support_y - 14, support_x + 14, support_y + 14),
        outline=(30, 255, 80),
        width=5,
    )
    draw.text((query_x + 18, query_y - 20), "CLAUDE QUERY", fill=(255, 55, 55), font=font)
    draw.text((support_x + 18, support_y + 4), "GROUNDING SUPPORT", fill=(30, 255, 80), font=font)

    panel_width = 430
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (25, 25, 25))
    canvas.paste(image, (0, 0))
    panel = ImageDraw.Draw(canvas)
    x0 = image.width + 18
    panel.text((x0, 16), "Camera A grounding debug", fill=(255, 255, 255), font=font)
    panel.text((x0, 40), f"query pixel:   [{query_x}, {query_y}]", fill=(255, 70, 70), font=font)
    panel.text((x0, 58), f"support pixel: [{support_x}, {support_y}]", fill=(70, 255, 100), font=font)
    panel.text((x0, 76), f"shift px:      {measurement.get('support_pixel_diagnostic', {}).get('shift_px', 0.0):.2f}", fill=(255, 220, 0), font=font)
    panel.text((x0, 110), "query/support measurement:", fill=(255, 255, 255), font=font)
    diag = measurement.get("support_pixel_diagnostic", {})
    qpatch = diag.get("query_patch", {})
    spatch = diag.get("support_patch", {})
    rows = [
        ("query height", qpatch.get("height_median_mm")),
        ("support height", spatch.get("height_median_mm")),
        ("query z spread", qpatch.get("base_z_spread_mm")),
        ("support z spread", spatch.get("base_z_spread_mm")),
        ("final height", measurement.get("height_above_table_median_mm")),
        ("final table z", measurement.get("table_z_median_mm")),
    ]
    for index, (label, value) in enumerate(rows):
        rendered = "n/a" if value is None else f"{float(value):.3f} mm"
        panel.text((x0, 132 + 18 * index), f"{label:16s} {rendered}", fill=(220, 220, 220), font=font)

    crop_size = 120
    cx = int(round((query_x + support_x) / 2))
    cy = int(round((query_y + support_y) / 2))
    left = max(0, min(image.width - crop_size, cx - crop_size // 2))
    top = max(0, min(image.height - crop_size, cy - crop_size // 2))
    crop = image.crop((left, top, left + crop_size, top + crop_size)).resize((360, 360))
    canvas.paste(crop, (image.width + 18, 270))
    panel = ImageDraw.Draw(canvas)
    panel.rectangle((image.width + 18, 270, image.width + 378, 630), outline=(255, 255, 255), width=2)
    panel.text((x0, 650), "red=query, green=support,", fill=(220, 220, 220), font=font)
    panel.text((x0, 668), "yellow=shift, blue=search window", fill=(220, 220, 220), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "perception_result": str(result_path),
                "measurement": measurement,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--perception-result", type=Path)
    parser.add_argument("--query-x", type=int, required=True)
    parser.add_argument("--query-y", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/grounding_debug.png"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = args.perception_result.resolve() if args.perception_result else _latest_result(root)
    output = args.output.resolve()
    print(render(result, args.query_x, args.query_y, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
