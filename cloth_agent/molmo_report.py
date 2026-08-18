"""Bounded Molmo all-parts inference and deterministic report annotations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont


class MolmoReportError(RuntimeError):
    """Raised when report-only Molmo inference cannot be completed."""


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def annotate_molmo_all_parts(
    image_path: Path,
    records: list[dict[str, Any]],
    overlay_path: Path,
    legend_path: Path,
    *,
    camera_label: str = "A",
) -> dict[str, Any]:
    """Draw exact Molmo points and explicit UNKNOWN entries without redrawing RGB."""

    image = Image.open(image_path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    point_font = _font(14, bold=True)
    returned = 0
    for index, record in enumerate(records, start=1):
        if record.get("status") != "point_returned":
            continue
        pixel = record.get("selected_pixel_xy")
        if not isinstance(pixel, list) or len(pixel) != 2:
            continue
        x_px, y_px = float(pixel[0]), float(pixel[1])
        color = tuple(int(value) for value in record.get("color", [255, 255, 255]))
        radius = 11
        draw.ellipse(
            (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=5,
        )
        draw.ellipse(
            (
                x_px - radius + 3,
                y_px - radius + 3,
                x_px + radius - 3,
                y_px + radius - 3,
            ),
            outline=color,
            width=4,
        )
        text = str(index)
        box = draw.textbbox((0, 0), text, font=point_font)
        draw.text(
            (
                x_px - (box[2] - box[0]) / 2,
                y_px - (box[3] - box[1]) / 2 - 2,
            ),
            text,
            fill=(255, 255, 255),
            font=point_font,
        )
        returned += 1
    draw.rectangle((8, 8, 270, 36), fill=(0, 0, 0))
    draw.text(
        (14, 13),
        f"Molmo parts: {returned}/{len(records)} points",
        fill=(255, 220, 80),
        font=_font(15, bold=True),
    )

    panel_width = 440
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (25, 25, 25))
    canvas.paste(overlay, (0, 0))
    panel = ImageDraw.Draw(canvas)
    panel.text(
        (image.width + 14, 12),
        f"Camera {camera_label} | MolmoPoint all parts",
        fill=(255, 255, 255),
        font=_font(17, bold=True),
    )
    panel.text(
        (image.width + 14, 39),
        "Zero-shot observation: one point or UNKNOWN",
        fill=(255, 200, 80),
        font=_font(13),
    )
    y = 70
    for index, record in enumerate(records, start=1):
        color = tuple(int(value) for value in record.get("color", [255, 255, 255]))
        panel.ellipse((image.width + 14, y + 2, image.width + 26, y + 14), fill=color)
        if record.get("status") == "point_returned":
            x_px, y_px = record["selected_pixel_xy"]
            detail = f"({float(x_px):.1f}, {float(y_px):.1f})"
        else:
            detail = "UNKNOWN"
        panel.text(
            (image.width + 34, y),
            f"{index}. {record.get('name', 'part')}: {detail}",
            fill=(235, 235, 235),
            font=_font(13),
        )
        y += 34

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(overlay_path)
    canvas.save(legend_path)
    return {
        "camera": camera_label,
        "point_count": returned,
        "unknown_count": len(records) - returned,
        "overlay_image": str(overlay_path),
        "legend_image": str(legend_path),
        "records": records,
    }


def run_molmo_all_parts_report(
    project_root: Path,
    image_path: Path,
    output_dir: Path,
    *,
    python_path: Path,
    camera_label: str = "A",
    timeout_s: int = 600,
    model: str = "allenai/MolmoPoint-8B",
    dtype: str = "bf16",
    max_crops: int = 1,
    max_new_tokens: int = 96,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Invoke one report-only Molmo process, then draw its saved point records."""

    root = Path(project_root).resolve()
    image = Path(image_path).resolve()
    output = Path(output_dir).resolve()
    python = Path(python_path).expanduser().resolve()
    if not python.is_file():
        raise MolmoReportError(f"Molmo Python does not exist: {python}")
    if not image.is_file():
        raise MolmoReportError(f"Molmo input image does not exist: {image}")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "molmo_all_parts.json"
    worker = root / "cloth_agent" / "molmo_all_parts_worker.py"
    command = [
        str(python),
        str(worker),
        "--image",
        str(image),
        "--label",
        camera_label,
        "--output",
        str(result_path),
        "--model",
        model,
        "--dtype",
        dtype,
        "--max-crops",
        str(max_crops),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if local_files_only:
        command.append("--local-files-only")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MolmoReportError(f"Molmo all-parts invocation failed: {exc}") from exc
    duration_s = time.monotonic() - started
    log_path = output / "molmo_all_parts.stdout.txt"
    log_path.write_text(
        completed.stdout + ("\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise MolmoReportError(
            f"Molmo all-parts exited with {completed.returncode}; inspect {log_path}"
        )
    if not result_path.is_file():
        raise MolmoReportError("Molmo all-parts returned without a JSON result")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    views = payload.get("views", [])
    if not isinstance(views, list) or len(views) != 1:
        raise MolmoReportError("Molmo all-parts result must contain exactly one view")
    records = views[0].get("records", [])
    if not isinstance(records, list):
        raise MolmoReportError("Molmo all-parts records are malformed")
    overlay_path = output / f"molmo_all_parts_camera_{camera_label}.png"
    legend_path = output / f"molmo_all_parts_camera_{camera_label}_with_legend.png"
    annotation = annotate_molmo_all_parts(
        image,
        records,
        overlay_path,
        legend_path,
        camera_label=camera_label,
    )
    manifest = {
        "model": payload.get("model", model),
        "query_mode": payload.get("query_mode"),
        "duration_s": duration_s,
        "input_image": str(image),
        "result_json": str(result_path),
        "stdout_log": str(log_path),
        **annotation,
    }
    (output / "molmo_all_parts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
