"""Deterministic report figures assembled from saved garment-perception images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _camera_view(perception: dict[str, Any], camera: str) -> dict[str, Any]:
    label = str(camera).strip().upper()
    for view in perception.get("views", []):
        if str(view.get("label", "")).upper() == label:
            return dict(view)
    raise ValueError(f"perception result has no Camera {label} view")


def _source_path(result_path: Path, value: Any) -> Path:
    path = result_path.parent / str(value or "")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def compose_camera_perception_report(
    perception: dict[str, Any],
    result_path: Path,
    output_path: Path,
    *,
    camera: str = "A",
    run_name: str,
    iteration: int,
    target_overlay_path: Path | None = None,
    molmo_annotation_path: Path | None = None,
    selected_reference: dict[str, Any] | None = None,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one exact-pixel RGB/depth/edge/target report sheet.

    The source panels are copied from saved perception artifacts. No generative
    image operation, recoloring, or scientific-data interpolation is applied.
    """

    label = str(camera).strip().upper()
    view = _camera_view(perception, label)
    rgb_path = _source_path(result_path, view.get("image"))
    height_path = _source_path(
        result_path, view.get("height_map", view.get("depth_heatmap"))
    )
    boundary_path = _source_path(
        result_path,
        view.get("height_map_boundary", view.get("depth_heatmap_boundary")),
    )
    edges_path = _source_path(
        result_path,
        view.get("height_gradient_overlay", view.get("fold_edge_overlay")),
    )
    coordinate_path = _source_path(result_path, view.get("coordinate_overlay"))

    rgb = Image.open(rgb_path).convert("RGB")
    panel_w, image_h = rgb.size
    if target_overlay_path is not None and Path(target_overlay_path).is_file():
        target_image = Image.open(target_overlay_path).convert("RGB")
    else:
        target_image = rgb.copy()
        banner_h = max(34, image_h // 12)
        banner = ImageDraw.Draw(target_image)
        banner.rectangle((8, 8, min(panel_w - 8, 390), 8 + banner_h), fill=(0, 0, 0))
        banner.text(
            (16, 15),
            "GRASP TARGET: unknown",
            fill=(255, 210, 40),
            font=_font(max(16, image_h // 28), bold=True),
        )

    molmo_path = Path(molmo_annotation_path) if molmo_annotation_path else None
    molmo_available = bool(molmo_path is not None and molmo_path.is_file())
    if molmo_available:
        height_global_value = view.get(
            "height_map_global", view.get("depth_heatmap_global")
        )
        try:
            height_global_path = _source_path(result_path, height_global_value)
        except FileNotFoundError:
            height_global_path = height_path
        panels = [
            ("(a) RGB observation", rgb),
            ("(b) Height above table", Image.open(height_path).convert("RGB")),
            ("(c) Global height scale", Image.open(height_global_path).convert("RGB")),
            ("(d) Garment boundary", Image.open(boundary_path).convert("RGB")),
            ("(e) Height-gradient / fold edges", Image.open(edges_path).convert("RGB")),
            ("(f) Uniform Rxx references", Image.open(coordinate_path).convert("RGB")),
            ("(g) Molmo zero-shot parts", Image.open(molmo_path).convert("RGB")),
            ("(h) Selected grasp target", target_image),
        ]
        source_paths = [
            rgb_path,
            height_path,
            height_global_path,
            boundary_path,
            edges_path,
            coordinate_path,
            molmo_path,
            Path(target_overlay_path) if target_overlay_path is not None else rgb_path,
        ]
        cols, rows = 4, 2
    else:
        panels = [
            ("(a) RGB observation", rgb),
            ("(b) Height above table", Image.open(height_path).convert("RGB")),
            ("(c) Garment boundary", Image.open(boundary_path).convert("RGB")),
            ("(d) Height-gradient / fold edges", Image.open(edges_path).convert("RGB")),
            ("(e) Uniform Rxx references", Image.open(coordinate_path).convert("RGB")),
            ("(f) Selected grasp target", target_image),
        ]
        source_paths = [
            rgb_path,
            height_path,
            boundary_path,
            edges_path,
            coordinate_path,
            Path(target_overlay_path) if target_overlay_path is not None else rgb_path,
        ]
        cols, rows = 3, 2

    label_h = max(46, image_h // 10)
    panel_h = image_h + label_h
    margin, gap = 30, 22
    header_h, footer_h = 112, 72
    width = margin * 2 + cols * panel_w + (cols - 1) * gap
    height = header_h + rows * panel_h + (rows - 1) * gap + footer_h
    canvas = Image.new("RGB", (width, height), (244, 247, 250))
    draw = ImageDraw.Draw(canvas)

    reference_text = "unknown"
    if selected_reference:
        reference_text = (
            f"{str(selected_reference.get('camera', label)).upper()}/"
            f"{selected_reference.get('reference_id', 'unknown')}"
        )
    draw.text(
        (margin, 20),
        f"Camera {label} Garment Perception and Grasp Planning",
        fill=(20, 33, 50),
        font=_font(34, bold=True),
    )
    draw.text(
        (margin, 66),
        f"Run {run_name} | Iteration {iteration:03d} | Primary RGB-D view | "
        f"Selected reference: {reference_text}",
        fill=(72, 88, 108),
        font=_font(19),
    )

    for index, (panel_label, source) in enumerate(panels):
        row, col = divmod(index, cols)
        x = margin + col * (panel_w + gap)
        y = header_h + row * (panel_h + gap)
        if source.size != (panel_w, image_h):
            source = source.resize((panel_w, image_h), Image.Resampling.LANCZOS)
        canvas.paste(source, (x, y + label_h))
        draw.rounded_rectangle(
            (x, y, x + panel_w - 1, y + panel_h - 1),
            radius=8,
            outline=(157, 169, 184),
            width=2,
        )
        draw.rectangle(
            (x + 1, y + 1, x + panel_w - 2, y + label_h), fill=(25, 42, 62)
        )
        draw.text(
            (x + 16, y + max(8, label_h // 5)),
            panel_label,
            fill=(255, 255, 255),
            font=_font(max(20, label_h // 2), bold=True),
        )

    min_height = view.get("height_map_min_mm")
    max_height = view.get("height_map_max_mm")
    if isinstance(min_height, (int, float)) and isinstance(max_height, (int, float)):
        height_text = f"Height range: {float(min_height):.1f} to {float(max_height):.1f} mm"
    else:
        height_text = "Height range: saved Camera height-above-table scale"
    footer_y = height - footer_h + 15
    draw.text(
        (margin, footer_y),
        height_text
        + " | White: garment boundary | Cyan: strong height gradient / occlusion "
        "| Red: final grasp target",
        fill=(53, 68, 86),
        font=_font(17),
    )
    if target:
        target_text = (
            "Grasp target (robot base): "
            f"X={float(target['x']):.3f} mm, Y={float(target['y']):.3f} mm, "
            f"TCP Z={float(target['z']):.3f} mm, yaw={float(target['yaw']):.1f} deg"
        )
    else:
        target_text = "Grasp target: unknown (planning not completed or target rejected)"
    draw.text(
        (margin, footer_y + 28),
        target_text,
        fill=(53, 68, 86),
        font=_font(17),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.png")
    canvas.save(temporary, format="PNG", dpi=(300, 300), optimize=True)
    temporary.replace(output)
    return {
        "camera": label,
        "iteration": int(iteration),
        "image": str(output),
        "size_px": [width, height],
        "selected_reference": dict(selected_reference or {}),
        "target": dict(target or {}),
        "molmo_annotation": str(molmo_path) if molmo_available else None,
        "source_result": str(Path(result_path)),
        "source_panels": [str(path) for path in source_paths],
        "generation_mode": "deterministic_exact_pixel_composite",
    }
