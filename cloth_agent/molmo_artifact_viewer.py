"""Lightweight live image viewer for the headless Molmo keypoint CLI.

This process only polls already-saved JSON/PNG artifacts.  It does not open a
camera, connect to the robot, load Molmo/Claude, or create a WebGL/Viser server.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


class ArtifactViewerError(RuntimeError):
    """Raised when no readable CLI artifact run can be found."""


@dataclass(frozen=True)
class ArtifactPage:
    name: str
    tiles: tuple[tuple[str, "ArtifactSource"], ...]


@dataclass(frozen=True)
class RawDepthArtifact:
    path: Path


@dataclass(frozen=True)
class RefinedHeightArtifact:
    heatmap_path: Path
    rgb_path: Path
    height_path: Path
    minimum_table_color_distance: float


@dataclass(frozen=True)
class RefinedCoordinateArtifact:
    rgb_path: Path
    heatmap_path: Path
    height_path: Path
    guide_path: Path
    minimum_table_color_distance: float


ArtifactSource = (
    Path | RawDepthArtifact | RefinedHeightArtifact | RefinedCoordinateArtifact | None
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def discover_output_dir(source: Path) -> Path | None:
    """Resolve a run directory or exact CLI result directory to its latest run."""

    source = Path(source).expanduser().resolve()
    if (source / "events.jsonl").is_file() or (source / "summary.json").is_file():
        return source

    search_roots = [
        source / "results" / "molmo_keypoint_cli",
        source / "molmo_keypoint_cli",
        source,
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "events.jsonl").is_file() or (child / "summary.json").is_file():
                candidates.append(child.resolve())
        if candidates:
            break
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: max(
            (path / "events.jsonl").stat().st_mtime_ns
            if (path / "events.jsonl").exists()
            else 0,
            (path / "summary.json").stat().st_mtime_ns
            if (path / "summary.json").exists()
            else 0,
        ),
    )


def _latest_iteration_dir(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("iteration_[0-9][0-9][0-9]"):
        if not path.is_dir():
            continue
        try:
            number = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        candidates.append((number, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _first_existing(*paths: Path) -> Path | None:
    return next((path.resolve() for path in paths if path.is_file()), None)


def _camera_raw(directory: Path, camera: str) -> Path | None:
    return next(
        (
            path.resolve()
            for path in sorted(directory.glob(f"camera_[0-9]*_{camera}.png"))
            if path.is_file()
        ),
        None,
    )


def _last_event(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            return event
    return {}


def build_pages(output_dir: Path) -> tuple[list[ArtifactPage], dict[str, Any]]:
    """Return the current viewer pages and status without opening a GUI."""

    output_dir = output_dir.resolve()
    summary = _load_json(output_dir / "summary.json")
    iteration_dir = _latest_iteration_dir(output_dir)
    record = _load_json(iteration_dir / "result.json") if iteration_dir else {}

    saved_result_text = record.get("saved_perception_result")
    saved_result_path = (
        Path(saved_result_text).expanduser().resolve()
        if isinstance(saved_result_text, str) and saved_result_text
        else None
    )
    perception_dir = (
        saved_result_path.parent
        if saved_result_path is not None
        else Path()
    )
    if not perception_dir.is_dir():
        run_dir_text = summary.get("run_dir")
        perception_dir = (
            Path(run_dir_text).expanduser().resolve() / "workspace" / "perception_views"
            if isinstance(run_dir_text, str) and run_dir_text
            else output_dir
        )

    molmo_dir = iteration_dir / "molmo_keypoints" if iteration_dir else output_dir
    after_dir = iteration_dir / "after_capture" if iteration_dir else output_dir
    before_a = _camera_raw(perception_dir, "A")
    before_b = _camera_raw(perception_dir, "B")
    after_a = _camera_raw(after_dir, "A")
    after_b = _camera_raw(after_dir, "B")
    perception_result = (
        _load_json(saved_result_path)
        if saved_result_path is not None and saved_result_path.is_file()
        else {}
    )
    depth_fusion = perception_result.get("depth_fusion", {})
    minimum_color_distance = (
        float(depth_fusion.get("garment_color_distance_threshold", 24.0))
        if isinstance(depth_fusion, dict)
        else 24.0
    )

    def raw_depth(camera: str, index: int) -> ArtifactSource:
        path = perception_dir / f"camera_{index}_{camera}_depth_m.npy"
        return RawDepthArtifact(path.resolve()) if path.is_file() else None

    def refined_height(camera: str, rgb_path: Path | None) -> ArtifactSource:
        heatmap_path = perception_dir / f"camera_{camera}_height_map_heatmap.png"
        height_path = perception_dir / f"camera_{camera}_height_above_table_mm.npy"
        if rgb_path is None or not heatmap_path.is_file() or not height_path.is_file():
            return _first_existing(
                perception_dir / f"camera_{camera}_height_map_boundary.png",
                heatmap_path,
            )
        return RefinedHeightArtifact(
            heatmap_path=heatmap_path.resolve(),
            rgb_path=rgb_path.resolve(),
            height_path=height_path.resolve(),
            minimum_table_color_distance=minimum_color_distance,
        )

    def refined_coordinates(camera: str, rgb_path: Path | None) -> ArtifactSource:
        heatmap_path = perception_dir / f"camera_{camera}_height_map_heatmap.png"
        height_path = perception_dir / f"camera_{camera}_height_above_table_mm.npy"
        guide_path = perception_dir / f"camera_{camera}_coordinate_guide.json"
        required = (heatmap_path, height_path, guide_path)
        if rgb_path is None or not all(path.is_file() for path in required):
            return _first_existing(
                perception_dir / f"camera_{camera}_coordinate_overlay.png"
            )
        return RefinedCoordinateArtifact(
            rgb_path=rgb_path.resolve(),
            heatmap_path=heatmap_path.resolve(),
            height_path=height_path.resolve(),
            guide_path=guide_path.resolve(),
            minimum_table_color_distance=minimum_color_distance,
        )

    pages = [
        ArtifactPage(
            "Overview",
            (
                ("Camera A · before RGB", before_a),
                ("Camera B · before RGB", before_b),
                (
                    "Fused height/boundary",
                    _first_existing(
                        perception_dir / "fused_height_map_boundary.png",
                        perception_dir / "fused_height_map_heatmap.png",
                    ),
                ),
                (
                    "Camera A · accepted Molmo Rxxx",
                    _first_existing(
                        molmo_dir / "camera_A_molmo_keypoint_references.png"
                    ),
                ),
                (
                    "Camera B · accepted Molmo Rxxx",
                    _first_existing(
                        molmo_dir / "camera_B_molmo_keypoint_references.png"
                    ),
                ),
                ("Camera A · after RGB", after_a),
            ),
        ),
        ArtifactPage(
            "Perception",
            (
                ("Camera A · RGB", before_a),
                ("Camera A · raw depth (camera Z)", raw_depth("A", 0)),
                ("Camera A · height above table (garment only)", refined_height("A", before_a)),
                ("Camera B · RGB", before_b),
                ("Camera B · raw depth (camera Z)", raw_depth("B", 1)),
                ("Camera B · height above table (garment only)", refined_height("B", before_b)),
            ),
        ),
        ArtifactPage(
            "Molmo keypoints",
            (
                (
                    "Camera A · all candidates",
                    _first_existing(
                        molmo_dir / "camera_A_molmo_keypoint_candidates.png"
                    ),
                ),
                (
                    "Camera A · accepted only",
                    _first_existing(
                        molmo_dir / "camera_A_molmo_keypoint_references.png"
                    ),
                ),
                (
                    "Camera A · calibrated coordinates",
                    refined_coordinates("A", before_a),
                ),
                (
                    "Camera B · all candidates",
                    _first_existing(
                        molmo_dir / "camera_B_molmo_keypoint_candidates.png"
                    ),
                ),
                (
                    "Camera B · accepted only",
                    _first_existing(
                        molmo_dir / "camera_B_molmo_keypoint_references.png"
                    ),
                ),
                (
                    "Camera B · calibrated coordinates",
                    refined_coordinates("B", before_b),
                ),
            ),
        ),
        ArtifactPage(
            "Before / after",
            (
                ("Camera A · before", before_a),
                ("Camera A · after", after_a),
                (
                    "Camera A · selected references",
                    _first_existing(
                        molmo_dir / "camera_A_molmo_keypoint_references.png"
                    ),
                ),
                ("Camera B · before", before_b),
                ("Camera B · after", after_b),
                (
                    "Camera B · selected references",
                    _first_existing(
                        molmo_dir / "camera_B_molmo_keypoint_references.png"
                    ),
                ),
            ),
        ),
    ]
    event = _last_event(output_dir / "events.jsonl")
    status = {
        "run_status": summary.get("status", record.get("status", "WAITING")),
        "iteration": event.get("iteration", record.get("iteration")),
        "phase": event.get("phase", record.get("last_completed_stage", "waiting")),
        "level": event.get("level", "INFO"),
        "message": event.get("message", "waiting for saved artifacts"),
        "local_time": event.get("local_time"),
        "run_elapsed_s": event.get("run_elapsed_s"),
        "iteration_dir": str(iteration_dir) if iteration_dir else None,
    }
    return pages, status


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _raw_depth_image(artifact: RawDepthArtifact) -> Image.Image:
    values = np.asarray(np.load(artifact.path), dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not valid.any():
        return Image.fromarray(rgb)
    low, high = np.percentile(values[valid], [2.0, 98.0])
    scale = max(1e-9, float(high - low))
    normalized = np.clip((values - low) / scale, 0.0, 1.0)
    # Near surfaces are warm/bright and far surfaces are cool/dark. This is
    # camera-Z depth in metres, not height relative to the table.
    centers = np.linspace(0.0, 1.0, 5)
    colors = np.asarray(
        [
            [255, 245, 180],
            [255, 135, 45],
            [195, 40, 70],
            [75, 25, 125],
            [20, 10, 50],
        ],
        dtype=np.float64,
    )
    for channel in range(3):
        rgb[..., channel][valid] = np.rint(
            np.interp(normalized[valid], centers, colors[:, channel])
        ).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 188, 33), radius=5, fill=(8, 10, 14))
    draw.text(
        (15, 13),
        f"camera Z: {low:.3f}-{high:.3f} m",
        font=_font(14),
        fill=(235, 240, 245),
    )
    return image


def _refined_garment_mask(
    heatmap_path: Path,
    rgb_path: Path,
    height_path: Path,
    minimum_table_color_distance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from .perception import (
        _camera_table_appearance_mask,
        _solidify_largest_mask,
    )

    heatmap = np.asarray(Image.open(heatmap_path).convert("RGB"))
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    height = np.asarray(np.load(height_path), dtype=np.float64)
    valid = np.isfinite(height)
    appearance, diagnostics = _camera_table_appearance_mask(
        rgb,
        height,
        valid,
        minimum_color_distance=minimum_table_color_distance,
    )
    rendered = np.any(heatmap != 0, axis=2)
    garment = _solidify_largest_mask(
        rendered & appearance,
        dilation_iterations=0,
        closing_iterations=0,
        fill_holes=False,
    )
    return garment, diagnostics


def _refined_height_image(artifact: RefinedHeightArtifact) -> Image.Image:
    heatmap = np.asarray(Image.open(artifact.heatmap_path).convert("RGB")).copy()
    garment, diagnostics = _refined_garment_mask(
        artifact.heatmap_path,
        artifact.rgb_path,
        artifact.height_path,
        artifact.minimum_table_color_distance,
    )
    heatmap[~garment] = 0
    image = Image.fromarray(heatmap)
    draw = ImageDraw.Draw(image)
    threshold = diagnostics.get("applied_color_distance")
    annotation = "height = surface Z - table Z"
    if threshold is not None:
        annotation += f" · table filter {float(threshold):.0f}"
    draw.rounded_rectangle((8, 8, 286, 33), radius=5, fill=(8, 10, 14))
    draw.text((15, 13), annotation, font=_font(14), fill=(235, 240, 245))
    return image


def _refined_coordinate_image(artifact: RefinedCoordinateArtifact) -> Image.Image:
    image = Image.open(artifact.rgb_path).convert("RGB")
    garment, _ = _refined_garment_mask(
        artifact.heatmap_path,
        artifact.rgb_path,
        artifact.height_path,
        artifact.minimum_table_color_distance,
    )
    guide = _load_json(artifact.guide_path)
    samples = guide.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    draw = ImageDraw.Draw(image)
    kept = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        pixel = sample.get("pixel_xy")
        if not isinstance(pixel, list) or len(pixel) != 2:
            continue
        x_px, y_px = int(pixel[0]), int(pixel[1])
        if not (
            0 <= y_px < garment.shape[0]
            and 0 <= x_px < garment.shape[1]
            and garment[y_px, x_px]
        ):
            continue
        reference_id = str(sample.get("reference_id", f"R{kept + 1:03d}"))
        radius = 4
        draw.ellipse(
            (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
            fill=(0, 255, 255),
            outline=(0, 0, 0),
            width=1,
        )
        draw.text(
            (x_px + 5, y_px - 7),
            reference_id,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            font=_font(13),
        )
        kept += 1
    draw.rounded_rectangle((8, 8, 260, 33), radius=5, fill=(8, 10, 14))
    draw.text(
        (15, 13),
        f"mask-filtered references: {kept}/{len(samples)}",
        font=_font(14),
        fill=(235, 240, 245),
    )
    return image


def _artifact_image(source: ArtifactSource) -> Image.Image | None:
    if source is None:
        return None
    if isinstance(source, RawDepthArtifact):
        return _raw_depth_image(source)
    if isinstance(source, RefinedHeightArtifact):
        return _refined_height_image(source)
    if isinstance(source, RefinedCoordinateArtifact):
        return _refined_coordinate_image(source)
    with Image.open(source) as image:
        return image.convert("RGB").copy()


def _artifact_name(source: ArtifactSource) -> str | None:
    if isinstance(source, RawDepthArtifact):
        return source.path.name
    if isinstance(source, RefinedHeightArtifact):
        return f"{source.heatmap_path.name} · appearance-refined"
    if isinstance(source, RefinedCoordinateArtifact):
        return f"{source.guide_path.name} · mask-filtered"
    if isinstance(source, Path):
        return source.name
    return None


def _fit_tile(source: ArtifactSource, size: tuple[int, int]) -> Image.Image:
    width, height = size
    background = Image.new("RGB", size, (18, 22, 28))
    if source is None:
        draw = ImageDraw.Draw(background)
        text = "Waiting for artifact..."
        box = draw.textbbox((0, 0), text, font=_font(18))
        draw.text(
            ((width - (box[2] - box[0])) // 2, (height - (box[3] - box[1])) // 2),
            text,
            fill=(125, 135, 148),
            font=_font(18),
        )
        return background
    try:
        source_image = _artifact_image(source)
        if source_image is None:
            return _fit_tile(None, size)
        image = ImageOps.contain(source_image, size)
    except (OSError, ValueError, RuntimeError):
        return _fit_tile(None, size)
    background.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return background


def render_dashboard(
    output_dir: Path,
    page_index: int = 0,
    *,
    width: int = 1440,
    height: int = 900,
) -> tuple[Image.Image, int]:
    """Render one current dashboard frame for the GUI or a PNG snapshot."""

    if width < 720 or height < 480:
        raise ValueError("viewer canvas must be at least 720x480")
    pages, status = build_pages(output_dir)
    page_index = page_index % len(pages)
    page = pages[page_index]
    canvas = Image.new("RGB", (width, height), (10, 14, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25)
    body_font = _font(17)
    label_font = _font(16)
    colors = {
        "START": (84, 167, 255),
        "DONE": (71, 203, 129),
        "PASS": (71, 203, 129),
        "WAIT": (191, 128, 255),
        "WARNING": (255, 196, 77),
        "ERROR": (255, 91, 91),
    }
    accent = colors.get(str(status["level"]), (120, 200, 220))
    iteration = status["iteration"]
    iteration_text = f"I{int(iteration):03d}" if iteration else "RUN"
    elapsed = status.get("run_elapsed_s")
    elapsed_text = f"+{float(elapsed):.1f}s" if elapsed is not None else "+--"
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    draw.text(
        (22, 14),
        "ClothAgent · Lightweight Artifact Viewer",
        font=title_font,
        fill=(235, 241, 248),
    )
    draw.text(
        (width - 310, 18),
        now,
        font=body_font,
        fill=(155, 165, 178),
    )
    status_line = (
        f"{iteration_text}   {str(status['phase']).upper()}   {status['level']}   "
        f"{elapsed_text}   run={status['run_status']}"
    )
    draw.rounded_rectangle((20, 52, width - 20, 86), radius=7, fill=(25, 31, 40))
    draw.text((32, 59), status_line, font=body_font, fill=accent)
    message = str(status.get("message") or "")
    if len(message) > 155:
        message = message[:152] + "..."
    draw.text((24, 94), message, font=body_font, fill=(190, 199, 210))
    draw.text(
        (24, 119),
        f"Page {page_index + 1}/{len(pages)} · {page.name}    keys: 1-4 pages  ←/→ switch  r refresh  q/Esc close",
        font=body_font,
        fill=(125, 185, 225),
    )

    columns, rows = 3, 2
    gap = 12
    top = 151
    tile_width = (width - gap * (columns + 1)) // columns
    tile_height = (height - top - gap * (rows + 1)) // rows
    label_height = 28
    for index in range(columns * rows):
        row, column = divmod(index, columns)
        x = gap + column * (tile_width + gap)
        y = top + gap + row * (tile_height + gap)
        label, source = page.tiles[index]
        draw.rounded_rectangle(
            (x, y, x + tile_width, y + tile_height),
            radius=6,
            fill=(25, 31, 40),
        )
        draw.text((x + 9, y + 5), label, font=label_font, fill=(222, 228, 235))
        tile = _fit_tile(source, (tile_width - 4, tile_height - label_height - 4))
        canvas.paste(tile, (x + 2, y + label_height))
        filename = _artifact_name(source)
        if filename is not None:
            draw.rectangle(
                (x + 2, y + tile_height - 23, x + tile_width - 2, y + tile_height - 2),
                fill=(10, 14, 20),
            )
            draw.text(
                (x + 8, y + tile_height - 22),
                filename,
                font=_font(13),
                fill=(145, 155, 168),
            )
    return canvas, page_index


def run_viewer(
    source: Path,
    *,
    refresh_s: float = 1.0,
    page_index: int = 0,
    width: int = 1440,
    height: int = 900,
) -> int:
    """Open the read-only OpenCV window and follow the latest CLI artifacts."""

    if not 0.1 <= refresh_s <= 30.0:
        raise ValueError("refresh_s must be between 0.1 and 30 seconds")
    import cv2

    # Some OpenCV wheels point Qt at a bundled font directory that is absent.
    # Use the host DejaVu fonts so starting this tiny viewer does not spam the
    # CLI with repeated QFontDatabase warnings.
    qt_font_dir = Path(str(os.environ.get("QT_QPA_FONTDIR", "")))
    system_font_dir = Path("/usr/share/fonts/truetype/dejavu")
    if not qt_font_dir.is_dir() and system_font_dir.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(system_font_dir)

    window = "ClothAgent lightweight artifacts (q/Esc to close)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, width, height)
    print(
        "Lightweight viewer started: saved PNG/JSON only; no camera, robot, "
        "Molmo, CUDA, browser, or Viser.",
        flush=True,
    )
    last_output: Path | None = None
    try:
        while True:
            output_dir = discover_output_dir(source)
            if output_dir is None:
                frame = Image.new("RGB", (width, height), (10, 14, 20))
                draw = ImageDraw.Draw(frame)
                draw.text(
                    (35, 35),
                    f"Waiting for a Molmo CLI run under:\n{source.expanduser().resolve()}",
                    fill=(220, 225, 232),
                    font=_font(22),
                )
            else:
                if output_dir != last_output:
                    print(f"Watching: {output_dir}", flush=True)
                    last_output = output_dir
                frame, page_index = render_dashboard(
                    output_dir,
                    page_index,
                    width=width,
                    height=height,
                )
            bgr = np.asarray(frame, dtype=np.uint8)[:, :, ::-1]
            cv2.imshow(window, bgr)
            key = cv2.waitKey(max(100, int(refresh_s * 1000))) & 0xFF
            if key in {27, ord("q")}:
                break
            if ord("1") <= key <= ord("4"):
                page_index = key - ord("1")
            elif key in {81, ord("a")}:
                page_index -= 1
            elif key in {83, ord("d")}:
                page_index += 1
            elif key == ord("r"):
                continue
    finally:
        cv2.destroyAllWindows()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="run directory or exact results/molmo_keypoint_cli/<timestamp> directory",
    )
    parser.add_argument("--refresh-s", type=float, default=1.0)
    parser.add_argument("--page", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="save the selected dashboard page as PNG and exit without opening a window",
    )
    args = parser.parse_args(argv)
    if args.snapshot:
        output_dir = discover_output_dir(args.source)
        if output_dir is None:
            raise ArtifactViewerError(f"no Molmo CLI artifacts found under {args.source}")
        image, _ = render_dashboard(
            output_dir,
            args.page - 1,
            width=args.width,
            height=args.height,
        )
        snapshot = args.snapshot.expanduser().resolve()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        image.save(snapshot)
        print(f"Saved artifact dashboard: {snapshot}", flush=True)
        return 0
    try:
        return run_viewer(
            args.source,
            refresh_s=args.refresh_s,
            page_index=args.page - 1,
            width=args.width,
            height=args.height,
        )
    except Exception as exc:
        print(f"Artifact viewer failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
