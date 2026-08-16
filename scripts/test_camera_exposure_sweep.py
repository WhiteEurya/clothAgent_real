#!/usr/bin/env python3
"""Capture a labeled RGB exposure sweep from one configured RealSense camera."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.perception import PerceptionConfig


def _resolve(root: Path, raw: Path) -> Path:
    return raw.expanduser().resolve() if raw.is_absolute() else (root / raw).resolve()


def _save_grid(
    captures: list[tuple[int, Image.Image]],
    output_path: Path,
    *,
    columns: int = 5,
    tile_width: int = 320,
) -> None:
    if not captures:
        raise RuntimeError("exposure sweep produced no images")
    source_width, source_height = captures[0][1].size
    tile_height = int(round(tile_width * source_height / source_width))
    label_height = 32
    rows = (len(captures) + columns - 1) // columns
    grid = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        (12, 12, 14),
    )
    draw = ImageDraw.Draw(grid)
    for index, (exposure, image) in enumerate(captures):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        resized = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        grid.paste(resized, (x, y + label_height))
        draw.text((x + 10, y + 9), f"Exposure {exposure}", fill=(255, 255, 255))
    grid.save(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument("--camera", default="A")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=800)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--settle-frames", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.step <= 0 or args.end < args.start or args.settle_frames < 1:
        raise ValueError("expected positive step/settle-frames and end >= start")
    root = args.project_root.expanduser().resolve()
    perception = PerceptionConfig.load(root, _resolve(root, args.perception_config))
    label = str(args.camera).strip().upper()
    spec = next((item for item in perception.cameras if item.label == label), None)
    if spec is None:
        raise ValueError(f"camera {label!r} is not configured")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "results" / "exposure_sweep" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(spec.serial)
    rs_config.enable_stream(
        rs.stream.color,
        perception.width,
        perception.height,
        rs.format.rgb8,
        perception.fps,
    )
    profile = pipeline.start(rs_config)
    sensor = next(
        (
            item
            for item in profile.get_device().query_sensors()
            if item.get_info(rs.camera_info.name) == "RGB Camera"
        ),
        None,
    )
    if sensor is None:
        pipeline.stop()
        raise RuntimeError(f"camera {label} has no RGB Camera sensor")

    exposures = list(range(args.start, args.end + 1, args.step))
    captures: list[tuple[int, Image.Image]] = []
    records: list[dict[str, object]] = []
    restore_exposure = float(spec.color_exposure or args.end)
    try:
        sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        for _ in range(args.settle_frames):
            pipeline.wait_for_frames()
        for requested in exposures:
            sensor.set_option(rs.option.exposure, float(requested))
            for _ in range(args.settle_frames):
                frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"camera {label} returned no RGB frame at {requested}")
            actual = float(sensor.get_option(rs.option.exposure))
            rgb = np.asanyarray(color_frame.get_data()).copy()
            image = Image.fromarray(rgb).convert("RGB")
            image_name = f"exposure_{requested:04d}.png"
            image.save(output_dir / image_name)
            captures.append((requested, image))
            records.append(
                {
                    "requested_exposure": requested,
                    "actual_exposure": actual,
                    "image": image_name,
                    "rgb_mean": float(rgb.mean()),
                    "rgb_p95": float(np.percentile(rgb, 95)),
                    "saturated_fraction": float(np.mean(rgb >= 250)),
                }
            )
    finally:
        try:
            sensor.set_option(rs.option.exposure, restore_exposure)
        finally:
            pipeline.stop()

    grid_path = output_dir / "exposure_grid.png"
    _save_grid(captures, grid_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "camera_label": label,
                "camera_serial": spec.serial,
                "auto_exposure": False,
                "exposures": exposures,
                "settle_frames": args.settle_frames,
                "restored_exposure": restore_exposure,
                "grid": grid_path.name,
                "captures": records,
                "robot_motion": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Exposure sweep complete: {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
