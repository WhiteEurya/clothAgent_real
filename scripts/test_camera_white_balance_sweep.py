#!/usr/bin/env python3
"""Capture a labeled manual-white-balance sweep from one RealSense RGB camera."""

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
        raise RuntimeError("white-balance sweep produced no images")
    source_width, source_height = captures[0][1].size
    tile_height = int(round(tile_width * source_height / source_width))
    label_height = 34
    rows = (len(captures) + columns - 1) // columns
    grid = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        (12, 12, 14),
    )
    draw = ImageDraw.Draw(grid)
    for index, (white_balance, image) in enumerate(captures):
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        resized = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        grid.paste(resized, (x, y + label_height))
        draw.text((x + 10, y + 10), f"White balance {white_balance} K", fill=(255, 255, 255))
    grid.save(output_path)


def _bright_region_metrics(rgb: np.ndarray) -> dict[str, object]:
    values = rgb.astype(np.float64)
    luminance = values.mean(axis=2)
    threshold = float(np.percentile(luminance, 75))
    mask = luminance >= threshold
    mean_rgb = values[mask].mean(axis=0)
    normalized = mean_rgb / max(float(mean_rgb.mean()), 1.0)
    return {
        "bright_region_threshold": threshold,
        "bright_region_mean_rgb": mean_rgb.tolist(),
        "bright_region_normalized_rgb": normalized.tolist(),
        "bright_region_neutral_error": float(np.abs(normalized - 1.0).sum()),
        "saturated_fraction": float(np.mean(rgb >= 250)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument("--camera", default="A")
    parser.add_argument("--start", type=int, default=2800)
    parser.add_argument("--end", type=int, default=6400)
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--settle-frames", type=int, default=12)
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
        else root / "results" / "white_balance_sweep" / stamp / f"camera_{label}"
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
    required = (rs.option.enable_auto_white_balance, rs.option.white_balance)
    if any(not sensor.supports(option) for option in required):
        pipeline.stop()
        raise RuntimeError(f"camera {label} does not support manual white balance")

    white_balances = list(range(args.start, args.end + 1, args.step))
    captures: list[tuple[int, Image.Image]] = []
    records: list[dict[str, object]] = []
    restore_auto = float(sensor.get_option(rs.option.enable_auto_white_balance))
    restore_white_balance = float(sensor.get_option(rs.option.white_balance))
    exposure = (
        float(sensor.get_option(rs.option.exposure))
        if sensor.supports(rs.option.exposure)
        else None
    )
    gain = (
        float(sensor.get_option(rs.option.gain))
        if sensor.supports(rs.option.gain)
        else None
    )
    try:
        sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
        for _ in range(args.settle_frames):
            pipeline.wait_for_frames()
        for requested in white_balances:
            sensor.set_option(rs.option.white_balance, float(requested))
            for _ in range(args.settle_frames):
                frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"camera {label} returned no RGB frame at {requested} K")
            actual = float(sensor.get_option(rs.option.white_balance))
            rgb = np.asanyarray(color_frame.get_data()).copy()
            image = Image.fromarray(rgb).convert("RGB")
            image_name = f"white_balance_{requested:04d}K.png"
            image.save(output_dir / image_name)
            captures.append((requested, image))
            records.append(
                {
                    "requested_white_balance_k": requested,
                    "actual_white_balance_k": actual,
                    "image": image_name,
                    **_bright_region_metrics(rgb),
                }
            )
    finally:
        try:
            sensor.set_option(rs.option.white_balance, restore_white_balance)
            sensor.set_option(rs.option.enable_auto_white_balance, restore_auto)
        finally:
            pipeline.stop()

    grid_path = output_dir / "white_balance_grid.png"
    _save_grid(captures, grid_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "camera_label": label,
                "camera_serial": spec.serial,
                "exposure": exposure,
                "gain": gain,
                "white_balances_k": white_balances,
                "settle_frames": args.settle_frames,
                "restored_auto_white_balance": restore_auto,
                "restored_white_balance_k": restore_white_balance,
                "grid": grid_path.name,
                "captures": records,
                "robot_motion": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"White-balance sweep complete: {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
