#!/usr/bin/env python3
"""Recompute the complete garment height map across an RGB exposure sweep."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import ExperimentConfig, RobotConfig
from cloth_agent.perception import (
    ClothCenterPerception,
    PerceptionConfig,
    RGBDFrame,
    _scalar_heatmap_rgb,
    capture_two_view_rgbd,
)


def _resolve(root: Path, raw: Path) -> Path:
    return raw.expanduser().resolve() if raw.is_absolute() else (root / raw).resolve()


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _save_raw_capture(frames: list[RGBDFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        image_name = f"camera_{index}_{frame.label}.png"
        depth_name = f"camera_{index}_{frame.label}_depth_m.npy"
        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(
            output_dir / image_name
        )
        np.save(output_dir / depth_name, np.asarray(frame.depth_m, dtype=np.float32))
        records.append(
            {
                "label": frame.label,
                "serial": frame.serial,
                "image": image_name,
                "depth_m": depth_name,
                "intrinsics": np.asarray(frame.intrinsics, dtype=np.float64).tolist(),
                "X_base_camera": np.asarray(
                    frame.X_base_camera, dtype=np.float64
                ).tolist(),
            }
        )
    _save_json(
        output_dir / "capture_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "frames": records,
        },
    )


def _save_grid(
    items: list[tuple[int, str, Path]],
    output_path: Path,
    *,
    columns: int = 5,
    tile_size: tuple[int, int] = (320, 240),
) -> None:
    if not items:
        raise RuntimeError("height-map exposure sweep produced no grid images")
    label_height = 38
    rows = (len(items) + columns - 1) // columns
    grid = Image.new(
        "RGB",
        (columns * tile_size[0], rows * (tile_size[1] + label_height)),
        (10, 10, 12),
    )
    draw = ImageDraw.Draw(grid)
    for index, (exposure, detail, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail(tile_size, Image.Resampling.LANCZOS)
        x0 = (index % columns) * tile_size[0]
        y0 = (index // columns) * (tile_size[1] + label_height)
        x = x0 + (tile_size[0] - image.width) // 2
        y = y0 + label_height + (tile_size[1] - image.height) // 2
        grid.paste(image, (x, y))
        draw.text(
            (x0 + 9, y0 + 7),
            f"Exposure {exposure} | {detail}",
            fill=(255, 255, 255),
        )
    grid.save(output_path)


def _restore_exposure(serial: str, exposure: float) -> None:
    import pyrealsense2 as rs

    device = next(
        (
            item
            for item in rs.context().query_devices()
            if item.get_info(rs.camera_info.serial_number) == serial
        ),
        None,
    )
    if device is None:
        return
    sensor = next(
        (
            item
            for item in device.query_sensors()
            if item.get_info(rs.camera_info.name) == "RGB Camera"
        ),
        None,
    )
    if sensor is not None:
        sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        sensor.set_option(rs.option.exposure, float(exposure))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument(
        "--robot-config", type=Path, default=Path("config/robot.example.json")
    )
    parser.add_argument("--camera", default="A")
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=800)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--temporal-median-frames", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.step <= 0 or args.end < args.start:
        raise ValueError("expected positive step and end >= start")
    root = args.project_root.expanduser().resolve()
    perception = PerceptionConfig.load(root, _resolve(root, args.perception_config))
    robot = RobotConfig.load(root, _resolve(root, args.robot_config))
    if args.temporal_median_frames is not None:
        perception = replace(
            perception,
            temporal_median_frames=int(args.temporal_median_frames),
        )
        perception.validate()
    label = str(args.camera).strip().upper()
    target = next((item for item in perception.cameras if item.label == label), None)
    if target is None:
        raise ValueError(f"camera {label!r} is not configured")
    restore_exposure = float(target.color_exposure or args.end)
    exposures = list(range(args.start, args.end + 1, args.step))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "results" / "heightmap_exposure_sweep" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    heatmap_items: list[tuple[int, str, Path]] = []
    rgb_items: list[tuple[int, str, Path]] = []
    records: list[dict[str, Any]] = []
    try:
        for exposure in exposures:
            print(f"Capturing and computing height map at exposure {exposure}...")
            cameras = tuple(
                replace(item, color_exposure=float(exposure))
                if item.label == label
                else item
                for item in perception.cameras
            )
            exposure_config = replace(perception, cameras=cameras)
            exposure_dir = output_dir / f"exposure_{exposure:04d}"
            exposure_dir.mkdir(parents=True, exist_ok=False)
            try:
                frames = capture_two_view_rgbd(exposure_config)
                _save_raw_capture(frames, exposure_dir / "raw_capture")
                service = ClothCenterPerception(root, robot, exposure_config)
                result, _ = service.locate(
                    exposure_dir / "pipeline",
                    ExperimentConfig(),
                    frames=frames,
                )
                view = next(item for item in result["views"] if item["label"] == label)
                fusion = result["depth_fusion"]
                p95 = float(fusion["garment_height_p95_mm"])
                detail = f"p95={p95:.1f} mm"
                heatmap_path = exposure_dir / "pipeline" / view["height_map"]
                rgb_path = exposure_dir / "pipeline" / view["image"]
                heatmap_items.append((exposure, detail, heatmap_path))
                rgb_items.append((exposure, detail, rgb_path))
                records.append(
                    {
                        "exposure": exposure,
                        "status": "COMPLETE",
                        "result": str(
                            (exposure_dir / "pipeline" / "result.json").relative_to(
                                output_dir
                            )
                        ),
                        "height_map": str(heatmap_path.relative_to(output_dir)),
                        "height_map_scalar": str(
                            (
                                exposure_dir
                                / "pipeline"
                                / view["height_map_path"]
                            ).relative_to(output_dir)
                        ),
                        "rgb": str(rgb_path.relative_to(output_dir)),
                        "garment_point_count": fusion["garment_point_count"],
                        "garment_height_p50_mm": fusion["garment_height_p50_mm"],
                        "garment_height_p95_mm": p95,
                        "heatmap_display_max_mm": fusion["heatmap_display_max_mm"],
                        "table_plane": fusion["table_plane"],
                    }
                )
            except BaseException as exc:
                failure = {
                    "exposure": exposure,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                records.append(failure)
                _save_json(exposure_dir / "failure.json", failure)
    finally:
        _restore_exposure(target.serial, restore_exposure)

    per_capture_grid = output_dir / f"camera_{label}_heightmap_per_capture_scale_grid.png"
    _save_grid(heatmap_items, per_capture_grid)
    shared_display_max_mm = max(
        float(item["heatmap_display_max_mm"])
        for item in records
        if item["status"] == "COMPLETE"
    )
    shared_heatmap_items: list[tuple[int, str, Path]] = []
    for item in records:
        if item["status"] != "COMPLETE":
            continue
        exposure = int(item["exposure"])
        scalar = np.load(output_dir / str(item["height_map_scalar"]))
        original_heatmap = np.asarray(
            Image.open(output_dir / str(item["height_map"])).convert("RGB")
        )
        garment_mask = np.any(original_heatmap != 0, axis=2)
        shared_heatmap = _scalar_heatmap_rgb(
            scalar,
            np.isfinite(scalar),
            focus_mask=garment_mask,
            higher_is_bright=True,
            value_range_mm=(0.0, shared_display_max_mm),
        )
        shared_path = (
            output_dir
            / f"exposure_{exposure:04d}"
            / "pipeline"
            / f"camera_{label}_heightmap_shared_scale.png"
        )
        Image.fromarray(shared_heatmap).save(shared_path)
        item["shared_scale_height_map"] = str(shared_path.relative_to(output_dir))
        shared_heatmap_items.append(
            (
                exposure,
                f"p95={float(item['garment_height_p95_mm']):.1f} | 0..{shared_display_max_mm:g} mm",
                shared_path,
            )
        )
    heatmap_grid = output_dir / f"camera_{label}_heightmap_grid.png"
    _save_grid(shared_heatmap_items, heatmap_grid)
    rgb_grid = output_dir / f"camera_{label}_rgb_grid.png"
    _save_grid(rgb_items, rgb_grid)
    _save_json(
        output_dir / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "camera_label": label,
            "camera_serial": target.serial,
            "exposures": exposures,
            "temporal_median_frames": perception.temporal_median_frames,
            "restored_exposure": restore_exposure,
            "heightmap_grid": heatmap_grid.name,
            "heightmap_per_capture_scale_grid": per_capture_grid.name,
            "shared_display_min_mm": 0.0,
            "shared_display_max_mm": shared_display_max_mm,
            "rgb_grid": rgb_grid.name,
            "robot_motion": False,
            "runs": records,
        },
    )
    complete = sum(item["status"] == "COMPLETE" for item in records)
    print(f"Height-map exposure sweep complete: {complete}/{len(exposures)}")
    print(f"Height-map grid: {heatmap_grid}")
    return 0 if complete == len(exposures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
