#!/usr/bin/env python3
"""Run and archive the complete garment height-map/heatmap pipeline.

This diagnostic is perception-only: it captures or reloads calibrated A/B
RGB-D frames, fits the table, segments the garment, computes per-pixel and
fused height-above-table maps, and saves every intermediate/result artifact.
It never connects to or commands the xArm.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    capture_two_view_rgbd,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _resolve(root: Path, raw: Path) -> Path:
    return raw.expanduser().resolve() if raw.is_absolute() else (root / raw).resolve()


def _save_raw_capture(frames: list[RGBDFrame], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        image_name = f"camera_{index}_{frame.label}.png"
        depth_name = f"camera_{index}_{frame.label}_depth_m.npy"
        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(output_dir / image_name)
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
                "rgb_shape": list(frame.rgb.shape),
                "depth_shape": list(frame.depth_m.shape),
            }
        )
    manifest = output_dir / "capture_manifest.json"
    _save_json(
        manifest,
        {
            "created_at": _now(),
            "coordinate_frame": "robot_base",
            "depth_unit": "metres",
            "frames": records,
        },
    )
    return manifest


def _load_raw_capture(capture_dir: Path) -> list[RGBDFrame]:
    manifest_path = capture_dir / "capture_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: list[RGBDFrame] = []
    for item in payload.get("frames", []):
        frames.append(
            RGBDFrame(
                label=str(item["label"]),
                serial=str(item["serial"]),
                rgb=np.asarray(Image.open(capture_dir / item["image"]).convert("RGB")),
                depth_m=np.load(capture_dir / item["depth_m"]).astype(np.float32),
                intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
                X_base_camera=np.asarray(item["X_base_camera"], dtype=np.float64),
            )
        )
    if len(frames) != 2:
        raise RuntimeError(
            f"offline capture must contain exactly two frames, found {len(frames)}"
        )
    return frames


def _labeled_tile(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size[0], size[1] + 38), (20, 20, 24))
    x = (size[0] - image.width) // 2
    y = 38 + (size[1] - image.height) // 2
    tile.paste(image, (x, y))
    draw = ImageDraw.Draw(tile)
    draw.text((10, 11), label, fill=(255, 255, 255))
    return tile


def _save_contact_sheet(
    result: dict[str, Any], result_path: Path, output_path: Path
) -> list[dict[str, str]]:
    result_dir = result_path.parent
    display_max = result.get("depth_fusion", {}).get("heatmap_display_max_mm")
    scale_suffix = f" [0..{float(display_max):g} mm]" if display_max is not None else ""
    items: list[tuple[str, Path]] = []
    for view in result.get("views", []):
        if not isinstance(view, dict):
            continue
        label = str(view.get("label", "?"))
        for key, title in (
            ("image", "RGB"),
            ("height_map", f"garment-focused height heatmap{scale_suffix}"),
            ("height_map_global", f"global height heatmap{scale_suffix}"),
            ("height_map_boundary", f"height heatmap + garment boundary{scale_suffix}"),
            ("height_gradient_overlay", "height-gradient/occlusion edges"),
            ("table_reference_overlay", "table corner/edge depth references"),
            ("coordinate_overlay", "base-XYZ coordinate references"),
        ):
            path = result_dir / str(view.get(key, ""))
            if path.is_file():
                items.append((f"Camera {label}: {title}", path))

    artifacts = result.get("depth_fusion", {}).get("artifacts", {})
    for key, title in (
        ("preview", "Fused height map: grayscale preview"),
        ("heatmap", f"Fused height map: heatmap{scale_suffix}"),
        ("boundary_overlay", f"Fused height map: garment boundary{scale_suffix}"),
        ("height_gradient_overlay", "Fused height map: gradient edges"),
    ):
        path = result_dir / str(artifacts.get(key, ""))
        if path.is_file():
            items.append((title, path))

    if not items:
        raise RuntimeError("perception result contains no heatmap images for report")
    tile_size = (420, 315)
    columns = 3
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_size[0], rows * (tile_size[1] + 38)),
        (10, 10, 12),
    )
    records: list[dict[str, str]] = []
    for index, (label, path) in enumerate(items):
        tile = _labeled_tile(path, label, tile_size)
        x = (index % columns) * tile_size[0]
        y = (index // columns) * (tile_size[1] + 38)
        sheet.paste(tile, (x, y))
        records.append({"label": label, "path": str(path.relative_to(result_dir))})
    sheet.save(output_path)
    return records


def _write_pipeline_readme(output_dir: Path) -> None:
    (output_dir / "PIPELINE.md").write_text(
        """# Standalone garment height-map test

This directory was generated without commanding the robot.

Pipeline:

1. Capture/reload calibrated A/B RGB-D frames.
2. Transform both point clouds into the robot base frame.
3. Sample table depths at four corners and four edge midpoints in each view,
   reject robot/fixture outliers, and interpolate `table_z = a*x + b*y + c`.
4. Segment the garment from table-relative height and appearance evidence.
5. Compute every height value as `surface_z_mm - table_z_mm`.
6. Render every heatmap against one absolute table-zero scale (`0..max mm`),
   and save camera-pixel height maps, focused/global heatmaps, boundaries,
   height-gradient overlays, coordinate maps, and the fused top-down map.
7. Assemble `heatmap_contact_sheet.png` for visual inspection.

Important files:

- `raw_capture/`: reusable RGB-D input plus calibration manifest.
- `pipeline/result.json`: authoritative perception result and table fit.
- `pipeline/camera_*_height_above_table_mm.npy`: full per-pixel scalar maps.
- `pipeline/camera_*_table_references.json`: sampled corner/edge depths.
- `pipeline/camera_*_table_z_mm.npy`: interpolated table surface per pixel.
- `pipeline/fused_height_map_mm.npy`: fused top-down scalar map.
- `pipeline/*heatmap*.png`: heatmap outputs.
- `heatmap_manifest.json`: compact artifact/range summary.
- `heatmap_contact_sheet.png`: all visual outputs in one image.
""",
        encoding="utf-8",
    )


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new output directory; defaults to results/heightmap_test/<timestamp>",
    )
    parser.add_argument(
        "--input-capture",
        type=Path,
        help="reuse a previous raw_capture directory instead of opening cameras",
    )
    parser.add_argument(
        "--temporal-median-frames",
        type=int,
        help="override the perception config's live-capture median frame count",
    )
    args = parser.parse_args(argv)

    root = args.project_root.expanduser().resolve()
    perception_path = _resolve(root, args.perception_config)
    robot_path = _resolve(root, args.robot_config)
    perception_config = PerceptionConfig.load(root, perception_path)
    if args.temporal_median_frames is not None:
        perception_config = replace(
            perception_config,
            temporal_median_frames=int(args.temporal_median_frames),
        )
        perception_config.validate()
    robot_config = RobotConfig.load(root, robot_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "results" / "heightmap_test" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_pipeline_readme(output_dir)
    shutil.copy2(perception_path, output_dir / "perception_config.json")
    shutil.copy2(robot_path, output_dir / "robot_config.json")

    raw_capture_dir = output_dir / "raw_capture"
    pipeline_dir = output_dir / "pipeline"
    try:
        if args.input_capture:
            source_capture = args.input_capture.expanduser().resolve()
            print(f"Reloading calibrated RGB-D capture: {source_capture}")
            frames = _load_raw_capture(source_capture)
            shutil.copytree(source_capture, raw_capture_dir)
            capture_source = "offline_replay"
        else:
            print("Capturing calibrated A/B RGB-D frames. No robot command will be sent.")
            frames = capture_two_view_rgbd(perception_config)
            _save_raw_capture(frames, raw_capture_dir)
            capture_source = "live_realsense"

        print("Running full table-fit, garment segmentation, height-map and heatmap flow...")
        service = ClothCenterPerception(root, robot_config, perception_config)
        result, _ = service.locate(
            pipeline_dir,
            ExperimentConfig(),
            frames=frames,
        )
        result_path = pipeline_dir / "result.json"
        contact_sheet = output_dir / "heatmap_contact_sheet.png"
        report_items = _save_contact_sheet(result, result_path, contact_sheet)

        fusion = result.get("depth_fusion", {})
        artifacts = fusion.get("artifacts", {})
        camera_maps = artifacts.get("camera_height_maps", {})
        manifest = {
            "created_at": _now(),
            "status": "COMPLETE",
            "robot_motion": False,
            "capture_source": capture_source,
            "output_dir": str(output_dir),
            "raw_capture_manifest": "raw_capture/capture_manifest.json",
            "perception_result": "pipeline/result.json",
            "contact_sheet": contact_sheet.name,
            "table_plane": fusion.get("table_plane"),
            "garment_point_count": fusion.get("garment_point_count"),
            "garment_height_p50_mm": fusion.get("garment_height_p50_mm"),
            "garment_height_p95_mm": fusion.get("garment_height_p95_mm"),
            "heatmap_display_min_mm": fusion.get("heatmap_display_min_mm", 0.0),
            "heatmap_display_max_mm": fusion.get("heatmap_display_max_mm"),
            "heatmap_normalization": fusion.get(
                "heatmap_normalization", "absolute_table_zero_shared"
            ),
            "camera_height_maps": camera_maps,
            "fused_height_map": {
                key: artifacts.get(key)
                for key in (
                    "path",
                    "preview",
                    "heatmap",
                    "boundary_overlay",
                    "height_gradient_overlay",
                    "heatmap_quantity",
                    "heatmap_display_min_mm",
                    "heatmap_display_max_mm",
                    "heatmap_normalization",
                    "height_min_mm",
                    "height_max_mm",
                    "grid_size_mm",
                    "origin_xy_mm",
                    "shape_yx",
                )
            },
            "contact_sheet_items": report_items,
        }
        _save_json(output_dir / "heatmap_manifest.json", manifest)
        print("Height-map test complete.")
        print(f"Output directory: {output_dir}")
        print(f"Visual report: {contact_sheet}")
        print(f"Manifest: {output_dir / 'heatmap_manifest.json'}")
        return 0
    except BaseException as exc:
        _save_json(
            output_dir / "failure.json",
            {
                "created_at": _now(),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "raw_capture_saved": raw_capture_dir.is_dir(),
                "pipeline_dir": str(pipeline_dir),
            },
        )
        print(f"Height-map test failed; diagnostics saved in {output_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
