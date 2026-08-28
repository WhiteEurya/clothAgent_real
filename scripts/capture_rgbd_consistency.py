#!/usr/bin/env python3
"""Capture one calibrated RealSense A/B pair for offline perception diagnosis.

This module deliberately does *only* acquisition. It does not fit a table,
segment the garment, start Claude, open Viser, move the robot, or overwrite a
previous capture. Feed its output to ``scripts/test_height_map_pipeline.py``
with ``--input-capture`` and then to
``scripts/diagnose_perception_consistency.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.perception import PerceptionConfig, RGBDFrame, capture_two_view_rgbd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _resolve(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _save_capture(frames: list[RGBDFrame], output_dir: Path) -> dict[str, Any]:
    if len(frames) != 2:
        raise RuntimeError(f"expected exactly two calibrated frames, got {len(frames)}")
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        image_name = f"camera_{index}_{frame.label}.png"
        depth_name = f"camera_{index}_{frame.label}_depth_m.npy"
        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(output_dir / image_name)
        np.save(output_dir / depth_name, np.asarray(frame.depth_m, dtype=np.float32))
        records.append(
            {
                "label": str(frame.label),
                "serial": str(frame.serial),
                "image": image_name,
                "depth_m": depth_name,
                "intrinsics": np.asarray(frame.intrinsics, dtype=np.float64).tolist(),
                "X_base_camera": np.asarray(frame.X_base_camera, dtype=np.float64).tolist(),
                "rgb_shape": list(np.asarray(frame.rgb).shape),
                "depth_shape": list(np.asarray(frame.depth_m).shape),
            }
        )
    manifest = {
        "created_at": _now(),
        "capture_source": "live_realsense",
        "robot_motion": False,
        "coordinate_frame": "robot_base_mm",
        "depth_unit": "metres",
        "frames": records,
    }
    (output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("config/perception.free_exploration.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new capture directory; defaults to results/rgbd_captures/<timestamp>",
    )
    parser.add_argument(
        "--temporal-median-frames",
        type=int,
        help="override live capture temporal median frame count",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    perception_path = _resolve(root, args.perception_config)
    config = PerceptionConfig.load(root, perception_path)
    if args.temporal_median_frames is not None:
        from dataclasses import replace

        config = replace(config, temporal_median_frames=int(args.temporal_median_frames))
        config.validate()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "results" / "rgbd_captures" / stamp
    )
    if output_dir.exists():
        existing = {path.name for path in output_dir.iterdir()}
        if "capture_manifest.json" in existing or existing - {"perception_config.json"}:
            raise FileExistsError(
                f"capture output already contains data: {output_dir}; choose a new --output-dir"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "perception_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("Capturing calibrated Camera A/B RGB-D frames; no robot command will be sent.", flush=True)
    frames = capture_two_view_rgbd(config)
    manifest = _save_capture(frames, output_dir)
    print(json.dumps({"output_dir": str(output_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
