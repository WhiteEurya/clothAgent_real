#!/usr/bin/env python3
"""Interactively capture six Camera-A RGB fold states from another garment.

The operator places a different shirt in each requested state and presses
Enter.  The camera remains open for the whole collection.  The output contains
only RGB PNGs and a manifest; it contains no robot actions or calibration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIAL = "317222073552"
STATES = (
    ("state_00_unfolded", "unfolded shirt"),
    ("state_01_left_sleeve", "after folding the image-left sleeve inward"),
    ("state_02_right_sleeve", "after folding the image-right sleeve inward"),
    ("state_03_left_side", "after folding the first torso side inward"),
    ("state_04_right_side", "after folding the second torso side inward"),
    ("state_05_bottom_hem", "after folding the bottom hem upward"),
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument(
        "--name",
        default="active",
        help="collection directory name (default: active; planner reads this collection automatically)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/reference/fold_states"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--exposure", type=float, default=700.0)
    parser.add_argument("--white-balance", type=float, default=3800.0)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="disable the live Camera-A window and use terminal Enter capture",
    )
    return parser


def _capture_pipeline(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("numpy and Pillow are required; run this script in the cali environment") from exc
    cv2 = None
    preview = not bool(args.no_preview)
    if preview:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            cv2.namedWindow("Camera A fold reference capture", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera A fold reference capture", 960, 540)
        except Exception as exc:
            print(f"Live preview unavailable ({exc}); using terminal capture.", file=sys.stderr)
            preview = False
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required; run this script in the cali environment") from exc
    context = rs.context()
    serial = str(args.serial).strip()
    available = {d.get_info(rs.camera_info.serial_number) for d in context.query_devices()}
    if serial not in available:
        raise RuntimeError(f"Camera A serial {serial} is not connected; available={sorted(available)}")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.rgb8, int(args.fps))
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        sensor = next((s for s in device.query_sensors() if s.supports(rs.option.enable_auto_exposure)), None)
        if sensor is not None:
            try:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
            except Exception:
                pass
            for option, value in ((rs.option.exposure, float(args.exposure)), (rs.option.enable_auto_white_balance, 0.0), (rs.option.white_balance, float(args.white_balance))):
                try:
                    if sensor.supports(option):
                        sensor.set_option(option, value)
                except Exception:
                    pass
        for _ in range(int(args.warmup_frames)):
            pipeline.wait_for_frames(2000)
        states: list[dict[str, Any]] = []
        for index, (state_id, instruction) in enumerate(STATES):
            print(f"\n[{index + 1}/{len(STATES)}] Place the reference shirt in the {instruction} state.")
            print("Live preview: press Enter/Space in the preview window to capture; press q to cancel.")
            image = None
            color = None
            while image is None:
                frames = pipeline.wait_for_frames(2000)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError(f"Camera A returned no color frame for {state_id}")
                image = np.asanyarray(color.get_data()).copy()
                if preview and cv2 is not None:
                    cv2.imshow("Camera A fold reference capture", image[:, :, ::-1])
                    key = cv2.waitKey(30) & 0xFF
                    if key in (13, 32):
                        break
                    if key in (ord("q"), ord("Q"), 27):
                        raise KeyboardInterrupt("capture cancelled from preview window")
                    image = None
                else:
                    input("Press Enter to capture this state (Ctrl-C to cancel)... ")
            filename = f"{state_id}.png"
            # The fold planner always reasons over the canonical Camera-A
            # clockwise-90 upright view.  Save references in that same frame
            # so left/right and state geometry are visually comparable.
            Image.fromarray(image).rotate(-90, expand=True).save(output / filename)
            states.append({
                "state_id": state_id,
                "step": None if index == 0 else state_id.removeprefix("state_%02d_" % index),
                "description": instruction,
                "filename": filename,
                "captured_at": _utc_stamp(),
                "frame_number": int(color.get_frame_number()),
                "device_timestamp_ms": float(color.get_timestamp()),
            })
            print(f"Saved {output / filename}")
        manifest = {
            "schema_version": 1,
            "reference_type": "static_cross_garment_fold_states",
            "collection_name": str(args.name).strip(),
            "created_at": _utc_stamp(),
            "source_garment": "operator-provided reference garment; no trajectory data",
            "camera": {
                "label": "A",
                "serial": serial,
                "stream": "RGB color only",
                "raw_resolution": [int(args.width), int(args.height)],
                "image_resolution": [int(args.height), int(args.width)],
                "orientation": "clockwise90_upright (same as fold planner Camera-A RGB)",
                "fps": int(args.fps),
            },
            "states": states,
            "coordinate_policy": "static semantic reference only; no pixel transfer, depth, XYZ, calibration, or robot action",
        }
        manifest["manifest_path"] = str((output / "manifest.json").resolve())
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest
    finally:
        pipeline.stop()
        if preview and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not str(args.name).strip() or Path(str(args.name)).name != str(args.name):
        raise SystemExit("--name must be a simple directory name")
    if min(args.width, args.height, args.fps) <= 0 or args.warmup_frames < 0:
        raise SystemExit("width, height, fps must be positive and warmup-frames non-negative")
    output = (args.output_root.expanduser() / str(args.name).strip()).resolve()
    if output.exists():
        backup = output.with_name(f"{output.name}_previous_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        suffix = 2
        while backup.exists():
            backup = output.with_name(f"{output.name}_previous_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{suffix}")
            suffix += 1
        shutil.move(str(output), str(backup))
        print(f"Existing collection backed up to {backup}", file=sys.stderr)
    output.mkdir(parents=True, exist_ok=False)
    try:
        manifest = _capture_pipeline(args, output)
    except BaseException:
        # Preserve already captured states for recovery, but do not claim a
        # complete collection without a manifest.
        print(f"Collection incomplete; partial images remain in {output}", file=sys.stderr)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
