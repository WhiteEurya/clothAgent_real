#!/usr/bin/env python3
"""Manually capture one or more RGB photos from a RealSense camera.

The default device is the Camera-A wrist camera used by the project
(``317222073552``).  Each shot receives its own timestamped directory, so
repeated invocations never overwrite an earlier photo.  This utility opens the
selected camera, warms it up, captures RGB only, writes a PNG plus a manifest,
and closes the pipeline.  It does not move the robot and does not participate
in A/B perception or geometry.

Examples::

    # One Camera-A photo (default)
    python scripts/capture_camera_photo.py

    # Three photos, two seconds apart
    python scripts/capture_camera_photo.py --count 3 --interval-s 2

    # Use a different RealSense serial and output directory
    python scripts/capture_camera_photo.py \
      --serial 317222073552 --output-dir runs/my_camera_a_photos

Run it with the project's ``cali`` Python environment, which contains
``pyrealsense2``::

    /home/CNS2026330003/miniconda3/envs/cali/bin/python \
      scripts/capture_camera_photo.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_A_SERIAL = "317222073552"
CALIBRATED_PYTHON = Path("/home/CNS2026330003/miniconda3/envs/cali/bin/python")

# When invoked as ``python scripts/capture_camera_photo.py``, Python puts only
# the ``scripts`` directory on ``sys.path``. Add the project root explicitly so
# the local ``cloth_agent`` package can be imported without requiring
# ``PYTHONPATH=.``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _shot_directory(root: Path, *, label: str, index: int, count: int) -> Path:
    """Return a fresh per-shot directory without overwriting existing output."""

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"camera_{label}_{_utc_stamp()}"
    if count > 1:
        prefix += f"_{index:02d}"
    candidate = root / prefix
    suffix = 2
    while candidate.exists():
        candidate = root / f"{prefix}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        default=DEFAULT_CAMERA_A_SERIAL,
        help=f"RealSense serial (default: Camera A {DEFAULT_CAMERA_A_SERIAL})",
    )
    parser.add_argument("--label", default="A", help="label used in filenames/manifests")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/camera_A_photos"),
        help="parent directory for timestamped shot directories",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--exposure", type=float, default=700.0)
    parser.add_argument("--white-balance", type=float, default=3800.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--count", type=int, default=1, help="number of photos to capture")
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.0,
        help="delay between photos when --count is greater than one",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.serial).strip():
        raise SystemExit("--serial must be non-empty")
    if not str(args.label).strip():
        raise SystemExit("--label must be non-empty")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("--width, --height, and --fps must be positive")
    if args.warmup_frames < 0:
        raise SystemExit("--warmup-frames must be non-negative")
    if args.count < 1:
        raise SystemExit("--count must be at least one")
    if args.interval_s < 0:
        raise SystemExit("--interval-s must be non-negative")


def _reexec_in_cali_environment() -> None:
    """Use the project's RealSense-tested Python when plain ``python`` is used.

    On this machine the system interpreter can import ``pyrealsense2`` but its
    librealsense build fails while initializing the udev monitor. The cali
    environment is the project's known-good camera environment. An environment
    guard prevents recursion when the replacement interpreter starts.
    """

    if os.environ.get("CLOTH_AGENT_CAPTURE_CALI_REEXEC") == "1":
        return
    cali = CALIBRATED_PYTHON.expanduser().resolve()
    current = Path(sys.executable).expanduser().resolve()
    if not cali.is_file() or current == cali:
        return
    environment = os.environ.copy()
    environment["CLOTH_AGENT_CAPTURE_CALI_REEXEC"] = "1"
    script = Path(__file__).resolve()
    os.execve(str(cali), [str(cali), str(script), *sys.argv[1:]], environment)


def _capture_one(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    # Import only after argument validation so ``--help`` and validation work
    # even in a Python environment without RealSense installed.
    from cloth_agent.rollout_recorder import capture_observer_rgb

    manifest = capture_observer_rgb(
        str(args.serial).strip(),
        output_dir,
        label=str(args.label).strip(),
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        color_exposure=float(args.exposure) if args.exposure is not None else None,
        color_white_balance=(
            float(args.white_balance) if args.white_balance is not None else None
        ),
        warmup_frames=int(args.warmup_frames),
    )
    return dict(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    _reexec_in_cali_environment()
    parent = args.output_dir.expanduser().resolve()
    results: list[dict[str, Any]] = []
    for index in range(1, int(args.count) + 1):
        shot_dir = _shot_directory(
            parent,
            label=str(args.label).strip(),
            index=index,
            count=int(args.count),
        )
        try:
            manifest = _capture_one(args, shot_dir)
        except Exception:
            # Remove only the empty directory created for this failed shot;
            # never touch existing photos or manifests.
            try:
                shot_dir.rmdir()
            except OSError:
                pass
            raise
        manifest["shot_index"] = index
        manifest["shot_count"] = int(args.count)
        manifest["output_directory"] = str(shot_dir)
        results.append(manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        if index < int(args.count) and args.interval_s:
            time.sleep(float(args.interval_s))
    print(
        json.dumps(
            {"status": "completed", "count": len(results), "shots": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
