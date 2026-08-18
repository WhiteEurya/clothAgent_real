"""Native OpenCV RGB preview for one RealSense camera.

The camera can be selected from an existing A/B perception config, or opened
directly by serial with ``--serial``. The direct-serial form is intended for an
independent overview camera and does not modify the grasp perception pipeline.
Press ``q``/Esc or close the window to exit cleanly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--perception-config",
        default="config/perception.free_exploration.json",
        help="existing two-camera config used when --serial is not supplied",
    )
    parser.add_argument("--camera", default="A", help="configured camera label (A/B)")
    parser.add_argument(
        "--serial",
        help="open an arbitrary RealSense serial directly; useful for the overview camera",
    )
    parser.add_argument(
        "--label",
        default="Overview",
        help="window label when --serial is supplied",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--white-balance",
        type=float,
        default=None,
        help="manual RGB white balance in Kelvin; overrides the configured value",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="manual RGB exposure; overrides the configured value",
    )
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    from .perception import CameraSpec, PerceptionConfig, RealSenseRGBD

    root = Path(args.project_root).resolve()
    config = None
    if args.serial:
        label = args.label.strip() or "Overview"
        serial = args.serial.strip()
        if not serial:
            raise ValueError("--serial must be non-empty")
        config_path = (root / args.perception_config).resolve()
        if config_path.is_file():
            config = PerceptionConfig.load(root, config_path)
        configured = (
            next((camera for camera in config.cameras if camera.serial == serial), None)
            if config is not None
            else None
        )
        if configured is not None:
            spec = replace(configured, label=label)
        else:
            spec = CameraSpec(label, serial, Path("."))
        width = args.width or 640
        height = args.height or 480
        fps = args.fps or 30
    else:
        config = PerceptionConfig.load(root, Path(args.perception_config).resolve())
        label = args.camera.strip().upper()
        spec = next((camera for camera in config.cameras if camera.label == label), None)
        if spec is None:
            raise ValueError(f"camera {label!r} is not configured")
        width = args.width or config.width
        height = args.height or config.height
        fps = args.fps or config.fps
    if args.white_balance is not None or args.exposure is not None:
        spec = replace(
            spec,
            color_white_balance=(
                args.white_balance
                if args.white_balance is not None
                else spec.color_white_balance
            ),
            color_exposure=(
                args.exposure if args.exposure is not None else spec.color_exposure
            ),
        )

    camera = RealSenseRGBD(spec, width, height, fps)
    white_balance_label = (
        f"{spec.color_white_balance:.0f}K"
        if spec.color_white_balance is not None
        else "auto-WB"
    )
    exposure_label = (
        f"{spec.color_exposure:.0f}"
        if spec.color_exposure is not None
        else "auto-exp"
    )
    window_name = (
        f"Cam{label} live monitor ({spec.serial}) "
        f"WB={white_balance_label} EXP={exposure_label}"
    )
    camera.start()
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        while True:
            rgb, _ = camera.read()
            cv2.imshow(window_name, cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            try:
                visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break
            if visible >= 0 and visible < 1:
                break
    finally:
        camera.stop()
        try:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
