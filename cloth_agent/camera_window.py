"""Native OpenCV RGB preview for one RealSense camera.

The camera can be selected from an existing A/B perception config, or opened
directly by serial with ``--serial``. The direct-serial form is intended for an
independent overview camera and does not modify the grasp perception pipeline.
Press ``q``/Esc or close the window to exit cleanly.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    from .perception import CameraSpec, PerceptionConfig, RealSenseRGBD

    root = Path(args.project_root).resolve()
    if args.serial:
        label = args.label.strip() or "Overview"
        serial = args.serial.strip()
        if not serial:
            raise ValueError("--serial must be non-empty")
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

    camera = RealSenseRGBD(spec, width, height, fps)
    window_name = f"Cam{label} live monitor ({spec.serial})"
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
