#!/usr/bin/env python3
"""Apply saved RGB controls and verify every value before a fold run."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_agent.realsense_tuner import CameraControls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.preset.read_text(encoding="utf-8"))
    import pyrealsense2 as rs

    context = rs.context()
    devices = [device for device in context.query_devices()
               if device.get_info(rs.camera_info.serial_number) == payload["serial"]]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one RealSense {payload['serial']}; found {len(devices)}")
    sensor = devices[0].first_color_sensor()
    controls = CameraControls(sensor, serial=payload["serial"],
                              sensor_name=sensor.get_info(rs.camera_info.name))
    controls.apply_preset(payload)
    # Other controls (including power-line frequency) can reset exposure.
    # Restore manual exposure/gain/WB last, without disabling saved auto modes.
    options = payload["options"]
    for name, automatic in (("exposure", "enable_auto_exposure"),
                            ("gain", "enable_auto_exposure"),
                            ("white_balance", "enable_auto_white_balance")):
        if name in options and options.get(automatic) == 0:
            controls.set_value(name, options[name])
    time.sleep(0.5)
    actual = controls.preset()["options"]
    mismatches = {name: {"expected": value, "actual": actual.get(name)}
                  for name, value in options.items()
                  if actual.get(name) != value}
    if mismatches:
        raise RuntimeError(f"RealSense preset readback mismatch: {mismatches}")
    print(f"[camera-preset] applied and verified {len(options)} options "
          f"for {payload['serial']}: {args.preset}", flush=True)


if __name__ == "__main__":
    main()
