#!/usr/bin/env python3
"""Perform one deliberately small, low-speed xArm joint test.

The script never clears faults, enables motors, or changes the controller state.
It only moves if the controller is already in normal motion state (0).
"""

from __future__ import annotations

import argparse
import sys

from xarm.wrapper import XArmAPI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.2.232")
    parser.add_argument("--joint", type=int, default=7, choices=range(1, 8))
    parser.add_argument("--delta-deg", type=float, default=1.0)
    parser.add_argument("--speed-deg-s", type=float, default=1.0)
    parser.add_argument("--accel-deg-s2", type=float, default=5.0)
    args = parser.parse_args()

    if not 0 < abs(args.delta_deg) <= 1.0:
        parser.error("--delta-deg must be within (-1, 0) or (0, 1] degrees")

    arm = XArmAPI(args.ip, is_radian=False)
    try:
        if not arm.connected:
            raise RuntimeError(f"cannot connect to xArm at {args.ip}")

        code, state = arm.get_state()
        error_code, errors = arm.get_err_warn_code()
        if code != 0 or error_code != 0 or errors != [0, 0]:
            raise RuntimeError(f"controller reports errors: state={state}, errors={errors}")
        if state != 0:
            raise RuntimeError(
                f"controller is not in motion state (state={state}); refusing to change state"
            )
        if arm.mode != 0:
            raise RuntimeError(f"controller mode is {arm.mode}, expected position mode 0")

        code, angles = arm.get_servo_angle(is_radian=False)
        if code != 0:
            raise RuntimeError(f"could not read joint angles (code={code})")
        current = angles[args.joint - 1]
        target = current + args.delta_deg
        print(
            f"moving joint {args.joint}: {current:.3f}° -> {target:.3f}° "
            f"at {args.speed_deg_s:.1f}°/s"
        )
        code = arm.set_servo_angle(
            servo_id=args.joint,
            angle=target,
            speed=args.speed_deg_s,
            mvacc=args.accel_deg_s2,
            wait=True,
            timeout=20,
            is_radian=False,
        )
        if code != 0:
            raise RuntimeError(f"motion command failed with code {code}")
        code, final_angles = arm.get_servo_angle(is_radian=False)
        if code != 0:
            raise RuntimeError(f"could not verify final angle (code={code})")
        print(f"completed: joint {args.joint} is now {final_angles[args.joint - 1]:.3f}°")
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"micro-jog cancelled: {exc}", file=sys.stderr)
        raise SystemExit(1)
