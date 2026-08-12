#!/usr/bin/env python3
"""Return the xArm to its configured home pose with the gripper open.

This is intentionally a recovery-only script: it performs no Cartesian moves,
perception, or garment interaction.  The arm goes to the configured joint home
pose and the gripper is commanded to its configured open position.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import RobotConfig
from cloth_agent.robot_api import XArmBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/robot.example.json"),
        help="robot configuration JSON (default: config/robot.example.json)",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = RobotConfig.load(PROJECT_ROOT, config_path)
    config.validate_for_real()

    backend = XArmBackend(config)
    try:
        print(f"Connected to xArm at {config.robot_ip}")
        print("Moving to configured home pose...")
        home_pose, _ = backend.home(config)
        print(f"Home pose: {home_pose}")
        print("Opening gripper...")
        gripper_result, _ = backend.open_gripper(config)
        print(f"Gripper result: {gripper_result}")
        print("Recovery complete.")
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
