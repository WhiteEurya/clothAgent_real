#!/usr/bin/env python3
"""Test composite shake-open on an already grasped, lifted garment.

Dry-run is the default. Real execution requires --enable-real.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import RobotConfig
from cloth_agent.robot_api import RobotExecutionError
from cloth_agent.shake_open import _timestamp, build_shake_open_plan, shake_open


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _default_output(project_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "results" / "shake_open_test" / f"shake_open_{stamp}.json"


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.robot_config).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = RobotConfig.load(project_root, config_path)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output(project_root)
    )
    artifact: dict[str, Any] = {
        "created_at": _timestamp(),
        "mode": "real" if args.enable_real else "dry_run",
        "robot_ip": config.robot_ip,
        "gripper_commands": [],
        "home_commanded": False,
        "physical_commands_sent": False,
    }
    if not args.enable_real:
        dry_pose = (
            config.init_pose_mm_deg[0],
            config.init_pose_mm_deg[1],
            config.init_pose_mm_deg[2],
            config.orientation_roll_deg,
            config.orientation_pitch_deg,
            config.init_pose_mm_deg[5],
        )
        plan = build_shake_open_plan(dry_pose, config)
        artifact.update(
            {
                "status": "DRY_RUN",
                "pose_source": "configured_init_pose",
                "plan": plan.as_dict(),
                "note": "No robot connection was made. Real mode rebuilds from live TCP pose.",
            }
        )
        _write_json(output, artifact)
        print(json.dumps(artifact, ensure_ascii=False, indent=2))
        print(f"Saved dry-run plan: {output}")
        return 0

    config.validate_for_real()
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise RobotExecutionError("xarm package is required for real execution") from exc
    arm = XArmAPI(config.robot_ip)
    try:
        if not getattr(arm, "connected", True):
            raise RobotExecutionError(f"unable to connect to xArm at {config.robot_ip}")
        artifact["physical_commands_sent"] = True
        result = shake_open(arm, config)
        artifact.update(result)
        artifact["status"] = "COMPLETED"
        artifact["completed_at"] = _timestamp()
        _write_json(output, artifact)
        print(f"Composite shake-open completed; gripper unchanged. Log: {output}")
        return 0
    except BaseException as exc:
        artifact["status"] = "FAILED"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        artifact["completed_at"] = _timestamp()
        _write_json(output, artifact)
        raise
    finally:
        if getattr(arm, "connected", False):
            arm.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--robot-config", default="config/robot.example.json")
    parser.add_argument("--output")
    parser.add_argument("--enable-real", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
