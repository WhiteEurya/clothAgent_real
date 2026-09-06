"""Standalone, bounded three-cycle Y-axis shake for an already lifted garment.

This module is deliberately separate from generated RobotAPI programs and the
collar workflow.  It preserves the live TCP X/Z/orientation and gripper state,
centres at Y=0, performs three fixed positive/negative Y cycles, and stops at
Y=0. Real execution requires the explicit ``--enable-real`` flag.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import RobotConfig, SafetyError
from .robot_api import RobotExecutionError, _validated_live_tcp_offset


SHAKE_AMPLITUDE_MM = 20.0
SHAKE_CYCLES = 3
MINIMUM_SHAKE_Z_MM = 250.0
PRELOAD_RETURN_SPEED_MM_S = 150.0
PRELOAD_RETURN_ACCELERATION_MM_S2 = 360.0
FLICK_SPEED_MM_S = 300.0
FLICK_ACCELERATION_MM_S2 = 1200.0
MAX_EXPERIMENTAL_AMPLITUDE_MM = 60.0
MAX_EXPERIMENTAL_CYCLES = 3
MAX_EXPERIMENTAL_SPEED_MM_S = 330.0
MAX_EXPERIMENTAL_ACCELERATION_MM_S2 = 1350.0
IK_SAMPLE_STEP_MM = 5.0
ORIENTATION_TOLERANCE_DEG = 8.0
POSITION_VERIFICATION_TOLERANCE_MM = 5.0


@dataclass(frozen=True)
class ShakeStep:
    name: str
    target_pose_mm_deg: tuple[float, float, float, float, float, float]
    speed_mm_s: float
    acceleration_mm_s2: float


@dataclass(frozen=True)
class ShakePlan:
    start_pose_mm_deg: tuple[float, float, float, float, float, float]
    amplitude_mm: float
    cycles: int
    minimum_z_mm: float
    steps: tuple[ShakeStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _shortest_angle_delta_deg(start: float, target: float) -> float:
    return (target - start + 180.0) % 360.0 - 180.0


def _pose(values: Any, *, label: str) -> tuple[float, float, float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise RobotExecutionError(f"{label} must contain six values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise RobotExecutionError(f"{label} contains non-finite values")
    return result  # type: ignore[return-value]


def _validate_experimental_profile() -> None:
    if not 0.0 < SHAKE_AMPLITUDE_MM <= MAX_EXPERIMENTAL_AMPLITUDE_MM:
        raise SafetyError(
            f"shake amplitude must be in (0, {MAX_EXPERIMENTAL_AMPLITUDE_MM:g}] mm"
        )
    if not 1 <= SHAKE_CYCLES <= MAX_EXPERIMENTAL_CYCLES:
        raise SafetyError(
            f"shake cycles must be in [1, {MAX_EXPERIMENTAL_CYCLES}]"
        )
    for name, value in (
        ("preload/return speed", PRELOAD_RETURN_SPEED_MM_S),
        ("flick speed", FLICK_SPEED_MM_S),
    ):
        if not 0.0 < value <= MAX_EXPERIMENTAL_SPEED_MM_S:
            raise SafetyError(
                f"{name} must be in (0, {MAX_EXPERIMENTAL_SPEED_MM_S:g}] mm/s"
            )
    for name, value in (
        ("preload/return acceleration", PRELOAD_RETURN_ACCELERATION_MM_S2),
        ("flick acceleration", FLICK_ACCELERATION_MM_S2),
    ):
        if not 0.0 < value <= MAX_EXPERIMENTAL_ACCELERATION_MM_S2:
            raise SafetyError(
                f"{name} must be in (0, {MAX_EXPERIMENTAL_ACCELERATION_MM_S2:g}] mm/s^2"
            )


def build_shake_plan(
    current_pose_mm_deg: Sequence[float],
    config: RobotConfig,
    *,
    require_complete_workspace: bool = True,
) -> ShakePlan:
    """Create the fixed Y=0 -> (+20 <-> -20)x3 -> 0 shake plan."""

    _validate_experimental_profile()
    current = _pose(current_pose_mm_deg, label="current TCP pose")
    x, y, z, roll, pitch, yaw = current
    if z < MINIMUM_SHAKE_Z_MM:
        raise SafetyError(
            f"shake requires an already lifted TCP at z>={MINIMUM_SHAKE_Z_MM:g} mm; "
            f"current z={z:.3f} mm"
        )
    roll_error = abs(_shortest_angle_delta_deg(roll, config.orientation_roll_deg))
    pitch_error = abs(_shortest_angle_delta_deg(pitch, config.orientation_pitch_deg))
    if roll_error > ORIENTATION_TOLERANCE_DEG or pitch_error > ORIENTATION_TOLERANCE_DEG:
        raise SafetyError(
            "current TCP orientation does not match the calibrated top-down tool orientation: "
            f"roll error={roll_error:.3f} deg, pitch error={pitch_error:.3f} deg"
        )

    targets = [
        # Y=0 centring is a prerequisite, not part of the shake.  Keep it on
        # the repository's normal configured motion profile.
        (
            "center_y0",
            (x, 0.0, z, roll, pitch, yaw),
            config.speed_mm_s,
            config.acceleration_mm_s2,
        ),
        (
            "preload_positive_y",
            (x, SHAKE_AMPLITUDE_MM, z, roll, pitch, yaw),
            PRELOAD_RETURN_SPEED_MM_S,
            PRELOAD_RETURN_ACCELERATION_MM_S2,
        ),
    ]
    for cycle in range(1, SHAKE_CYCLES + 1):
        targets.extend(
            [
                (
                    f"flick_{cycle}_negative_y",
                    (x, -SHAKE_AMPLITUDE_MM, z, roll, pitch, yaw),
                    FLICK_SPEED_MM_S,
                    FLICK_ACCELERATION_MM_S2,
                ),
                (
                    f"flick_{cycle}_positive_y",
                    (x, SHAKE_AMPLITUDE_MM, z, roll, pitch, yaw),
                    FLICK_SPEED_MM_S,
                    FLICK_ACCELERATION_MM_S2,
                ),
            ]
        )
    targets.append(
        (
            "return_y0",
            (x, 0.0, z, roll, pitch, yaw),
            PRELOAD_RETURN_SPEED_MM_S,
            PRELOAD_RETURN_ACCELERATION_MM_S2,
        )
    )
    for label, target, _, _ in targets:
        config.validate_workspace_pose(
            target[0],
            target[1],
            target[2],
            relative_yaw_deg=config.relative_yaw_from_absolute_deg(target[5]),
            require_complete=require_complete_workspace,
        )
        if target[0] != x or target[2] != z or target[3:] != current[3:]:
            raise SafetyError(f"{label} is not a pure Y-axis motion")

    # Validate the observed start as well; no command should begin from outside
    # the configured workspace.
    config.validate_workspace_pose(
        x,
        y,
        z,
        relative_yaw_deg=config.relative_yaw_from_absolute_deg(current[5]),
        require_complete=require_complete_workspace,
    )
    return ShakePlan(
        start_pose_mm_deg=current,
        amplitude_mm=SHAKE_AMPLITUDE_MM,
        cycles=SHAKE_CYCLES,
        minimum_z_mm=MINIMUM_SHAKE_Z_MM,
        steps=tuple(
            ShakeStep(
                name=label,
                target_pose_mm_deg=target,
                speed_mm_s=speed,
                acceleration_mm_s2=acceleration,
            )
            for label, target, speed, acceleration in targets
        ),
    )


def read_live_pose(arm: Any) -> tuple[float, float, float, float, float, float]:
    result = arm.get_position(is_radian=False)
    if (
        not isinstance(result, (list, tuple))
        or len(result) < 2
        or int(result[0]) != 0
    ):
        raise RobotExecutionError(f"get_position failed: result={result}")
    return _pose(result[1], label="live TCP pose")


def _read_live_joints_deg(arm: Any) -> list[float]:
    result = arm.get_servo_angle(is_radian=False)
    if (
        not isinstance(result, (list, tuple))
        or len(result) < 2
        or int(result[0]) != 0
        or not isinstance(result[1], (list, tuple))
        or len(result[1]) != 7
    ):
        raise RobotExecutionError(f"get_servo_angle failed: result={result}")
    joints = [float(value) for value in result[1]]
    if not all(math.isfinite(value) for value in joints):
        raise RobotExecutionError("get_servo_angle returned non-finite values")
    return joints


def _controller_error_warning(arm: Any) -> tuple[int, int]:
    result = arm.get_err_warn_code()
    if (
        not isinstance(result, (list, tuple))
        or len(result) < 2
        or int(result[0]) != 0
        or not isinstance(result[1], (list, tuple))
        or len(result[1]) < 2
    ):
        raise RobotExecutionError(f"get_err_warn_code failed: result={result}")
    error_code, warning_code = int(result[1][0]), int(result[1][1])
    if error_code != 0:
        raise RobotExecutionError(f"xArm controller has active error code {error_code}")
    return error_code, warning_code


def validate_shake_plan_with_controller(arm: Any, plan: ShakePlan) -> dict[str, Any]:
    """Use limited controller IK on every 5 mm of the live-pose trajectory."""

    _, warning_code = _controller_error_warning(arm)
    reference_deg = _read_live_joints_deg(arm)
    segment_start = plan.start_pose_mm_deg
    sample_count_total = 0
    for step_index, step in enumerate(plan.steps, start=1):
        target = step.target_pose_mm_deg
        distance_mm = math.dist(segment_start[:3], target[:3])
        samples = max(1, int(math.ceil(distance_mm / IK_SAMPLE_STEP_MM)))
        for sample_index in range(1, samples + 1):
            fraction = sample_index / samples
            sample_pose = [
                segment_start[index] + fraction * (target[index] - segment_start[index])
                for index in range(6)
            ]
            for index in range(3, 6):
                delta = _shortest_angle_delta_deg(segment_start[index], target[index])
                sample_pose[index] = segment_start[index] + fraction * delta
            if sample_index == samples:
                sample_pose = list(target)
            result = arm.get_inverse_kinematics(
                sample_pose,
                input_is_radian=False,
                return_is_radian=False,
                limited=True,
                ref_angles=reference_deg,
            )
            if (
                not isinstance(result, (list, tuple))
                or len(result) < 2
                or int(result[0]) != 0
                or not isinstance(result[1], (list, tuple))
                or len(result[1]) != 7
            ):
                code = result[0] if isinstance(result, (list, tuple)) and result else result
                raise SafetyError(
                    f"controller IK rejected shake step {step_index} ({step.name}) "
                    f"sample {sample_index}/{samples}, pose={sample_pose}, code={code}"
                )
            reference_deg = [float(value) for value in result[1]]
            if not all(math.isfinite(value) for value in reference_deg):
                raise RobotExecutionError("controller IK returned non-finite joint values")
            sample_count_total += 1
        segment_start = target
    return {
        "controller_warning_code": warning_code,
        "validated_sample_count": sample_count_total,
        "ik_sample_step_mm": IK_SAMPLE_STEP_MM,
    }


def _sdk_code(value: Any) -> int:
    return int(value[0]) if isinstance(value, tuple) else int(value)


def _check_sdk(method: str, result: Any) -> None:
    code = _sdk_code(result)
    if code != 0:
        raise RobotExecutionError(f"{method} failed, code={code}")


def execute_shake_plan(arm: Any, plan: ShakePlan) -> list[dict[str, Any]]:
    """Execute the fixed Y-only shake sequence; never command gripper or Home."""

    _check_sdk("motion_enable", arm.motion_enable(enable=True))
    _check_sdk("set_mode", arm.set_mode(0))
    _check_sdk("set_state", arm.set_state(0))
    records: list[dict[str, Any]] = []
    motion_started = False
    try:
        for step in plan.steps:
            requested_at = _timestamp()
            motion_started = True
            target = step.target_pose_mm_deg
            result = arm.set_position(
                x=target[0],
                y=target[1],
                z=target[2],
                roll=target[3],
                pitch=target[4],
                yaw=target[5],
                speed=step.speed_mm_s,
                mvacc=step.acceleration_mm_s2,
                wait=True,
                is_radian=False,
            )
            _check_sdk(f"set_position({step.name})", result)
            actual = read_live_pose(arm)
            position_error = math.dist(actual[:3], target[:3])
            if position_error > POSITION_VERIFICATION_TOLERANCE_MM:
                raise RobotExecutionError(
                    f"{step.name} ended {position_error:.3f} mm from its target; "
                    f"limit={POSITION_VERIFICATION_TOLERANCE_MM:.3f} mm"
                )
            _controller_error_warning(arm)
            records.append(
                {
                    "name": step.name,
                    "requested_at": requested_at,
                    "completed_at": _timestamp(),
                    "target_pose_mm_deg": list(target),
                    "actual_pose_mm_deg": list(actual),
                    "position_error_mm": position_error,
                    "speed_mm_s": step.speed_mm_s,
                    "acceleration_mm_s2": step.acceleration_mm_s2,
                }
            )
        return records
    except BaseException:
        if motion_started:
            try:
                arm.set_state(4)
            except BaseException:
                pass
        raise


def shake(
    arm: Any,
    config: RobotConfig,
    *,
    preflight_callback: Callable[[ShakePlan, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Validate and execute the fixed three-cycle shake from the live TCP pose."""

    config.validate_for_real()
    tcp_offset = _validated_live_tcp_offset(arm, config)
    plan = build_shake_plan(read_live_pose(arm), config)
    controller_validation = validate_shake_plan_with_controller(arm, plan)
    controller_validation["tcp_offset_mm_deg"] = list(tcp_offset)
    if preflight_callback is not None:
        preflight_callback(plan, controller_validation)
    execution = execute_shake_plan(arm, plan)
    return {
        "plan": plan.as_dict(),
        "controller_validation": controller_validation,
        "execution": execution,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _default_output(project_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "results" / "shake_once" / f"shake_once_{stamp}.json"


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
        plan = build_shake_plan(config.init_pose_mm_deg, config)
        artifact.update(
            {
                "status": "DRY_RUN",
                "pose_source": "configured_init_pose",
                "plan": plan.as_dict(),
                "note": "No robot connection was made. Real mode rebuilds this plan from the live TCP pose.",
            }
        )
        _write_json(output, artifact)
        print(json.dumps(artifact, indent=2, ensure_ascii=False))
        print(f"Saved dry-run plan: {output}")
        return 0

    config.validate_for_real()
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise RobotExecutionError("xarm package is required for real shake execution") from exc

    arm = XArmAPI(config.robot_ip)
    try:
        if not getattr(arm, "connected", True):
            raise RobotExecutionError(f"unable to connect to xArm at {config.robot_ip}")
        def record_preflight(
            plan: ShakePlan,
            controller_validation: dict[str, Any],
        ) -> None:
            artifact.update(
                {
                    "status": "PREFLIGHT_VALIDATED",
                    "pose_source": "live_controller",
                    "tcp_offset_mm_deg": controller_validation[
                        "tcp_offset_mm_deg"
                    ],
                    "plan": plan.as_dict(),
                    "controller_validation": controller_validation,
                }
            )
            _write_json(output, artifact)
            artifact["physical_commands_sent"] = True

        shake_result = shake(arm, config, preflight_callback=record_preflight)
        artifact["execution"] = shake_result["execution"]
        artifact["status"] = "COMPLETED"
        artifact["completed_at"] = _timestamp()
        _write_json(output, artifact)
        print(f"Three-cycle shake completed; gripper unchanged; stopped at Y=0. Log: {output}")
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
    parser.add_argument(
        "--enable-real",
        action="store_true",
        help="Connect to the xArm and execute the fixed three-cycle shake.",
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
