"""Standalone composite shake-open test for an already grasped, lifted garment.

The test centres at Y=0, lifts to a bounded test-high pose when needed, moves
down to create motion room, applies two vertical slow-down/fast-up snaps, then queues two blended 3-D diagonal
figure-eight cycles. It returns to the high Y=0 pose without opening the
gripper or returning Home.  Real execution requires ``--enable-real``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .config import RobotConfig, SafetyError
from .robot_api import RobotExecutionError, _validated_live_tcp_offset


MINIMUM_ENTRY_Z_MM = 250.0
TARGET_TEST_HIGH_Z_MM = 450.0
CEILING_MARGIN_MM = 5.0
WORK_Z_DROP_MM = 50.0
VERTICAL_SNAP_DROP_MM = 30.0
VERTICAL_SNAP_CYCLES = 2
DIAGONAL_X_MM = 15.0
DIAGONAL_Y_MM = 20.0
DIAGONAL_Z_MM = 10.0
DIAGONAL_CYCLES = 2
DIAGONAL_SCALE_CANDIDATES = (1.0, 0.75, 0.5, 0.25)

SETUP_SPEED_MM_S = 60.0
SETUP_ACCELERATION_MM_S2 = 120.0
SLOW_DROP_SPEED_MM_S = 150.0
SLOW_DROP_ACCELERATION_MM_S2 = 360.0
FAST_RISE_SPEED_MM_S = 300.0
FAST_RISE_ACCELERATION_MM_S2 = 1200.0
DIAGONAL_SPEED_MM_S = 220.0
DIAGONAL_ACCELERATION_MM_S2 = 900.0
RETURN_SPEED_MM_S = 150.0
RETURN_ACCELERATION_MM_S2 = 360.0

VERTICAL_BLEND_RADIUS_MM = 6.0
DIAGONAL_BLEND_RADIUS_MM = 8.0
IK_SAMPLE_STEP_MM = 5.0
ORIENTATION_TOLERANCE_DEG = 8.0
FINAL_POSITION_TOLERANCE_MM = 5.0

MAX_EXPERIMENTAL_SPEED_MM_S = 320.0
MAX_EXPERIMENTAL_ACCELERATION_MM_S2 = 1300.0
MAX_EXPERIMENTAL_OFFSET_MM = 60.0


@dataclass(frozen=True)
class ShakeOpenStep:
    name: str
    target_pose_mm_deg: tuple[float, float, float, float, float, float]
    speed_mm_s: float
    acceleration_mm_s2: float
    blend_radius_mm: float | None
    wait: bool


@dataclass(frozen=True)
class ShakeOpenPlan:
    start_pose_mm_deg: tuple[float, float, float, float, float, float]
    work_pose_mm_deg: tuple[float, float, float, float, float, float]
    vertical_snap_cycles: int
    diagonal_cycles: int
    diagonal_scale: float
    steps: tuple[ShakeOpenStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _shortest_angle_delta_deg(start: float, target: float) -> float:
    return (target - start + 180.0) % 360.0 - 180.0


def _pose(
    values: Sequence[float],
    *,
    label: str,
) -> tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise RobotExecutionError(f"{label} must contain six values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise RobotExecutionError(f"{label} contains non-finite values")
    return result  # type: ignore[return-value]


def _validate_profile() -> None:
    speeds = (
        SETUP_SPEED_MM_S,
        SLOW_DROP_SPEED_MM_S,
        FAST_RISE_SPEED_MM_S,
        DIAGONAL_SPEED_MM_S,
        RETURN_SPEED_MM_S,
    )
    accelerations = (
        SETUP_ACCELERATION_MM_S2,
        SLOW_DROP_ACCELERATION_MM_S2,
        FAST_RISE_ACCELERATION_MM_S2,
        DIAGONAL_ACCELERATION_MM_S2,
        RETURN_ACCELERATION_MM_S2,
    )
    if any(not 0.0 < value <= MAX_EXPERIMENTAL_SPEED_MM_S for value in speeds):
        raise SafetyError(
            f"shake-open speeds must be in (0, {MAX_EXPERIMENTAL_SPEED_MM_S:g}] mm/s"
        )
    if any(
        not 0.0 < value <= MAX_EXPERIMENTAL_ACCELERATION_MM_S2
        for value in accelerations
    ):
        raise SafetyError(
            "shake-open accelerations must be in "
            f"(0, {MAX_EXPERIMENTAL_ACCELERATION_MM_S2:g}] mm/s^2"
        )
    if max(
        WORK_Z_DROP_MM,
        VERTICAL_SNAP_DROP_MM,
        DIAGONAL_X_MM,
        DIAGONAL_Y_MM,
        DIAGONAL_Z_MM,
    ) > MAX_EXPERIMENTAL_OFFSET_MM:
        raise SafetyError("shake-open offset exceeds the fixed experimental cap")


def build_shake_open_plan(
    current_pose_mm_deg: Sequence[float],
    config: RobotConfig,
    *,
    diagonal_scale: float = 1.0,
) -> ShakeOpenPlan:
    """Build the fixed centre -> vertical snaps -> diagonal figure-eight plan."""

    _validate_profile()
    scale = float(diagonal_scale)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise SafetyError("shake-open diagonal_scale must be in (0, 1]")
    current = _pose(current_pose_mm_deg, label="current TCP pose")
    x, y, z, roll, pitch, yaw = current
    if z < MINIMUM_ENTRY_Z_MM:
        raise SafetyError(
            f"shake-open test requires entry z>={MINIMUM_ENTRY_Z_MM:g} mm; "
            f"current z={z:.3f} mm"
        )
    roll_error = abs(_shortest_angle_delta_deg(roll, config.orientation_roll_deg))
    pitch_error = abs(_shortest_angle_delta_deg(pitch, config.orientation_pitch_deg))
    if roll_error > ORIENTATION_TOLERANCE_DEG or pitch_error > ORIENTATION_TOLERANCE_DEG:
        raise SafetyError(
            "current TCP orientation does not match the calibrated top-down orientation: "
            f"roll error={roll_error:.3f}, pitch error={pitch_error:.3f} deg"
        )

    if config.boundaries.z_max is None:
        raise SafetyError("shake-open test requires a configured z_max")
    highest_allowed_z = float(
        config.boundaries.z_max
        - config.workspace_margin_mm
        - CEILING_MARGIN_MM
    )
    if highest_allowed_z < MINIMUM_ENTRY_Z_MM + WORK_Z_DROP_MM:
        raise SafetyError("configured z_max leaves no room for the shake-open test")
    test_high_z = min(highest_allowed_z, max(z, TARGET_TEST_HIGH_Z_MM))
    work_z = test_high_z - WORK_Z_DROP_MM
    work_pose = (x, 0.0, work_z, roll, pitch, yaw)
    steps: list[ShakeOpenStep] = [
        ShakeOpenStep(
            "center_y0",
            (x, 0.0, z, roll, pitch, yaw),
            config.speed_mm_s,
            config.acceleration_mm_s2,
            None,
            True,
        ),
        ShakeOpenStep(
            "move_to_test_high",
            (x, 0.0, test_high_z, roll, pitch, yaw),
            SETUP_SPEED_MM_S,
            SETUP_ACCELERATION_MM_S2,
            None,
            True,
        ),
        ShakeOpenStep(
            "move_to_shake_height",
            work_pose,
            SETUP_SPEED_MM_S,
            SETUP_ACCELERATION_MM_S2,
            None,
            True,
        ),
    ]
    for cycle in range(1, VERTICAL_SNAP_CYCLES + 1):
        steps.extend(
            [
                ShakeOpenStep(
                    f"vertical_{cycle}_slow_drop",
                    (x, 0.0, work_z - VERTICAL_SNAP_DROP_MM, roll, pitch, yaw),
                    SLOW_DROP_SPEED_MM_S,
                    SLOW_DROP_ACCELERATION_MM_S2,
                    VERTICAL_BLEND_RADIUS_MM,
                    False,
                ),
                ShakeOpenStep(
                    f"vertical_{cycle}_fast_rise",
                    work_pose,
                    FAST_RISE_SPEED_MM_S,
                    FAST_RISE_ACCELERATION_MM_S2,
                    VERTICAL_BLEND_RADIUS_MM,
                    False,
                ),
            ]
        )

    diagonal_offsets = (
        (-DIAGONAL_X_MM * scale, DIAGONAL_Y_MM * scale, -DIAGONAL_Z_MM * scale),
        (DIAGONAL_X_MM * scale, -DIAGONAL_Y_MM * scale, DIAGONAL_Z_MM * scale),
        (-DIAGONAL_X_MM * scale, -DIAGONAL_Y_MM * scale, -DIAGONAL_Z_MM * scale),
        (DIAGONAL_X_MM * scale, DIAGONAL_Y_MM * scale, DIAGONAL_Z_MM * scale),
    )
    diagonal_blend_radius = max(2.0, DIAGONAL_BLEND_RADIUS_MM * scale)
    for cycle in range(1, DIAGONAL_CYCLES + 1):
        for point_index, (dx, dy, dz) in enumerate(diagonal_offsets, start=1):
            steps.append(
                ShakeOpenStep(
                    f"diagonal_{cycle}_{point_index}",
                    (x + dx, dy, work_z + dz, roll, pitch, yaw),
                    DIAGONAL_SPEED_MM_S,
                    DIAGONAL_ACCELERATION_MM_S2,
                    diagonal_blend_radius,
                    False,
                )
            )
    steps.append(
        ShakeOpenStep(
            "return_center",
            work_pose,
            RETURN_SPEED_MM_S,
            RETURN_ACCELERATION_MM_S2,
            diagonal_blend_radius,
            False,
        )
    )
    steps.append(
        ShakeOpenStep(
            "return_test_high",
            (x, 0.0, test_high_z, roll, pitch, yaw),
            SETUP_SPEED_MM_S,
            SETUP_ACCELERATION_MM_S2,
            None,
            True,
        )
    )

    config.validate_workspace_pose(
        x,
        y,
        z,
        relative_yaw_deg=config.relative_yaw_from_absolute_deg(current[5]),
        require_complete=True,
    )
    for step in steps:
        target = step.target_pose_mm_deg
        config.validate_workspace_pose(
            target[0],
            target[1],
            target[2],
            relative_yaw_deg=config.relative_yaw_from_absolute_deg(target[5]),
            require_complete=True,
        )
        if target[3:] != current[3:]:
            raise SafetyError(f"{step.name} changed the fixed TCP orientation")
    return ShakeOpenPlan(
        start_pose_mm_deg=current,
        work_pose_mm_deg=work_pose,
        vertical_snap_cycles=VERTICAL_SNAP_CYCLES,
        diagonal_cycles=DIAGONAL_CYCLES,
        diagonal_scale=scale,
        steps=tuple(steps),
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


def validate_shake_open_with_controller(
    arm: Any,
    plan: ShakeOpenPlan,
) -> dict[str, Any]:
    """Validate every straight segment at 5 mm spacing with limited live-pose IK."""

    _, warning_code = _controller_error_warning(arm)
    reference_deg = _read_live_joints_deg(arm)
    segment_start = plan.start_pose_mm_deg
    sample_total = 0
    for step_index, step in enumerate(plan.steps, start=1):
        target = step.target_pose_mm_deg
        distance = math.dist(segment_start[:3], target[:3])
        samples = max(1, int(math.ceil(distance / IK_SAMPLE_STEP_MM)))
        for sample_index in range(1, samples + 1):
            fraction = sample_index / samples
            sample_pose = [
                segment_start[index]
                + fraction * (target[index] - segment_start[index])
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
                    f"controller IK rejected shake-open step {step_index} ({step.name}) "
                    f"sample {sample_index}/{samples}, pose={sample_pose}, code={code}"
                )
            reference_deg = [float(value) for value in result[1]]
            if not all(math.isfinite(value) for value in reference_deg):
                raise RobotExecutionError("controller IK returned non-finite joints")
            sample_total += 1
        segment_start = target
    return {
        "controller_warning_code": warning_code,
        "validated_sample_count": sample_total,
        "ik_sample_step_mm": IK_SAMPLE_STEP_MM,
        "blended_intermediate_steps": sum(
            step.blend_radius_mm is not None for step in plan.steps
        ),
    }


def select_controller_valid_shake_open_plan(
    arm: Any,
    current_pose_mm_deg: Sequence[float],
    config: RobotConfig,
    *,
    scale_candidates: Sequence[float] = DIAGONAL_SCALE_CANDIDATES,
) -> tuple[ShakeOpenPlan, dict[str, Any], list[dict[str, Any]]]:
    """Select the strongest controller-valid diagonal amplitude before motion."""

    candidates = tuple(float(value) for value in scale_candidates)
    if not candidates:
        raise SafetyError("shake-open requires at least one diagonal scale candidate")
    trials: list[dict[str, Any]] = []
    last_error: SafetyError | None = None
    for scale in candidates:
        plan = build_shake_open_plan(
            current_pose_mm_deg,
            config,
            diagonal_scale=scale,
        )
        try:
            validation = validate_shake_open_with_controller(arm, plan)
        except SafetyError as exc:
            last_error = exc
            trials.append(
                {
                    "diagonal_scale": scale,
                    "status": "IK_REJECTED",
                    "error": str(exc),
                }
            )
            continue
        trials.append({"diagonal_scale": scale, "status": "IK_ACCEPTED"})
        validation["selected_diagonal_scale"] = scale
        validation["adaptive_scale_trials"] = trials
        return plan, validation, trials
    assert last_error is not None
    raise SafetyError(
        "controller IK rejected every shake-open diagonal scale "
        f"{list(candidates)}; last error: {last_error}"
    ) from last_error


def _sdk_code(result: Any) -> int:
    return int(result[0]) if isinstance(result, tuple) else int(result)


def _check_sdk(method: str, result: Any) -> None:
    code = _sdk_code(result)
    if code != 0:
        raise RobotExecutionError(f"{method} failed, code={code}")


def execute_shake_open(arm: Any, plan: ShakeOpenPlan) -> list[dict[str, Any]]:
    """Queue blended intermediate moves and wait only at setup/final stop points."""

    _check_sdk("motion_enable", arm.motion_enable(enable=True))
    _check_sdk("set_mode", arm.set_mode(0))
    _check_sdk("set_state", arm.set_state(0))
    records: list[dict[str, Any]] = []
    motion_started = False
    try:
        for step in plan.steps:
            motion_started = True
            target = step.target_pose_mm_deg
            queued_at = _timestamp()
            result = arm.set_position(
                x=target[0],
                y=target[1],
                z=target[2],
                roll=target[3],
                pitch=target[4],
                yaw=target[5],
                radius=step.blend_radius_mm,
                speed=step.speed_mm_s,
                mvacc=step.acceleration_mm_s2,
                wait=step.wait,
                is_radian=False,
            )
            _check_sdk(f"set_position({step.name})", result)
            records.append(
                {
                    "name": step.name,
                    "queued_at": queued_at,
                    "command_returned_at": _timestamp(),
                    "target_pose_mm_deg": list(target),
                    "speed_mm_s": step.speed_mm_s,
                    "acceleration_mm_s2": step.acceleration_mm_s2,
                    "blend_radius_mm": step.blend_radius_mm,
                    "wait": step.wait,
                }
            )
        actual = read_live_pose(arm)
        final_target = plan.steps[-1].target_pose_mm_deg
        final_error = math.dist(actual[:3], final_target[:3])
        if final_error > FINAL_POSITION_TOLERANCE_MM:
            raise RobotExecutionError(
                f"shake-open final pose error {final_error:.3f} mm exceeds "
                f"{FINAL_POSITION_TOLERANCE_MM:.3f} mm"
            )
        _controller_error_warning(arm)
        records[-1]["actual_pose_mm_deg"] = list(actual)
        records[-1]["final_position_error_mm"] = final_error
        return records
    except BaseException:
        if motion_started:
            try:
                arm.set_state(4)
            except BaseException:
                pass
        raise


def shake_open(arm: Any, config: RobotConfig) -> dict[str, Any]:
    """Run the complete live-pose validation and composite shake-open test."""

    config.validate_for_real()
    tcp_offset = _validated_live_tcp_offset(arm, config)
    plan, validation, _ = select_controller_valid_shake_open_plan(
        arm,
        read_live_pose(arm),
        config,
    )
    validation["tcp_offset_mm_deg"] = list(tcp_offset)
    return {
        "plan": plan.as_dict(),
        "controller_validation": validation,
        "execution": execute_shake_open(arm, plan),
    }


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
