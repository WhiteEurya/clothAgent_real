"""The only robot surface visible to generated experiment programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import io
import sys
import time
from typing import Any, Protocol

from .config import RobotConfig, SafetyError


class RobotExecutionError(RuntimeError):
    """Raised when the xArm SDK reports a command failure."""


@dataclass(frozen=True)
class ControllerTrajectoryValidation:
    """Read-only controller IK result used for validation and URDF animation."""

    joint_targets_rad: dict[int, tuple[float, ...]]
    controller_warning_code: int
    tcp_offset_mm_deg: tuple[float, ...]
    validated_sample_count: int
    shake_open_diagonal_scales: dict[int, float] = field(default_factory=dict)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _gripper_log(message: str) -> None:
    """Keep the experiment log and make feedback waits visible during capture."""
    print(message, flush=True)
    # Experiment execution redirects stdout to StringIO. Also show waits on
    # the terminal so an unbounded feedback wait never looks like a frozen run.
    if isinstance(sys.stdout, io.StringIO) and sys.__stdout__ is not None:
        print(message, file=sys.__stdout__, flush=True)


def _validated_live_tcp_offset(arm: Any, config: RobotConfig) -> tuple[float, ...]:
    """Wait briefly for the xArm report thread before checking the saved TCP."""

    expected = config.expected_tcp_offset_mm_deg
    actual = getattr(arm, "tcp_offset", None)

    def matches(value: Any) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) == 6
            and all(
                abs(float(observed) - target) <= config.tcp_offset_tolerance
                for observed, target in zip(value, expected)
            )
        )

    if not matches(actual):
        reader = getattr(arm, "get_position", None)
        if callable(reader):
            reader(is_radian=False)
        for _ in range(20):
            actual = getattr(arm, "tcp_offset", None)
            if matches(actual):
                break
            time.sleep(0.05)
    config.validate_live_tcp_offset(actual)
    return tuple(float(value) for value in actual)


def _read_gripper_feedback(arm: Any) -> dict[str, Any]:
    """Read vendor telemetry and retain failures for the caller to interpret.

    xArm gripper firmware >= 3.4.3 exposes a status register whose low two
    bits distinguish stop (0), motion (1), and catch/grasp (2).  Reads are
    best-effort for general diagnostics. The gripper command completion gate
    requires valid feedback and retries read failures while holding the arm.
    """

    feedback: dict[str, Any] = {
        "sampled_at": _timestamp(),
        "status_code": None,
        "status_raw": None,
        "state_code": None,
        "state": "unavailable",
        "position_pulse": None,
        "error_code": None,
        "read_errors": [],
    }

    def result_value(name: str, getter: Any) -> Any:
        try:
            result = getter()
        except Exception as exc:
            feedback["read_errors"].append(
                {"field": name, "error": f"{type(exc).__name__}: {exc}"}
            )
            return None
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            try:
                code = int(result[0])
            except (TypeError, ValueError, OverflowError):
                feedback['read_errors'].append({'field': name, 'error': 'invalid SDK return code'})
                return None
            feedback[f"{name}_code"] = code
            if code != 0:
                feedback["read_errors"].append(
                    {"field": name, "code": code}
                )
                return None
            return result[1]
        feedback["read_errors"].append({"field": name, "error": "invalid SDK result", "result": repr(result)})
        return None

    position = result_value("position", getattr(arm, "get_gripper_position", None))
    if position is not None:
        try:
            feedback["position_pulse"] = int(position)
        except (TypeError, ValueError, OverflowError):
            feedback["read_errors"].append(
                {"field": "position", "error": "non-numeric position"}
            )

    status_getter = getattr(arm, "get_gripper_status", None)
    if callable(status_getter):
        raw_status = result_value("status", status_getter)
        if raw_status is not None:
            try:
                raw_status = int(raw_status)
                state_code = raw_status & 0x03
                feedback["status_raw"] = raw_status
                feedback["state_code"] = state_code
                feedback["state"] = {
                    0: "stop",
                    1: "moving",
                    2: "grasp",
                    3: "error",
                }.get(state_code, "unknown")
            except (TypeError, ValueError, OverflowError):
                feedback["read_errors"].append(
                    {"field": "status", "error": "non-numeric status"}
                )
    else:
        feedback["read_errors"].append(
            {"field": "status", "error": "getter unavailable"}
        )

    error_getter = getattr(arm, "get_gripper_err_code", None)
    if callable(error_getter):
        error_code = result_value("error", error_getter)
        if error_code is not None:
            try:
                feedback["error_code"] = int(error_code)
            except (TypeError, ValueError, OverflowError):
                feedback["read_errors"].append(
                    {"field": "error", "error": "non-numeric error code"}
                )
    else:
        feedback["read_errors"].append(
            {"field": "error", "error": "getter unavailable"}
        )
    feedback["mechanical_grasp_detected"] = feedback["state"] == "grasp"
    feedback["available"] = bool(
        feedback["position_pulse"] is not None
        or feedback["status_raw"] is not None
        or feedback["error_code"] is not None
    )
    return feedback


def _controller_home_pose(arm: Any, config: RobotConfig) -> list[float]:
    """Read the controller's Cartesian pose for the configured home joints."""

    forward_kinematics = getattr(arm, "get_forward_kinematics", None)
    if not callable(forward_kinematics):
        raise RobotExecutionError(
            "xArm controller does not expose forward kinematics for the configured home"
        )
    result = forward_kinematics(
        list(config.init_joints_deg),
        input_is_radian=False,
        return_is_radian=False,
    )
    if (
        not isinstance(result, (list, tuple))
        or len(result) < 2
        or int(result[0]) != 0
        or not isinstance(result[1], (list, tuple))
        or len(result[1]) != 6
    ):
        raise RobotExecutionError(
            "xArm forward kinematics rejected the configured home joints: "
            f"result={result}"
        )
    pose = [float(value) for value in result[1]]
    if not all(math.isfinite(value) for value in pose):
        raise RobotExecutionError(
            "xArm forward kinematics returned non-finite home pose values"
        )
    return pose


def _shortest_angle_delta_deg(start: float, target: float) -> float:
    """Return the shortest signed Euler-angle delta in degrees."""

    return (target - start + 180.0) % 360.0 - 180.0


def _controller_trajectory_with_arm(
    arm: Any,
    config: RobotConfig,
    actions: list[dict[str, Any]],
) -> ControllerTrajectoryValidation:
    """Validate sampled Cartesian segments with joint-limited controller IK."""

    live_tcp_offset = _validated_live_tcp_offset(arm, config)
    err_warn = arm.get_err_warn_code()
    error_code = 0
    warning_code = 0
    if isinstance(err_warn, tuple) and len(err_warn) >= 2 and int(err_warn[0]) == 0:
        values = err_warn[1]
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            error_code, warning_code = int(values[0]), int(values[1])
    if error_code != 0:
        raise RobotExecutionError(f"xArm controller has active error code {error_code}")

    targets: dict[int, tuple[float, ...]] = {}
    reference_deg = [float(value) for value in config.init_joints_deg]
    home_pose = _controller_home_pose(arm, config)
    current_pose = list(home_pose)
    validated_sample_count = 0
    shake_open_diagonal_scales: dict[int, float] = {}

    def validate_pose_segment(
        pose: list[float],
        *,
        action_index: int,
        substep_label: str | None = None,
    ) -> None:
        nonlocal current_pose, reference_deg, validated_sample_count

        cartesian_distance_mm = math.dist(current_pose[:3], pose[:3])
        angular_deltas = [
            _shortest_angle_delta_deg(current_pose[index], pose[index])
            for index in range(3, 6)
        ]
        angular_distance_deg = max(abs(value) for value in angular_deltas)
        sample_count = max(
            1,
            int(math.ceil(cartesian_distance_mm / 10.0)),
            int(math.ceil(angular_distance_deg / 10.0)),
        )
        if cartesian_distance_mm < 1e-9 and angular_distance_deg < 1e-9:
            current_pose = list(pose)
            return
        segment_start = list(current_pose)
        action_description = f"action {action_index + 1}"
        if substep_label is not None:
            action_description += f" ({substep_label})"
        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / sample_count
            sample_pose = [
                segment_start[index]
                + fraction * (pose[index] - segment_start[index])
                for index in range(6)
            ]
            for index, delta in zip(range(3, 6), angular_deltas):
                sample_pose[index] = segment_start[index] + fraction * delta
            if sample_index == sample_count:
                sample_pose = list(pose)
            # Validate every interpolated TCP sample against the same
            # yaw-dependent Y envelope used by RobotAPI.move().  The xArm IK
            # check remains the final authority, but a segment must not pass
            # through the static Y boundary while rotating the gripper.
            config.validate_workspace_pose(
                sample_pose[0],
                sample_pose[1],
                sample_pose[2],
                relative_yaw_deg=config.relative_yaw_from_absolute_deg(
                    sample_pose[5]
                ),
            )
            code, angles_deg = arm.get_inverse_kinematics(
                sample_pose,
                input_is_radian=False,
                return_is_radian=False,
                limited=True,
                ref_angles=reference_deg,
            )
            if (
                int(code) != 0
                or not isinstance(angles_deg, (list, tuple))
                or len(angles_deg) != 7
            ):
                raise SafetyError(
                    f"controller IK rejected {action_description} "
                    f"segment sample {sample_index}/{sample_count} "
                    f"pose={sample_pose}, code={code}"
                )
            next_reference = [float(value) for value in angles_deg]
            if not all(math.isfinite(value) for value in next_reference):
                raise RobotExecutionError(
                    "controller IK returned non-finite joints for "
                    f"{action_description} segment sample "
                    f"{sample_index}/{sample_count}"
                )
            reference_deg = next_reference
            validated_sample_count += 1
        current_pose = list(pose)

    for action_index, action in enumerate(actions):
        name = action.get("name")
        if name == "home":
            targets[action_index] = tuple(math.radians(value) for value in config.init_joints_deg)
            reference_deg = [float(value) for value in config.init_joints_deg]
            current_pose = list(home_pose)
            continue
        if name == "shake_open":
            from .shake_open_test import (
                DIAGONAL_SCALE_CANDIDATES,
                build_shake_open_plan,
            )

            start_pose = list(current_pose)
            start_reference = list(reference_deg)
            start_sample_count = validated_sample_count
            last_error: SafetyError | None = None
            selected_plan = None
            for scale in DIAGONAL_SCALE_CANDIDATES:
                current_pose = list(start_pose)
                reference_deg = list(start_reference)
                validated_sample_count = start_sample_count
                plan = build_shake_open_plan(
                    current_pose,
                    config,
                    diagonal_scale=scale,
                )
                try:
                    for step in plan.steps:
                        validate_pose_segment(
                            list(step.target_pose_mm_deg),
                            action_index=action_index,
                            substep_label=f"shake_open:{step.name}",
                        )
                except SafetyError as exc:
                    last_error = exc
                    continue
                selected_plan = plan
                shake_open_diagonal_scales[action_index] = scale
                break
            if selected_plan is None:
                current_pose = start_pose
                reference_deg = start_reference
                validated_sample_count = start_sample_count
                assert last_error is not None
                raise SafetyError(
                    "controller IK rejected every shake_open diagonal scale "
                    f"{list(DIAGONAL_SCALE_CANDIDATES)}; last error: {last_error}"
                ) from last_error
            targets[action_index] = tuple(
                math.radians(value) for value in reference_deg
            )
            continue
        poses: list[tuple[str | None, list[float]]] = []
        if name == "move":
            args = action.get("args", {})
            poses.append(
                (
                    None,
                    [
                        float(args["x"]),
                        float(args["y"]),
                        float(args["z"]),
                        config.orientation_roll_deg,
                        config.orientation_pitch_deg,
                        config.command_yaw_deg(float(args["yaw"])),
                    ],
                )
            )
        elif name == "shake":
            from .shake_once import build_shake_plan

            plan = build_shake_plan(current_pose, config)
            poses.extend(
                (
                    f"shake:{step.name}",
                    list(step.target_pose_mm_deg),
                )
                for step in plan.steps
            )
        else:
            continue
        for substep_label, pose in poses:
            validate_pose_segment(
                pose,
                action_index=action_index,
                substep_label=substep_label,
            )
        targets[action_index] = tuple(math.radians(value) for value in reference_deg)
    return ControllerTrajectoryValidation(
        joint_targets_rad=targets,
        controller_warning_code=warning_code,
        tcp_offset_mm_deg=live_tcp_offset,
        validated_sample_count=validated_sample_count,
        shake_open_diagonal_scales=shake_open_diagonal_scales,
    )


def validate_controller_trajectory(
    config: RobotConfig,
    actions: list[dict[str, Any]],
) -> ControllerTrajectoryValidation:
    """Connect read-only, validate TCP/tool state and solve all planned IK targets."""

    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise RobotExecutionError("xarm package is required for controller IK validation") from exc
    arm = XArmAPI(config.robot_ip)
    try:
        if not getattr(arm, "connected", True):
            raise RobotExecutionError(f"unable to connect to xArm at {config.robot_ip}")
        return _controller_trajectory_with_arm(arm, config, actions)
    finally:
        if getattr(arm, "connected", False):
            arm.disconnect()


@dataclass
class ActionRecord:
    name: str
    args: dict[str, Any]
    requested_at: str
    completed_at: str | None = None
    success: bool = False
    error: str | None = None
    actual_ee_pose: list[float] | None = None
    robot_state: Any = None
    gripper_result: Any = None


class Backend(Protocol):
    def move(self, x: float, y: float, z: float, yaw: float, config: RobotConfig) -> tuple[list[float] | None, Any]: ...
    def open_gripper(self, config: RobotConfig) -> tuple[Any, tuple[list[float] | None, Any]]: ...
    def close_gripper(self, config: RobotConfig) -> tuple[Any, tuple[list[float] | None, Any]]: ...
    def shake(self, config: RobotConfig) -> tuple[list[float] | None, Any]: ...
    def shake_open(self, config: RobotConfig) -> tuple[list[float] | None, Any]: ...
    def home(self, config: RobotConfig) -> tuple[list[float] | None, Any]: ...
    def perception_position(self, config: RobotConfig) -> tuple[list[float] | None, Any]: ...
    def close(self) -> None: ...


class SimulatedBackend:
    """No-motion backend used for preflight and tests."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.pose = list(config.init_pose_mm_deg)
        self.gripper = config.gripper_open
        self.state = "simulated"

    def _gripper_feedback(self) -> dict[str, Any]:
        return {
            "sampled_at": _timestamp(),
            "available": True,
            "status_code": 0,
            "status_raw": 0,
            "state_code": 0,
            "state": "stop",
            "position_pulse": int(self.gripper),
            "error_code": 0,
            "mechanical_grasp_detected": False,
            "simulated": True,
            "read_errors": [],
        }

    def _state(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "gripper_feedback": self._gripper_feedback(),
        }

    def move(self, x: float, y: float, z: float, yaw: float, config: RobotConfig):
        self.pose = [
            x,
            y,
            z,
            config.orientation_roll_deg,
            config.orientation_pitch_deg,
            config.command_yaw_deg(yaw),
        ]
        return list(self.pose), self._state()

    def open_gripper(self, config: RobotConfig):
        self.gripper = config.gripper_open
        return {"position": self.gripper, "simulated": True, "feedback": self._gripper_feedback()}, (list(self.pose), self._state())

    def close_gripper(self, config: RobotConfig):
        self.gripper = config.gripper_close
        return {"position": self.gripper, "simulated": True, "feedback": self._gripper_feedback()}, (list(self.pose), self._state())

    def shake(self, config: RobotConfig):
        from .shake_once import build_shake_plan

        plan = build_shake_plan(self.pose, config)
        self.pose = list(plan.steps[-1].target_pose_mm_deg)
        return list(self.pose), {"state": self.state, "shake": plan.as_dict(), "gripper_feedback": self._gripper_feedback()}

    def shake_open(self, config: RobotConfig):
        from .shake_open_test import build_shake_open_plan

        plan = build_shake_open_plan(self.pose, config)
        self.pose = list(plan.steps[-1].target_pose_mm_deg)
        return list(self.pose), {"state": self.state, "shake_open": plan.as_dict(), "gripper_feedback": self._gripper_feedback()}

    def home(self, config: RobotConfig):
        self.pose = list(config.init_pose_mm_deg)
        return list(self.pose), self._state()

    def perception_position(self, config: RobotConfig):
        if config.perception_pose_mm_deg is None:
            raise RobotExecutionError("perception_position is not configured")
        self.pose = list(config.perception_pose_mm_deg)
        return list(self.pose), self._state()

    def close(self) -> None:
        return None


class XArmBackend:
    """Thin adapter around the project's actual xArm SDK interface.

    The SDK is imported lazily so validation and dry-run operation work on a
    machine without the vendor package or a connected robot.
    """

    def __init__(self, config: RobotConfig):
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:  # pragma: no cover - depends on robot host
            raise RobotExecutionError(
                "xarm package is not installed; install the vendor SDK on the robot host"
            ) from exc
        self.arm = XArmAPI(config.robot_ip)
        try:
            if not getattr(self.arm, "connected", True):
                raise RobotExecutionError(f"unable to connect to xArm at {config.robot_ip}")
            _validated_live_tcp_offset(self.arm, config)
            self._check("motion_enable", self.arm.motion_enable(enable=True))
            self._check("set_mode", self.arm.set_mode(0))
            self._check("set_state", self.arm.set_state(0))
            # These are the concrete gripper APIs used by the existing project.
            self._check("set_gripper_mode", self.arm.set_gripper_mode(0))
            self._check("set_gripper_enable", self.arm.set_gripper_enable(True))
            self._check("set_gripper_speed", self.arm.set_gripper_speed(config.gripper_speed))
        except BaseException:
            if getattr(self.arm, "connected", False):
                self.arm.disconnect()
            raise

    @staticmethod
    def _code(value: Any) -> int:
        if isinstance(value, tuple):
            return int(value[0])
        return int(value)

    @classmethod
    def _check(cls, method: str, value: Any) -> Any:
        code = cls._code(value)
        if code != 0:
            raise RobotExecutionError(f"{method} failed, code={code}")
        return value

    def _state(self) -> tuple[list[float] | None, Any]:
        position = self.arm.get_position()
        pose: list[float] | None = None
        if isinstance(position, tuple) and len(position) >= 2 and int(position[0]) == 0:
            pose = [float(v) for v in position[1]]
        state = {
            "state": self.arm.get_state(),
            "error_warn": self.arm.get_err_warn_code(),
            "gripper_feedback": _read_gripper_feedback(self.arm),
        }
        angles = self.arm.get_servo_angle(is_radian=False)
        if (
            isinstance(angles, tuple)
            and len(angles) >= 2
            and int(angles[0]) == 0
            and isinstance(angles[1], (list, tuple))
        ):
            state["servo_angles_deg"] = [float(value) for value in angles[1]]
        return pose, state

    def _checked_gripper_feedback(self):
        feedback = _read_gripper_feedback(self.arm)
        # A failed read is not a hardware fault or completion. Keep polling.
        # Only successfully decoded fault feedback stops the command here.
        if feedback['error_code'] not in (None, 0) or feedback['state'] == 'error':
            exc = RobotExecutionError(f'xArm gripper hardware fault: {feedback}')
            exc.gripper_feedback = feedback
            raise exc
        feedback['usable_for_completion'] = bool(
            getattr(self.arm, 'connected', True) and not feedback['read_errors'] and
            feedback['position_pulse'] is not None and feedback['error_code'] == 0 and
            feedback['state'] in {'stop', 'moving', 'grasp'})
        return feedback

    def _command_gripper(self, config: RobotConfig, *, target: str):
        """Poll until measured completion or operator interrupt; never time out into motion."""
        target_position = config.gripper_open if target == 'open' else config.gripper_close
        started = time.monotonic()
        trace = {'target': target, 'target_position_pulse': target_position,
                 'position_tolerance_pulse': config.gripper_open_tolerance_pulse if target == 'open' else 5.0,
                 'samples': [], 'status': 'WAITING',
                 'timeout_s': None, 'wait_policy': 'feedback_until_complete_or_interrupt',
                 'warning_after_s': config.gripper_completion_timeout_s,
                 'sample_count': 0, 'dropped_sample_count': 0}
        last_print = -1.
        before = None

        def record_sample(feedback, phase, *, progress=None, reason=None):
            nonlocal last_print
            elapsed = time.monotonic() - started
            trace['sample_count'] += 1
            trace['samples'].append({'elapsed_s': elapsed, 'phase': phase, 'feedback': feedback,
                                    'progress_pulse': progress, 'completion_reason': reason})
            # Waiting can last indefinitely. Keep a bounded recent history plus
            # the initial state, total count, and final result.
            if len(trace['samples']) > 200:
                del trace['samples'][0]
                trace['dropped_sample_count'] += 1
            interval = .5 if elapsed < trace['warning_after_s'] else 5.
            if elapsed - last_print >= interval or reason:
                waiting = 'READ_RETRY' if not feedback['usable_for_completion'] else 'WAITING'
                _gripper_log(f'[gripper] {target}: phase={phase}, elapsed={elapsed:.2f}s, '
                    f'position={feedback["position_pulse"]}, target={target_position}, '
                    f'tolerance={trace["position_tolerance_pulse"]}, state={feedback["state"]}, '
                    f'progress={progress}, result={reason or waiting}, '
                    f'read_errors={feedback["read_errors"]}; '
                    + ('completion confirmed' if reason else 'arm stays still; Ctrl+C to interrupt'))
                last_print = elapsed
            return elapsed

        try:
            _gripper_log(f'[gripper] {target}: acquiring initial feedback; arm stays still')
            while True:
                before = self._checked_gripper_feedback()
                if before['usable_for_completion'] and before['state'] != 'moving':
                    break
                record_sample(before, 'before_command')
                time.sleep(.05)
            trace['before_command'] = before
            initial = before['position_pulse']
            direction = 1 if target_position > initial else -1
            _gripper_log(f'[gripper] {target}: before={initial}, target={target_position}; waiting for measured completion')
            # The SDK's wait=True itself has status/no-progress success paths.
            # Keep its default wait_motion=True, but own gripper completion here.
            result = self._check('set_gripper_position', self.arm.set_gripper_position(
                target_position, speed=config.gripper_speed, wait=False))
            trace['command_result'] = result
            while True:
                feedback = self._checked_gripper_feedback()
                if not feedback['usable_for_completion']:
                    record_sample(feedback, 'after_command')
                    time.sleep(.05)
                    continue
                position = feedback['position_pulse']
                state = feedback['state']
                progress = direction * (position - initial)
                at_target = abs(position - target_position) <= trace['position_tolerance_pulse']
                reason = None
                if state in {'stop', 'grasp'} and at_target:
                    reason = 'measured_target_reached'
                # Real telemetry reports grasp even near the open endpoint
                # and during travel. It must never bypass measured position.
                elapsed = record_sample(feedback, 'after_command', progress=progress, reason=reason)
                if reason:
                    trace.update(status='COMPLETED', reason=reason, duration_s=elapsed)
                    pose, robot_state = self._state()
                    robot_state['gripper_feedback'] = feedback
                    robot_state['gripper_completion'] = trace
                    return {'command_result': result, 'feedback': feedback, 'completion': trace}, (pose, robot_state)
                time.sleep(.05)
        except BaseException as exc:
            trace.update(status='FAILED', duration_s=time.monotonic()-started,
                         error=f'{type(exc).__name__}: {exc}')
            if hasattr(exc, 'gripper_feedback'):
                trace['failed_feedback'] = exc.gripper_feedback
            exc.gripper_completion = trace
            _gripper_log(f'[gripper] {target}: FAILED; next robot action blocked: {exc}')
            raise

    def move(self, x: float, y: float, z: float, yaw: float, config: RobotConfig):
        code = self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=config.orientation_roll_deg,
            pitch=config.orientation_pitch_deg,
            yaw=config.command_yaw_deg(yaw),
            speed=config.speed_mm_s,
            mvacc=config.acceleration_mm_s2,
            wait=True,
            is_radian=False,
        )
        self._check("set_position", code)
        return self._state()

    def open_gripper(self, config: RobotConfig):
        return self._command_gripper(config, target='open')

    def close_gripper(self, config: RobotConfig):
        return self._command_gripper(config, target='close')

    def shake(self, config: RobotConfig):
        from .shake_once import shake

        shake_result = shake(self.arm, config)
        pose, state = self._state()
        if isinstance(state, dict):
            state["shake"] = shake_result
        return pose, state

    def shake_open(self, config: RobotConfig):
        from .shake_open_test import shake_open

        shake_open_result = shake_open(self.arm, config)
        pose, state = self._state()
        if isinstance(state, dict):
            state["shake_open"] = shake_open_result
        return pose, state

    def home(self, config: RobotConfig):
        code = self.arm.set_servo_angle(
            angle=list(config.init_joints_deg),
            speed=config.home_speed_deg_s,
            mvacc=config.home_acceleration_deg_s2,
            wait=True,
            is_radian=False,
        )
        self._check("set_servo_angle(home)", code)
        return self._state()

    def perception_position(self, config: RobotConfig):
        target = config.perception_joints_deg
        if target is None or len(target) != 7:
            raise RobotExecutionError(
                "perception_position requires seven configured joint angles"
            )
        code = self.arm.set_servo_angle(
            angle=list(target),
            speed=config.home_speed_deg_s,
            mvacc=config.home_acceleration_deg_s2,
            wait=True,
            is_radian=False,
        )
        self._check("set_servo_angle(perception_position)", code)
        actual = self._state()
        state = actual[1]
        actual_joints = state.get("servo_angles_deg") if isinstance(state, dict) else None
        if not isinstance(actual_joints, list) or len(actual_joints) != 7:
            raise RobotExecutionError(
                "xArm did not report seven joint angles at perception_position"
            )
        max_error = max(
            abs(observed - expected)
            for observed, expected in zip(actual_joints, target)
        )
        if max_error > 1.0:
            raise RobotExecutionError(
                "perception_position joint verification failed: "
                f"maximum error {max_error:.3f} deg exceeds 1.000 deg"
            )
        state["perception_position_max_joint_error_deg"] = max_error
        return actual

    def close(self) -> None:
        if getattr(self.arm, "connected", False):
            self.arm.disconnect()


def move_robot_to_perception_position(
    config: RobotConfig,
    *,
    backend: Backend | None = None,
) -> dict[str, Any]:
    """Move through Home into the recorded camera-clear perception position.

    This is an internal acquisition transition and is deliberately not exposed
    to generated RobotAPI programs.  Joint-space motion preserves the recorded
    arm shape; the stored TCP pose is retained as an audit/reference value.
    """

    if config.perception_joints_deg is None or config.perception_pose_mm_deg is None:
        raise RobotExecutionError(
            "perception_position is not configured; provide perception_pose_file"
        )
    if len(config.perception_joints_deg) != 7:
        raise RobotExecutionError(
            "perception_position requires seven configured joint angles"
        )
    if len(config.perception_pose_mm_deg) != 6:
        raise RobotExecutionError(
            "perception_position requires a six-value recorded TCP pose"
        )
    controller = backend or XArmBackend(config)
    started_at = _timestamp()
    try:
        home_pose, home_state = controller.home(config)
        perception_pose, perception_state = controller.perception_position(config)
        return {
            "name": "perception_position",
            "started_at": started_at,
            "completed_at": _timestamp(),
            "sequence": ["home", "perception_position"],
            "target_joint_angles_deg": list(config.perception_joints_deg),
            "recorded_tcp_pose_mm_deg": list(config.perception_pose_mm_deg),
            "home_actual_tcp_pose_mm_deg": home_pose,
            "home_robot_state": home_state,
            "actual_tcp_pose_mm_deg": perception_pose,
            "robot_state": perception_state,
        }
    finally:
        controller.close()


class RobotAPI:
    """Safety-checked facade injected into generated experiment code.

    Generated code never receives ``XArmAPI`` or a backend. It can only call
    the six methods below. Every command is recorded, and the first failure
    permanently halts the run so a script cannot continue after an error.
    """

    ALLOWED_METHODS = frozenset(
        {"move", "open_gripper", "close_gripper", "shake", "shake_open", "home"}
    )

    def __init__(self, config: RobotConfig, backend: Backend):
        self.config = config
        self.backend = backend
        self.actions: list[ActionRecord] = []
        self.halted = False

    def _begin(self, name: str, args: dict[str, Any]) -> ActionRecord:
        if self.halted:
            raise RobotExecutionError("robot execution is halted after a previous failure")
        record = ActionRecord(name=name, args=args, requested_at=_timestamp())
        self.actions.append(record)
        return record

    def _finish(self, record: ActionRecord, *, actual: tuple[list[float] | None, Any] | None = None, gripper: Any = None) -> None:
        record.completed_at = _timestamp()
        record.success = True
        record.gripper_result = gripper
        if actual is not None:
            record.actual_ee_pose, record.robot_state = actual

    def _fail(self, record: ActionRecord, exc: BaseException) -> None:
        record.completed_at = _timestamp()
        record.error = f"{type(exc).__name__}: {exc}"
        if hasattr(exc, 'gripper_completion'):
            record.gripper_result = {'completion': exc.gripper_completion}
        self.halted = True

    def move(self, x: float, y: float, z: float, yaw: float) -> None:
        record = self._begin("move", {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw)})
        try:
            self.config.validate_workspace_pose(
                record.args["x"],
                record.args["y"],
                record.args["z"],
                relative_yaw_deg=record.args["yaw"],
            )
            actual = self.backend.move(record.args["x"], record.args["y"], record.args["z"], record.args["yaw"], self.config)
            self._finish(record, actual=actual)
        except BaseException as exc:
            self._fail(record, exc)
            raise

    def open_gripper(self) -> None:
        record = self._begin("open_gripper", {})
        try:
            gripper, actual = self.backend.open_gripper(self.config)
            self._finish(record, actual=actual, gripper=gripper)
        except BaseException as exc:
            self._fail(record, exc)
            raise

    def close_gripper(self) -> None:
        record = self._begin("close_gripper", {})
        try:
            gripper, actual = self.backend.close_gripper(self.config)
            self._finish(record, actual=actual, gripper=gripper)
        except BaseException as exc:
            self._fail(record, exc)
            raise

    def shake(self) -> None:
        record = self._begin("shake", {})
        try:
            self._finish(record, actual=self.backend.shake(self.config))
        except BaseException as exc:
            self._fail(record, exc)
            raise

    def shake_open(self) -> None:
        record = self._begin("shake_open", {})
        try:
            self._finish(record, actual=self.backend.shake_open(self.config))
        except BaseException as exc:
            self._fail(record, exc)
            raise

    def home(self) -> None:
        record = self._begin("home", {})
        try:
            self._finish(record, actual=self.backend.home(self.config))
        except BaseException as exc:
            self._fail(record, exc)
            raise

    def close(self) -> None:
        self.backend.close()

    def action_dicts(self) -> list[dict[str, Any]]:
        return [asdict(action) for action in self.actions]
