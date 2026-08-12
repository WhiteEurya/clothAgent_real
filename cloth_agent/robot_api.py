"""The only robot surface visible to generated experiment programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
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


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


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


def _controller_trajectory_with_arm(
    arm: Any,
    config: RobotConfig,
    actions: list[dict[str, Any]],
) -> ControllerTrajectoryValidation:
    """Validate every Cartesian target with controller IK without enabling motion."""

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
    cached_pose_targets: dict[tuple[float, ...], tuple[float, ...]] = {}
    reference_deg = [float(value) for value in config.init_joints_deg]
    for action_index, action in enumerate(actions):
        name = action.get("name")
        if name == "home":
            targets[action_index] = tuple(math.radians(value) for value in config.init_joints_deg)
            reference_deg = [float(value) for value in config.init_joints_deg]
            continue
        if name != "move":
            continue
        args = action.get("args", {})
        pose = [
            float(args["x"]),
            float(args["y"]),
            float(args["z"]),
            config.orientation_roll_deg,
            config.orientation_pitch_deg,
            float(args["yaw"]),
        ]
        pose_key = tuple(pose)
        if pose_key in cached_pose_targets:
            targets[action_index] = cached_pose_targets[pose_key]
            reference_deg = [math.degrees(value) for value in targets[action_index]]
            continue
        code, angles_deg = arm.get_inverse_kinematics(
            pose,
            input_is_radian=False,
            return_is_radian=False,
            limited=False,
            ref_angles=reference_deg,
        )
        if int(code) != 0 or not isinstance(angles_deg, (list, tuple)) or len(angles_deg) != 7:
            raise SafetyError(
                f"controller IK rejected action {action_index + 1} "
                f"pose={pose}, code={code}"
            )
        reference_deg = [float(value) for value in angles_deg]
        if not all(math.isfinite(value) for value in reference_deg):
            raise RobotExecutionError(
                f"controller IK returned non-finite joints for action {action_index + 1}"
            )
        targets[action_index] = tuple(math.radians(value) for value in reference_deg)
        cached_pose_targets[pose_key] = targets[action_index]
    return ControllerTrajectoryValidation(
        joint_targets_rad=targets,
        controller_warning_code=warning_code,
        tcp_offset_mm_deg=live_tcp_offset,
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
    def home(self, config: RobotConfig) -> tuple[list[float] | None, Any]: ...
    def close(self) -> None: ...


class SimulatedBackend:
    """No-motion backend used for preflight and tests."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.pose = list(config.init_pose_mm_deg)
        self.gripper = config.gripper_open
        self.state = "simulated"

    def move(self, x: float, y: float, z: float, yaw: float, config: RobotConfig):
        self.pose = [x, y, z, config.orientation_roll_deg, config.orientation_pitch_deg, yaw]
        return list(self.pose), self.state

    def open_gripper(self, config: RobotConfig):
        self.gripper = config.gripper_open
        return {"position": self.gripper, "simulated": True}, (list(self.pose), self.state)

    def close_gripper(self, config: RobotConfig):
        self.gripper = config.gripper_close
        return {"position": self.gripper, "simulated": True}, (list(self.pose), self.state)

    def home(self, config: RobotConfig):
        self.pose = list(config.init_pose_mm_deg)
        return list(self.pose), self.state

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
        }
        return pose, state

    def move(self, x: float, y: float, z: float, yaw: float, config: RobotConfig):
        code = self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=config.orientation_roll_deg,
            pitch=config.orientation_pitch_deg,
            yaw=yaw,
            speed=config.speed_mm_s,
            mvacc=config.acceleration_mm_s2,
            wait=True,
            is_radian=False,
        )
        self._check("set_position", code)
        return self._state()

    def open_gripper(self, config: RobotConfig):
        result = self._check(
            "set_gripper_position",
            self.arm.set_gripper_position(config.gripper_open, speed=config.gripper_speed, wait=True),
        )
        position = self.arm.get_gripper_position()
        return {"command_result": result, "position_result": position}, self._state()

    def close_gripper(self, config: RobotConfig):
        result = self._check(
            "set_gripper_position",
            self.arm.set_gripper_position(config.gripper_close, speed=config.gripper_speed, wait=True),
        )
        position = self.arm.get_gripper_position()
        return {"command_result": result, "position_result": position}, self._state()

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

    def close(self) -> None:
        if getattr(self.arm, "connected", False):
            self.arm.disconnect()


class RobotAPI:
    """Safety-checked facade injected into generated experiment code.

    Generated code never receives ``XArmAPI`` or a backend.  It can only call
    the four methods below.  Every command is recorded, and the first failure
    permanently halts the run so a script cannot continue after an error.
    """

    ALLOWED_METHODS = frozenset({"move", "open_gripper", "close_gripper", "home"})

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
        self.halted = True

    def move(self, x: float, y: float, z: float, yaw: float) -> None:
        record = self._begin("move", {"x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw)})
        try:
            self.config.boundaries.validate(
                record.args["x"],
                record.args["y"],
                record.args["z"],
                self.config.workspace_margin_mm,
                z_lower_margin_mm=self.config.lower_z_margin_mm,
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
