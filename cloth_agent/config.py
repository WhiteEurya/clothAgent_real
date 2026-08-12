"""Configuration and safety contracts for the real xArm runtime."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a robot or experiment configuration is unsafe/incomplete."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class WorkspaceBounds:
    """Cartesian limits in xArm base coordinates, in millimetres.

    ``x_max`` is optional. When it is absent, upper-X reachability is delegated
    to the xArm controller's read-only inverse-kinematics validation and final
    motion command. The locally measured lower-X, Y, and Z limits remain
    mandatory for real execution.
    """

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WorkspaceBounds":
        names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        parsed: dict[str, float | None] = {}
        for name in names:
            value = values.get(name)
            parsed[name] = None if value is None else _number(value, name)
        bounds = cls(**parsed)
        bounds.validate_order()
        return bounds

    def validate_order(self) -> None:
        for axis in ("x", "y", "z"):
            low, high = getattr(self, f"{axis}_min"), getattr(self, f"{axis}_max")
            if low is not None and high is not None and low >= high:
                raise ConfigError(f"{axis}_min must be less than {axis}_max")

    @property
    def complete(self) -> bool:
        return all(
            getattr(self, name) is not None
            for name in ("x_min", "y_min", "y_max", "z_min", "z_max")
        )

    def validate(
        self,
        x: float,
        y: float,
        z: float,
        margin_mm: float = 0.0,
        *,
        require_complete: bool = False,
        z_lower_margin_mm: float | None = None,
    ) -> None:
        """Validate one target before it reaches the xArm SDK."""

        if require_complete and not self.complete:
            raise ConfigError("x_min, y_min, y_max, z_min, and z_max are required for a real run")
        margin = _number(margin_mm, "workspace margin")
        if margin < 0:
            raise ConfigError("workspace margin must be non-negative")
        z_lower_margin = (
            margin
            if z_lower_margin_mm is None
            else _number(z_lower_margin_mm, "lower-z workspace margin")
        )
        if z_lower_margin < 0:
            raise ConfigError("lower-z workspace margin must be non-negative")
        target = {"x": _number(x, "x"), "y": _number(y, "y"), "z": _number(z, "z")}
        for axis, value in target.items():
            low = getattr(self, f"{axis}_min")
            high = getattr(self, f"{axis}_max")
            lower_margin = z_lower_margin if axis == "z" else margin
            if low is not None and value < low + lower_margin:
                raise SafetyError(
                    f"{axis}={value:g} is below the safe lower bound {low + lower_margin:g}"
                )
            if high is not None and value > high - margin:
                raise SafetyError(f"{axis}={value:g} is above the safe upper bound {high - margin:g}")


@dataclass(frozen=True)
class RobotConfig:
    robot_ip: str
    boundaries: WorkspaceBounds
    init_joints_deg: tuple[float, ...]
    init_pose_mm_deg: tuple[float, ...]
    orientation_roll_deg: float
    orientation_pitch_deg: float
    expected_tcp_offset_mm_deg: tuple[float, ...] = (0.0, 0.0, 172.0, 0.0, 0.0, 0.0)
    tcp_offset_tolerance: float = 1.0
    workspace_margin_mm: float = 10.0
    lower_z_margin_mm: float = 0.0
    speed_mm_s: float = 15.0
    acceleration_mm_s2: float = 30.0
    home_speed_deg_s: float = 5.0
    home_acceleration_deg_s2: float = 10.0
    gripper_speed: float = 500.0
    gripper_open: float = 850.0
    gripper_close: float = 0.0

    MAX_SAFE_SPEED_MM_S = 30.0
    MAX_SAFE_ACCELERATION_MM_S2 = 60.0
    MAX_SAFE_HOME_SPEED_DEG_S = 10.0
    MAX_SAFE_HOME_ACCELERATION_DEG_S2 = 20.0

    @classmethod
    def load(cls, project_root: Path, config_path: Path | None = None) -> "RobotConfig":
        project_root = project_root.resolve()
        raw: dict[str, Any] = {}
        if config_path is not None:
            raw = json.loads(config_path.expanduser().resolve().read_text(encoding="utf-8"))
        boundaries_path = Path(raw.get("boundaries_file", "xarm_boundaries.json"))
        if not boundaries_path.is_absolute():
            boundaries_path = project_root / boundaries_path
        boundary_doc = json.loads(boundaries_path.read_text(encoding="utf-8"))
        boundary_values = dict(boundary_doc.get("boundary_mm", boundary_doc))
        boundary_values.update(raw.get("boundaries", {}))
        boundaries = WorkspaceBounds.from_mapping(boundary_values)

        pose_path = Path(raw.get("init_pose_file", "data/robot/xarm_init_pose.json"))
        if not pose_path.is_absolute():
            pose_path = project_root / pose_path
        pose_doc = json.loads(pose_path.read_text(encoding="utf-8"))
        joints = tuple(_number(v, "init joint") for v in pose_doc["joint_angles_deg"])
        pose = tuple(_number(v, "init pose") for v in pose_doc["tcp_pose_mm_deg"])
        orientation = raw.get("fixed_orientation_deg", {})
        roll = _number(orientation.get("roll", pose[3]), "fixed roll")
        pitch = _number(orientation.get("pitch", pose[4]), "fixed pitch")
        tcp_offset = tuple(
            _number(v, "expected TCP offset")
            for v in raw.get("expected_tcp_offset_mm_deg", [0, 0, 172, 0, 0, 0])
        )
        motion = raw.get("motion", {})
        gripper = raw.get("gripper", {})
        return cls(
            robot_ip=str(raw.get("robot_ip", boundary_doc.get("robot_ip", "192.168.1.200"))),
            boundaries=boundaries,
            init_joints_deg=joints,
            init_pose_mm_deg=pose,
            orientation_roll_deg=roll,
            orientation_pitch_deg=pitch,
            expected_tcp_offset_mm_deg=tcp_offset,
            tcp_offset_tolerance=_number(
                raw.get("tcp_offset_tolerance", 1.0), "TCP offset tolerance"
            ),
            workspace_margin_mm=_number(raw.get("workspace_margin_mm", 10.0), "workspace margin"),
            lower_z_margin_mm=_number(
                raw.get("lower_z_margin_mm", 0.0), "lower-z workspace margin"
            ),
            speed_mm_s=_number(motion.get("speed_mm_s", 15.0), "speed_mm_s"),
            acceleration_mm_s2=_number(motion.get("acceleration_mm_s2", 30.0), "acceleration_mm_s2"),
            home_speed_deg_s=_number(motion.get("home_speed_deg_s", 5.0), "home_speed_deg_s"),
            home_acceleration_deg_s2=_number(
                motion.get("home_acceleration_deg_s2", 10.0), "home_acceleration_deg_s2"
            ),
            gripper_speed=_number(gripper.get("speed", 500.0), "gripper speed"),
            gripper_open=_number(gripper.get("open", 850.0), "gripper open"),
            gripper_close=_number(gripper.get("close", 0.0), "gripper close"),
        )

    def validate_for_real(self) -> None:
        if not self.boundaries.complete:
            raise ConfigError("x_min, y_min, y_max, z_min, and z_max are required for a real run")
        self.boundaries.validate_order()
        self.boundaries.validate(
            self.init_pose_mm_deg[0],
            self.init_pose_mm_deg[1],
            self.init_pose_mm_deg[2],
            self.workspace_margin_mm,
            require_complete=True,
            z_lower_margin_mm=self.lower_z_margin_mm,
        )
        if not 0 < self.speed_mm_s <= self.MAX_SAFE_SPEED_MM_S:
            raise ConfigError(f"real motion speed must be in (0, {self.MAX_SAFE_SPEED_MM_S:g}] mm/s")
        if not 0 < self.acceleration_mm_s2 <= self.MAX_SAFE_ACCELERATION_MM_S2:
            raise ConfigError(
                f"real motion acceleration must be in (0, {self.MAX_SAFE_ACCELERATION_MM_S2:g}] mm/s^2"
            )
        if not 0 < self.home_speed_deg_s <= self.MAX_SAFE_HOME_SPEED_DEG_S:
            raise ConfigError(
                f"home speed must be in (0, {self.MAX_SAFE_HOME_SPEED_DEG_S:g}] deg/s"
            )
        if not 0 < self.home_acceleration_deg_s2 <= self.MAX_SAFE_HOME_ACCELERATION_DEG_S2:
            raise ConfigError(
                "home acceleration must be in "
                f"(0, {self.MAX_SAFE_HOME_ACCELERATION_DEG_S2:g}] deg/s^2"
            )
        if self.gripper_speed <= 0:
            raise ConfigError("real gripper speed must be positive")
        if len(self.expected_tcp_offset_mm_deg) != 6:
            raise ConfigError("expected_tcp_offset_mm_deg must contain six values")
        if self.tcp_offset_tolerance <= 0:
            raise ConfigError("TCP offset tolerance must be positive")
        if self.lower_z_margin_mm < 0:
            raise ConfigError("lower-z workspace margin must be non-negative")

    def validate_live_tcp_offset(self, actual: Any) -> None:
        """Reject real execution if the controller's saved tool frame changed."""

        if not isinstance(actual, (list, tuple)) or len(actual) != 6:
            raise ConfigError("xArm did not report a valid six-value TCP offset")
        actual_values = tuple(_number(v, "live TCP offset") for v in actual)
        differences = [
            abs(observed - expected)
            for observed, expected in zip(actual_values, self.expected_tcp_offset_mm_deg)
        ]
        if any(delta > self.tcp_offset_tolerance for delta in differences):
            raise ConfigError(
                "xArm TCP offset changed: "
                f"expected {list(self.expected_tcp_offset_mm_deg)}, got {list(actual_values)}; "
                "recalibrate/verify the gripper tool frame before physical execution"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    """Per-run scene values.

    All fields may be deferred until perception.  A no-perception/manual run
    must call :meth:`require_ready` before code generation.
    """

    cloth_center_x: float | None = None
    cloth_center_y: float | None = None
    grasp_z: float | None = None
    approach_z: float | None = None
    lift_z: float | None = None
    yaw_deg: float | None = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        require_center: bool = True,
        allow_deferred: bool | None = None,
    ) -> "ExperimentConfig":
        if allow_deferred is None:
            allow_deferred = not require_center
        cloth = values.get("cloth", values)
        yaw = cloth.get("yaw_deg", cloth.get("yaw"))
        center_x = cloth.get("center_x", cloth.get("cloth_center_x"))
        center_y = cloth.get("center_y", cloth.get("cloth_center_y"))

        def optional_number(value: Any, name: str) -> float | None:
            if value is None and allow_deferred:
                return None
            return _number(value, name)

        result = cls(
            cloth_center_x=optional_number(center_x, "cloth center x"),
            cloth_center_y=optional_number(center_y, "cloth center y"),
            grasp_z=optional_number(cloth.get("grasp_z"), "grasp_z"),
            approach_z=optional_number(cloth.get("approach_z"), "approach_z"),
            lift_z=optional_number(cloth.get("lift_z"), "lift_z"),
            yaw_deg=optional_number(yaw, "yaw"),
        )
        result.validate()
        return result

    def validate(self) -> None:
        z_values = (self.grasp_z, self.approach_z, self.lift_z)
        if all(value is not None for value in z_values) and not (
            self.grasp_z <= self.approach_z <= self.lift_z  # type: ignore[operator]
        ):
            raise ConfigError("expected grasp_z <= approach_z <= lift_z")
        if self.yaw_deg is not None and not -360.0 <= self.yaw_deg <= 360.0:
            raise ConfigError("yaw must be between -360 and 360 degrees")

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)

    def require_center(self) -> tuple[float, float]:
        if self.cloth_center_x is None or self.cloth_center_y is None:
            raise ConfigError("cloth center is unset; run perception or provide center_x/center_y")
        return self.cloth_center_x, self.cloth_center_y

    def require_ready(self) -> tuple[float, float, float, float, float, float]:
        values = (
            self.cloth_center_x,
            self.cloth_center_y,
            self.grasp_z,
            self.approach_z,
            self.lift_z,
            self.yaw_deg,
        )
        names = ("center_x", "center_y", "grasp_z", "approach_z", "lift_z", "yaw")
        missing = [name for name, value in zip(names, values) if value is None]
        if missing:
            raise ConfigError(
                "experiment plan is incomplete; run perception or provide: " + ", ".join(missing)
            )
        self.validate()
        return values  # type: ignore[return-value]


class SafetyError(RuntimeError):
    """Raised before an unsafe command is sent to the robot."""
