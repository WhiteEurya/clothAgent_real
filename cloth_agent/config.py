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
    motion command. Real execution requires lower-X/Y/Z limits, or a pair of
    lateral XY points defining a fixed strip together with a Z floor; a Z ceiling is optional.
    """

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    lateral_points_mm: tuple[tuple[float, float], tuple[float, float]] | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "WorkspaceBounds":
        names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        parsed: dict[str, Any] = {}
        for name in names:
            value = values.get(name)
            parsed[name] = None if value is None else _number(value, name)
        points = values.get("lateral_points_mm")
        if points is not None:
            if not isinstance(points, (list, tuple)) or len(points) != 2 or any(
                not isinstance(p, (list, tuple)) or len(p) != 2 for p in points
            ):
                raise ConfigError("lateral_points_mm requires two XY points")
            parsed["lateral_points_mm"] = tuple(
                tuple(_number(v, "lateral point") for v in p) for p in points
            )
        bounds = cls(**parsed)
        bounds.validate_order()
        return bounds

    def validate_order(self) -> None:
        if self.lateral_points_mm is not None:
            self.lateral_geometry()
        for axis in ("x", "y", "z"):
            low, high = getattr(self, f"{axis}_min"), getattr(self, f"{axis}_max")
            if low is not None and high is not None and low >= high:
                raise ConfigError(f"{axis}_min must be less than {axis}_max")

    @property
    def complete(self) -> bool:
        if self.lateral_points_mm is not None:
            return self.z_min is not None
        return all(
            getattr(self, name) is not None
            for name in ("x_min", "y_min", "y_max", "z_min")
        )

    def lateral_geometry(self) -> tuple[float, float, float, float]:
        """Return the fixed XY normal and projection limits of the two sides."""
        if self.lateral_points_mm is None:
            raise ConfigError("lateral points are not configured")
        a, b = self.lateral_points_mm
        dx, dy = b[0] - a[0], b[1] - a[1]
        width = math.hypot(dx, dy)
        if not math.isfinite(width) or width < 1.0:
            raise ConfigError("lateral points must be at least 1 mm apart in XY")
        nx, ny = dx / width, dy / width
        low = nx * a[0] + ny * a[1]
        return nx, ny, low, low + width

    def validate_lateral(self, x: float, y: float, margin_mm: float = 0.0) -> None:
        if self.lateral_points_mm is None:
            return
        nx, ny, low, high = self.lateral_geometry()
        position = nx * x + ny * y
        if position < low + margin_mm - 1e-6 or position > high - margin_mm + 1e-6:
            raise SafetyError("TCP is outside the selected left/right boundaries")

    def validate(
        self,
        x: float,
        y: float,
        z: float,
        margin_mm: float = 0.0,
        *,
        require_complete: bool = False,
        z_lower_margin_mm: float | None = None,
        y_extension_mm: float = 0.0,
    ) -> None:
        """Validate one target before it reaches the xArm SDK.

        ``y_extension_mm`` is intentionally explicit: callers that know the
        relative wrist yaw may widen only the Y envelope for the TCP center.
        The default remains the calibrated static workspace.
        """

        if require_complete and not self.complete:
            raise ConfigError("real execution requires x_min, y_min, y_max, z_min or lateral points with z_min")
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
        y_extension = _number(y_extension_mm, "Y workspace extension")
        if y_extension < 0:
            raise ConfigError("Y workspace extension must be non-negative")
        target = {"x": _number(x, "x"), "y": _number(y, "y"), "z": _number(z, "z")}
        self.validate_lateral(target["x"], target["y"], margin)
        for axis, value in target.items():
            low = getattr(self, f"{axis}_min")
            high = getattr(self, f"{axis}_max")
            if axis == "y":
                # The TCP center may use a yaw-dependent allowance for the
                # physical gripper span.  X/Z remain the calibrated limits.
                if low is not None:
                    low -= y_extension
                if high is not None:
                    high += y_extension
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
    perception_joints_deg: tuple[float, ...] | None = None
    perception_pose_mm_deg: tuple[float, ...] | None = None
    expected_tcp_offset_mm_deg: tuple[float, ...] = (0.0, 0.0, 172.0, 0.0, 0.0, 0.0)
    tcp_offset_tolerance: float = 1.0
    workspace_margin_mm: float = 0.0
    lower_z_margin_mm: float = 0.0
    # Use the deepest value allowed by the shared shallow-engagement policy.
    # The robot lower bound and controller checks still prevent illegal motion.
    grasp_surface_compression_mm: float = 3.0
    grasp_min_compression_mm: float = 0.75
    grasp_max_compression_mm: float = 3.0
    grasp_table_clearance_mm: float = 0.0
    # Optional compliant support beneath the garment. The support layer is
    # deliberately separate from the legacy tabletop floor. It can be enabled
    # by an explicit operator declaration (useful when the sponge is hidden
    # underneath the garment), or by a local support-ring confirmation.
    support_layer_type: str = "none"
    support_layer_thickness_mm: float = 0.0
    support_layer_press_mm: float = 0.0
    support_layer_max_compression_mm: float = 0.0
    support_layer_hard_clearance_mm: float = 0.0
    support_layer_presence_threshold_mm: float = 3.0
    # Explicitly confirmed by the operator/configuration.  This is useful when
    # the compliant layer is only underneath the garment: a ring sampled just
    # outside the garment can legitimately see the exposed hard table and is
    # therefore not a reliable presence test by itself.
    support_layer_confirmed: bool = False
    # A live tabletop is not a reliable absolute-Z reference on the current
    # RealSense setup.  When disabled, perception keeps table geometry for
    # segmentation/diagnostics but does not apply the per-camera online Z-bias
    # correction to the calibrated camera depth.
    online_camera_z_bias_correction: bool = True
    # When disabled, grasp-height resolution uses the measured absolute camera
    # surface Z and the robot lower bound only.  Table estimates remain in the
    # audit record, but cannot veto an otherwise engaged grasp.
    grasp_use_table_clearance_floor: bool = True
    speed_mm_s: float = 15.0
    acceleration_mm_s2: float = 30.0
    home_speed_deg_s: float = 5.0
    home_acceleration_deg_s2: float = 10.0
    gripper_speed: float = 500.0
    gripper_open: float = 850.0
    gripper_close: float = 0.0
    # Legacy field retained for saved configurations; no fixed settle delay.
    gripper_settle_s: float = 0.5
    # Legacy name: now the slow-wait reporting threshold, not an abort deadline.
    # Feedback waits continue until completion, hardware fault or Ctrl+C.
    gripper_completion_timeout_s: float = 10.0
    # Effective jaw-to-jaw span used only to account for the gripper body when
    # the TCP is close to a Y workspace edge.  A value of 0 keeps the legacy
    # fixed Y bounds until the installed tool is measured and configured.
    gripper_width_mm: float = 0.0

    MAX_SAFE_SPEED_MM_S = 30.0
    MAX_SAFE_ACCELERATION_MM_S2 = 60.0
    MAX_SAFE_HOME_SPEED_DEG_S = 10.0
    MAX_SAFE_HOME_ACCELERATION_DEG_S2 = 20.0

    def command_yaw_deg(self, relative_yaw_deg: float) -> float:
        """Convert a generated relative wrist yaw into the calibrated TCP yaw.

        The xArm home joint configuration has a non-zero TCP yaw on this tool
        frame.  Generated programs use ``yaw=0`` to mean "keep the calibrated
        home/gripper orientation"; adding the saved home yaw prevents every
        grasp from first making an unnecessary ~180-degree wrist turn.
        """

        value = float(relative_yaw_deg) + float(self.init_pose_mm_deg[5])
        return (value + 180.0) % 360.0 - 180.0

    def y_workspace_extension_mm(self, relative_yaw_deg: float) -> float:
        """Return the extra Y allowance for a relative wrist yaw.

        The allowance models the TCP center moving toward an edge while one
        jaw remains supported by the garment/table.  It is zero for the
        calibrated orientation (``yaw=0``) and reaches half the configured
        effective gripper width at ``+/-90`` degrees.  The controller's IK and
        the physical gripper geometry remain authoritative; this is only a
        host-side workspace envelope adjustment.
        """

        yaw = _number(relative_yaw_deg, "relative yaw")
        width = _number(self.gripper_width_mm, "gripper width")
        if width < 0:
            raise ConfigError("gripper width must be non-negative")
        return 0.5 * width * abs(math.sin(math.radians(yaw)))

    def relative_yaw_from_absolute_deg(self, absolute_yaw_deg: float) -> float:
        """Convert an absolute TCP yaw into the shortest Home-relative delta."""

        absolute = _number(absolute_yaw_deg, "absolute yaw")
        home = _number(self.init_pose_mm_deg[5], "home yaw")
        return (absolute - home + 180.0) % 360.0 - 180.0

    def y_workspace_bounds_mm(
        self, relative_yaw_deg: float
    ) -> tuple[float | None, float | None]:
        """Return the effective Y limits for one relative wrist yaw."""

        extension = self.y_workspace_extension_mm(relative_yaw_deg)
        low = (
            None
            if self.boundaries.y_min is None
            else float(self.boundaries.y_min) - extension
        )
        high = (
            None
            if self.boundaries.y_max is None
            else float(self.boundaries.y_max) + extension
        )
        return low, high

    def validate_workspace_pose(
        self,
        x: float,
        y: float,
        z: float,
        relative_yaw_deg: float = 0.0,
        *,
        require_complete: bool = False,
    ) -> None:
        """Validate a TCP target with yaw-dependent Y clearance."""

        self.boundaries.validate(
            x,
            y,
            z,
            self.workspace_margin_mm,
            require_complete=require_complete,
            z_lower_margin_mm=self.lower_z_margin_mm,
            y_extension_mm=self.y_workspace_extension_mm(relative_yaw_deg),
        )

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
        perception_joints: tuple[float, ...] | None = None
        perception_pose: tuple[float, ...] | None = None
        perception_path_value: str | None = None
        inline_perception = raw.get("perception_position")
        if isinstance(inline_perception, dict) and inline_perception.get(
            "joint_angles_deg"
        ) is not None:
            perception_joints = tuple(
                _number(v, "perception joint")
                for v in inline_perception["joint_angles_deg"]
            )
            perception_pose = tuple(
                _number(v, "perception pose")
                for v in inline_perception["tcp_pose_mm_deg"]
            )
        else:
            perception_path_value = raw.get(
                "perception_pose_file", "data/robot/xarm_perception_pose.json"
            )
        if perception_joints is None and perception_path_value is not None:
            perception_path = Path(perception_path_value)
            if not perception_path.is_absolute():
                perception_path = project_root / perception_path
            if perception_path.is_file():
                perception_doc = json.loads(
                    perception_path.read_text(encoding="utf-8")
                )
                perception_joints = tuple(
                    _number(v, "perception joint")
                    for v in perception_doc["joint_angles_deg"]
                )
                perception_pose = tuple(
                    _number(v, "perception pose")
                    for v in perception_doc["tcp_pose_mm_deg"]
                )
        orientation = raw.get("fixed_orientation_deg", {})
        roll = _number(orientation.get("roll", pose[3]), "fixed roll")
        pitch = _number(orientation.get("pitch", pose[4]), "fixed pitch")
        tcp_offset = tuple(
            _number(v, "expected TCP offset")
            for v in raw.get("expected_tcp_offset_mm_deg", [0, 0, 172, 0, 0, 0])
        )
        motion = raw.get("motion", {})
        gripper = raw.get("gripper", {})
        grasp_height = raw.get("grasp_height", {})
        support_layer = raw.get("support_layer", {})
        if not isinstance(support_layer, Mapping):
            raise ConfigError("support_layer must be a JSON object")
        support_layer_type = str(support_layer.get("type", "none")).strip().lower()
        gripper_width_mm = gripper.get(
            "width_mm",
            gripper.get(
                "effective_width_mm",
                raw.get("gripper_width_mm", 0.0),
            ),
        )
        return cls(
            robot_ip=str(raw.get("robot_ip", boundary_doc.get("robot_ip", "192.168.2.232"))),
            boundaries=boundaries,
            init_joints_deg=joints,
            init_pose_mm_deg=pose,
            orientation_roll_deg=roll,
            orientation_pitch_deg=pitch,
            perception_joints_deg=perception_joints,
            perception_pose_mm_deg=perception_pose,
            expected_tcp_offset_mm_deg=tcp_offset,
            tcp_offset_tolerance=_number(
                raw.get("tcp_offset_tolerance", 1.0), "TCP offset tolerance"
            ),
            workspace_margin_mm=_number(raw.get("workspace_margin_mm", 0.0), "workspace margin"),
            lower_z_margin_mm=_number(
                raw.get("lower_z_margin_mm", 0.0), "lower-z workspace margin"
            ),
            grasp_surface_compression_mm=_number(
                grasp_height.get("surface_compression_mm", 3.0),
                "grasp surface compression",
            ),
            grasp_min_compression_mm=_number(
                grasp_height.get("min_compression_mm", 0.75),
                "minimum grasp compression",
            ),
            grasp_max_compression_mm=_number(
                grasp_height.get("max_compression_mm", 3.0),
                "maximum grasp compression",
            ),
            grasp_table_clearance_mm=_number(
                grasp_height.get("table_clearance_mm", 0.0),
                "grasp table clearance",
            ),
            support_layer_type=support_layer_type,
            support_layer_thickness_mm=_number(
                support_layer.get("thickness_mm", 0.0),
                "support layer thickness",
            ),
            support_layer_press_mm=_number(
                support_layer.get("press_mm", 0.0),
                "support layer press depth",
            ),
            support_layer_max_compression_mm=_number(
                support_layer.get("max_compression_mm", 0.0),
                "support layer maximum compression",
            ),
            support_layer_hard_clearance_mm=_number(
                support_layer.get("hard_table_clearance_mm", 0.0),
                "support layer hard-table clearance",
            ),
            support_layer_presence_threshold_mm=_number(
                support_layer.get("presence_threshold_mm", 3.0),
                "support layer presence threshold",
            ),
            support_layer_confirmed=bool(
                support_layer.get("confirmed", False)
            ),
            online_camera_z_bias_correction=bool(
                raw.get("online_camera_z_bias_correction", True)
            ),
            grasp_use_table_clearance_floor=bool(
                grasp_height.get("use_table_clearance_floor", True)
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
            gripper_settle_s=_number(gripper.get("settle_s", 0.5), "gripper settle time"),
            gripper_completion_timeout_s=_number(gripper.get("completion_timeout_s", 10.0), "gripper completion timeout"),
            gripper_width_mm=_number(gripper_width_mm, "gripper width"),
        )

    def validate_for_real(self) -> None:
        if not self.boundaries.complete:
            raise ConfigError("real execution requires x_min, y_min, y_max, z_min or lateral points with z_min")
        self.boundaries.validate_order()
        self.validate_workspace_pose(
            self.init_pose_mm_deg[0],
            self.init_pose_mm_deg[1],
            self.init_pose_mm_deg[2],
            relative_yaw_deg=0.0,
            require_complete=True,
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
        if not math.isfinite(self.gripper_settle_s) or self.gripper_settle_s < 0:
            raise ConfigError("gripper settle time must be finite and non-negative")
        if not math.isfinite(self.gripper_completion_timeout_s) or self.gripper_completion_timeout_s <= 0:
            raise ConfigError("gripper completion timeout must be finite and positive")
        if not math.isfinite(self.gripper_width_mm) or self.gripper_width_mm < 0:
            raise ConfigError("gripper width must be finite and non-negative")
        if self.perception_joints_deg is not None and len(
            self.perception_joints_deg
        ) != 7:
            raise ConfigError("perception position must contain seven joint angles")
        if self.perception_pose_mm_deg is not None and len(
            self.perception_pose_mm_deg
        ) != 6:
            raise ConfigError("perception TCP pose must contain six values")
        if len(self.expected_tcp_offset_mm_deg) != 6:
            raise ConfigError("expected_tcp_offset_mm_deg must contain six values")
        if self.tcp_offset_tolerance <= 0:
            raise ConfigError("TCP offset tolerance must be positive")
        if self.lower_z_margin_mm < 0:
            raise ConfigError("lower-z workspace margin must be non-negative")
        if self.grasp_min_compression_mm <= 0:
            raise ConfigError("minimum grasp compression must be positive")
        if self.grasp_max_compression_mm < self.grasp_min_compression_mm:
            raise ConfigError(
                "maximum grasp compression must be at least the minimum compression"
            )
        if not (
            self.grasp_min_compression_mm
            <= self.grasp_surface_compression_mm
            <= self.grasp_max_compression_mm
        ):
            raise ConfigError(
                "grasp surface compression must lie between its minimum and maximum"
            )
        if self.grasp_table_clearance_mm < 0:
            raise ConfigError("grasp table clearance must be non-negative")
        if self.support_layer_type not in {"none", "sponge"}:
            raise ConfigError(
                "support_layer.type must be either 'none' or 'sponge'"
            )
        for value, name in (
            (self.support_layer_thickness_mm, "support layer thickness"),
            (self.support_layer_press_mm, "support layer press depth"),
            (self.support_layer_max_compression_mm, "support layer maximum compression"),
            (self.support_layer_hard_clearance_mm, "support layer hard-table clearance"),
            (self.support_layer_presence_threshold_mm, "support layer presence threshold"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"{name} must be finite and non-negative")
        if self.support_layer_type == "sponge":
            if self.support_layer_thickness_mm <= 0:
                raise ConfigError("sponge support layer thickness must be positive")
            if self.support_layer_press_mm <= 0:
                raise ConfigError("sponge support layer press depth must be positive")
            if self.support_layer_max_compression_mm < self.support_layer_press_mm:
                raise ConfigError(
                    "sponge support layer maximum compression must be at least its press depth"
                )

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

    Perception fills ``cloth_center_x``, ``cloth_center_y``, and the observed
    surface height stored in the legacy ``grasp_z`` slot.  The motion fields
    (``approach_z``, ``lift_z``, and ``yaw_deg``) may remain deferred until
    Claude writes an action program.  A no-perception/manual run must call
    :meth:`require_ready` before code generation.
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
        surface_z = cloth.get("grasp_z", cloth.get("surface_z", cloth.get("surface_z_mm")))

        def optional_number(value: Any, name: str) -> float | None:
            if value is None and allow_deferred:
                return None
            return _number(value, name)

        result = cls(
            cloth_center_x=optional_number(center_x, "cloth center x"),
            cloth_center_y=optional_number(center_y, "cloth center y"),
            # ``grasp_z`` remains the compatibility key written by older
            # runs; new observation documents may call it ``surface_z``.
            grasp_z=optional_number(surface_z, "surface_z"),
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
            raise ConfigError(
                "experiment plan is incomplete; cloth center is unset; "
                "run perception or provide center_x/center_y"
            )
        return self.cloth_center_x, self.cloth_center_y

    @property
    def surface_z_mm(self) -> float | None:
        """Observed fused garment surface height, not a commanded grasp pose."""

        return self.grasp_z

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
