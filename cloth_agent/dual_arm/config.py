"""Dual-arm configuration; unknown physical calibration never gets a default."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .geometry import DualArmError, finite, transform


def number(value, name, low, high):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not np.isfinite(value)
        or not low <= value <= high
    ):
        raise DualArmError(f"{name} must be in [{low}, {high}]")
    return float(value)


def workspace_bound(value, name, *, lower):
    """A null coordinate explicitly leaves that side of an axis unrestricted."""
    if not isinstance(value, list) or len(value) != 3:
        raise DualArmError(f"{name} requires three coordinates")
    bounded = finite([0 if v is None else v for v in value], (3,), name)
    for axis, coordinate in enumerate(value):
        if coordinate is None:
            bounded[axis] = -np.inf if lower else np.inf
    return bounded


@dataclass
class ArmConfig:
    arm_id: str
    ip: str
    axis: int
    serial: str
    urdf: Path
    world_from_base: np.ndarray
    home_joints: np.ndarray
    tcp_offset: np.ndarray
    grasp_rpy: np.ndarray | None
    workspace: dict
    capsules: list[dict]
    gripper: SimpleNamespace
    raw: dict

    def base_pose(self, world_pose):
        from .geometry import matrix_pose, pose_matrix

        return matrix_pose(
            np.linalg.inv(self.world_from_base) @ pose_matrix(world_pose)
        )

    def validate_point(self, point):
        p = finite(point, (3,), "base point")
        if np.any(p < self.workspace["min_mm"]) or np.any(p > self.workspace["max_mm"]):
            raise DualArmError(f"{self.arm_id}: target outside measured workspace")


@dataclass
class DualConfig:
    root: Path
    arms: dict[str, ArmConfig]
    limits: dict
    obstacles: list[dict]
    raw: dict
    execution_mode: str = 'servo_stream'

    @classmethod
    def load(cls, path, *, root=None, homing_only=False):
        path = Path(path).resolve()
        return cls.parse(
            json.loads(path.read_text()), Path(root or path.parent).resolve(), homing_only=homing_only
        )

    @classmethod
    def parse(cls, raw, root, *, homing_only=False):
        if raw.get("schema_version") != 1 or raw.get("frame_id") != "world_mm":
            raise DualArmError("schema_version=1 and frame_id=world_mm required")
        if raw.get("calibration_status") not in {"measured", "synthetic"}:
            raise DualArmError(
                "complete the measured base transforms/workspaces/collision geometry first"
            )
        records = raw.get("arms", {})
        if set(records) != {"left", "right"}:
            raise DualArmError(
                "exactly left and right arms required; assign physical devices explicitly"
            )
        arms = {}
        for arm_id, data in records.items():
            axis = data["axis"]
            if type(axis) is not int or axis not in (6, 7):
                raise DualArmError("axis must be 6 or 7")
            workspace = data["workspace"]
            low = workspace_bound(workspace["min_mm"], "workspace min", lower=True)
            high = workspace_bound(workspace["max_mm"], "workspace max", lower=False)
            if np.any(low >= high):
                raise DualArmError("workspace min must be below max")
            capsules = data["collision_capsules"]
            if len({c["name"] for c in capsules}) != len(capsules):
                raise DualArmError("collision capsule names must be unique")
            required_links = {
                "link_base",
                *(f"link{i}" for i in range(1, axis + 1)),
                "tool",
            }
            if not required_links.issubset({c["name"] for c in capsules}):
                raise DualArmError(
                    f"{arm_id}: collision geometry must cover every link and the tool"
                )
            for capsule in capsules:
                finite(capsule["start_mm"], (3,), "capsule start")
                finite(capsule["end_mm"], (3,), "capsule end")
                number(capsule["radius_mm"], "capsule radius", 1, 500)
            g = data["gripper"]
            gripper = SimpleNamespace(
                gripper_open=number(g["open"], "gripper open", 0, 850),
                gripper_close=number(g["close"], "gripper close", 0, 850),
                gripper_speed=number(g["speed"], "gripper speed", 1, 500),
                gripper_open_tolerance_pulse=number(
                    g["open_tolerance"], "open tolerance", 1, 30
                ),
                gripper_completion_timeout_s=number(
                    g["timeout_s"], "gripper timeout", 4, 60
                ),
            )
            if gripper.gripper_close >= gripper.gripper_open:
                raise DualArmError("gripper close must be below open")
            arms[arm_id] = ArmConfig(
                arm_id,
                str(data["ip"]),
                axis,
                str(data["control_box_sn"]),
                (root / data["urdf"]).resolve(),
                transform(data["world_from_base_mm"], f"{arm_id} base transform"),
                finite(data["home_joints_deg"], (axis,), "home joints"),
                finite(data["tcp_offset_mm_deg"], (6,), "TCP offset"),
                (None if homing_only and data.get("grasp_rpy_world_deg") is None
                 else finite(data["grasp_rpy_world_deg"], (3,), "grasp orientation")),
                {"min_mm": low, "max_mm": high},
                capsules,
                gripper,
                data,
            )
            if not arms[arm_id].serial or not arms[arm_id].urdf.is_file():
                raise DualArmError(
                    f"{arm_id}: controller identity and an existing URDF required"
                )
        if (
            arms["left"].ip == arms["right"].ip
            or arms["left"].serial == arms["right"].serial
        ):
            raise DualArmError("arms must have distinct IPs and controller identities")
        bounds = {
            "rate_hz": (20, 100),
            "cartesian_speed_mm_s": (1, 30),
            "cartesian_accel_mm_s2": (1, 60),
            "joint_speed_deg_s": (1, 20),
            "joint_accel_deg_s2": (1, 60),
            "max_ik_step_deg": (0.1, 5),
            "tracking_error_mm": (0, 15),
            "tracking_error_deg": (0, 5),
            "arrival_error_mm": (0.1, 3),
            "dispatch_skew_s": (0.001, 0.05),
            "max_tick_lateness_s": (0.001, 0.1),
            "clearance_mm": (0, 100),
            "max_grasp_age_s": (1, 300),
            "max_span_mm": (20, 1500),
            "min_span_mm": (10, 500),
            "max_spread_mm": (1, 200),
            "max_spread_step_mm": (1, 20),
            "max_lift_mm": (10, 300),
            "max_approach_mm": (10, 150),
            "contact_descent_mm": (0, 8),
            "max_vision_wait_s": (1, 180),
            "stationary_tolerance_deg": (0.01, 1),
            "max_plan_samples": (100, 100000),
            "max_surface_spread_mm": (0.1, 20),
            "center_radius_mm": (20, 1000),
            "center_line_tolerance_mm": (1, 100),
        }
        limits = {
            key: number(raw["limits"][key], key, *limits)
            for key, limits in bounds.items()
        }
        if limits["min_span_mm"] >= limits["max_span_mm"]:
            raise DualArmError("min span must be below max span")
        obstacles = raw["obstacles"]
        for box in obstacles:
            lo, hi = (
                finite(box["min_mm"], (3,), "obstacle min"),
                finite(box["max_mm"], (3,), "obstacle max"),
            )
            if np.any(lo >= hi) or not box.get("name"):
                raise DualArmError("invalid obstacle box")
            if any(
                name not in {"left/link_base", "right/link_base"}
                for name in box.get("excluded_capsules", [])
            ):
                raise DualArmError("only fixed base mounting contacts may be excluded")
            for name in box.get("excluded_capsules", []):
                arm_id, capsule_name = name.split("/")
                capsule = next(
                    c for c in arms[arm_id].capsules if c["name"] == capsule_name
                )
                if capsule["frame"] != "link_base":
                    raise DualArmError(
                        "excluded base capsule must actually be fixed in link_base"
                    )
            if "tool_contact_allowance_mm" in box:
                number(
                    box["tool_contact_allowance_mm"],
                    "tool/table contact allowance",
                    0,
                    8,
                )
        from .safety import validate_safety

        validate_safety(raw.get("safety"), arms, limits)
        if (
            raw["calibration_status"] == "measured"
            and raw["safety"]["status"] != "measured"
        ):
            raise DualArmError(
                "measured configuration cannot use synthetic safety bounds"
            )
        return cls(root, arms, limits, obstacles, raw)

    def require_real(self, *, commissioning=False, homing=False):
        from .safety import validate_safety

        if not homing and any(a.grasp_rpy is None for a in self.arms.values()):
            raise DualArmError("grasp orientation required outside joint Home")

        validate_safety(self.raw.get("safety"), self.arms, self.limits, real=True)
        if self.raw["calibration_status"] != "measured" or not self.raw.get(
            "calibration_id"
        ):
            raise DualArmError(
                "real execution requires measured calibration with an ID"
            )
        if (
            self.raw.get("collision_geometry_verified") is not True
            or not self.obstacles
        ):
            raise DualArmError(
                "real execution requires verified link/tool envelopes and table/obstacle geometry"
            )
        if any(box.get("tool_contact_allowance_mm", 0) > 0 for box in self.obstacles):
            raise DualArmError(
                "physical obstacle penetration allowances require a force-limited contact controller; unsupported"
            )
        if (self.raw.get("servo_commissioned") is not True and not commissioning
                and not (homing and self.execution_mode == 'controller_sequential')):
            raise DualArmError(
                "commission the shared-time servo path without cloth before enabling this runtime"
            )
        description = self.raw.get("arm_layout_description", "").strip()
        if not description or description.startswith("SET "):
            raise DualArmError("describe the actual physical left/right arm assignment")
        if not commissioning and not homing:
            camera = self.raw["camera"]
            if not camera.get("serial") or camera.get("mount_arm") not in {
                None,
                *self.arms,
            }:
                raise DualArmError("camera identity/owner required")
            transform(
                camera["world_from_camera_mm"]
                if camera["mount_arm"] is None
                else camera["mount_from_camera_mm"],
                "camera calibration",
            )
            for key in ("width", "height", "fps"):
                number(camera[key], f"camera {key}", 1, 4096)
