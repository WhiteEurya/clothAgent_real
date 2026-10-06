"""Offline xArm kinematics and animation frames for the Viser dashboard."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any

import numpy as np


class KinematicsError(RuntimeError):
    """Raised when the offline URDF cannot reproduce a requested TCP pose."""


@dataclass(frozen=True)
class AnimationFrame:
    configuration_rad: np.ndarray
    action_index: int
    label: str


class XArm7Kinematics:
    """Small scipy/yourdfpy IK wrapper matching the project's xArm7 URDF."""

    def __init__(self, urdf_path: Path):
        try:
            import yourdfpy
        except ImportError as exc:
            raise KinematicsError(
                "yourdfpy is required for xArm URDF animation"
            ) from exc
        self.urdf_path = urdf_path.expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.urdf = yourdfpy.URDF.load(
            self.urdf_path,
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
            filename_handler=partial(
                yourdfpy.filename_handler_magic, dir=self.urdf_path.parent
            ),
        )
        names = list(self.urdf.actuated_joint_names)
        if names[:7] != [f"joint{index}" for index in range(1, 8)] or names[7:] != [
            "drive_joint"
        ]:
            raise KinematicsError(f"unexpected xArm7 actuated joints: {names}")
        lower: list[float] = []
        upper: list[float] = []
        for joint in self.urdf.actuated_joints[:7]:
            if joint.limit is None:
                lower.append(-2.0 * np.pi)
                upper.append(2.0 * np.pi)
            else:
                lower.append(float(joint.limit.lower))
                upper.append(float(joint.limit.upper))
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

    def forward(self, joints_rad: np.ndarray, gripper_rad: float = 0.0) -> np.ndarray:
        joints = np.asarray(joints_rad, dtype=np.float64)
        if joints.shape != (7,):
            raise KinematicsError("xArm joint vector must contain seven values")
        actuated = np.asarray(joints, dtype=np.float64)
        if len(self.urdf.actuated_joint_names) > 6:
            actuated = np.concatenate([actuated, [float(gripper_rad)]])
        self.urdf.update_cfg(actuated)
        tip = "link_tcp" if "link_tcp" in self.urdf.link_map else "link6"
        return np.asarray(self.urdf.get_transform(tip, "world"), dtype=np.float64)

    def solve(
        self,
        pose_mm_deg: tuple[float, float, float, float, float, float] | list[float],
        seed_rad: np.ndarray,
    ) -> np.ndarray:
        try:
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
        except ImportError as exc:
            raise KinematicsError(
                "scipy is required for offline xArm inverse kinematics"
            ) from exc

        pose = np.asarray(pose_mm_deg, dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise KinematicsError("target TCP pose must be six finite xyz/rpy values")
        target_position = pose[:3] / 1000.0
        target_rotation = Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix()
        seed = np.clip(np.asarray(seed_rad, dtype=np.float64), self.lower, self.upper)

        def residual(joints: np.ndarray) -> np.ndarray:
            transform = self.forward(joints)
            position_error = (transform[:3, 3] - target_position) * 8.0
            rotation_error = Rotation.from_matrix(
                target_rotation @ transform[:3, :3].T
            ).as_rotvec()
            return np.concatenate([position_error, rotation_error])

        solved = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            max_nfev=350,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        transform = self.forward(solved.x)
        position_error_mm = float(
            np.linalg.norm(transform[:3, 3] - target_position) * 1000.0
        )
        orientation_error_deg = float(
            np.linalg.norm(
                Rotation.from_matrix(target_rotation @ transform[:3, :3].T).as_rotvec()
            )
            * 180.0
            / np.pi
        )
        if (
            not np.all(np.isfinite(solved.x))
            or position_error_mm > 2.0
            or orientation_error_deg > 2.0
        ):
            raise KinematicsError(
                "offline IK did not converge: "
                f"position_error={position_error_mm:.2f} mm, "
                f"orientation_error={orientation_error_deg:.2f} deg"
            )
        return solved.x.astype(np.float64)


class XArm6Kinematics(XArm7Kinematics):
    """Six-axis xArm wrapper using the same pose/IK interface as xArm7."""

    def __init__(self, urdf_path: Path):
        try:
            import yourdfpy
        except ImportError as exc:
            raise KinematicsError(
                "yourdfpy is required for xArm URDF animation"
            ) from exc
        self.urdf_path = urdf_path.expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.urdf = yourdfpy.URDF.load(
            self.urdf_path,
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
            filename_handler=partial(
                yourdfpy.filename_handler_magic, dir=self.urdf_path.parent
            ),
        )
        names = list(self.urdf.actuated_joint_names)
        if names[:6] != [f"joint{index}" for index in range(1, 7)]:
            raise KinematicsError(f"unexpected xArm6 actuated joints: {names}")
        self.joint_count = 6
        self.lower = np.asarray(
            [
                float(j.limit.lower) if j.limit is not None else -2.0 * np.pi
                for j in self.urdf.actuated_joints[:6]
            ],
            dtype=np.float64,
        )
        self.upper = np.asarray(
            [
                float(j.limit.upper) if j.limit is not None else 2.0 * np.pi
                for j in self.urdf.actuated_joints[:6]
            ],
            dtype=np.float64,
        )

    def forward(self, joints_rad: np.ndarray, gripper_rad: float = 0.0) -> np.ndarray:
        joints = np.asarray(joints_rad, dtype=np.float64)
        if joints.shape != (6,):
            raise KinematicsError("xArm6 joint vector must contain six values")
        self.urdf.update_cfg(joints)
        tip = "link_tcp" if "link_tcp" in self.urdf.link_map else "link6"
        return np.asarray(self.urdf.get_transform(tip, "world"), dtype=np.float64)

    def build_animation(
        self,
        actions: list[dict[str, Any]],
        home_joints_deg: tuple[float, ...],
        fixed_roll_deg: float,
        fixed_pitch_deg: float,
        *,
        joint_targets_rad: dict[int, tuple[float, ...]] | None = None,
        yaw_offset_deg: float = 0.0,
        arm_steps: int = 24,
        gripper_steps: int = 10,
    ) -> list[AnimationFrame]:
        if len(home_joints_deg) not in {6, 7}:
            raise KinematicsError(
                "home joint configuration must contain six or seven values"
            )
        home = np.radians(np.asarray(home_joints_deg, dtype=np.float64))
        current_joints = home.copy()
        current_gripper = 0.0
        frames = [
            AnimationFrame(
                np.concatenate([current_joints, [current_gripper]]), -1, "start"
            )
        ]

        def append_interpolation(
            target_joints: np.ndarray,
            target_gripper: float,
            steps: int,
            action_index: int,
            label: str,
        ) -> None:
            nonlocal current_joints, current_gripper
            start_joints = current_joints.copy()
            start_gripper = current_gripper
            for step in range(1, max(1, steps) + 1):
                alpha = step / max(1, steps)
                joints = start_joints + alpha * (target_joints - start_joints)
                gripper = start_gripper + alpha * (target_gripper - start_gripper)
                frames.append(
                    AnimationFrame(
                        np.concatenate([joints, [gripper]]), action_index, label
                    )
                )
            current_joints = target_joints.copy()
            current_gripper = float(target_gripper)

        for action_index, action in enumerate(actions):
            name = action.get("name")
            if name == "home":
                target_home = (
                    np.asarray(joint_targets_rad[action_index], dtype=np.float64)
                    if joint_targets_rad is not None
                    and action_index in joint_targets_rad
                    else home
                )
                append_interpolation(
                    target_home, current_gripper, arm_steps, action_index, "home"
                )
            elif name == "move":
                args = action.get("args", {})
                target_pose = [
                    float(args["x"]),
                    float(args["y"]),
                    float(args["z"]),
                    fixed_roll_deg,
                    fixed_pitch_deg,
                    (float(args["yaw"]) + float(yaw_offset_deg) + 180.0) % 360.0
                    - 180.0,
                ]
                if joint_targets_rad is not None:
                    if action_index not in joint_targets_rad:
                        raise KinematicsError(
                            f"controller IK is missing action {action_index + 1}"
                        )
                    target_joints = np.asarray(
                        joint_targets_rad[action_index], dtype=np.float64
                    )
                else:
                    target_joints = self.solve(target_pose, current_joints)
                append_interpolation(
                    target_joints, current_gripper, arm_steps, action_index, "move"
                )
            elif name == "open_gripper":
                append_interpolation(
                    current_joints, 0.0, gripper_steps, action_index, "open_gripper"
                )
            elif name == "close_gripper":
                append_interpolation(
                    current_joints, 0.85, gripper_steps, action_index, "close_gripper"
                )
        return frames


# Keep the historical seven-axis import available for archived experiments.
# New runtime code selects XArm6Kinematics explicitly.
if not hasattr(XArm7Kinematics, "build_animation"):
    XArm7Kinematics.build_animation = XArm6Kinematics.build_animation


def _model_locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return call


class ArmModel:
    """Axis-aware FK/IK and measured collision envelopes for the dual runtime.

    The historical classes above are retained in this independent source copy.
    This model uses the physical flange plus the configured SDK TCP offset.
    All public geometry is in millimetres; URDF translations are metres.
    """

    def __init__(self, config):
        import yourdfpy

        from .geometry import DualArmError

        self.config = config
        self._lock = threading.RLock()
        self.urdf = yourdfpy.URDF.load(
            config.urdf, load_meshes=False, load_collision_meshes=False
        )
        self.names = [f"joint{i}" for i in range(1, config.axis + 1)]
        if not set(self.names).issubset(self.urdf.joint_map):
            raise DualArmError(f"{config.arm_id}: model does not match axis count")
        self.flange = config.raw.get("flange_link", "link_eef")
        if self.flange not in self.urdf.link_map:
            raise DualArmError(f"model has no flange link {self.flange}")
        self.lower = np.array([self.urdf.joint_map[n].limit.lower for n in self.names])
        self.upper = np.array([self.urdf.joint_map[n].limit.upper for n in self.names])
        for capsule in config.capsules:
            if capsule["frame"] not in self.urdf.link_map:
                raise DualArmError(f"unknown capsule frame: {capsule['frame']}")
        self.capsule_reaches = {}
        parents = {j.child: j for j in self.urdf.robot.joints}
        for capsule in config.capsules:
            # Triangle inequality gives a configuration-independent upper bound
            # from each ancestor's axis to either capsule endpoint.
            reach = max(np.linalg.norm(capsule[key]) for key in ("start_mm", "end_mm"))
            weights = np.zeros(config.axis)
            frame = capsule["frame"]
            while frame != "link_base":
                joint = parents[frame]
                if joint.type not in {"fixed", "revolute", "continuous"}:
                    raise DualArmError("unsupported joint type in collision envelope")
                if joint.type != "fixed":
                    if joint.name not in self.names:
                        raise DualArmError(
                            "gripper envelopes must cover all openings from a fixed arm link"
                        )
                    weights[self.names.index(joint.name)] = reach
                if joint.origin is not None:
                    reach += np.linalg.norm(joint.origin[:3, 3]) * 1000
                frame = joint.parent
            self.capsule_reaches[capsule["name"]] = weights

    def capsule_motion_bounds(self, delta_deg):
        from .geometry import finite

        delta = np.radians(
            np.abs(finite(delta_deg, (self.config.axis,), "joint envelope"))
        )
        return {
            name: float(weights @ delta)
            for name, weights in self.capsule_reaches.items()
        }

    @_model_locked
    def update(self, joints_deg):
        from .geometry import DualArmError, finite

        joints = np.radians(finite(joints_deg, (self.config.axis,), "joints"))
        if np.any(joints < self.lower - 1e-6) or np.any(joints > self.upper + 1e-6):
            raise DualArmError(f"{self.config.arm_id}: joint limits exceeded")
        self.urdf.update_cfg(dict(zip(self.names, joints)))

    @_model_locked
    def frame(self, name):
        result = np.array(self.urdf.get_transform(name, "link_base"), dtype=float)
        result[:3, 3] *= 1000
        return result

    @_model_locked
    def frame_at(self, joints_deg, name):
        self.update(joints_deg)
        return self.frame(name)

    @_model_locked
    def forward(self, joints_deg):
        from .geometry import matrix_pose, pose_matrix

        self.update(joints_deg)
        return matrix_pose(
            self.frame(self.flange) @ pose_matrix(self.config.tcp_offset)
        )

    @_model_locked
    def inverse(self, pose_mm_deg, seed_deg):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        from .geometry import DualArmError, pose_error, pose_matrix

        target = pose_matrix(pose_mm_deg)
        seed = np.radians(seed_deg)

        def residual(joints):
            actual = pose_matrix(self.forward(np.degrees(joints)))
            return np.r_[
                (actual[:3, 3] - target[:3, 3]) / 1000,
                Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec(),
                # Resolve the seventh joint's redundant freedom toward the
                # preceding solution; otherwise tiny TCP steps can drift along
                # the null space despite a good Cartesian residual.
                1e-4 * (joints - seed),
            ]

        result = least_squares(
            residual,
            np.clip(seed, self.lower, self.upper),
            bounds=(self.lower, self.upper),
            max_nfev=100,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        q = np.degrees(result.x)
        distance, angle = pose_error(self.forward(q), pose_mm_deg)
        if not result.success or distance > 0.5 or angle > 0.2:
            raise DualArmError(
                f"{self.config.arm_id}: offline IK rejected target ({distance:.3f} mm, {angle:.3f} deg)"
            )
        return q

    @_model_locked
    def capsules(self, joints_deg):
        from .geometry import Capsule, apply

        self.update(joints_deg)
        output = []
        for c in self.config.capsules:
            frame = self.config.world_from_base @ self.frame(c["frame"])
            output.append(
                Capsule(
                    c["name"],
                    apply(frame, c["start_mm"]),
                    apply(frame, c["end_mm"]),
                    c["radius_mm"],
                )
            )
        return output
