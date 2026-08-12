"""Offline xArm7 kinematics and animation frames for the Viser dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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
            raise KinematicsError("yourdfpy is required for xArm URDF animation") from exc
        self.urdf_path = urdf_path.expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.urdf = yourdfpy.URDF.load(
            self.urdf_path,
            build_scene_graph=True,
            build_collision_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=self.urdf_path.parent),
        )
        names = list(self.urdf.actuated_joint_names)
        if names[:7] != [f"joint{index}" for index in range(1, 8)] or names[7:] != ["drive_joint"]:
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
        self.urdf.update_cfg(np.concatenate([joints, [float(gripper_rad)]]))
        return np.asarray(self.urdf.get_transform("link_tcp", "world"), dtype=np.float64)

    def solve(
        self,
        pose_mm_deg: tuple[float, float, float, float, float, float] | list[float],
        seed_rad: np.ndarray,
    ) -> np.ndarray:
        try:
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
        except ImportError as exc:
            raise KinematicsError("scipy is required for offline xArm inverse kinematics") from exc

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
        position_error_mm = float(np.linalg.norm(transform[:3, 3] - target_position) * 1000.0)
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

    def build_animation(
        self,
        actions: list[dict[str, Any]],
        home_joints_deg: tuple[float, ...],
        fixed_roll_deg: float,
        fixed_pitch_deg: float,
        *,
        joint_targets_rad: dict[int, tuple[float, ...]] | None = None,
        arm_steps: int = 24,
        gripper_steps: int = 10,
    ) -> list[AnimationFrame]:
        if len(home_joints_deg) != 7:
            raise KinematicsError("home joint configuration must contain seven values")
        home = np.radians(np.asarray(home_joints_deg, dtype=np.float64))
        current_joints = home.copy()
        current_gripper = 0.0
        frames = [
            AnimationFrame(np.concatenate([current_joints, [current_gripper]]), -1, "start")
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
                    if joint_targets_rad is not None and action_index in joint_targets_rad
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
                    float(args["yaw"]),
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
