"""Two grounded grasp targets compiled into a checked, paired phase program."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from .config import number
from .geometry import (
    DualArmError,
    finite,
    matrix_pose,
    pose_error,
    pose_matrix,
)


@dataclass
class Phase:
    name: str
    kind: str
    targets: dict = field(default_factory=dict)
    holding: bool = False
    checkpoint: str | None = None
    tool_contact: bool = False
    gripper_arms: tuple[str, ...] = ("left", "right")
    fixed_arms: tuple[str, ...] = ()
    pin_arm: str | None = None


@dataclass
class Motion:
    phase: Phase
    times: np.ndarray
    joints: dict[str, np.ndarray]
    poses: dict[str, np.ndarray]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_targets(raw, config):
    if (
        set(raw)
        != {
            "schema_version",
            "observation_id",
            "grasps",
            "lift_mm",
            "spread_mm",
            "approach_mm",
            "mode",
            "center",
            "pin_arm",
        }
        or raw["schema_version"] != 2
    ):
        raise DualArmError(
            "invalid dual grasp proposal fields; schema_version=2 required"
        )
    if raw["mode"] not in {"pin_pull", "center_pair"}:
        raise DualArmError("choose pin_pull or center_pair")
    if raw["mode"] == "pin_pull":
        if raw["pin_arm"] not in config.arms or raw["center"] is not None:
            raise DualArmError("pin_pull requires a pin arm and no center")
    elif raw["pin_arm"] is not None or not isinstance(raw["center"], dict):
        raise DualArmError("center_pair requires a center and no pin arm")
    if not isinstance(raw["observation_id"], str) or not raw["observation_id"]:
        raise DualArmError("observation_id required")
    if set(raw["grasps"]) != set(config.arms):
        raise DualArmError("one grasp per arm required")
    points = dict(raw["grasps"])
    if raw["center"] is not None:
        points["center"] = raw["center"]
    for arm_id, grasp in points.items():
        if (
            set(grasp) != {"pixel_xy", "reason"}
            or not isinstance(grasp["reason"], str)
            or not grasp["reason"]
        ):
            raise DualArmError(f"{arm_id}: pixel and reason required")
        xy = finite(grasp["pixel_xy"], (2,), "grasp pixel")
        if np.any(xy < 0) or not np.all(xy == xy.astype(int)):
            raise DualArmError("integer original-image pixels required")
    for name, low, limit in [
        ("lift_mm", 10, "max_lift_mm"),
        ("spread_mm", 0, "max_spread_mm"),
        ("approach_mm", 10, "max_approach_mm"),
    ]:
        number(raw[name], name, low, config.limits[limit])
    return raw


def ground_targets(proposal, observation, config):
    read_targets(proposal, config)
    if proposal["observation_id"] != observation.meta["observation_id"]:
        raise DualArmError("proposal belongs to a different observation")
    targets = {}
    for arm_id, grasp in proposal["grasps"].items():
        xyz = observation.sample(
            grasp["pixel_xy"], config.limits["max_surface_spread_mm"]
        )
        # A pin is placed on the observed surface, never blindly driven below it.
        if arm_id != proposal["pin_arm"]:
            xyz[2] -= config.limits["contact_descent_mm"]
        pose = np.r_[xyz, config.arms[arm_id].grasp_rpy]
        config.arms[arm_id].validate_point(config.arms[arm_id].base_pose(pose)[:3])
        targets[arm_id] = pose
    distance = float(np.linalg.norm(targets["right"][:3] - targets["left"][:3]))
    if not config.limits["min_span_mm"] <= distance <= config.limits["max_span_mm"]:
        raise DualArmError("initial grasp span outside configured limits")
    if proposal["mode"] == "center_pair":
        center = observation.sample(
            proposal["center"]["pixel_xy"], config.limits["max_surface_spread_mm"]
        )
        points = [
            observation.sample(
                proposal["grasps"][k]["pixel_xy"],
                config.limits["max_surface_spread_mm"],
            )
            for k in config.arms
        ]
        if any(
            np.linalg.norm(p - center) > config.limits["center_radius_mm"]
            for p in points
        ):
            raise DualArmError("grasp lies outside the fixed center neighborhood")
        from .geometry import segment_distance

        if np.dot(points[0] - center, points[1] - center) >= 0:
            raise DualArmError(
                "grasp points must lie on opposite sides of the fixed center"
            )
        if (
            segment_distance(center, center, *points)
            > config.limits["center_line_tolerance_mm"]
        ):
            raise DualArmError("grasp pair does not straddle the fixed center")
    return targets


def build_phases(proposal, targets, initial_world, config):
    """Explicit task primitive: paired acquire, trial lift, spread, return, release.

    No automatic Home on failure. Successful retreat reverses the checked
    Cartesian approach to each arm's observed starting pose.
    """

    if proposal["mode"] == "pin_pull":
        return build_pin_phases(proposal, targets, initial_world, config)

    def offset(poses, z=0, spread=0):
        result = {k: np.array(p, dtype=float).copy() for k, p in poses.items()}
        direction = targets["right"][:3] - targets["left"][:3]
        direction[2] = 0
        length = np.linalg.norm(direction)
        if length < config.limits["min_span_mm"]:
            raise DualArmError("grasp points need horizontal separation")
        direction /= length
        for k, sign in [("left", -1), ("right", 1)]:
            result[k][:3] += direction * sign * spread / 2 + [0, 0, z]
        return result

    approach = offset(targets, z=proposal["approach_mm"])
    # Raise first at the existing XY before crossing toward either grasp.
    travel = {k: p.copy() for k, p in initial_world.items()}
    for k, pose in travel.items():
        pose[2] = max(pose[2], approach[k][2])
    phases = [
        Phase("open", "open"),
        Phase("clearance", "move", travel),
        Phase("approach", "move", approach),
        Phase("descend", "move", targets),
        Phase("grasp", "close"),
        Phase("grasp_check", "observe", holding=True, checkpoint="grasp"),
        Phase(
            "trial_lift",
            "move",
            offset(
                targets,
                z=min(
                    proposal["lift_mm"],
                    max(
                        10,
                        config.limits["clearance_mm"]
                        + config.limits["contact_descent_mm"]
                        + 1,
                    ),
                ),
            ),
            holding=True,
        ),
        Phase("trial_check", "observe", holding=True, checkpoint="trial_lift"),
        Phase("lift", "move", offset(targets, z=proposal["lift_mm"]), holding=True),
    ]
    count = max(
        1, math.ceil(proposal["spread_mm"] / config.limits["max_spread_step_mm"])
    )
    for index in range(1, count + 1):
        spread = proposal["spread_mm"] * index / count
        poses = offset(targets, z=proposal["lift_mm"], spread=spread)
        phases.extend(
            [
                Phase(f"spread_{index}", "move", poses, holding=True),
                Phase(
                    f"spread_check_{index}",
                    "observe",
                    holding=True,
                    checkpoint="spread",
                ),
            ]
        )
    # Undo stretch before lowering; both hold throughout this coordinated return.
    phases.extend(
        [
            Phase(
                "unspread", "move", offset(targets, z=proposal["lift_mm"]), holding=True
            ),
            Phase("lower", "move", targets, holding=True),
            Phase("release", "open"),
            Phase("retreat", "move", approach),
            Phase("clearance_return", "move", travel),
            Phase("return_start", "move", initial_world),
        ]
    )
    for phase in phases:
        phase.tool_contact = phase.name in {
            "descend",
            "grasp",
            "grasp_check",
            "trial_lift",
            "lower",
            "release",
            "retreat",
        }
    return phases


def build_pin_phases(proposal, targets, initial_world, config):
    """Pin with closed fingers, acquire with the other hand, then pull outward.

    This positional contact primitive is simulated only until a measured force
    controller is integrated. The physical entry point explicitly rejects it.
    """
    pin = proposal["pin_arm"]
    moving = next(k for k in config.arms if k != pin)
    direction = targets[moving][:3] - targets[pin][:3]
    direction[2] = 0
    distance = np.linalg.norm(direction)
    if distance < config.limits["min_span_mm"]:
        raise DualArmError("pin and grasp need horizontal separation")
    direction /= distance

    def poses(source, arm=None, z=0, pull=0):
        result = {k: p.copy() for k, p in source.items()}
        for k in [arm] if arm else config.arms:
            result[k][:3] += [0, 0, z]
            if k == moving:
                result[k][:3] += direction * pull
        return result

    approach = poses(targets, z=proposal["approach_mm"])
    travel = poses(initial_world)
    for k in travel:
        travel[k][2] = max(travel[k][2], approach[k][2])
    pin_down = poses(approach)
    pin_down[pin] = targets[pin].copy()
    phases = [
        Phase("open", "open"),
        Phase("clearance", "move", travel),
        Phase("approach", "move", approach),
        Phase("prepare_pin", "close", gripper_arms=(pin,)),
        Phase(
            "pin_descend", "move", pin_down, tool_contact=(pin,), fixed_arms=(moving,)
        ),
        Phase(
            "pin_check",
            "observe",
            checkpoint="pin_only",
            tool_contact=(pin,),
            pin_arm=pin,
        ),
        Phase(
            "moving_descend",
            "move",
            targets,
            tool_contact=True,
            fixed_arms=(pin,),
            pin_arm=pin,
        ),
        Phase(
            "moving_grasp",
            "close",
            holding=True,
            tool_contact=True,
            gripper_arms=(moving,),
            pin_arm=pin,
        ),
        Phase(
            "grasp_check",
            "observe",
            holding=True,
            checkpoint="pin_hold",
            tool_contact=True,
            pin_arm=pin,
        ),
        Phase(
            "trial_lift",
            "move",
            poses(targets, moving, z=min(10, proposal["lift_mm"])),
            holding=True,
            tool_contact=True,
            fixed_arms=(pin,),
            pin_arm=pin,
        ),
        Phase(
            "trial_check",
            "observe",
            holding=True,
            checkpoint="pin_hold",
            tool_contact=True,
            pin_arm=pin,
        ),
        Phase(
            "lift",
            "move",
            poses(targets, moving, z=proposal["lift_mm"]),
            holding=True,
            tool_contact=True,
            fixed_arms=(pin,),
            pin_arm=pin,
        ),
    ]
    count = max(
        1, math.ceil(proposal["spread_mm"] / config.limits["max_spread_step_mm"])
    )
    for i in range(1, count + 1):
        phases.extend(
            [
                Phase(
                    f"pull_{i}",
                    "move",
                    poses(
                        targets,
                        moving,
                        z=proposal["lift_mm"],
                        pull=proposal["spread_mm"] * i / count,
                    ),
                    holding=True,
                    tool_contact=(pin,),
                    fixed_arms=(pin,),
                    pin_arm=pin,
                ),
                Phase(
                    f"pull_check_{i}",
                    "observe",
                    holding=True,
                    checkpoint="pin_hold",
                    tool_contact=(pin,),
                    pin_arm=pin,
                ),
            ]
        )
    phases.extend(
        [
            Phase(
                "unpull",
                "move",
                poses(targets, moving, z=proposal["lift_mm"]),
                holding=True,
                tool_contact=(pin,),
                fixed_arms=(pin,),
                pin_arm=pin,
            ),
            Phase(
                "lower",
                "move",
                targets,
                holding=True,
                tool_contact=True,
                fixed_arms=(pin,),
                pin_arm=pin,
            ),
            Phase(
                "release",
                "open",
                tool_contact=True,
                gripper_arms=(moving,),
                pin_arm=pin,
            ),
            Phase(
                "moving_retreat",
                "move",
                pin_down,
                tool_contact=True,
                fixed_arms=(pin,),
                pin_arm=pin,
            ),
            Phase("unpin", "move", approach, tool_contact=(pin,), fixed_arms=(moving,)),
            Phase("open_pin", "open", gripper_arms=(pin,)),
            Phase("clearance_return", "move", travel),
            Phase("return_start", "move", initial_world),
        ]
    )
    return phases


def validate_sample(config, models, joints, *, holding, tool_contact=False):
    from .safety import check_state

    check_state(config, models, joints, tool_contact=tool_contact)
    poses = {}
    for k, arm in config.arms.items():
        base_pose = models[k].forward(joints[k])
        arm.validate_point(base_pose[:3])
        poses[k] = matrix_pose(arm.world_from_base @ pose_matrix(base_pose))
    if holding:
        span = np.linalg.norm(poses["left"][:3] - poses["right"][:3])
        if not config.limits["min_span_mm"] <= span <= config.limits["max_span_mm"]:
            raise DualArmError("paired trajectory exceeds cloth span limits")
    return poses


def compile_motion(phase, current_joints, config, models, inverse):
    from scipy.spatial.transform import Rotation, Slerp

    starts = validate_sample(
        config,
        models,
        current_joints,
        holding=phase.holding,
        tool_contact=phase.tool_contact,
    )
    limits = config.limits
    distance = max(np.linalg.norm(phase.targets[k][:3] - starts[k][:3]) for k in starts)
    duration = max(
        0.5,
        1.875 * distance / limits["cartesian_speed_mm_s"],
        math.sqrt(5.78 * distance / limits["cartesian_accel_mm_s2"]),
    )
    # Validate on a fine geometric grid; resample joint path on a shared clock.
    angle = max(pose_error(phase.targets[k], starts[k])[1] for k in starts)
    n = max(2, math.ceil(distance / 2), math.ceil(angle / 1))
    if n + 1 > limits["max_plan_samples"]:
        raise DualArmError("trajectory sample budget exceeded")
    progress = np.linspace(0, 1, n + 1)
    joints = {k: [np.array(q)] for k, q in current_joints.items()}
    rotations = {
        k: Slerp(
            [0, 1],
            Rotation.from_euler(
                "xyz", [starts[k][3:], phase.targets[k][3:]], degrees=True
            ),
        )
        for k in starts
    }
    for s in progress[1:]:
        for k, arm in config.arms.items():
            pose = np.r_[
                starts[k][:3] * (1 - s) + phase.targets[k][:3] * s,
                rotations[k](s).as_euler("xyz", degrees=True),
            ]
            if k in phase.fixed_arms:
                q = joints[k][-1].copy()
            else:
                q = finite(
                    inverse[k](arm.base_pose(pose), joints[k][-1]),
                    (arm.axis,),
                    "IK result",
                )
            if np.max(np.abs(q - joints[k][-1])) > limits["max_ik_step_deg"]:
                raise DualArmError(f"{k}: IK branch discontinuity")
            error = pose_error(models[k].forward(q), arm.base_pose(pose))
            if error[0] > 0.75 or error[1] > 0.5:
                raise DualArmError(f"{k}: IK/FK disagreement")
            joints[k].append(q)
        validate_sample(
            config,
            models,
            {k: v[-1] for k, v in joints.items()},
            holding=phase.holding,
            tool_contact=phase.tool_contact,
        )
    geometric = {k: np.asarray(v) for k, v in joints.items()}
    # Scale shared time until joint and actual FK Cartesian derivatives fit.
    for _ in range(8):
        count = math.ceil(duration * limits["rate_hz"])
        if count + 1 > limits["max_plan_samples"]:
            raise DualArmError("timed trajectory sample budget exceeded")
        times = np.linspace(0, duration, count + 1)
        u = times / duration
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        timed = {
            k: np.column_stack(
                [np.interp(s, progress, q[:, i]) for i in range(q.shape[1])]
            )
            for k, q in geometric.items()
        }
        factor = 1.0
        for q in timed.values():
            velocity = np.diff(q, axis=0) / (times[1] - times[0])
            accel = np.diff(
                np.vstack([np.zeros_like(q[0]), velocity, np.zeros_like(q[0])]), axis=0
            ) / (times[1] - times[0])
            factor = max(
                factor,
                np.max(np.abs(velocity)) / limits["joint_speed_deg_s"],
                math.sqrt(np.max(np.abs(accel)) / limits["joint_accel_deg_s2"]),
            )
        if factor <= 1.001:
            poses = {k: [] for k in config.arms}
            # Check actual joint-interpolated samples, not only desired waypoints.
            for index in range(len(times)):
                world = validate_sample(
                    config,
                    models,
                    {k: q[index] for k, q in timed.items()},
                    holding=phase.holding,
                    tool_contact=phase.tool_contact,
                )
                for k, points in poses.items():
                    points.append(world[k])
            for values in poses.values():
                xyz = np.asarray(values)[:, :3]
                velocity = np.diff(xyz, axis=0) / (times[1] - times[0])
                accel = np.diff(
                    np.vstack([np.zeros(3), velocity, np.zeros(3)]), axis=0
                ) / (times[1] - times[0])
                factor = max(
                    factor,
                    np.linalg.norm(velocity, axis=1).max()
                    / limits["cartesian_speed_mm_s"],
                    math.sqrt(
                        np.linalg.norm(accel, axis=1).max()
                        / limits["cartesian_accel_mm_s2"]
                    ),
                )
            if factor <= 1.001:
                break
        duration *= factor * 1.05
    else:
        raise DualArmError("unable to time-scale joint trajectory")
    from .safety import validate_sweep

    if config.execution_mode == 'controller_sequential':
        from .stop_sweep import validate_native_path
        validate_native_path(config, models, timed, tool_contact=phase.tool_contact)
    for index in (range(1, len(times)) if config.execution_mode != 'controller_sequential' else ()):
        validate_sweep(
            config,
            models,
            {k: q[index - 1] for k, q in timed.items()},
            {k: q[index] for k, q in timed.items()},
            tool_contact=phase.tool_contact,
        )
    return Motion(phase, times, timed, {k: np.array(v) for k, v in poses.items()})


def compile_program(phases, initial_joints, config, models, inverse):
    program, joints = [], {k: np.array(q).copy() for k, q in initial_joints.items()}
    for phase in phases:
        if phase.kind == "move":
            motion = compile_motion(phase, joints, config, models, inverse)
            joints = {k: v[-1].copy() for k, v in motion.joints.items()}
            program.append(motion)
            if (
                sum(len(m.times) for m in program if isinstance(m, Motion))
                > config.limits["max_plan_samples"]
            ):
                raise DualArmError("whole-program trajectory sample budget exceeded")
        else:
            validate_sample(
                config,
                models,
                joints,
                holding=phase.holding,
                tool_contact=phase.tool_contact,
            )
            program.append(phase)
    return program
