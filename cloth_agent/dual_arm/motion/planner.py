"""ClothAgent-facing offline planner; no SDK connection or motor commands."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from itertools import pairwise

import numpy as np

from ..collision import CollisionChecker
from ..geometry import DualArmError, finite
from .ik import GraspIK
from .path import PathPlanner
from .trajectory import parameterize, validate_timed
from .validation import IntervalValidator, scene_digest


class DualArmPlanner:
    def __init__(self, scene, *, ik_options=None):
        self.scene = scene
        self.checker = CollisionChecker(scene)
        self.ik = GraspIK(scene, self.checker, ik_options)
        self.validator = IntervalValidator(scene, self.checker)
        self.paths = PathPlanner(scene, self.checker, self.validator)

    def solve_grasp_pose(
        self, point_a, point_b, current_q_a, current_q_b, **directions
    ):
        return self.ik.solve(point_a, point_b, current_q_a, current_q_b, **directions)

    def plan(
        self,
        grasp_a,
        grasp_b,
        current_q_a,
        current_q_b,
        task="dual_grasp",
        *,
        approach_m=0.05,
        lift_m=0.05,
        spread_m=0.05,
        order="auto",
        velocity_limits_rad_s=None,
        acceleration_limits_rad_s2=None,
        path_timeout_s=15,
        return_after_release=False,
        **directions,
    ):
        phase, began = "input", time.monotonic()
        try:
            if task not in {"dual_grasp", "center_pair"}:
                raise DualArmError(
                    "this planner supports dual_grasp/center_pair; pin_pull requires a contact controller"
                )
            lengths = finite([approach_m, lift_m, spread_m], (3,), "phase distances")
            if np.any(lengths < 0) or np.any(lengths > 0.3) or approach_m < 0.005:
                raise DualArmError(
                    "phase distances must be 0..0.3 m, approach >= 0.005 m"
                )
            if not np.isfinite(path_timeout_s) or not 0 < path_timeout_s <= 120:
                raise DualArmError("path budget must be 0..120 seconds")
            if order not in {"auto", "simultaneous", "left_first", "right_first"}:
                raise DualArmError("unknown execution order")
            start = np.r_[
                finite(current_q_a, (6,), "q_a"), finite(current_q_b, (7,), "q_b")
            ]
            if not self.checker.check(start[:6], start[6:])["safe"]:
                raise DualArmError(
                    "current state is unsafe; automatic recovery is not planned"
                )
            phase = "grasp_ik"
            ik = self.solve_grasp_pose(
                grasp_a, grasp_b, start[:6], start[6:], **directions
            )
            if not ik["success"]:
                return dict(ik, failed_phase=phase)
            grasp = np.r_[ik["q_a"], ik["q_b"]]
            frames = self.scene.tcp_world(grasp[:6], grasp[6:])
            phase = "pregrasp_cartesian"
            pre = {k: f.copy() for k, f in frames.items()}
            for frame in pre.values():
                frame[:3, 3] += [0, 0, approach_m]
            reverse, corridors = self.paths.cartesian(
                grasp, pre, timeout_s=path_timeout_s
            )
            pre_q = reverse[-1]
            phase = "transit"
            failures = []
            for chosen in (
                ["simultaneous", "left_first", "right_first"]
                if order == "auto"
                else [order]
            ):
                try:
                    transit = self.paths.joint_path(
                        start, pre_q, order=chosen, timeout_s=path_timeout_s
                    )
                    break
                except DualArmError as exc:
                    failures.append(f"{chosen}: {exc}")
            else:
                raise DualArmError("; ".join(failures))
            edges, events = [], [{"kind": "open_grippers", "waypoint": 0}]

            def append(path, name, cartesian=None):
                for i, (a, b) in enumerate(pairwise(path)):
                    row = {
                        "phase": name,
                        "start_q_rad": a.tolist(),
                        "end_q_rad": b.tolist(),
                    }
                    if cartesian is not None:
                        row["cartesian"] = cartesian[i]
                    edges.append(row)

            append(transit, "transit")
            descend_corridors = [
                dict(c, start=c["end"], end=c["start"]) for c in corridors[::-1]
            ]
            append(reverse[::-1], "descend", descend_corridors)
            events += [
                {"kind": "close_grippers", "waypoint": len(edges)},
                {"kind": "require_hold_confirmation", "waypoint": len(edges)},
            ]
            q = grasp
            for name in ("lift", "spread"):
                phase = name
                destination = self.scene.tcp_world(q[:6], q[6:])
                if name == "lift":
                    for f in destination.values():
                        f[:3, 3] += [0, 0, lift_m]
                else:
                    direction = destination["right"][:3, 3] - destination["left"][:3, 3]
                    direction[2] = 0
                    if np.linalg.norm(direction) < 0.001:
                        raise DualArmError(
                            "grasp pair needs horizontal separation for spread"
                        )
                    direction /= np.linalg.norm(direction)
                    destination["left"][:3, 3] -= direction * spread_m / 2
                    destination["right"][:3, 3] += direction * spread_m / 2
                path, corridor = self.paths.cartesian(
                    q, destination, timeout_s=path_timeout_s
                )
                append(path, name, corridor)
                q = path[-1]
                events.append(
                    {"kind": "require_hold_confirmation", "waypoint": len(edges)}
                )
            if return_after_release:
                phase = "return"
                events.append(
                    {"kind": "require_release_confirmation", "waypoint": len(edges)}
                )
                append(
                    self.paths.joint_path(q, start, timeout_s=path_timeout_s),
                    "return_after_release",
                )
            phase = "time_parameterization"
            velocity = (
                np.full(13, 0.25)
                if velocity_limits_rad_s is None
                else finite(velocity_limits_rad_s, (13,), "velocity")
            )
            # URDF velocity is also a hard upper bound; acceleration must be
            # commissioned separately since the official URDF lacks it.
            urdf_velocity = np.array(
                [
                    m.urdf.joint_map[n].limit.velocity
                    for m in self.scene.models.values()
                    for n in m.names
                ]
            )
            velocity = np.minimum(velocity, urdf_velocity)
            acceleration = (
                np.full(13, 0.5)
                if acceleration_limits_rad_s2 is None
                else acceleration_limits_rad_s2
            )
            timed = parameterize(edges, velocity, acceleration)
            phase = "final_continuous_verification"
            verification = validate_timed(timed, self.validator)
            for event in events:
                event["time_s"] = timed["waypoint_times_s"][event["waypoint"]]
            result = {
                "success": True,
                "schema_version": 1,
                "units": "m_rad_s",
                "task": task,
                "world_frame": "xArm6 link_base",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "valid_for_s": 120.0,
                "scene_digest": scene_digest(self.scene),
                "initial_q_a": start[:6].tolist(),
                "initial_q_b": start[6:].tolist(),
                "grasp_pose_a": ik["grasp_pose_a"],
                "grasp_pose_b": ik["grasp_pose_b"],
                "minimum_clearance": verification["minimum_clearance_bound_m"],
                "execution_order": {
                    "transit": chosen,
                    "grasp_lift_spread": "simultaneous",
                },
                "ik": ik,
                "verification": verification,
                "events": events,
                **timed,
                "elapsed_s": time.monotonic() - began,
                "execution_authorized": False,
                "physical_execution_supported": False,
                "calibration_status": self.scene.raw.get(
                    "calibration_status", "unverified"
                ),
                "geometry_verified": self.scene.raw.get("geometry_verified") is True,
                "limitations": [
                    "nominal robot geometry only; cloth forces/deformation unmodeled",
                    "gripper and observation events require a separate guarded executor",
                    "SDK interpolation, latency and stopping bounds are not commissioned",
                ],
            }
            from .preflight import plan_digest

            result["plan_digest"] = plan_digest(result)
            return result
        except (DualArmError, ValueError, TypeError, KeyError) as exc:
            return {
                "success": False,
                "failed_phase": phase,
                "reason": str(exc),
                "elapsed_s": time.monotonic() - began,
                "execution_authorized": False,
            }
