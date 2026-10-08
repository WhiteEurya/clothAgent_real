"""OMPL RRTConnect with continuous FCL interval checking and Cartesian continuation."""

from __future__ import annotations

import time
from itertools import pairwise

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..geometry import DualArmError


class PathPlanner:
    def __init__(self, scene, checker, validator):
        self.scene, self.checker, self.validator = scene, checker, validator
        self.lower = np.r_[scene.models["left"].lower, scene.models["right"].lower]
        self.upper = np.r_[scene.models["left"].upper, scene.models["right"].upper]

    def joint_path(self, start, goal, *, order="simultaneous", timeout_s=10):
        start, goal = np.asarray(start), np.asarray(goal)
        deadline = time.monotonic() + timeout_s
        for q in (start, goal):
            if not self.checker.check(q[:6], q[6:])["safe"]:
                raise DualArmError("path start or goal is unsafe")
        if order != "simultaneous":
            if order not in {"left_first", "right_first"}:
                raise DualArmError("unknown arm execution order")
            mid = start.copy()
            sl = slice(0, 6) if order == "left_first" else slice(6, 13)
            mid[sl] = goal[sl]
            first = self._rrt(start, mid, sl, deadline)
            second = self._rrt(
                mid,
                goal,
                slice(6, 13) if order == "left_first" else slice(0, 6),
                deadline,
            )
            return first + second[1:]
        return self._rrt(start, goal, slice(0, 13), deadline)

    def _rrt(self, start, goal, active, deadline):
        try:
            self.validator.certify(start, goal, deadline=deadline)
            return [start.copy(), goal.copy()]
        except DualArmError:
            pass
        try:
            from ompl import base as ob
            from ompl import geometric as og
        except ImportError as exc:
            raise DualArmError(
                "OMPL is required for detours; install requirements-motion.txt"
            ) from exc
        indices = np.arange(13)[active]
        space = ob.RealVectorStateSpace(len(indices))
        bounds = ob.RealVectorBounds(len(indices))
        for i, j in enumerate(indices):
            bounds.setLow(i, float(self.lower[j]))
            bounds.setHigh(i, float(self.upper[j]))
        space.setBounds(bounds)
        si = ob.SpaceInformation(space)

        def unpack(state):
            q = start.copy()
            q[indices] = [state[i] for i in range(len(indices))]
            return q

        def valid(state):
            if time.monotonic() >= deadline:
                return False
            q = unpack(state)
            return self.checker.check(q[:6], q[6:])["safe"]

        validator = self.validator

        class MotionCheck(ob.MotionValidator):
            def checkMotion(self, first, last):
                try:
                    validator.certify(unpack(first), unpack(last), deadline=deadline)
                    return True
                except DualArmError:
                    return False

        motion = MotionCheck(si)
        si.setStateValidityChecker(valid)
        si.setMotionValidator(motion)
        si.setup()
        problem = ob.ProblemDefinition(si)
        first, last = space.allocState(), space.allocState()
        for i, j in enumerate(indices):
            first[i], last[i] = float(start[j]), float(goal[j])
        problem.setStartAndGoalStates(first, last)
        planner = og.RRTConnect(si)
        planner.setRange(0.2)
        planner.setProblemDefinition(problem)
        planner.setup()
        planner.solve(max(0.001, deadline - time.monotonic()))
        if not problem.hasExactSolution():
            raise DualArmError(
                "OMPL found no exact collision-certified path in the time budget"
            )
        result = [unpack(s) for s in problem.getSolutionPath().getStates()]
        if not np.allclose(result[0], start, atol=1e-8) or not np.allclose(
            result[-1], goal, atol=1e-8
        ):
            raise DualArmError("OMPL path endpoints do not match requested states")
        # Every extracted edge gets a fresh certificate; no approximate solution.
        for a, b in pairwise(result):
            validator.certify(a, b)
        return result

    def cartesian(self, start, targets, *, step_m=0.005, timeout_s=30):
        q = np.asarray(start).copy()
        origin = self.scene.tcp_world(q[:6], q[6:])
        steps = max(
            1,
            int(
                np.ceil(
                    max(
                        np.linalg.norm(targets[k][:3, 3] - origin[k][:3, 3])
                        for k in origin
                    )
                    / step_m
                )
            ),
        )
        if steps > 500:
            raise DualArmError("Cartesian path exceeds waypoint budget")
        deadline = time.monotonic() + timeout_s
        result, constraints = [q.copy()], []
        previous = origin
        for i in range(1, steps + 1):
            desired = {}
            for k in origin:
                desired[k] = origin[k].copy()
                desired[k][:3, 3] += (targets[k][:3, 3] - origin[k][:3, 3]) * i / steps
                if not np.allclose(targets[k][:3, :3], origin[k][:3, :3], atol=1e-8):
                    raise DualArmError(
                        "Cartesian move must preserve optimized gripper orientation"
                    )

            def residual(candidate, desired=desired, seed=q):
                if time.monotonic() >= deadline:
                    raise DualArmError("Cartesian IK time budget exceeded")
                frames = self.scene.tcp_world(candidate[:6], candidate[6:])
                values = []
                for k, f in frames.items():
                    values.extend((f[:3, 3] - desired[k][:3, 3]) * 20)
                    values.extend(
                        Rotation.from_matrix(
                            desired[k][:3, :3].T @ f[:3, :3]
                        ).as_rotvec()
                    )
                return np.r_[values, 1e-5 * (candidate - seed)]

            solved = least_squares(
                residual,
                q,
                bounds=(self.lower, self.upper),
                max_nfev=100,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if np.max(np.abs(residual(solved.x)[:12])) > 1e-4:
                raise DualArmError(
                    "Cartesian continuation cannot satisfy TCP constraints"
                )
            if np.max(np.abs(solved.x - q)) > 0.15:
                raise DualArmError("Cartesian continuation changed IK branch")
            corridor = {
                "start": {k: v.tolist() for k, v in previous.items()},
                "end": {k: v.tolist() for k, v in desired.items()},
                "position_tolerance_m": 0.001,
                "orientation_tolerance_rad": np.deg2rad(1),
            }
            self.validator.certify(q, solved.x, deadline=deadline, cartesian=corridor)
            q = solved.x.copy()
            result.append(q)
            constraints.append(corridor)
            previous = desired
        return result, constraints
