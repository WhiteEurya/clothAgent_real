"""Multi-start constrained 13DoF IK, sharing the production FK and FCL model."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize

from ..geometry import DualArmError, finite


def unit(value, name):
    value = finite(value, (3,), name)
    norm = np.linalg.norm(value)
    if norm < 1e-9:
        raise DualArmError(f"{name}: nonzero direction required")
    return value / norm


@dataclass(frozen=True)
class IKOptions:
    starts: int = 5
    iterations: int = 120
    timeout_s: float = 30.0
    position_tolerance_m: float = 0.001
    approach_tolerance_rad: float = np.deg2rad(15)
    edge_tolerance_rad: float = np.deg2rad(25)
    clearance_buffer_m: float = 0.001
    preferred_margin_m: float = 0.04
    motion_weight: float = 1.0
    clearance_weight: float = 0.3
    grasp_weight: float = 0.2
    joint_weight: float = 0.05
    seed: int = 7

    def __post_init__(self):
        if type(self.starts) is not int or not 1 <= self.starts <= 32:
            raise DualArmError("IK starts must be 1..32")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 1000:
            raise DualArmError("IK iterations must be 1..1000")
        for name in (
            "timeout_s",
            "position_tolerance_m",
            "approach_tolerance_rad",
            "edge_tolerance_rad",
            "clearance_buffer_m",
            "preferred_margin_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise DualArmError(f"positive finite {name} required")
        if max(self.approach_tolerance_rad, self.edge_tolerance_rad) >= np.pi / 2:
            raise DualArmError("orientation tolerances must be below pi/2")
        if self.position_tolerance_m > 0.005:
            raise DualArmError("grasp position tolerance must be <= 5 mm")
        for name in (
            "motion_weight",
            "clearance_weight",
            "grasp_weight",
            "joint_weight",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise DualArmError("objective weights must be finite and nonnegative")


class BudgetExpired(Exception):
    pass


class GraspIK:
    def __init__(self, scene, checker, options=None):
        self.scene, self.checker = scene, checker
        self.options = options or IKOptions()
        self.lower = np.r_[scene.models["left"].lower, scene.models["right"].lower]
        self.upper = np.r_[scene.models["left"].upper, scene.models["right"].upper]

    def solve(
        self,
        point_a,
        point_b,
        current_q_a,
        current_q_b,
        *,
        approach_a=(0, 0, -1),
        approach_b=(0, 0, -1),
        edge_a=None,
        edge_b=None,
    ):
        opt = self.options
        points = [finite(p, (3,), "World grasp point") for p in (point_a, point_b)]
        axes = [unit(v, "World approach") for v in (approach_a, approach_b)]
        edges = []
        for edge, axis in zip((edge_a, edge_b), axes):
            if edge is None:
                edges.append(None)
            else:
                edge = unit(edge, "cloth edge")
                edges.append(
                    unit(edge - axis * np.dot(edge, axis), "projected cloth edge")
                )
        current = np.r_[
            finite(current_q_a, (6,), "q_a"), finite(current_q_b, (7,), "q_b")
        ]
        self.scene.transforms(current[:6], current[6:])  # validates limits
        began = time.monotonic()
        deadline = began + opt.timeout_s
        cache = {}
        best = None
        attempts = []
        span = self.upper - self.lower

        def tick():
            if time.monotonic() >= deadline:
                raise BudgetExpired

        def fk(q):
            tick()
            return list(self.scene.tcp_world(q[:6], q[6:]).values())

        def evaluate(q):
            nonlocal best
            tick()
            key = np.asarray(q, dtype=float).tobytes()
            if key not in cache:
                frames = fk(q)
                report = self.checker.check(q[:6], q[6:], details=True)
                positions = np.concatenate(
                    [f[:3, 3] - p for f, p in zip(frames, points)]
                )
                directions = np.array(
                    [np.dot(f[:3, 2], a) for f, a in zip(frames, axes)]
                )
                aligned = [
                    np.dot(f[:3, 0], e) ** 2
                    for f, e in zip(frames, edges)
                    if e is not None
                ]
                margins = np.array([r["margin_m"] for r in report["pairs"]])
                rotation_cost = sum(1 - d for d in directions) + sum(
                    1 - d for d in aligned
                )
                score = (
                    opt.motion_weight * np.sum(((q - current) / span) ** 2)
                    + opt.joint_weight
                    * np.sum((2 * (q - (self.upper + self.lower) / 2) / span) ** 4)
                    + opt.grasp_weight * rotation_cost
                    + opt.clearance_weight
                    * np.mean(np.maximum(0, 1 - margins / opt.preferred_margin_m) ** 2)
                )
                feasible = (
                    np.max(np.linalg.norm(positions.reshape(2, 3), axis=1))
                    <= opt.position_tolerance_m
                    and np.min(directions) >= np.cos(opt.approach_tolerance_rad) - 1e-8
                    and all(
                        d >= np.cos(opt.edge_tolerance_rad) ** 2 - 1e-8 for d in aligned
                    )
                    and report["safe"]
                    and margins.min() >= opt.clearance_buffer_m - 1e-8
                )
                row = (
                    float(score),
                    positions,
                    directions,
                    np.array(aligned),
                    margins,
                    frames,
                    report,
                )
                if feasible and (best is None or score < best[0]):
                    best = (float(score), q.copy(), row)
                if len(cache) > 128:
                    cache.clear()
                cache[key] = row
            return cache[key]

        def seed_residual(q):
            frames = fk(q)
            residual = []
            for f, p, axis, edge in zip(frames, points, axes, edges):
                residual.extend((f[:3, 3] - p) * 20)
                residual.extend((f[:3, 2] - axis) * 0.5)
                if edge is not None:
                    residual.append(0.5 * (1 - np.dot(f[:3, 0], edge) ** 2))
            return np.r_[residual, 1e-4 * (q - current) / span]

        constraints = [
            {"type": "eq", "fun": lambda q: evaluate(q)[1]},
            {
                "type": "ineq",
                "fun": lambda q: evaluate(q)[2] - np.cos(opt.approach_tolerance_rad),
            },
            {"type": "ineq", "fun": lambda q: evaluate(q)[4] - opt.clearance_buffer_m},
        ]
        if any(e is not None for e in edges):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda q: (
                        evaluate(q)[3] - np.cos(opt.edge_tolerance_rad) ** 2
                    ),
                }
            )
        rng = np.random.default_rng(opt.seed)
        for index in range(opt.starts):
            try:
                # Joint-space initial guesses only; no discrete grasp yaw candidates.
                seed = (
                    current
                    if index == 0
                    else np.clip(
                        current + rng.normal(0, 0.45 if index < 3 else 1.2, 13),
                        self.lower + 1e-8,
                        self.upper - 1e-8,
                    )
                )
                seeded = least_squares(
                    seed_residual,
                    seed,
                    bounds=(self.lower, self.upper),
                    max_nfev=opt.iterations,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                )
                evaluate(seeded.x)
                solved = minimize(
                    lambda q: evaluate(q)[0],
                    seeded.x,
                    method="SLSQP",
                    bounds=list(zip(self.lower, self.upper)),
                    constraints=constraints,
                    options={"maxiter": opt.iterations, "ftol": 1e-8},
                )
                final = evaluate(solved.x)
                attempts.append(
                    {
                        "start": index,
                        "solver_message": str(solved.message),
                        "position_error_m": np.linalg.norm(
                            final[1].reshape(2, 3), axis=1
                        ).tolist(),
                        "minimum_margin_m": float(final[4].min()),
                    }
                )
            except BudgetExpired:
                attempts.append(
                    {"start": index, "solver_message": "time budget exhausted"}
                )
                break
        metadata = {
            "backend": "scipy SLSQP constrained 13DoF IK with multi-start least-squares seeds",
            "attempts": attempts,
            "elapsed_s": time.monotonic() - began,
            "execution_authorized": False,
            "units": "m_rad",
            "edge_direction_supplied": [e is not None for e in edges],
        }
        if best is None:
            return {
                "success": False,
                "reason": "No feasible grasp pose found within the search budget; not a proof of unreachability.",
                **metadata,
            }
        score, q, row = best
        # Recheck independently of optimizer convergence flags/cached evaluations.
        report = self.checker.check(q[:6], q[6:])
        if not report["safe"]:
            return {
                "success": False,
                "reason": "Final collision verification failed",
                **metadata,
            }
        return {
            "success": True,
            "q_a": q[:6].tolist(),
            "q_b": q[6:].tolist(),
            "grasp_pose_a": row[5][0].tolist(),
            "grasp_pose_b": row[5][1].tolist(),
            "position_error_m": np.linalg.norm(row[1].reshape(2, 3), axis=1).tolist(),
            "approach_error_rad": np.arccos(np.clip(row[2], -1, 1)).tolist(),
            "minimum_clearance": report["min_distance"],
            "objective": score,
            **metadata,
        }
