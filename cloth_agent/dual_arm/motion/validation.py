"""Conservative certificates for complete joint intervals, not endpoint samples."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import DualArmError, finite, pose_matrix


def scene_digest(scene):
    h = hashlib.sha256(json.dumps(scene.raw, sort_keys=True, allow_nan=False).encode())
    for key, model in scene.models.items():
        h.update(key.encode())
        h.update(model.config.urdf.read_bytes())
    for shape in scene.shapes:
        h.update(shape.name.encode())
        for value in (shape.local, shape.mesh.vertices, shape.mesh.faces):
            h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def reaches(model, link, points):
    """Triangle-inequality upper bound from each ancestor axis to every point."""
    radius = float(np.linalg.norm(points, axis=1).max())
    weights = np.zeros(model.config.axis)
    parents = {j.child: j for j in model.urdf.robot.joints}
    while link != "link_base":
        joint = parents[link]
        if joint.type not in {"fixed", "revolute", "continuous"}:
            raise DualArmError("unsupported joint in motion bound")
        if joint.type != "fixed":
            if joint.name not in model.names:
                raise DualArmError(
                    "moving gripper must be replaced by a fixed envelope"
                )
            weights[model.names.index(joint.name)] = radius
        if joint.origin is not None:
            radius += np.linalg.norm(joint.origin[:3, 3])
        link = joint.parent
    return weights


class IntervalValidator:
    def __init__(self, scene, checker, *, max_nodes=8192):
        self.scene, self.checker = scene, checker
        self.max_nodes = max_nodes
        self.weights = {}
        for s in scene.shapes:
            weights = np.zeros(13)
            if s.arm is not None:
                points = s.mesh.vertices @ s.local[:3, :3].T + s.local[:3, 3]
                weights[:6] = (
                    reaches(scene.models[s.arm], s.link, points)
                    if s.arm == "left"
                    else 0
                )
                if s.arm == "right":
                    weights[6:] = reaches(scene.models[s.arm], s.link, points)
            self.weights[s.name] = weights
        self.tcp_weights = {}
        for key, model in scene.models.items():
            local = pose_matrix(model.config.tcp_offset)[:3, 3] / 1000
            self.tcp_weights[key] = reaches(model, model.flange, local[None, :])

    def certify(self, start, end, *, deadline=None, padding=None, cartesian=None):
        a, b = finite(start, (13,), "start q"), finite(end, (13,), "end q")
        padding = (
            np.zeros(13)
            if padding is None
            else finite(padding, (13,), "joint uncertainty")
        )
        if np.any(padding < 0):
            raise DualArmError("joint uncertainty must be nonnegative")
        for q in (a - padding, a + padding, b - padding, b + padding):
            self.scene.transforms(q[:6], q[6:])
        minimum, nodes = float("inf"), 0
        pending = [(a, b, 0.0, 1.0, 0)]
        while pending:
            if deadline is not None and time.monotonic() >= deadline:
                raise DualArmError("continuous interval check exceeded time budget")
            lo, hi, u0, u1, depth = pending.pop()
            nodes += 1
            if nodes > self.max_nodes or depth > 24:
                raise DualArmError(
                    "continuous interval cannot be certified within node budget"
                )
            center, half = (lo + hi) / 2, np.abs(hi - lo) / 2 + padding
            report = self.checker.check(center[:6], center[6:], details=True)
            if not report["safe"]:
                raise DualArmError(
                    f"interval intersects unsafe state: {report['limiting_pair']}"
                )
            motion = {name: float(w @ half) for name, w in self.weights.items()}
            lower_bounds = []
            certified = True
            for row in report["pairs"]:
                bound = row["distance_m"] - sum(motion[n] for n in row["pair"])
                lower_bounds.append(bound)
                if bound <= row["required_distance_m"] + 1e-9:
                    certified = False
            if cartesian is not None:
                frames = self.scene.tcp_world(center[:6], center[6:])
                for key, sl in (("left", slice(0, 6)), ("right", slice(6, 13))):
                    first, last = (
                        np.asarray(cartesian["start"][key]),
                        np.asarray(cartesian["end"][key]),
                    )
                    # Cartesian phases keep the chosen gripper orientation fixed.
                    if not np.allclose(first[:3, :3], last[:3, :3], atol=1e-8):
                        raise DualArmError(
                            "Cartesian certificate requires constant target orientation"
                        )
                    u = (u0 + u1) / 2
                    desired = first[:3, 3] * (1 - u) + last[:3, 3] * u
                    error = np.linalg.norm(frames[key][:3, 3] - desired)
                    bound = (
                        error
                        + self.tcp_weights[key] @ half[sl]
                        + np.linalg.norm(last[:3, 3] - first[:3, 3]) * (u1 - u0) / 2
                    )
                    angle = Rotation.from_matrix(
                        first[:3, :3].T @ frames[key][:3, :3]
                    ).magnitude()
                    if (
                        bound > cartesian["position_tolerance_m"]
                        or angle + half[sl].sum()
                        > cartesian["orientation_tolerance_rad"]
                    ):
                        certified = False
            if certified:
                minimum = min(minimum, min(lower_bounds))
            else:
                mid = (u0 + u1) / 2
                pending.extend(
                    [(lo, center, u0, mid, depth + 1), (center, hi, mid, u1, depth + 1)]
                )
        return {
            "minimum_clearance_bound_m": minimum,
            "nodes": nodes,
            "scope": "continuous nominal joint interval; independent joint ranges inside each certified cell",
        }
