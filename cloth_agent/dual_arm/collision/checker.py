"""FCL solid-shape collision/distance queries in metres, independent of LLMs."""

from __future__ import annotations

import threading

import numpy as np

from ..geometry import DualArmError


class CollisionChecker:
    def __init__(self, scene):
        try:
            import fcl
        except ImportError as exc:
            raise DualArmError(
                "Install requirements-collision.txt in the viewer environment; python-fcl is required"
            ) from exc
        self.fcl, self.scene = fcl, scene
        self._lock = threading.RLock()
        self.objects = {}
        for shape in scene.shapes:
            if shape.size is not None:
                geometry = fcl.Box(*shape.size)
            else:
                faces = np.column_stack(
                    (np.full(len(shape.mesh.faces), 3), shape.mesh.faces)
                ).flatten()
                geometry = fcl.Convex(
                    np.asarray(shape.mesh.vertices), len(shape.mesh.faces), faces
                )
            self.objects[shape.name] = fcl.CollisionObject(geometry)

    def check(self, q_a, q_b, *, details=False):
        """q_a[6], q_b[7] in radians; safe is only for this modeled STATIC state.

        This does not certify trajectories, stopping distances or physical
        calibration and never grants execution permission.
        """
        with self._lock:
            frames = self.scene.transforms(q_a, q_b)
            for name, frame in frames.items():
                self.objects[name].setTransform(
                    self.fcl.Transform(frame[:3, :3], frame[:3, 3])
                )
            rows = []
            for a, b in self.scene.pairs():
                objects = self.objects[a.name], self.objects[b.name]
                hit = self.fcl.CollisionResult()
                self.fcl.collide(
                    *objects, self.fcl.CollisionRequest(num_max_contacts=1), hit
                )
                if hit.is_collision:
                    distance, nearest = (
                        0.0,
                        None,
                    )  # Unsigned distance: overlap clamps to zero.
                else:
                    result = self.fcl.DistanceResult()
                    distance = float(
                        self.fcl.distance(
                            *objects,
                            self.fcl.DistanceRequest(enable_nearest_points=True),
                            result,
                        )
                    )
                    if not np.isfinite(distance) or distance < 0:
                        raise DualArmError(
                            "FCL returned an invalid separation; refusing a safe result"
                        )
                    points = np.asarray(result.nearest_points)
                    if points.shape != (2, 3) or not np.isfinite(points).all():
                        raise DualArmError("FCL returned invalid witness points")
                    nearest = points.tolist()
                required = self.scene.clearance + a.inflation_m + b.inflation_m
                rows.append(
                    {
                        "pair": [a.name, b.name],
                        "distance_m": distance,
                        "required_distance_m": required,
                        "margin_m": distance - required,
                        "collision": bool(hit.is_collision),
                        "nearest_points_m": nearest,
                    }
                )
            if not rows:
                raise DualArmError("no collision pairs were checked")
            closest = min(rows, key=lambda r: r["distance_m"])
            limiting = min(rows, key=lambda r: r["margin_m"])
            collision = any(r["collision"] for r in rows)
            report = {
                "collision": collision,
                "min_distance": closest["distance_m"],
                "closest_pair": closest["pair"],
                "nearest_points_m": closest["nearest_points_m"],
                "safe": not collision and limiting["margin_m"] > 1e-9,
                "minimum_margin_m": limiting["margin_m"],
                "limiting_pair": limiting["pair"],
                "unsafe_pairs": [
                    r for r in rows if r["collision"] or r["margin_m"] <= 1e-9
                ],
                "checked_pairs": len(rows),
                "distance_units": "m",
                "joint_units": "rad",
                "distance_type": "unsigned solid distance; zero on contact/overlap",
                "scope": "configured static geometry only; convex hulls enclose each URDF collision mesh",
                "calibration_status": self.scene.raw.get(
                    "calibration_status", "unverified"
                ),
                "geometry_verified": self.scene.raw.get("geometry_verified") is True,
                "execution_authorized": False,
                "gripper_meshes_replaced_by_envelopes": self.scene.gripper_mesh_replacements,
                "excluded_pairs": [
                    {"pair": sorted(pair), "reason": reason}
                    for pair, reason in self.scene.exclusions.items()
                ],
            }
            if details:
                report["pairs"] = rows
            return report
