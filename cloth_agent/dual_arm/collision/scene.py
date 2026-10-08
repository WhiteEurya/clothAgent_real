"""SI-unit collision scene using the existing ArmModel/yourdfpy kinematics.

URDFs remain untouched. Camera optical frames and physical housing frames are
separate. All public joints are radians, positions and distances are metres.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from ..geometry import DualArmError, finite, pose_matrix, transform
from ..kinematics import ArmModel

ROOT = Path(__file__).resolve().parents[3]


def positive(value, shape, name):
    result = finite(value, shape, name)
    if np.any(result <= 0):
        raise DualArmError(f"{name}: strictly positive values required")
    return result


def matrix_m(value, name):
    return transform(value, name)


def model_for(arm_id, row, root):
    """Only the adapter crosses the existing mm/degree API boundary."""
    frame = matrix_m(row["world_from_base_m"], f"{arm_id} base")
    frame_mm = frame.copy()
    frame_mm[:3, 3] *= 1000
    tcp = finite(row["tcp_offset_m_rad"], (6,), "TCP offset")
    cfg = SimpleNamespace(
        axis=row["axis"],
        arm_id=arm_id,
        urdf=(root / row["urdf"]).resolve(),
        raw={"flange_link": row.get("flange_link", "link_eef")},
        capsules=[],
        world_from_base=frame_mm,
        tcp_offset=np.r_[tcp[:3] * 1000, np.degrees(tcp[3:])],
    )
    return ArmModel(cfg)


def collision_mesh(geometry, directory):
    if geometry.mesh is not None:
        path = (directory / geometry.mesh.filename).resolve()
        if not path.is_file():
            raise DualArmError(f"collision mesh missing: {path}")
        mesh = trimesh.load(path, force="mesh", process=True)
        if geometry.mesh.scale is not None:
            mesh.apply_scale(positive(geometry.mesh.scale, (3,), "mesh scale"))
    elif geometry.box is not None:
        mesh = trimesh.creation.box(positive(geometry.box.size, (3,), "box size"))
    else:
        raise DualArmError("unsupported URDF collision primitive; never omit geometry")
    if len(mesh.vertices) < 4 or not np.isfinite(mesh.vertices).all():
        raise DualArmError("invalid collision mesh")
    # A solid convex hull encloses the original mesh, including its interior;
    # triangle-only intersection queries can miss complete containment.
    return mesh.convex_hull


@dataclass
class Shape:
    name: str
    arm: str | None
    link: str | None
    kind: str
    mesh: trimesh.Trimesh
    local: np.ndarray
    size: np.ndarray | None = None
    inflation_m: float = 0.0


class CollisionScene:
    def __init__(self, raw, root=ROOT):
        self.raw, self.root = copy.deepcopy(raw), Path(root).resolve()
        if raw.get("schema_version") != 1 or raw.get("units") != "m_rad":
            raise DualArmError(
                "collision scene requires schema_version=1 and units=m_rad"
            )
        if set(raw.get("arms", {})) != {"left", "right"}:
            raise DualArmError("exactly left (xArm6) and right (xArm7) required")
        self.models, self.shapes, self.exclusions = {}, [], {}
        self.gripper_mesh_replacements = {}
        self.clearance = float(
            finite([raw["safety_distance_m"]], (1,), "safety distance")[0]
        )
        if not 0 <= self.clearance <= 0.5:
            raise DualArmError("safety distance must be in [0, 0.5] metres")
        self.initial = {}
        for key, axis in (("left", 6), ("right", 7)):
            row = raw["arms"][key]
            if type(row["axis"]) is not int or row["axis"] != axis:
                raise DualArmError(f"{key}: expected {axis} arm joints")
            model = self.models[key] = model_for(key, row, self.root)
            if model.urdf.actuated_joint_names[:axis] != model.names:
                raise DualArmError(
                    f"{key}: URDF joint order differs from controller order"
                )
            q = finite(row["initial_q_rad"], (axis,), "initial joints")
            model.update(np.degrees(q))
            self.initial[key] = q
            links = ["link_base", *[f"link{i}" for i in range(1, axis + 1)]]
            gripper_root = row.get("native_gripper_root")
            descendants = {gripper_root} if gripper_root is not None else set()
            if gripper_root is not None:
                if gripper_root not in model.urdf.link_map or gripper_root in links:
                    raise DualArmError("invalid native gripper subtree root")
                parents = {j.child: j for j in model.urdf.robot.joints}
                frame = gripper_root
                while frame != model.flange:
                    joint = parents.get(frame)
                    if joint is None or joint.type != "fixed":
                        raise DualArmError(
                            "native gripper root must be fixed below the flange"
                        )
                    frame = joint.parent
                if gripper_root == model.flange:
                    raise DualArmError("native gripper root cannot replace the flange")
                while True:
                    children = {
                        j.child
                        for j in model.urdf.robot.joints
                        if j.parent in descendants
                    }
                    if children <= descendants:
                        break
                    descendants |= children
            replaced = []
            for candidate in model.urdf.robot.links:
                if candidate.collisions and candidate.name not in links:
                    if candidate.name not in descendants:
                        raise DualArmError(
                            f"{key}/{candidate.name}: collision link has no coverage"
                        )
                    replaced.append(candidate.name)
            self.gripper_mesh_replacements[key] = replaced
            for link in links:
                collisions = model.urdf.link_map[link].collisions
                if not collisions:
                    raise DualArmError(f"{key}/{link}: collision geometry missing")
                for i, c in enumerate(collisions):
                    mesh = collision_mesh(c.geometry, model.config.urdf.parent)
                    local = (
                        np.eye(4)
                        if c.origin is None
                        else matrix_m(c.origin, "collision origin")
                    )
                    self.shapes.append(
                        Shape(f"{key}/{link}/{i}", key, link, "link", mesh, local)
                    )
            # Only direct mechanical neighbours are exempt by topology. No
            # arbitrary pair-disable list or cross-arm exemption is accepted.
            neighbors = {
                frozenset((j.parent, j.child)) for j in model.urdf.robot.joints
            }
            for a in self.shapes:
                for b in self.shapes:
                    if (
                        a.arm == b.arm == key
                        and a.name != b.name
                        and (
                            a.link == b.link or frozenset((a.link, b.link)) in neighbors
                        )
                    ):
                        self.exclusions[frozenset((a.name, b.name))] = (
                            "same link or direct URDF joint neighbours"
                        )
        if raw.get("disabled_pairs"):
            raise DualArmError("arbitrary disabled collision pairs are forbidden")
        attachments = raw.get("attachments", [])
        for key in self.models:
            for kind in ("camera", "gripper"):
                if not any(
                    a.get("arm") == key and a.get("kind") == kind for a in attachments
                ):
                    raise DualArmError(f"{key}: {kind} collision envelope required")
        for row in attachments:
            key, link, kind = row["arm"], row["link"], row["kind"]
            if key not in self.models or kind not in {
                "camera",
                "gripper",
                "bracket",
                "cable",
            }:
                raise DualArmError("invalid attachment owner or kind")
            model = self.models[key]
            if link not in model.urdf.link_map:
                raise DualArmError("unknown attachment link")
            # Attachments must be fixed to arm links, never moving gripper jaws.
            frame = link
            parents = {j.child: j for j in model.urdf.robot.joints}
            while frame not in {
                "link_base",
                *[f"link{i}" for i in range(1, model.config.axis + 1)],
            }:
                joint = parents[frame]
                if joint.type != "fixed":
                    raise DualArmError(
                        "attachment must be rigidly mounted to an arm link"
                    )
                frame = joint.parent
            if kind == "camera":
                optical = matrix_m(
                    row["link_from_optical_m"], "hand-eye optical transform"
                )
                housing = matrix_m(
                    row["optical_from_housing_m"], "optical-to-housing transform"
                )
                local = optical @ housing
            else:
                local = matrix_m(row["link_from_box_m"], "attachment transform")
            size = positive(row["size_m"], (3,), "attachment size")
            mesh = trimesh.creation.box(size)
            shape = Shape(row["name"], key, link, kind, mesh, local, size)
            # Intentional mount intersection is restricted to the actual rigid
            # host link. Camera-vs-gripper and all cross-arm pairs stay enabled.
            for a in self.shapes:
                if a.arm == key and a.kind == "link" and a.link == frame:
                    self.exclusions[frozenset((a.name, shape.name))] = (
                        "attachment contact with its rigid host link"
                    )
            self.shapes.append(shape)
        if not raw.get("obstacles"):
            raise DualArmError("table/environment geometry required")
        for row in raw["obstacles"]:
            size = positive(row["size_m"], (3,), "obstacle size")
            self.shapes.append(
                Shape(
                    row["name"],
                    None,
                    None,
                    "obstacle",
                    trimesh.creation.box(size),
                    matrix_m(row["world_from_box_m"], "obstacle pose"),
                    size,
                )
            )
        names = [s.name for s in self.shapes]
        if len(set(names)) != len(names):
            raise DualArmError("geometry names must be unique")
        # Installation contacts must identify ONLY the fixed robot base.
        for row in raw.get("base_mount_contacts", []):
            obstacle, key = row["obstacle"], row["arm"]
            if key not in self.models or obstacle not in {
                s.name for s in self.shapes if s.kind == "obstacle"
            }:
                raise DualArmError("invalid fixed base contact")
            if not isinstance(row.get("reason"), str) or not row["reason"].strip():
                raise DualArmError("base mounting contact requires a reason")
            for a in self.shapes:
                if a.arm == key and a.link == "link_base" and a.kind == "link":
                    self.exclusions[frozenset((a.name, obstacle))] = row["reason"]
        for shape in self.shapes:
            margin = raw.get("geometry_uncertainty_m", {}).get(
                shape.arm or "environment", 0
            )
            shape.inflation_m = float(finite([margin], (1,), "geometry uncertainty")[0])
            if not 0 <= shape.inflation_m <= 0.5:
                raise DualArmError("geometry uncertainty must be nonnegative metres")

    @classmethod
    def load(cls, path, root=ROOT):
        return cls(json.loads(Path(path).read_text()), root)

    def configurations(self, q_a, q_b):
        return {
            key: finite(q, (self.models[key].config.axis,), key + " radians")
            for key, q in (("left", q_a), ("right", q_b))
        }

    def transforms(self, q_a, q_b):
        joints = self.configurations(q_a, q_b)
        frames = {}
        for key, model in self.models.items():
            with model._lock:
                model.update(np.degrees(joints[key]))
                for link in {s.link for s in self.shapes if s.arm == key}:
                    f = model.frame(link)
                    f[:3, 3] /= 1000
                    frames[key, link] = (
                        np.asarray(self.raw["arms"][key]["world_from_base_m"]) @ f
                    )
        return {
            s.name: (s.local if s.arm is None else frames[s.arm, s.link] @ s.local)
            for s in self.shapes
        }

    def pairs(self):
        for i, a in enumerate(self.shapes):
            for b in self.shapes[i + 1 :]:
                if a.arm is None and b.arm is None:
                    continue
                if frozenset((a.name, b.name)) not in self.exclusions:
                    yield a, b

    def tcp_world(self, q_a, q_b):
        result = {}
        for key, q in self.configurations(q_a, q_b).items():
            f = pose_matrix(self.models[key].forward(np.degrees(q)))
            f[:3, 3] /= 1000
            result[key] = np.asarray(self.raw["arms"][key]["world_from_base_m"]) @ f
        return result
