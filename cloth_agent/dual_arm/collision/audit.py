"""Offline model/calibration evidence report. Never queries or commands hardware."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..geometry import DualArmError, finite, pose_error, pose_matrix
from .calibration import load_calibration, template
from .scene import ROOT, model_for


def model_audit(root=ROOT):
    root = Path(root)
    raw = template(root)
    home = json.loads((root / "data/robot/dual_arm_home.json").read_text())
    report = {
        "joint_order": [],
        "dof": 13,
        "joint_units": "rad",
        "distance_units": "m",
        "state_source": "saved historical Home; not live",
        "arms": {},
        "physical_acceptance": False,
        "unresolved": [
            "Fresh paired controller joint/TCP records and independent world distance measurement required",
            "Camera housing-center, bracket/cable and all-open gripper extents require measurement",
            "Imported coordinate correction is pending live validation",
        ],
    }
    models, world_tcp = {}, {}
    for key, index in (("left", 0), ("right", 1)):
        row = raw["arms"][key]
        model = models[key] = model_for(key, row, root)
        saved = home["arms"][index]
        q = np.asarray(saved["joints"])
        actual = model.forward(q)
        error = pose_error(actual, saved["home"])
        f = pose_matrix(actual)
        f[:3, 3] /= 1000
        world_tcp[key] = np.asarray(row["world_from_base_m"]) @ f
        joint_rows, links = [], []
        for name in model.names:
            j = model.urdf.joint_map[name]
            joint_rows.append(
                {
                    "name": name,
                    "type": j.type,
                    "lower_rad": j.limit.lower,
                    "upper_rad": j.limit.upper,
                    "velocity_rad_s": j.limit.velocity,
                }
            )
            report["joint_order"].append(key + "/" + name)
        for link in model.urdf.robot.links:
            paths = []
            for c in link.collisions:
                if c.geometry.mesh is not None:
                    path = model.config.urdf.parent / c.geometry.mesh.filename
                    paths.append(
                        {"path": str(path.relative_to(root)), "exists": path.is_file()}
                    )
            links.append(
                {
                    "link": link.name,
                    "collision_count": len(link.collisions),
                    "meshes": paths,
                }
            )
        core = {"link_base", *[f"link{i}" for i in range(1, row["axis"] + 1)]}
        body_complete = all(model.urdf.link_map[n].collisions for n in core)
        native_gripper = any(
            "gripper" in l.name and l.collisions for l in model.urdf.robot.links
        )
        report["arms"][key] = {
            "urdf": row["urdf"],
            "urdf_sha256": hashlib.sha256(model.config.urdf.read_bytes()).hexdigest(),
            "joint_order_matches": model.urdf.actuated_joint_names[: row["axis"]]
            == model.names,
            "joints": joint_rows,
            "links": links,
            "core_collision_complete": body_complete,
            "native_gripper_collision_present": native_gripper,
            "saved_tcp_base_m_rad": [
                *(np.asarray(saved["home"][:3]) / 1000),
                *np.radians(saved["home"][3:]),
            ],
            "fk_position_error_m": error[0] / 1000,
            "fk_orientation_error_rad": float(np.radians(error[1])),
            "saved_fk_pass_2mm_1deg": bool(error[0] <= 2 and error[1] <= 1),
        }
        if not native_gripper:
            report["unresolved"].append(
                f"{key}: URDF has no native gripper; measured attachment envelope is mandatory"
            )
        if error[0] > 2 or error[1] > 1:
            report["unresolved"].append(
                f"{key}: official URDF does not match the historical TCP record within 2 mm / 1 degree"
            )
    data, _ = load_calibration(root)
    base = np.asarray(data["dual_arm_base.yaml"]["X_Base1Base2"])
    report["base_transform"] = {
        "definition": "p_world = X_Base1Base2 @ p_xarm7_base; world is xArm6 link_base",
        "translation_units": "m",
        "matrix": base.tolist(),
        "base_origin_distance_m": float(np.linalg.norm(base[:3, 3])),
        "tcp_distance_at_saved_home_m": float(
            np.linalg.norm(world_tcp["left"][:3, 3] - world_tcp["right"][:3, 3])
        ),
        "status": data["dual_arm_base.yaml"]["status"],
        "450mm_vs_1131mm": "Withdrawn by the user: 450 mm was an erroneous calculation, not a calibration acceptance requirement.",
    }
    # A fitted URDF is a separate model candidate, never silently substituted.
    fitted = root / "assets/robots/xarm7/xarm7_controller_fit.urdf"
    if fitted.exists():
        candidate = dict(raw["arms"]["right"], urdf=str(fitted.relative_to(root)))
        predicted = model_for("right", candidate, root).forward(
            home["arms"][1]["joints"]
        )
        error = pose_error(predicted, home["arms"][1]["home"])
        report["right_fitted_candidate"] = {
            "position_error_m": error[0] / 1000,
            "orientation_error_rad": float(np.radians(error[1])),
            "automatically_selected": False,
        }
    return report


def validate_evidence(scene, evidence):
    """Compare saved *simultaneous* readings and labeled world distances.

    Evidence records explicit units and source identity; passing these checks is
    numerical consistency only and cannot establish authenticity of input data.
    """
    if (
        evidence.get("units") != "m_rad"
        or not evidence.get("capture_id")
        or not evidence.get("captured_at")
    ):
        raise DualArmError("evidence requires SI units, capture ID and timestamp")
    if set(evidence.get("arms", {})) != set(scene.models):
        raise DualArmError("paired evidence for both arms required")
    joints, residuals = {}, {}
    for key, model in scene.models.items():
        row = evidence["arms"][key]
        if row.get("serial") != scene.raw["arms"][key].get("serial"):
            raise DualArmError("controller identity mismatch in evidence")
        q = finite(row["q_rad"], (model.config.axis,), "evidence joints")
        tcp = finite(row["tcp_base_m_rad"], (6,), "evidence TCP")
        actual = model.forward(np.degrees(q))
        error = pose_error(actual, np.r_[tcp[:3] * 1000, np.degrees(tcp[3:])])
        residuals[key] = {
            "position_error_m": error[0] / 1000,
            "orientation_error_rad": float(np.radians(error[1])),
            "pass": bool(error[0] <= 2 and error[1] <= 1),
        }
        joints[key] = q
    tcps = scene.tcp_world(joints["left"], joints["right"])

    def endpoint(row):
        key = row["arm"]
        if key not in scene.models:
            raise DualArmError("unknown endpoint arm")
        local = finite(row["point_m"], (3,), "measured endpoint")
        if row["frame"] == "tcp":
            frame = tcps[key]
        else:
            model = scene.models[key]
            frame = model.frame_at(np.degrees(joints[key]), row["frame"])
            frame[:3, 3] /= 1000
            frame = np.asarray(scene.raw["arms"][key]["world_from_base_m"]) @ frame
        return frame[:3, :3] @ local + frame[:3, 3]

    measurements = []
    for row in evidence.get("distances", []):
        values = finite(
            [row["measured_m"], row["tolerance_m"]], (2,), "distance evidence"
        )
        if values[0] <= 0 or not 0 < values[1] <= 0.01 or not row.get("label"):
            raise DualArmError(
                "positive labeled measurement and tolerance <= 10 mm required"
            )
        predicted = float(np.linalg.norm(endpoint(row["a"]) - endpoint(row["b"])))
        measurements.append(
            {
                "label": row["label"],
                "measured_m": values[0],
                "predicted_m": predicted,
                "error_m": abs(predicted - values[0]),
                "pass": bool(abs(predicted - values[0]) <= values[1]),
            }
        )
    return {
        "capture_id": evidence["capture_id"],
        "fk": residuals,
        "distances": measurements,
        "numerical_consistency": bool(measurements)
        and all(r["pass"] for r in residuals.values())
        and all(r["pass"] for r in measurements),
        "physical_acceptance": False,
        "note": "Offline consistency check; requires independent physical review, multiple poses and attachment verification.",
    }
