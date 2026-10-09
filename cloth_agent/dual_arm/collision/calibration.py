"""Import provenance explicitly; never relabel pending calibration as verified."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ..geometry import DualArmError, transform
from .scene import ROOT

CALIBRATION = "config/calibration/dual_arm_working_20261008"


def load_calibration(root=ROOT):
    directory = Path(root) / CALIBRATION
    manifest = json.loads((directory / "import_manifest.json").read_text())
    data = {}
    for record in manifest["files"]:
        path = directory / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise DualArmError(f"calibration checksum mismatch: {path}")
        data[record["file"]] = yaml.safe_load(path.read_text())
    for filename, field in (
        ("dual_arm_base.yaml", "X_Base1Base2"),
        ("camA_extrinsics.yaml", "X_CammountCam"),
        ("camB_extrinsics.yaml", "X_CammountCam"),
    ):
        row = data[filename]
        if row.get("translation_units") != "m":
            raise DualArmError(f"{filename}: explicit metre units required")
        transform(row[field], filename)
    if data["dual_arm_base.yaml"].get("world_frame") != "xArm6 link_base":
        raise DualArmError("base calibration world convention is not xArm6 link_base")
    return data, manifest


def template(root=ROOT, *, synthetic=False):
    root = Path(root)
    data, manifest = load_calibration(root)
    home = json.loads((root / "data/robot/dual_arm_home.json").read_text())
    raw = {
        "schema_version": 1,
        "units": "m_rad",
        "world_frame": "xArm6 link_base",
        "calibration_status": "synthetic"
        if synthetic
        else data["dual_arm_base.yaml"]["status"],
        "geometry_verified": False,
        "safety_distance_m": 0.01,
        "geometry_uncertainty_m": {"left": 0.0, "right": 0.0, "environment": 0.0},
        "provenance": {
            "calibration_directory": CALIBRATION,
            "import_manifest": manifest,
            "initial_state": "historical saved Home, NOT a live controller read",
            "initial_state_timestamp": home["updated_at"],
            "mount_link_assumption": "link_eef; must confirm both physical camera mounts",
        },
        "arms": {},
        "attachments": [],
        "obstacles": [],
        "base_mount_contacts": [],
    }
    for key, index, label in (("left", 0, "A"), ("right", 1, "B")):
        saved = home["arms"][index]
        base = (
            np.eye(4)
            if key == "left"
            else np.asarray(data["dual_arm_base.yaml"]["X_Base1Base2"])
        )
        raw["arms"][key] = {
            "axis": saved["axis"],
            "serial": saved["control_box_sn"],
            "urdf": f"assets/robots/xarm{saved['axis']}/"
            + ("xarm6_wo_ee.urdf" if key == "left" else "xarm7_controller_fit.urdf"),
            "world_from_base_m": base.tolist(),
            "flange_link": "link_eef",
            "native_gripper_root": "xarm_gripper_base_link" if key == "right" else None,
            "tcp_offset_m_rad": [
                *(np.asarray(saved["tcp_offset"][:3]) / 1000),
                *np.radians(saved["tcp_offset"][3:]),
            ],
            "initial_q_rad": np.radians(saved["joints"]).tolist(),
        }
        camera = {
            "name": "cam_" + label.lower(),
            "kind": "camera",
            "arm": key,
            "serial": data[f"cam{label}_intrinsics.yaml"]["serial"],
            "link": "link_eef",
            "link_from_optical_m": data[f"cam{label}_extrinsics.yaml"]["X_CammountCam"],
            "optical_from_housing_m": np.eye(4).tolist() if synthetic else None,
            "size_m": [0.09, 0.025, 0.025],
            "note": "Nominal D435 housing example; measure optical origin to housing center, connectors, bracket and cable envelope.",
        }
        tool_pose = np.eye(4)
        tool_pose[2, 3] = 0.10
        gripper = {
            "name": key + "_gripper",
            "kind": "gripper",
            "arm": key,
            "link": "link_eef",
            "link_from_box_m": tool_pose.tolist() if synthetic else None,
            "size_m": [0.08, 0.12, 0.20] if synthetic else None,
            "note": "Must enclose all jaw openings and installed fingertips; replaces native gripper meshes for 13DoF queries.",
        }
        raw["attachments"] += [camera, gripper]
    table = np.eye(4)
    table[:3, 3] = [0.55, 0, -0.075]
    raw["obstacles"] = [
        {
            "name": "table",
            "size_m": [1.8, 1.2, 0.1] if synthetic else None,
            "world_from_box_m": table.tolist() if synthetic else None,
        }
    ]
    return raw
