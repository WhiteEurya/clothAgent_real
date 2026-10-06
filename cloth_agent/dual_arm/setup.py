"""Generate reviewable configuration and fit independently measured bases."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import DualArmError, finite, transform

DEFAULT_LIMITS = {
    "rate_hz": 20,
    "cartesian_speed_mm_s": 15,
    "cartesian_accel_mm_s2": 30,
    "joint_speed_deg_s": 10,
    "joint_accel_deg_s2": 20,
    "max_ik_step_deg": 3,
    "tracking_error_mm": 5,
    "tracking_error_deg": 2,
    "arrival_error_mm": 2,
    "dispatch_skew_s": 0.02,
    "max_tick_lateness_s": 0.04,
    "clearance_mm": 10,
    "max_grasp_age_s": 180,
    "max_span_mm": 1000,
    "min_span_mm": 80,
    "max_spread_mm": 100,
    "max_spread_step_mm": 10,
    "max_lift_mm": 150,
    "max_approach_mm": 100,
    "contact_descent_mm": 1,
    "max_vision_wait_s": 120,
    "stationary_tolerance_deg": 0.2,
    "max_plan_samples": 40000,
    "max_surface_spread_mm": 8,
    "center_radius_mm": 150,
    "center_line_tolerance_mm": 30,
}


def initial_config(root, home_path):
    root = Path(root)
    homes = json.loads(Path(home_path).read_text())["arms"]
    if len(homes) != 2:
        raise DualArmError("two independently taught home records required")
    records = {}
    for arm_id, home in zip(("left", "right"), homes):
        axis = home["axis"]
        records[arm_id] = {
            "ip": home["ip"],
            "axis": axis,
            "control_box_sn": home["control_box_sn"],
            "home_joints_deg": home["joints"],
            "tcp_offset_mm_deg": home["tcp_offset"],
            "urdf": f"assets/robots/xarm{axis}/"
            + ("xarm6_wo_ee.urdf" if axis == 6 else "xarm7.urdf"),
            "flange_link": "link_eef",
            "world_from_base_mm": None,
            "grasp_rpy_world_deg": None,
            "workspace": {"min_mm": None, "max_mm": None},
            "collision_capsules": [],
            "gripper": {
                "open": 850,
                "close": 5,
                "speed": 200,
                "open_tolerance": 15,
                "timeout_s": 20,
            },
        }
    from .safety import safety_template

    return {
        "schema_version": 1,
        "frame_id": "world_mm",
        "calibration_status": "incomplete",
        "calibration_id": None,
        "collision_geometry_verified": False,
        "servo_commissioned": False,
        "arm_layout_description": "SET physical left/right ownership and their visible positions in Camera A.",
        "arms": records,
        "limits": DEFAULT_LIMITS.copy(),
        "safety": safety_template(records),
        "obstacles": [],
        "camera": {
            "serial": "317222073552",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "mount_arm": "left",
            "mount_link": "link_eef",
            "mount_from_camera_mm": None,
        },
    }


def fit_base(points):
    source = np.asarray(points["base_points_mm"], dtype=float)
    target = np.asarray(points["world_points_mm"], dtype=float)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 4
    ):
        raise DualArmError("at least four paired 3D calibration points required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise DualArmError("calibration points must be finite")
    a, b = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(a, tol=1) < 2 or np.linalg.matrix_rank(b, tol=1) < 2:
        raise DualArmError("calibration points are collinear/degenerate")
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = target.mean(0) - rotation @ source.mean(0)
    residual = np.linalg.norm(source @ rotation.T + result[:3, 3] - target, axis=1)
    check = points.get("validation")
    if check is None:
        raise DualArmError("independent validation points required")
    va, vb = np.asarray(check["base_points_mm"]), np.asarray(check["world_points_mm"])
    if (
        va.shape != vb.shape
        or va.ndim != 2
        or va.shape[1] != 3
        or len(va) < 2
        or not np.isfinite(va).all()
        or not np.isfinite(vb).all()
    ):
        raise DualArmError("at least two independent finite validation points required")
    errors = np.linalg.norm(va @ rotation.T + result[:3, 3] - vb, axis=1)
    if residual.max() > 2 or errors.max() > 2:
        raise DualArmError("base calibration exceeds 2 mm fitting/validation tolerance")
    return {
        "world_from_base_mm": transform(result).tolist(),
        "fit_errors_mm": residual.tolist(),
        "validation_errors_mm": errors.tolist(),
    }


def mesh_capsules(urdf_path, axis, *, tool_radius_mm, tcp_offset):
    """Conservative PCA capsules enclose all URDF collision-mesh vertices.

    The physical tool envelope still needs measurement and review; the URDF
    does not describe every installed camera, cable and gripper attachment.
    """
    import trimesh
    import yourdfpy

    urdf_path = Path(urdf_path)
    robot = yourdfpy.URDF.load(
        urdf_path, load_meshes=False, load_collision_meshes=False
    )
    output = []
    for name in ["link_base", *[f"link{i}" for i in range(1, axis + 1)]]:
        clouds = []
        for c in robot.link_map[name].collisions:
            mesh = c.geometry.mesh
            if mesh is None:
                raise DualArmError(
                    "envelope generation currently requires mesh collision geometry"
                )
            path = urdf_path.parent / mesh.filename
            loaded = trimesh.load(path, force="mesh")
            vertices = np.asarray(loaded.vertices, dtype=float)
            if mesh.scale is not None:
                vertices *= np.asarray(mesh.scale)
            origin = c.origin if c.origin is not None else np.eye(4)
            clouds.append((vertices @ origin[:3, :3].T + origin[:3, 3]) * 1000)
        if not clouds:
            raise DualArmError(f"URDF has no collision shape for {name}")
        vertices = np.concatenate(clouds)
        center = vertices.mean(0)
        _, _, vectors = np.linalg.svd(vertices - center, full_matrices=False)
        axis_vector = vectors[0]
        projected = (vertices - center) @ axis_vector
        radius = (
            np.max(
                np.linalg.norm(
                    vertices - center - projected[:, None] * axis_vector, axis=1
                )
            )
            + 2
        )
        output.append(
            {
                "name": name,
                "frame": name,
                "start_mm": (center + projected.min() * axis_vector).tolist(),
                "end_mm": (center + projected.max() * axis_vector).tolist(),
                "radius_mm": float(radius),
            }
        )
    tip = finite(tcp_offset, (6,), "TCP offset")[:3]
    length = float(np.linalg.norm(tip))
    if length <= tool_radius_mm:
        raise DualArmError(
            "tool envelope requires TCP extension larger than its radius; supply custom measured shapes otherwise"
        )
    # Distal spherical cap ends at the TCP rather than beyond the fingertips.
    output.append(
        {
            "name": "tool",
            "frame": "link_eef",
            "start_mm": [0, 0, 0],
            "end_mm": (tip * (length - tool_radius_mm) / length).tolist(),
            "radius_mm": float(tool_radius_mm),
        }
    )
    return output
