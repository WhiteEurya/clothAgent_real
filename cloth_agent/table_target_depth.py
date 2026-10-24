"""Bare-support-plane fallback for one-shot placement targets, never grasps."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull

from .dual_arm.geometry import DualArmError
from .dual_arm.observation import DepthUnavailableError, sha

ROOT = Path(__file__).resolve().parents[1]
APPEARANCE_CONFIG = ROOT / "config/perception.free_exploration.json"
INTRINSICS_FILE = ROOT / "config/calibration/dual_arm_working_20261008/camA_intrinsics.yaml"
POLICY = "BARE_TABLE_TARGET_PLANE_V1"
PROJECT_POLICY = "TABLE_TARGET_RAY_PLANE_V3"


def _require(condition, message):
    if not condition:
        raise DualArmError("table-target fallback rejected: " + message)


def _estimate(observation, pixel, grasp_pixel, audit, *, require_local_support=True):
    import yaml
    from .perception import (
        RGBDFrame, _camera_parallel_table, _table_background_candidates, _table_roi_masks,
    )

    meta, xyz = observation.meta, observation.xyz
    _require(meta.get("camera_serial") == "317222073552", "unsupported camera")
    _require(not any(meta.get(k) for k in (
        "support_height_patch", "world_tilt_correction",
        "corrected_world_from_camera_affine_mm")), "unsupported compensated cloud")
    rgb = np.asarray(Image.open(observation.image).convert("RGB"))
    h, w = rgb.shape[:2]
    calibration = yaml.safe_load(INTRINSICS_FILE.read_text())
    _require(calibration["serial"] == meta["camera_serial"]
             and calibration["image_size"] == [w, h], "intrinsics identity/size mismatch")
    K = np.asarray(calibration["K"], dtype=float)
    T = np.asarray(meta["world_from_camera_mm"], dtype=float)
    _require(T.shape == (4, 4) and np.isfinite(T).all()
             and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5),
             "invalid camera transform")
    camera_xyz = (xyz - T[:3, 3]) @ T[:3, :3]
    valid = np.isfinite(camera_xyz).all(axis=2) & (camera_xyz[:, :, 2] > 0)
    yy, xx = np.nonzero(valid)
    _require(len(xx) >= 1000, "insufficient measured geometry")
    # Check the factory RGB intrinsics against this observation, not just its filename.
    take = np.linspace(0, len(xx)-1, min(2000, len(xx)), dtype=int)
    cam = camera_xyz[yy[take], xx[take]]
    projected = cam @ K.T
    error = np.linalg.norm(projected[:, :2]/projected[:, 2:] -
                           np.c_[xx[take], yy[take]], axis=1)
    audit["intrinsics_reprojection_p99_px"] = float(np.percentile(error, 99))
    _require(audit["intrinsics_reprojection_p99_px"] < .1, "intrinsics/cloud inconsistency")

    config = json.loads(APPEARANCE_CONFIG.read_text())
    keys = ("table_roi_xyxy", "table_plane_mode", "table_reference_clearance_px",
            "background_plane_inlier_threshold_mm", "min_depth_m", "max_depth_m")
    settings = SimpleNamespace(**{k: config[k] for k in keys})
    audit["appearance_config"] = dict(path=str(APPEARANCE_CONFIG),
                                      sha256=sha(APPEARANCE_CONFIG), settings=vars(settings))
    audit["intrinsics_file"] = dict(path=str(INTRINSICS_FILE), sha256=sha(INTRINSICS_FILE))
    # Perception's frame extrinsics use metres, while Observation stores millimetres.
    T_m = T.copy()
    T_m[:3, 3] /= 1000.
    frame = RGBDFrame("A", meta["camera_serial"], rgb, camera_xyz[:, :, 2]/1000., K, T_m)
    coefficients, stats, _, references = _camera_parallel_table(frame, settings)
    audit.update(plane_z_equals_ax_by_c=coefficients.tolist(),
                 plane_fit=stats, measured_references=references)
    candidates, excluded = _table_background_candidates(frame, settings, stats["background"])
    roi, _ = _table_roi_masks((h, w), tuple(settings.table_roi_xyxy))
    allowed = binary_erosion(roi & ~excluded, structure=np.ones((5, 5), dtype=bool))
    x, y = map(int, pixel)
    _require(0 <= x < w and 0 <= y < h, "target pixel outside image")
    if require_local_support:
        _require(allowed[y, x], "target is outside bare-table ROI or near foreground")
    else:
        # The 24-pixel exclusion is for selecting uncontaminated plane-fit samples,
        # not for rejecting a placement pixel that is itself visible bare support.
        appearance = stats["background"]
        target_contrast = float(np.linalg.norm(rgb[y, x].astype(float) -
                                              np.asarray(appearance["table_rgb_median"])))
        audit.update(target_inside_table_roi=bool(roi[y, x]),
                     target_table_rgb_distance=target_contrast,
                     target_table_rgb_threshold=appearance["applied_color_distance"],
                     target_foreground_clearance_px=0,
                     plane_reference_clearance_px=settings.table_reference_clearance_px)
        _require(roi[y, x], "target outside bare-table ROI")
        _require(target_contrast < appearance["applied_color_distance"],
                 "target pixel itself is not classified as bare table")
    gx, gy = map(int, grasp_pixel)
    _require(0 <= gx < w and 0 <= gy < h, "grasp pixel outside image")
    appearance = stats["background"]
    contrast = float(np.linalg.norm(rgb[gy, gx].astype(float) -
                                    np.asarray(appearance["table_rgb_median"])))
    audit["grasp_table_rgb_distance"] = contrast
    _require(contrast > 1.5*appearance["applied_color_distance"],
             "garment/table appearance ambiguous; dark or matching cloth unsupported")

    if require_local_support:
        # Bound the local hole and require surrounding real support, including coverage
        # around the query. Never extrapolate beyond nearby measured table support.
        raw_v, raw_u = np.nonzero(candidates)
        _require(len(raw_u) > 0, "no measured table pixels")
        nearest = float(np.sqrt(((raw_u-x)**2+(raw_v-y)**2).min()))
        audit["nearest_measured_table_pixel_px"] = nearest
        _require(nearest <= 25, "hole exceeds supported local radius")
        # Match the existing plane fitter's complete 15x15 reference patches.
        # Isolated returns at an occlusion boundary are not plane evidence.
        reference_centers = binary_erosion(candidates, structure=np.ones((15, 15), dtype=bool))
        v, u = np.nonzero(reference_centers)
        nearby = (u-x)**2 + (v-y)**2 <= 50**2
        u, v = u[nearby], v[nearby]
        _require(len(u) >= 100, "fewer than 100 nearby measured table pixels")
        audit.update(local_reference_count=len(u), reference_patch_size_px=15)
        hull = ConvexHull(np.c_[u, v])
        _require(np.all(hull.equations[:, :2] @ [x, y] + hull.equations[:, 2] <= 1e-7),
                 "target outside surrounding measured support")
        points = xyz[v, u]
        residual = points[:, 2] - (points[:, :2] @ coefficients[:2] + coefficients[2])
        p95 = float(np.percentile(np.abs(residual), 95))
        audit["local_plane_residual_p95_mm"] = p95
        tolerance = settings.background_plane_inlier_threshold_mm
        _require(p95 <= tolerance, "nearby measured geometry disagrees with table plane")
        patch = xyz[y-2:y+3, x-2:x+3].reshape(-1, 3)
        patch = patch[np.isfinite(patch).all(axis=1)]
        if len(patch):
            disagreement = np.abs(patch[:, 2] -
                                  (patch[:, :2] @ coefficients[:2] + coefficients[2]))
            audit["existing_target_depth_max_plane_error_mm"] = float(disagreement.max())
            _require(disagreement.max() <= tolerance, "existing target depth contradicts table")

    ray = T[:3, :3] @ np.linalg.solve(K, [x, y, 1.])
    normal = np.r_[-coefficients[:2], 1.]
    denominator = float(normal @ ray)
    _require(abs(denominator) > 1e-6, "camera ray parallel to support plane")
    camera_z_mm = (coefficients[2] - normal @ T[:3, 3])/denominator
    _require(settings.min_depth_m*1000 < camera_z_mm < settings.max_depth_m*1000,
             "plane intersection outside camera depth range")
    result = T[:3, 3] + camera_z_mm*ray
    _require(np.isfinite(result).all(), "nonfinite plane intersection")
    audit.update(source="TABLE_PLANE_ESTIMATE", measured_depth=False,
                 world_xyz_mm=result.tolist(), pixel_unchanged=True,
                 limitation="RGB bare-table heuristic; not valid for matching/dark garments or hidden objects")
    return result


def sample_target(observation, pixel, grasp_pixel, max_spread, audit_path):
    """Prefer unchanged measured-depth checks; only missing Pb depth may fall back."""
    audit = dict(policy=POLICY, role="target_only", target_px=list(pixel),
                 grasp_px=list(grasp_pixel), source="UNRESOLVED", status="CHECKING")
    try:
        result = observation.sample(pixel, max_spread)
        audit.update(source="MEASURED_DEPTH", measured_depth=True,
                     world_xyz_mm=result.tolist(), status="PASS")
        return result
    except DepthUnavailableError as exc:
        audit["original_depth_error"] = str(exc)
        try:
            result = _estimate(observation, pixel, grasp_pixel, audit)
            audit["status"] = "PASS"
            return result
        except Exception as error:
            audit.update(status="REJECT", error=f"{type(error).__name__}: {error}")
            raise
    except Exception as error:
        audit.update(status="REJECT", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        Path(audit_path).write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")


def project_target(observation, pixel, grasp_pixel, max_spread, audit_path):
    """One-shot Pb: intersect its RGB ray with the observed global support plane.

    No measured depth at Pb, local hole bound, or surrounding-support hull required.
    The plane still comes from the measured visible support, not hard-table Z=0.
    """
    audit=dict(policy=PROJECT_POLICY,role='target_only',
               target_px=list(pixel),grasp_px=list(grasp_pixel),source='UNRESOLVED',
               status='CHECKING',target_depth_sampled=False,local_support_required=False,
               plane_source='measured visible background support')
    try:
        result=_estimate(observation,pixel,grasp_pixel,audit,require_local_support=False)
        audit['status']='PASS'
        return result
    except Exception as error:
        audit.update(status='REJECT',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        Path(audit_path).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
