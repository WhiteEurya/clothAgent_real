"""Metric single-tag camera alignment; camera transforms, never robot calibration."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Expected a finite proper rigid transform')
    return t


def fit_camera_transform(pairs):
    if len(pairs) < 3:
        raise ValueError('Need at least three fitting pairs')
    candidates = np.array([rigid(a) @ np.linalg.inv(rigid(b)) for a, b in pairs])
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(candidates[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = candidates[:, :3, 3].mean(axis=0)
    return rigid(result)


def compare_tag(a_from_tag, b_from_tag, a_from_b):
    a, b = rigid(a_from_tag), rigid(a_from_b) @ rigid(b_from_tag)
    delta = b[:3, 3] - a[:3, 3]
    return {'tag_A_xyz_mm': a[:3, 3].tolist(),
            'tag_B_in_A_xyz_mm': b[:3, 3].tolist(),
            'delta_B_minus_A_mm': delta.tolist(),
            'position_error_mm': float(np.linalg.norm(delta)),
            'orientation_error_deg': float(np.degrees(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))}


def summarize(rows):
    if not rows:
        return {'status': 'UNKNOWN', 'reason': 'No independent validation samples', 'count': 0}
    p=np.array([r['position_error_mm'] for r in rows]);a=np.array([r['orientation_error_deg'] for r in rows])
    return {'status': 'MEASURED_RESIDUALS_NOT_ACCURACY_CERTIFICATION', 'count': len(rows),
            'mean_delta_B_minus_A_mm': np.mean([r['delta_B_minus_A_mm'] for r in rows],axis=0).tolist(),
            'position_rmse_mm': float(np.sqrt(np.mean(p*p))), 'position_mean_mm': float(p.mean()),
            'position_p95_mm': float(np.percentile(p,95)), 'position_max_mm': float(p.max()),
            'orientation_mean_deg':float(a.mean()),'orientation_max_deg':float(a.max())}


def estimate_tag(gray, k, detector, tag_id, size_mm, max_error_px=2):
    """Rectified pixels; IPPE returns two square-PnP hypotheses, screen ambiguity."""
    import cv2
    detections = [d for d in detector.detect(gray) if int(d.tag_id)==tag_id]
    if len(detections)!=1:
        raise ValueError(f'Expected one tag {tag_id}; saw {len(detections)}')
    d=detections[0]
    if d.hamming != 0 or d.decision_margin < 20:
        raise ValueError('Tag decoding quality insufficient')
    h=size_mm/2
    obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],dtype=np.float64)
    corners=np.asarray(d.corners,dtype=np.float64)
    ok,rvecs,tvecs,_=cv2.solvePnPGeneric(obj,corners,k,None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok: raise ValueError('PnP failed')
    candidates=[]
    for rv,tv in zip(rvecs,tvecs):
        rotation=cv2.Rodrigues(rv)[0]
        if not np.all((obj @ rotation.T + tv.reshape(3))[:,2]>0): continue
        predicted=cv2.projectPoints(obj,rv,tv,k,None)[0].reshape(-1,2)
        err=float(np.sqrt(np.mean(np.sum((predicted-corners)**2,axis=1))))
        t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=tv.reshape(3)
        if np.isfinite(err) and np.isfinite(t).all(): candidates.append((err,t))
    candidates.sort(key=lambda c:c[0])
    if not candidates or candidates[0][0]>max_error_px: raise ValueError('Reprojection error too high')
    if len(candidates)>1:
        angle=np.degrees(Rotation.from_matrix(candidates[0][1][:3,:3].T@candidates[1][1][:3,:3]).magnitude())
        if angle>5 and candidates[1][0] <= max(candidates[0][0]*1.2, candidates[0][0]+.1):
            raise ValueError('Ambiguous planar pose; change tag viewing angle')
    err,t=candidates[0]
    return {'camera_from_tag_mm':rigid(t).tolist(),'corners_rectified_px':corners.tolist(),
            'reprojection_rmse_px':err,'decision_margin':float(d.decision_margin),
            'alternative_reprojection_rmse_px':[c[0] for c in candidates[1:]]}
