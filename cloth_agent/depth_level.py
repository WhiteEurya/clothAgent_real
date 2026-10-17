"""Empirical world-Y tilt correction about a fixed, measured world reference."""
import numpy as np


def bind_support_height_patch(raw, root):
    """Apply the shared policy to every configuration using measured Camera B."""
    import hashlib
    import json
    import re
    from copy import deepcopy
    from pathlib import Path

    if str(raw.get('camera', {}).get('serial')) != '233622079809':
        return raw
    raw = deepcopy(raw)
    patch = json.loads((Path(root)/'config/arm7_support_height_patch.json').read_text())
    if patch.get('mode') != 'support_plane_z' or type(patch.get('enabled')) is not bool:
        raise ValueError('invalid shared Camera B support-height policy')
    if raw['camera'].get('mount_arm') != 'right':
        raise ValueError('Camera B support-height policy requires its arm7 mount')
    raw['camera'].pop('support_height_patch', None)
    if patch['enabled']:
        raw['camera']['support_height_patch'] = patch
    digest = hashlib.sha256(json.dumps(patch, sort_keys=True).encode()).hexdigest()[:12]
    identity = re.sub(r'_supportZ_[0-9a-f]{12}$', '', raw['calibration_id'])
    raw['calibration_id'] = identity + '_supportZ_' + digest
    return raw


def level_support_cloud(xyz, correction, joints, world_from_camera):
    """Empirical height and optional XY correction at the measured camera pose.

    This is an affine height adjustment, not a rigid camera calibration.
    """
    plane = np.asarray(correction['plane_z_ax_by_c'], dtype=float)
    anchor = np.asarray(correction['anchor_xy_mm'], dtype=float)
    reference_q = np.asarray(correction['reference_joints_deg'], dtype=float)
    q = np.asarray(joints, dtype=float)
    reference = np.asarray(correction['reference_world_from_camera_mm'], dtype=float)
    current = np.asarray(world_from_camera, dtype=float)
    if (plane.shape != (3,) or anchor.shape != (2,) or q.shape != reference_q.shape
            or reference.shape != (4, 4) or current.shape != (4, 4)
            or not all(np.isfinite(v).all() for v in (plane, anchor, q, reference_q, reference, current))):
        raise ValueError('invalid support-height patch geometry')
    rotation = current[:3, :3] @ reference[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))
    if (np.max(np.abs(q-reference_q)) > correction['max_joint_delta_deg']
            or np.linalg.norm(current[:3, 3]-reference[:3, 3]) > correction['max_camera_translation_delta_mm']
            or angle > correction['max_camera_rotation_delta_deg']):
        raise ValueError('support-height patch requires the measured arm7 camera pose/calibration')
    matrix = np.eye(4)
    matrix[2, :2] = -plane[:2]
    policy = correction.get('z_policy', 'anchor_plane')
    if policy == 'raw_plane_to_reference_height':
        height = float(correction['reference_height_mm'])
        if not np.isfinite(height):
            raise ValueError('finite reference support height required')
        matrix[2, 3] = height-plane[2]
    elif policy == 'anchor_plane':
        matrix[2, 3] = plane[:2] @ anchor
    else:
        raise ValueError('unknown support Z policy')
    result = np.asarray(xyz, dtype=float).copy()
    result[..., 2] += matrix[2, 3]-result[..., :2] @ plane[:2]
    xy_patch = correction.get('xy_patch', {})
    if xy_patch.get('enabled', False):
        if (xy_patch.get('mode') != 'world_xy_linear_y'
                or xy_patch.get('outside_interval') != 'linear_extrapolation'):
            raise ValueError('unsupported XY support patch')
        xb, left, right, lb, rb = [float(xy_patch[k]) for k in
            ('x_bias_mm', 'left_y_mm', 'right_y_mm', 'left_y_bias_mm', 'right_y_bias_mm')]
        if not np.isfinite([xb, left, right, lb, rb]).all() or right-left <= 1e-6:
            raise ValueError('invalid XY patch endpoints or bias')
        slope = (rb-lb)/(right-left)
        if 1-slope <= 0:
            raise ValueError('XY patch must preserve Y ordering')
        # Both adjustments use the original measured coordinates. In particular,
        # do not evaluate the height plane again using corrected XY.
        result[..., 0] -= xb
        result[..., 1] -= lb+slope*(np.asarray(xyz)[..., 1]-left)
        matrix[0, 3] = -xb
        matrix[1, 1] = 1-slope
        matrix[1, 3] = slope*left-lb
    return result, matrix


def rotate_cloud(xyz, pivot, angle_deg):
    pivot = np.asarray(pivot, dtype=float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all() or not np.isfinite(angle_deg):
        raise ValueError('finite XYZ pivot and angle required')
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (np.asarray(xyz) - pivot) @ rotation.T + pivot, rotation
