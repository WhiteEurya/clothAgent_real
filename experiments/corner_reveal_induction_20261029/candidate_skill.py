"""Unvalidated, offline hypothesis proposal generator. No robot imports or dispatch.
Input RGB uint8 (not BGR). Domain: single purple T-shirt on dark background.
A segmentation PCA axis is a *visible shape* axis, not a semantic torso axis.
Two cases do not identify a grasp-side selector; do not silently invent one.
"""
import cv2
import numpy as np


def observable_geometry(image):
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Expected HxWx3 RGB uint8')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Appearance segmentation only, not a grasp rule. Domain restriction explicit.
    mask = cv2.inRange(hsv, np.array([120, 20, 55]), np.array([170, 200, 255]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise ValueError('No garment-color component')
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == k).astype(np.uint8)
    if mask.sum() < 0.005 * mask.size:
        raise ValueError('Garment component too small')
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('Garment component intersects image boundary')
    y, x = np.where(mask)
    points = np.column_stack([x, y]).astype(float)
    center = points.mean(axis=0)
    eig, basis = np.linalg.eigh(np.cov(points.T))
    if eig[1] / max(eig[0], 1e-9) < 1.25:
        raise ValueError('Visible shape axis ambiguous')
    longitudinal = basis[:, 1]
    transverse = np.array([longitudinal[1], -longitudinal[0]])
    # Sign deliberately immaterial: both branches are returned, no screen-left rule.
    q = (points - center) @ transverse
    bounds = np.quantile(q, [0.01, 0.99])
    return mask, points, center, longitudinal, transverse, q, bounds


def generate_action(image, target_corner):
    """Return abstention plus two concrete pixel-action hypotheses for human review.

    This is deliberately NOT a complete executable robot policy: material identity,
    occlusion, choice of side, grip angle, depth and safety are not identifiable here.
    Any side selection is an extra assumption, not a result of these two examples.
    Uses before RGB only; no history IDs, recorded coordinates, or after images.
    """
    if target_corner not in ('LS', 'RS'):
        return {'decision': 'ABSTAIN', 'reason': 'No hem evidence; unsupported target',
                'hypothesis_proposals': [], 'robot_executable': False}
    try:
        mask, points, c, v, u, q, bounds = observable_geometry(image)
    except ValueError as e:
        return {'decision': 'ABSTAIN', 'reason': str(e),
                'hypothesis_proposals': [], 'robot_executable': False}
    width = float(bounds[1] - bounds[0])
    proposals = []
    for side in [-1, 1]:
        # Median of a narrow extreme band avoids choosing a single noisy tip.
        band = (q <= bounds[0] + .05 * width) if side < 0 else (q >= bounds[1] - .05 * width)
        subset = points[band]
        median = np.median(subset, axis=0)
        # Choose an actual segmented pixel with interior support, not background.
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        support = dist[subset[:, 1].astype(int), subset[:, 0].astype(int)]
        safe = subset[support >= .01 * width]
        if not len(safe):
            continue
        grasp = safe[np.argmin(((safe - median) ** 2).sum(axis=1))]
        direction = side * u
        target = grasp + .35 * width * direction
        in_frame = bool(0 <= target[0] < image.shape[1] and 0 <= target[1] < image.shape[0])
        proposals.append({'hypothesis': 'H_WIDTH', 'grasp_px': grasp.tolist(),
                          'target_px': target.tolist(), 'direction_xy_unit': direction.tolist(),
                          'distance_px': .35 * width, 'distance_mm': None,
                          'lift_height_mm': None, 'gripper_angle_deg': None,
                          'target_in_image': in_frame,
                          'workspace_checked': False, 'collision_checked': False,
                          'target_semantic_identity_verified': False})
    return {'decision': 'ABSTAIN', 'robot_executable': False,
            'target_corner': target_corner,
            'reason': 'Unresolved grasp side and unverified semantic/occlusion preconditions',
            'hypothesis_proposals': proposals,
            'geometry': {'centroid_px': c.tolist(), 'visible_long_axis': v.tolist(),
                         'visible_cross_axis': u.tolist(), 'robust_cross_span_px': width},
            'engineering_assumptions': '5% edge band, 1% interior support, 0.35-span travel, '
            '1.25 eigenvalue ratio are provisional implementation choices, not learned physics.'}
