"""Fixed-board RGB/depth/hand-eye consistency diagnostics; no robot commands."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml


A4_SCALE = 2 ** -.5


def load_board(path, scale=A4_SCALE):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Board scale must be positive and finite')
    content = Path(path).read_bytes()
    data = yaml.safe_load(content)
    if (data['schema'] != 'robot_cam_calib.apriltag_board.v1' or data['units'] != 'mm'
            or data['detection_corner_order']['source'] != 'pupil_apriltags.Detection.corners'):
        raise ValueError('Unsupported board schema, units or corner convention')
    corners = {}
    for tag in data['tags']:
        points = np.asarray(tag['corners_board_mm'], float) * scale / 1000
        if points.shape != (4, 3) or not np.isfinite(points).all() or tag['id'] in corners:
            raise ValueError('Invalid/duplicate board tag')
        corners[int(tag['id'])] = points
    if len(corners) < 4:
        raise ValueError('At least four board tags required')
    return {'name': data['name'], 'family': data['family'], 'corners': corners,
            'scale': float(scale), 'tag_size_mm': data['geometry']['tag_size_mm'] * scale,
            'source_sha256': hashlib.sha256(content).hexdigest()}


def transform_points(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def angle_deg(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0}
    return {'count': int(len(values)), 'rms': float(np.sqrt(np.mean(values**2))),
            'median': float(np.median(values)), 'max_abs': float(np.max(np.abs(values)))}


def project(transform, points, k):
    camera = transform_points(transform, points)
    if np.any(camera[:, 2] <= 0):
        raise ValueError('PnP points behind camera')
    homogeneous = camera @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def solve_board_pose(objects, pixels, k):
    # Keep both planar solutions to flag near-equal fits with different normals.
    result = cv2.solvePnPGeneric(np.asarray(objects, float), np.asarray(pixels, float),
                                k, None, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    if result[0]:
        for rvec, tvec in zip(result[1], result[2]):
            t = np.eye(4)
            t[:3, :3] = cv2.Rodrigues(rvec)[0]
            t[:3, 3] = tvec.ravel()
            if not np.isfinite(t).all():
                continue
            try:
                error = project(t, objects, k) - pixels
            except ValueError:
                continue
            candidates.append((float(np.sqrt(np.mean(np.sum(error**2, axis=1)))), t))
    if not candidates:
        raise ValueError('No finite positive-depth planar PnP solution')
    candidates.sort(key=lambda item: item[0])
    best_error, best = candidates[0]
    ambiguous = False
    if len(candidates) > 1:
        other_error, other = candidates[1]
        ambiguous = other_error-best_error < .05 and angle_deg(best[:3, :3].T @ other[:3, :3]) > 2
    rvec, tvec = cv2.solvePnPRefineLM(np.asarray(objects, float), np.asarray(pixels, float), k, None,
                                   cv2.Rodrigues(best[:3, :3])[0], best[:3, 3].copy())
    refined = np.eye(4)
    refined[:3, :3] = cv2.Rodrigues(rvec)[0]
    refined[:3, 3] = tvec.ravel()
    if np.isfinite(refined).all() and np.all(transform_points(refined, objects)[:, 2] > 0):
        refined_error = np.sqrt(np.mean(np.sum((project(refined, objects, k)-pixels)**2, axis=1)))
        if refined_error <= best_error:
            best = refined
            best_error = refined_error
    # Near frontal views can leave IPPE's two seeds poorly conditioned. Keep
    # the ambiguity flag, but also fit the homography-initialized iterative pose.
    ok, rvec, tvec = cv2.solvePnP(np.asarray(objects, float), np.asarray(pixels, float), k, None,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        iterative = np.eye(4)
        iterative[:3, :3] = cv2.Rodrigues(rvec)[0]
        iterative[:3, 3] = tvec.ravel()
        if np.isfinite(iterative).all() and np.all(transform_points(iterative, objects)[:, 2] > 0):
            error = np.sqrt(np.mean(np.sum((project(iterative, objects, k)-pixels)**2, axis=1)))
            if error < best_error:
                best = iterative
    return best, ambiguous


def analyze_frame(rgb, depth, k, X_base_camera, board, detector, max_reprojection_px=2.0):
    k = np.asarray(k, float)
    detections = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), estimate_tag_pose=False)
    accepted = {}
    seen = set()
    unexpected = []
    for det in detections:
        tag_id = int(det.tag_id)
        if tag_id not in board['corners']:
            unexpected.append(tag_id)
            continue
        if tag_id in seen:
            raise ValueError(f'Duplicate tag ID {tag_id}; use only one board')
        seen.add(tag_id)
        corners = np.asarray(det.corners, float)
        if det.hamming != 0 or det.decision_margin < 20:
            continue
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            continue
        accepted[tag_id] = corners
    ids = sorted(accepted)
    if len(ids) < 4:
        raise ValueError(f'Need >=4 clear {board["family"]} board tags; accepted IDs={ids}, unexpected={unexpected}')
    objects = np.concatenate([board['corners'][i] for i in ids])
    pixels = np.concatenate([accepted[i] for i in ids])
    pose, ambiguous = solve_board_pose(objects, pixels, k)
    predicted = project(pose, objects, k)
    error = np.linalg.norm(predicted-pixels, axis=1)
    # Fit on alternate complete tags; evaluate on tags not used in that fit.
    train, held = ids[::2], ids[1::2]
    train_pose, train_ambiguous = solve_board_pose(
        np.concatenate([board['corners'][i] for i in train]),
        np.concatenate([accepted[i] for i in train]), k)
    held_error = np.linalg.norm(project(train_pose, np.concatenate([board['corners'][i] for i in held]), k)
                                - np.concatenate([accepted[i] for i in held]), axis=1)
    depth_records = []
    normal = pose[:3, 2]
    plane_distance = float(normal @ pose[:3, 3])
    # Sample 9 interior points per tag. These avoid the black marker's outer
    # edge; each actual depth sample is compared to PnP plane at its own ray.
    for tag_id in ids:
        canonical = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], np.float32)
        homography = cv2.getPerspectiveTransform(canonical, accepted[tag_id].astype(np.float32))
        interior = np.array([[[x, y] for x in (-.5, 0, .5) for y in (-.5, 0, .5)]], np.float32)
        locations = cv2.perspectiveTransform(interior, homography)[0]
        for uv in locations:
            u, v = np.rint(uv).astype(int)
            if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
                continue
            z = float(depth[v, u])
            if not np.isfinite(z) or z <= 0:
                continue
            ray = np.linalg.solve(k, [u, v, 1.0])
            denominator = float(normal @ ray)
            if abs(denominator) < 1e-6:
                continue
            expected_z = plane_distance / denominator
            if expected_z <= 0:
                continue
            depth_records.append({'tag_id': tag_id, 'pixel_xy': [int(u), int(v)],
                                  'depth_m': z, 'pnp_depth_m': float(expected_z),
                                  'depth_minus_pnp_mm': float((z-expected_z)*1000),
                                  'depth_over_pnp': float(z/expected_z)})
    base_board = np.asarray(X_base_camera) @ pose
    rms, held_rms = stats(error), stats(held_error)
    good = not ambiguous and not train_ambiguous and max(rms['rms'], held_rms['rms']) <= max_reprojection_px
    return {'status': 'USABLE' if good else 'RGB_POSE_UNRELIABLE', 'tag_ids': ids,
            'unexpected_ids': unexpected, 'planar_pose_ambiguous': ambiguous or train_ambiguous,
            'X_camera_board': pose.tolist(), 'X_base_board': base_board.tolist(),
            'X_base_camera': np.asarray(X_base_camera).tolist(),
            'reprojection_px': rms, 'held_out_reprojection_px': held_rms,
            'held_out_tag_ids': held, 'depth_samples': depth_records,
            'depth_minus_pnp_mm': stats([r['depth_minus_pnp_mm'] for r in depth_records]),
            'median_depth_over_pnp': float(np.median([r['depth_over_pnp'] for r in depth_records])) if depth_records else None,
            'observed_pixels': pixels.tolist(), 'projected_pixels': predicted.tolist()}


def draw_detection(rgb, result):
    image = rgb.copy()
    if 'observed_pixels' in result:
        for i, (actual, fitted) in enumerate(zip(result['observed_pixels'], result['projected_pixels'])):
            a, b = tuple(np.rint(actual).astype(int)), tuple(np.rint(fitted).astype(int))
            cv2.circle(image, a, 4, (0, 255, 0), 1)
            cv2.drawMarker(image, b, (255, 60, 60), cv2.MARKER_CROSS, 9, 1)
            cv2.line(image, a, b, (255, 230, 0), 1)
            if i % 4 == 0:
                cv2.putText(image, str(result['tag_ids'][i//4]), a, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv2.putText(image, result['status'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 150, 0), 2)
    return image


def summarize(records):
    usable = [r for r in records if r['status'] == 'USABLE']
    summary = {'sample_count': len(records), 'usable_count': len(usable),
               'status': 'INSUFFICIENT_DATA', 'absolute_accuracy': 'NOT_VERIFIED',
               'tcp_accuracy': 'NOT_TESTED', 'findings': [], 'drift_samples': []}
    if not usable:
        summary['findings'].append('No reliable board poses. Check detection, focus, board geometry and RGB model first.')
        return summary
    reference = np.asarray(usable[0]['X_base_board'])
    groups = {}
    for record in usable:
        pose = np.asarray(record['X_base_board'])
        delta = (pose[:3, 3]-reference[:3, 3])*1000
        summary['drift_samples'].append({'sample_id': record['sample_id'], 'pose_group': record['pose_group'],
                                         'delta_xyz_mm': delta.tolist(), 'distance_mm': float(np.linalg.norm(delta)),
                                         'rotation_deg': angle_deg(reference[:3, :3].T @ pose[:3, :3])})
        groups.setdefault(record['pose_group'], []).append(record)
    # Same-pose repeatability; not a hand-eye accuracy test.
    summary['repeatability'] = []
    for group, items in groups.items():
        centers = np.array([np.asarray(r['X_base_board'])[:3, 3]*1000 for r in items])
        distances = np.linalg.norm(centers-centers.mean(axis=0), axis=1)
        summary['repeatability'].append({'pose_group': group, 'samples': len(items),
                                         'center_scatter_mm': stats(distances) if len(items) >= 2 else {'count': 0}})
    camera_t, camera_r, board_t, board_r = [], [], [], []
    for index, a in enumerate(usable):
        for b in usable[:index]:
            ca, cb = np.asarray(a['X_base_camera']), np.asarray(b['X_base_camera'])
            ba, bb = np.asarray(a['X_base_board']), np.asarray(b['X_base_board'])
            camera_t.append(float(np.linalg.norm(ca[:3, 3]-cb[:3, 3])*1000))
            camera_r.append(angle_deg(ca[:3, :3].T @ cb[:3, :3]))
            board_t.append(float(np.linalg.norm(ba[:3, 3]-bb[:3, 3])*1000))
            board_r.append(angle_deg(ba[:3, :3].T @ bb[:3, :3]))
    summary['camera_pose_span'] = {'translation_mm': max(camera_t, default=0), 'rotation_deg': max(camera_r, default=0)}
    summary['max_pairwise_board_drift_mm'] = max(board_t, default=0)
    summary['max_pairwise_board_rotation_deg'] = max(board_r, default=0)
    variation = max(camera_t, default=0) >= 10 or max(camera_r, default=0) >= 5
    summary['status'] = 'CONSISTENCY_MEASURED' if len(groups) >= 3 and variation else 'INSUFFICIENT_POSE_VARIATION'
    depth = [d for r in usable for d in r['depth_samples']]
    summary['depth_minus_pnp_mm'] = stats([d['depth_minus_pnp_mm'] for d in depth])
    summary['median_depth_over_pnp'] = float(np.median([d['depth_over_pnp'] for d in depth])) if depth else None
    if any(r['status'] == 'RGB_POSE_UNRELIABLE' for r in records):
        summary['findings'].append('Some RGB poses rejected: inspect reprojection/held-out errors and planar ambiguity.')
    if depth and abs(np.median([d['depth_minus_pnp_mm'] for d in depth])) > 5:
        summary['findings'].append('RGB-board and depth disagree by >5 mm median: check print scale, depth, intrinsics and RGB-D alignment.')
    if summary['status'] == 'CONSISTENCY_MEASURED' and (max(board_t) > 5 or max(board_r) > 2):
        summary['findings'].append('Fixed-board pose varies across views: inspect hand-eye/FK/timing AND board stability/scale/RGB pose quality.')
    if not depth:
        summary['findings'].append('No valid interior depth samples; depth consistency was not tested.')
    if not summary['findings']:
        summary['findings'].append('No large residual flagged in available data; this is not an absolute calibration or TCP certification.')
    return summary


def save_plots(records, summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    drift = summary['drift_samples']
    if drift:
        xyz = np.array([r['delta_xyz_mm'] for r in drift])
        axes[0, 0].scatter(xyz[:, 0], xyz[:, 1])
        for row, (x, y, _) in zip(drift, xyz):
            axes[0, 0].annotate(str(row['sample_id']), (x, y))
        axes[0, 0].set_aspect('equal', adjustable='datalim')
        for i, label in enumerate(('dX', 'dY', 'dZ')):
            axes[0, 1].plot([r['sample_id'] for r in drift], xyz[:, i], '.-', label=label)
        axes[0, 1].legend()
    for field, label in [('reprojection_px', 'fit'), ('held_out_reprojection_px', 'held-out tags')]:
        rows = [r for r in records if field in r]
        axes[1, 0].plot([r['sample_id'] for r in rows], [r[field]['rms'] for r in rows], '.-', label=label)
    axes[1, 0].legend()
    rows = [r for r in records if r.get('depth_minus_pnp_mm', {}).get('count', 0)]
    axes[1, 1].plot([r['sample_id'] for r in rows], [r['depth_minus_pnp_mm']['median'] for r in rows], '.-')
    axes[0, 0].set(xlabel='dX (mm)', ylabel='dY (mm)', title='Fixed board center relative to first usable view')
    axes[0, 1].set(xlabel='Sample ID', ylabel='Drift (mm)', title='Board center drift')
    axes[1, 0].set(xlabel='Sample ID', ylabel='RMS (pixels)', title='RGB pinhole fit / held-out residuals')
    axes[1, 1].set(xlabel='Sample ID', ylabel='Median difference (mm)', title='Depth minus RGB-board depth')
    for axis in axes.flat:
        axis.grid(alpha=.3)
    fig.suptitle('AprilTag diagnostics | relative consistency, not absolute/TCP accuracy')
    fig.savefig(path, dpi=140)
    plt.close(fig)
