#!/usr/bin/env python3
"""Offline RGB-D calibration audit. Never opens a camera or commands a robot.

See docs/camera_calibration_check.md for independent-target measurements.
Pixels are in the original RGB, not rotated display images. Distances are mm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite_vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite numbers')
    return result


def load_capture(path, camera='A'):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= 'result.json'
    result = read_json(path)
    view = next((v for v in result['views'] if v['label'] == camera), None)
    if view is None:
        raise ValueError(f'Camera {camera} absent in {path}')
    root = path.parent
    rgb = Image.open(root / view['image']).convert('RGB')
    depth = np.load(root / view['depth_m'], allow_pickle=False)
    if depth.shape != (rgb.height, rgb.width):
        raise ValueError('RGB and aligned depth dimensions differ')
    k, transform = np.asarray(view['intrinsics'], float), np.asarray(view['X_base_camera'], float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8
            or not 0 <= k[0, 2] < rgb.width or not 0 <= k[1, 2] < rgb.height):
        raise ValueError('Invalid pinhole intrinsics for this RGB resolution')
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError('Invalid X_base_camera homogeneous transform')
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError('X_base_camera rotation is not orthonormal with determinant +1')
    offset = float(view.get('base_z_offset_mm', 0))
    if not np.isfinite(offset):
        raise ValueError('Invalid base_z_offset_mm')
    xyz_path = root / view.get('base_xyz_map', f'camera_{camera}_base_xyz_mm.npy')
    xyz = np.load(xyz_path, allow_pickle=False) if xyz_path.is_file() else None
    if xyz is not None and xyz.shape != (*depth.shape, 3):
        raise ValueError('Saved XYZ map dimensions differ from RGB/depth')
    return dict(path=path, root=root, view=view, rgb=rgb, depth=depth, k=k,
                transform=transform, offset=offset, xyz=xyz)


def deproject(k, uv, depth_m):
    uv = np.asarray(uv, float)
    z = np.asarray(depth_m, float)
    return np.stack(((uv[..., 0] - k[0, 2]) * z / k[0, 0],
                     (uv[..., 1] - k[1, 2]) * z / k[1, 1], z), axis=-1) * 1000


def to_base(capture, camera_mm):
    t = capture['transform']
    return np.asarray(camera_mm) @ t[:3, :3].T + t[:3, 3] * 1000


def stats(errors):
    values = np.asarray(errors, float)
    if values.size == 0:
        return {'count': 0}
    return {'count': int(values.size), 'rms': float(np.sqrt(np.mean(values ** 2))),
            'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values))}


def measure_point(capture, pixel, radius=3):
    uv = finite_vector(pixel, 2, 'raw pixel_xy')
    x, y = np.floor(uv + .5).astype(int)
    depth = capture['depth']
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        raise ValueError(f'Pixel {pixel} outside raw RGB')
    patch = depth[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    z = float(depth[y, x])
    result = {'pixel_xy': uv.tolist(), 'depth_sample_pixel_xy': [int(x), int(y)],
              'valid_patch_fraction': float(len(valid) / patch.size),
              'patch_depth_p10_p90_mm': (np.percentile(valid, [10, 90]) * 1000).tolist()
                  if len(valid) else None}
    if not np.isfinite(z) or z <= 0:
        return {**result, 'status': 'INVALID_DEPTH'}
    base = to_base(capture, deproject(capture['k'], uv, z))
    pipeline = base + [0, 0, capture['offset']]
    result.update(status='MEASURED', depth_m=z, base_xyz_mm=base.tolist(),
                  pipeline_xyz_mm=pipeline.tolist())
    if capture['xyz'] is not None:
        saved = capture['xyz'][y, x].astype(float)
        if np.isfinite(saved).all():
            result['saved_xyz_mm'] = saved.tolist()
            # Map uses the exact integer pixel, even for subpixel target inputs.
            expected = to_base(capture, deproject(capture['k'], [x, y], z)) + [0, 0, capture['offset']]
            result['map_error_mm'] = float(np.linalg.norm(saved - expected))
    return result


def audit_map(capture, tolerance_mm):
    xyz = capture['xyz']
    if xyz is None:
        return {'status': 'NOT_AVAILABLE'}
    depth = capture['depth']
    mask = np.isfinite(depth) & (depth > 0) & np.isfinite(xyz).all(axis=2)
    yy, xx = np.nonzero(mask)
    stride = max(1, (len(xx) + 19999) // 20000)
    yy, xx = yy[::stride], xx[::stride]
    if not len(xx):
        return {'status': 'INSUFFICIENT_DATA'}
    expected = to_base(capture, deproject(capture['k'], np.column_stack((xx, yy)), depth[yy, xx]))
    expected[:, 2] += capture['offset']
    errors = stats(np.linalg.norm(expected - xyz[yy, xx], axis=1))
    return {'status': 'CONSISTENT' if errors['max'] <= tolerance_mm else 'MISMATCH',
            'error_mm': errors, 'tolerance_mm': tolerance_mm,
            'note': 'Numerical consistency only; a wrong calibration can also be consistent.'}


def checkerboard(capture, columns, rows, square_mm, tolerance_px, tolerance_mm=5):
    import cv2
    gray = np.asarray(capture['rgb'].convert('L'))
    found, corners = cv2.findChessboardCornersSB(gray, (columns, rows))
    if not found:
        return {'status': 'NOT_DETECTED'}
    uv = corners.reshape(-1, 2).astype(float)
    points = np.zeros((rows * columns, 3), float)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_mm
    # Fit only alternate corners and score independent held-out corners.
    train = np.arange(len(points)) % 2 == 0
    ok, rvec, tvec = cv2.solvePnP(points[train], uv[train], capture['k'], None,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {'status': 'POSE_FIT_FAILED'}
    projected = cv2.projectPoints(points, rvec, tvec, capture['k'], None)[0].reshape(-1, 2)
    residuals = stats(np.linalg.norm(projected[~train] - uv[~train], axis=1))
    camera_points = []
    for pixel in uv:
        measurement = measure_point(capture, pixel)
        camera_points.append(deproject(capture['k'], pixel, measurement['depth_m'])
                             if measurement['status'] == 'MEASURED' else [np.nan] * 3)
    grid = np.asarray(camera_points).reshape(rows, columns, 3)
    lengths = np.concatenate((np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                              np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()))
    lengths = lengths[np.isfinite(lengths)]
    edge_errors = stats(abs(lengths-square_mm))
    return {'status': 'WITHIN_TOLERANCE' if residuals['max'] <= tolerance_px else 'HIGH_RESIDUAL',
            'held_out_reprojection_error_px': residuals, 'tolerance_px': tolerance_px,
            'square_size_mm': square_mm, 'depth_edge_length_error_mm': edge_errors,
            'depth_scale_status': ('INSUFFICIENT_DATA' if not len(lengths) else
                'WITHIN_TOLERANCE' if edge_errors['max'] <= tolerance_mm else 'OUT_OF_TOLERANCE'),
            'depth_edge_tolerance_mm': tolerance_mm,
            'corners_raw_xy': uv.tolist(),
            'note': 'Tests the pipeline pinhole model (no distortion correction). Pose fitting can absorb '
                    'intrinsic error; use several unseen board tilts/distances. Edge length also depends on depth.'}


def fixed_point_report(observations, tolerance_mm):
    measured = [row for row in observations if row['measurement']['status'] == 'MEASURED']
    groups = {}
    absolute = []
    for row in measured:
        groups.setdefault(row['point_id'], []).append(row)
        if 'known_base_xyz_mm' in row:
            known = finite_vector(row['known_base_xyz_mm'], 3, 'known_base_xyz_mm')
            for key in ('base_xyz_mm', 'pipeline_xyz_mm'):
                row[key + '_error_vector_mm'] = (np.asarray(row['measurement'][key]) - known).tolist()
            row['absolute_error_mm'] = float(np.linalg.norm(row['pipeline_xyz_mm_error_vector_mm']))
            absolute.append(row['absolute_error_mm'])
    drift = []
    for point_id, group in groups.items():
        poses = [np.asarray(row['X_base_camera']) for row in group]
        translations = [np.linalg.norm(a[:3, 3]-b[:3, 3])*1000
                        for i, a in enumerate(poses) for b in poses[i+1:]]
        rotations = [np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3])-1)/2, -1, 1)))
                     for i, a in enumerate(poses) for b in poses[i+1:]]
        diverse = max(translations, default=0) >= 10 or max(rotations, default=0) >= 5
        entry = {'point_id': point_id, 'count': len(group), 'distinct_captures': len({r['result_json'] for r in group}),
                 'max_camera_translation_mm': max(translations, default=0),
                 'max_camera_rotation_deg': max(rotations, default=0)}
        points = np.asarray([row['measurement']['pipeline_xyz_mm'] for row in group])
        maximum = max((float(np.linalg.norm(a-b)) for i, a in enumerate(points) for b in points[i+1:]), default=0)
        entry.update(max_pairwise_drift_mm=maximum,
                     status=('INSUFFICIENT_POSE_VARIATION' if not diverse or entry['distinct_captures'] < 2 else
                             'WITHIN_TOLERANCE' if maximum <= tolerance_mm else 'DRIFT_DETECTED'))
        drift.append(entry)
    return {'fixed_point_drift': drift,
            'absolute_accuracy': {'status': ('NOT_VERIFIED' if not absolute else
                'WITHIN_TOLERANCE_AT_TESTED_POINTS' if max(absolute) <= tolerance_mm else 'OUT_OF_TOLERANCE'),
                'error_mm': stats(absolute), 'tolerance_mm': tolerance_mm},
            'invalid_observations': len(observations) - len(measured),
            'note': 'Known XYZ must be independently measured, never copied from this camera. '
                    'Combined RGB-D/hand-eye/kinematic accuracy does not isolate extrinsics or validate gripper TCP.'}


def iteration_capture(iteration, camera):
    if camera != 'A':
        raise ValueError('--iteration resolves Camera A only; use --perception for other cameras')
    mappings = sorted((Path(iteration) / 'before_raw').glob('camera_A_upright_mapping.json'))
    if len(mappings) != 1:
        raise ValueError('Need before_raw/camera_A_upright_mapping.json; use --perception for relocated/retry captures')
    mapping = read_json(mappings[0])
    path = Path(mapping['coordinate_guide']).parent / 'result.json'
    if not path.is_file():
        raise ValueError(f'Original perception path unavailable: {path}; use --perception explicitly')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perception', type=Path, action='append', default=[], help='Exact result.json or its directory; repeatable')
    parser.add_argument('--iteration', type=Path, help='Resolve the BEFORE capture; never use latest workspace files')
    parser.add_argument('--camera', default='A')
    parser.add_argument('--reference', action='append', default=[], help='Saved reference ID, e.g. R081')
    parser.add_argument('--pixel', nargs=2, type=int, action='append', default=[], metavar=('U', 'V'))
    parser.add_argument('--observations', type=Path, help='JSON list of fixed point observations; see documentation')
    parser.add_argument('--checkerboard', nargs=2, type=int, metavar=('COLS', 'ROWS'), help='Inner corner counts')
    parser.add_argument('--square-mm', type=float, help='Measured checkerboard square size')
    parser.add_argument('--tolerance-mm', type=float, default=5, help='Physical error limit (default 5 mm; choose for your task)')
    parser.add_argument('--tolerance-px', type=float, default=1, help='Held-out checkerboard residual limit')
    parser.add_argument('--map-tolerance-mm', type=float, default=.05)
    parser.add_argument('--output-dir', type=Path, default=Path('results/calibration_checks'))
    args = parser.parse_args(argv)
    for name in ('tolerance_mm', 'tolerance_px', 'map_tolerance_mm'):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be positive and finite')
    if args.checkerboard and (min(args.checkerboard) < 3 or args.square_mm is None
                              or not np.isfinite(args.square_mm) or args.square_mm <= 0):
        parser.error('Checkerboard needs at least 3x3 inner corners and positive --square-mm')
    if not args.perception and not args.iteration and not args.observations:
        parser.error('Supply --perception, --iteration or --observations')
    output = args.output_dir.expanduser().resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'status': 'DIAGNOSTIC_ONLY', 'captures': [], 'observations': [], 'errors': [],
              'limitations': ['Saved K/X validity and reprojection do not establish real-world accuracy.',
                  'Raw images/depth are temporal aggregates; exact capture-time joints are not saved by old runs.',
                  'No robot/camera connection, calibration changes, or motion are performed.']}
    cache = {}
    def capture(path):
        path = Path(path).resolve()
        if path.is_dir():
            path /= 'result.json'
        if path not in cache:
            cache[path] = load_capture(path, args.camera)
        return cache[path]
    try:
        paths = list(args.perception)
        if args.iteration:
            paths.append(iteration_capture(args.iteration, args.camera))
        for path in paths:
            capture(path)
        if args.observations:
            inputs = read_json(args.observations)
            if not isinstance(inputs, list):
                raise ValueError('Observations JSON must be a list')
            for row in inputs:
                if not isinstance(row.get('point_id'), str) or not row['point_id'].strip():
                    raise ValueError('Each observation needs a non-empty physical point_id')
                path = Path(row['result_json'])
                if not path.is_absolute():
                    path = args.observations.resolve().parent / path
                c = capture(path)
                item = {'point_id': row['point_id'], 'result_json': str(c['path']),
                        'X_base_camera': c['transform'].tolist(), 'measurement': measure_point(c, row['pixel_xy'])}
                if 'known_base_xyz_mm' in row:
                    item['known_base_xyz_mm'] = finite_vector(row['known_base_xyz_mm'], 3, 'known_base_xyz_mm').tolist()
                report['observations'].append(item)
        for index, c in enumerate(cache.values()):
            entry = {'result_json': str(c['path']), 'camera': args.camera,
                     'result_sha256': hashlib.sha256(c['path'].read_bytes()).hexdigest(),
                     'intrinsics': c['k'].tolist(), 'X_base_camera': c['transform'].tolist(),
                     'base_z_offset_mm': c['offset'], 'matrix_validation': 'VALID',
                     'map_consistency': audit_map(c, args.map_tolerance_mm), 'points': []}
            pixels = [(f'pixel_{i}', pixel, None) for i, pixel in enumerate(args.pixel)]
            if args.reference:
                guide = read_json(c['root'] / c['view'].get('coordinate_guide', f'camera_{args.camera}_coordinate_guide.json'))
                for ref in args.reference:
                    sample = next((r for r in guide['samples'] if r['reference_id'] == ref), None)
                    if sample is None:
                        raise ValueError(f'{ref} not present in {c["path"]}')
                    pixels.append((ref, sample['pixel_xy'], sample['base_xyz_mm']))
            for row in report['observations']:
                if row['result_json'] == str(c['path']):
                    pixels.append((row['point_id'], row['measurement']['pixel_xy'], None))
            overlay = c['rgb'].copy()
            draw = ImageDraw.Draw(overlay)
            for name, pixel, saved in pixels:
                measurement = measure_point(c, pixel)
                if saved is not None and measurement['status'] == 'MEASURED':
                    measurement['reference_error_mm'] = float(np.linalg.norm(
                        finite_vector(saved, 3, 'reference XYZ') - measurement['pipeline_xyz_mm']))
                entry['points'].append({'name': name, **measurement})
                x, y = pixel
                draw.ellipse((x-6, y-6, x+6, y+6), outline='red', width=2)
                draw.text((x+8, y), name, fill='red', stroke_width=1, stroke_fill='white')
            if args.checkerboard:
                entry['checkerboard'] = checkerboard(c, *args.checkerboard, args.square_mm,
                                                    args.tolerance_px, args.tolerance_mm)
                for x, y in entry['checkerboard'].get('corners_raw_xy', []):
                    draw.ellipse((x-2, y-2, x+2, y+2), fill='lime')
            overlay.save(output / f'{index:03d}_raw_rgb_points.png')
            report['captures'].append(entry)
        report.update(fixed_point_report(report['observations'], args.tolerance_mm))
        failed = (report['absolute_accuracy']['status'] == 'OUT_OF_TOLERANCE'
                  or any(g['status'] == 'DRIFT_DETECTED' for g in report['fixed_point_drift'])
                  or any(c['map_consistency']['status'] == 'MISMATCH' or
                         c.get('checkerboard', {}).get('status') == 'HIGH_RESIDUAL' or
                         c.get('checkerboard', {}).get('depth_scale_status') == 'OUT_OF_TOLERANCE' or
                         any(p.get('reference_error_mm', 0) > args.map_tolerance_mm for p in c['points'])
                         for c in report['captures']))
        report['status'] = 'ISSUES_DETECTED' if failed else 'DIAGNOSTIC_COMPLETE'
    except Exception as exc:
        report['status'] = 'INPUT_ERROR'
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = [f"Status: {report['status']}", f"Absolute accuracy: {report.get('absolute_accuracy', {}).get('status', 'NOT_VERIFIED')}",
             'All error units are stated in report.json. No overall calibration certification is inferred.', '']
    for c in report['captures']:
        lines.append(f"{c['result_json']}: map={c['map_consistency']['status']}")
        if 'checkerboard' in c:
            board = c['checkerboard']
            lines.append(f"  Board reprojection: {board['status']}; depth scale: {board.get('depth_scale_status', 'NOT_VERIFIED')}")
        for point in c['points']:
            lines.append(f"  {point['name']}: {point['status']}, reference error mm={point.get('reference_error_mm', 'N/A')}")
    for g in report.get('fixed_point_drift', []):
        lines.append(f"Fixed point {g['point_id']}: {g['status']}, max drift={g['max_pairwise_drift_mm']:.3f} mm")
    lines.extend(report['errors'])
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:3]))
    print(f'Report: {output / "report.json"}')
    return 1 if report['status'] == 'INPUT_ERROR' else 2 if report['status'] == 'ISSUES_DETECTED' else 0


if __name__ == '__main__':
    sys.exit(main())
