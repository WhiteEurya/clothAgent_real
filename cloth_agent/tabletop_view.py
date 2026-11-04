"""Table-plane RGB view with explicit, verified mapping back to measured RGB-D.

Adapted from RobotCamCalib/reconstruct_tabletop.py. Appearance assumes zero
garment height; returned robot points always come from the source depth cloud.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cloth_agent.dual_arm.observation import Observation, sha


def fit_table(points, threshold_mm=5.):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1000 or not np.isfinite(points).all():
        raise ValueError('Too few finite neutral tabletop points (minimum 1000)')
    generator = np.random.default_rng(42)
    sampled = points[generator.choice(len(points), min(len(points), 12000), replace=False)]
    best, best_count = None, 0
    for _ in range(500):
        triple = sampled[generator.choice(len(sampled), 3, replace=False)]
        normal = np.cross(triple[1] - triple[0], triple[2] - triple[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if normal[2] < 0:
            normal = -normal
        if normal[2] < np.cos(np.deg2rad(15)):
            continue
        offset = -normal @ triple[0]
        count = int((np.abs(np.einsum('ij,j->i', sampled, normal) + offset) < threshold_mm).sum())
        if count > best_count:
            best, best_count = (normal, offset), count
    if best is None or best_count < len(sampled) * .4:
        raise ValueError('No reliable near-horizontal tabletop plane in ROI')
    normal, offset = best
    for _ in range(4):
        inliers = points[np.abs(points @ normal + offset) < threshold_mm]
        if len(inliers) < 1000:
            raise ValueError('Table plane refinement has too few inliers')
        center = inliers.mean(axis=0)
        _, _, directions = np.linalg.svd(inliers - center, full_matrices=False)
        normal = directions[-1]
        if normal[2] < 0:
            normal = -normal
        offset = -normal @ center
    residuals = points @ normal + offset
    inliers = np.abs(residuals) < threshold_mm
    if inliers.mean() < .4 or normal[2] < np.cos(np.deg2rad(15)):
        raise ValueError('Refined plane is unreliable')
    return normal, float(offset), inliers, residuals


@dataclass
class TabletopView:
    directory: Path
    meta: dict
    source_xy: np.ndarray
    mask: np.ndarray

    @property
    def image(self):
        return self.directory / 'rgb.png'

    @classmethod
    def load(cls, directory, observation):
        directory = Path(directory)
        meta = json.loads((directory / 'view.json').read_text())
        if (meta['observation_id'] != observation.meta['observation_id']
                or meta['source_sha256'] != observation.meta['sha256']
                or meta['source_metadata_sha256'] != sha(observation.directory / 'observation.json')):
            raise ValueError('Top-down view belongs to a different observation/geometry')
        for name, expected in meta['sha256'].items():
            if sha(directory / name) != expected:
                raise ValueError('Top-down artifact changed: ' + name)
        lookup = np.load(directory / 'source_xy.npy', allow_pickle=False)
        mask = np.asarray(Image.open(directory / 'mask.png')) > 0
        if lookup.shape != (*mask.shape, 2):
            raise ValueError('Invalid top-down mapping dimensions')
        return cls(directory, meta, lookup, mask)

    def source_pixel(self, pixel):
        if not isinstance(pixel, list) or len(pixel) != 2 or any(type(v) is not int for v in pixel):
            raise ValueError('Expected integer top-down [x, y] pixel')
        x, y = pixel
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h) or not self.mask[y, x]:
            raise ValueError('Top-down anchor outside visible garment mask')
        source = self.source_xy[y, x]
        if not np.isfinite(source).all():
            raise ValueError('Top-down anchor has no source correspondence')
        return np.rint(source).astype(int).tolist()


def reconstruct(observation: Observation, output, config):
    """Save flat appearance warp, source lookup and measured source-height map.

    No depth interpolation/filling, neural completion or robot connection.
    """
    rgb = np.asarray(Image.open(observation.image).convert('RGB'))
    h, w = rgb.shape[:2]
    if (config.get('version') != 1 or config.get('garment_mask') != 'red_hsv_largest_component'
            or config.get('camera_serial') != observation.meta.get('camera_serial')
            or config.get('image_size_wh') != [w, h]):
        raise ValueError('Top-down configuration camera/size/mask mismatch')
    resolution = config['resolution_mm']
    if isinstance(resolution, bool) or not np.isfinite(resolution) or not .25 <= resolution <= 5:
        raise ValueError('resolution_mm must be within [0.25, 5]')
    polygon = np.asarray(config['table_polygon_px'])
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.issubdtype(polygon.dtype, np.integer)
            or np.any(polygon < 0) or np.any(polygon >= [w, h])):
        raise ValueError('Invalid table polygon')
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [polygon.astype(np.int32)], 1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    valid = np.isfinite(observation.xyz).all(axis=-1)
    table = valid & (region > 0) & (hsv[..., 1] < 65) & (hsv[..., 2] > 90)
    neutral_candidates = int(table.sum())
    # Fit on explicitly selected visible tabletop patches near printed markers.
    # Untextured stereo regions can report finite but incorrect depths far below
    # the tabletop. Keep the existing RANSAC thresholds, and record both counts.
    if 'plane_fit_polygons_px' in config:
        polygons = config['plane_fit_polygons_px']
        if not isinstance(polygons, list) or not polygons:
            raise ValueError('Provide nonempty plane fit polygons')
        fit_region = np.zeros((h, w), np.uint8)
        for entry in polygons:
            patch = np.asarray(entry)
            if (patch.ndim != 2 or patch.shape[1] != 2 or len(patch) < 3
                    or not np.issubdtype(patch.dtype, np.integer)
                    or np.any(patch < 0) or np.any(patch >= [w, h])):
                raise ValueError('Invalid plane fit polygon')
            cv2.fillPoly(fit_region, [patch.astype(np.int32)], 1)
        table &= fit_region > 0
    normal, offset, inliers, residuals = fit_table(observation.xyz[table])
    red = ((region > 0) & ((hsv[..., 0] < 15) | (hsv[..., 0] > 170))
           & (hsv[..., 1] > 100) & (hsv[..., 2] > 45)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    if count < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 1000:
        raise ValueError('No sufficiently large red garment in table ROI')
    garment = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    heights = observation.xyz @ normal + offset
    retained = garment & valid & (heights >= -5) & (heights <= 100)
    if retained.sum() < 1000:
        raise ValueError('Too few measured garment points near table')
    K = np.asarray(observation.meta['intrinsics'], dtype=float)
    T = np.asarray(observation.meta['world_from_camera_mm'], dtype=float)
    if (K.shape != (3, 3) or T.shape != (4, 4) or not np.isfinite(K).all()
            or not np.isfinite(T).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or not np.allclose(T[3], [0, 0, 0, 1])):
        raise ValueError('Invalid camera geometry')
    if observation.meta.get('support_height_patch') or observation.meta.get('world_tilt_correction'):
        raise ValueError('Top-down requires unmodified rigid camera geometry')
    contours, _ = cv2.findContours(garment.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pixels = np.concatenate([c[:, 0] for c in contours])
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T @ T[:3, :3].T
    denominator = rays @ normal
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError('Camera rays parallel to table')
    distance = -(T[:3, 3] @ normal + offset) / denominator
    if np.any(distance <= 0):
        raise ValueError('Table lies behind camera')
    origin = -offset * normal
    horizontal = np.array([1., 0., 0.]) - normal[0] * normal
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    surface = T[:3, 3] + distance[:, None] * rays - origin
    uv = np.column_stack((surface @ horizontal, surface @ vertical))
    lower, upper = uv.min(axis=0) - 25, uv.max(axis=0) + 25
    width, height = np.ceil((upper - lower) / resolution).astype(int)
    if width * height > 4_000_000:
        raise ValueError('Top-down raster exceeds 4 million pixels')
    rows, columns = np.indices((height, width))
    points = (origin + (lower[0] + (columns[..., None] + .5) * resolution) * horizontal
              + (upper[1] - (rows[..., None] + .5) * resolution) * vertical)
    camera_points = (points - T[:3, 3]) @ T[:3, :3]
    projected = camera_points @ K.T
    lookup = (projected[..., :2] / projected[..., 2:]).astype(np.float32)
    mx, my = lookup[..., 0], lookup[..., 1]
    alpha = cv2.remap(garment.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    weighted = cv2.remap(rgb.astype(np.float32) * garment[..., None], mx, my, cv2.INTER_LINEAR)
    rgb_view = np.clip(np.rint(weighted + 235 * (1 - alpha[..., None])), 0, 255).astype(np.uint8)
    mask = (alpha >= .99) & (camera_points[..., 2] > 0)
    # Require the actual nearest source pixel to belong to the garment too.
    source_garment = cv2.remap(garment.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0
    mask &= source_garment
    measured_height = cv2.remap(np.where(retained, heights, np.nan).astype(np.float32),
        mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))
    measured_height[~mask] = np.nan
    safe = cv2.erode(region, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    clipped = bool(np.any(garment & (safe == 0)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb_view).save(output / 'rgb.png')
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'mask.png')
    Image.fromarray(garment.astype(np.uint8) * 255).save(output / 'source_garment_mask.png')
    fit_overlay = rgb.copy()
    fit_overlay[table] = [255, 0, 255]
    Image.fromarray(fit_overlay).save(output / 'table_fit_candidates.png')
    np.save(output / 'source_xy.npy', lookup, allow_pickle=False)
    np.save(output / 'source_height_mm.npy', measured_height, allow_pickle=False)
    meta = dict(version=1, view='table_plane_rgb', execution_allowed=False,
        observation_id=observation.meta['observation_id'], source_sha256=observation.meta['sha256'],
        source_metadata_sha256=sha(observation.directory / 'observation.json'),
        common_frame=observation.meta['common_frame'], config=config,
        table_normal_common=normal.tolist(), table_offset_mm=offset,
        neutral_table_candidates=neutral_candidates, selected_table_candidates=int(table.sum()),
        table_inlier_fraction=float(inliers.mean()), table_rms_mm=float(np.sqrt(np.mean(residuals[inliers]**2))),
        origin_common_mm=origin.tolist(), horizontal_common=horizontal.tolist(), vertical_common=vertical.tolist(),
        raster_left_top_mm=[float(lower[0]), float(upper[1])], resolution_mm=resolution,
        raster_size_wh=[int(width), int(height)], garment_clipped=clipped,
        garment_measured_depth_fraction=float(valid[garment].mean()),
        limitations=['Red garment on neutral tabletop only; fixed calibrated camera and configured ROI.',
            'Appearance projects onto the table plane; raised folds have parallax distortion.',
            'No recovery of cropped or occluded cloth; clipping is reported, not completed.',
            'Source height array is source depth remapped into appearance coordinates, not an orthographic height raster.',
            'Anchors map to source pixels and use measured depth; raster pitch is not physical accuracy.'])
    meta['sha256'] = {name: sha(output / name) for name in
        ('rgb.png', 'mask.png', 'source_garment_mask.png', 'source_xy.npy', 'source_height_mm.npy', 'table_fit_candidates.png')}
    (output / 'view.json').write_text(json.dumps(meta, indent=2) + '\n')
    return TabletopView.load(output, observation)
