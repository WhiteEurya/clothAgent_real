"""One opposite-edge pull along a visible silhouette normal.

The caller supplies the invisible-region hypothesis. CV never labels a hidden
hem corner or reconstructs a hidden contour. No robot is connected by planning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


class NormalGraspError(ValueError):
    pass


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def ray_intersections(contour, origin, direction):
    """Ordered exact intersections; a shared polygon vertex counts once."""
    hits = []
    for a, b in zip(contour, np.roll(contour, -1, axis=0)):
        edge = b - a
        denominator = _cross(direction, edge)
        if abs(denominator) < 1e-8:
            continue
        t = _cross(a - origin, edge) / denominator
        u = _cross(a - origin, direction) / denominator
        if t > 1.0 and -1e-7 <= u <= 1 + 1e-7:
            hits.append(t)
    unique = []
    for t in sorted(hits):
        if not unique or t - unique[-1] > .1:
            unique.append(t)
    return unique


def geometry(mask, region_point, *, tangent_window_px=25., inset_px=8.,
             pull_px=80., max_snap_px=80.):
    """Normal points from the selected side into cloth, then OUT past the far edge."""
    mask = np.asarray(mask)
    point = np.asarray(region_point, float)
    values = [tangent_window_px, inset_px, pull_px, max_snap_px]
    if mask.ndim != 2 or not np.isfinite(mask).all() or point.shape != (2,) or not np.isfinite(point).all():
        raise NormalGraspError('invalid mask or region point')
    if not np.isfinite(values).all() or min(values) <= 0 or inset_px < 3:
        raise NormalGraspError('positive finite geometry parameters and inset >=3 px required')
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    if not (0 <= point[0] < w and 0 <= point[1] < h):
        raise NormalGraspError('region point outside image')
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count != 2 or stats[1, cv2.CC_STAT_AREA] < 100:
        raise NormalGraspError('require one connected visible garment component')
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise NormalGraspError('garment contour clipped by image boundary')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0][:, 0].astype(float)
    idx = int(np.argmin(np.linalg.norm(contour - point, axis=1)))
    origin = contour[idx]
    if np.linalg.norm(origin - point) > max_snap_px:
        raise NormalGraspError('selected region is too far from a visible contour')
    # Arc-local PCA suppresses pixel staircases without bridging remote folds.
    local = [origin]
    for sign in (-1, 1):
        traveled = 0.
        previous = origin
        for offset in range(1, len(contour)//2):
            q = contour[(idx + sign * offset) % len(contour)]
            traveled += np.linalg.norm(q - previous)
            if traveled > tangent_window_px:
                break
            local.append(q)
            previous = q
    if len(local) < 7:
        raise NormalGraspError('insufficient local contour support')
    eigenvalues, vectors = np.linalg.eigh(np.cov(np.asarray(local).T))
    ratio = float(eigenvalues[0] / max(eigenvalues[1], 1e-9))
    if ratio > .2:
        raise NormalGraspError('corner/curvature makes the local normal ambiguous')
    tangent = vectors[:, -1]
    normal = np.array([-tangent[1], tangent[0]])
    polygon = contour.astype(np.float32).reshape(-1, 1, 2)
    inside = lambda p: cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) > 0
    for distance in (3., 5., 10.):
        plus, minus = inside(origin + distance*normal), inside(origin - distance*normal)
        if plus != minus:
            if minus:
                normal = -normal
            break
    else:
        raise NormalGraspError('cannot distinguish inward and outward normal')
    hits = ray_intersections(contour, origin, normal)
    if len(hits) != 1:
        raise NormalGraspError(f'ambiguous silhouette crossing: {len(hits)} forward intersections')
    chord = hits[0]
    if chord <= 2*inset_px + 6:
        raise NormalGraspError('opposite-edge chord too short for an interior grasp')
    intersection = origin + chord*normal
    grasp = np.rint(intersection - inset_px*normal).astype(int)
    target = np.rint(grasp + pull_px*normal).astype(int)
    for p in (grasp, target):
        if not (3 <= p[0] < w-3 and 3 <= p[1] < h-3):
            raise NormalGraspError('grasp/target outside supported image interior')
    # Test actual material membership, not a hole-filled silhouette.
    x, y = grasp
    if not mask[y-2:y+3, x-2:x+3].all():
        raise NormalGraspError('grasp patch is not fully on visible segmented material')
    if inside(target) or mask[target[1], target[0]]:
        raise NormalGraspError('pull distance does not place target beyond the opposite edge')
    bg = cv2.distanceTransform(1-mask, cv2.DIST_L2, 5)
    if bg[target[1], target[0]] < 5:
        raise NormalGraspError('target lacks clearance from garment')
    return dict(region_point_px=point.tolist(), origin_px=origin.tolist(),
                tangent_unit_xy=tangent.tolist(), normal_unit_xy=normal.tolist(),
                intersection_px=intersection.tolist(), grasp_px=grasp.tolist(),
                target_px=target.tolist(), chord_px=chord, inset_px=inset_px,
                requested_pull_px=pull_px, tangent_variance_ratio=ratio,
                contour_px=contour.astype(int).tolist(),
                direction='selected side -> opposite edge -> outward')


def build(image_path, region_point, region_name, evidence, folder, *, mask_path=None, **options):
    """Produce a traceable, image-bound candidate, never a dispatch authorization."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    if not region_name.strip() or not evidence.strip():
        raise NormalGraspError('region identity and visible evidence are required')
    rgb = np.asarray(Image.open(image_path).convert('RGB'))
    if mask_path is None:
        from cloth_agent.cloth_harness.cv_harness.segmentation import segment
        mask, segmentation = segment(rgb)
        if segmentation['status'] != 'HEURISTIC_OK':
            save(folder/'segmentation.json', segmentation)
            raise NormalGraspError('segmentation unavailable: '+str(segmentation['reasons']))
    else:
        mask = np.asarray(Image.open(mask_path).convert('L')) > 0
        segmentation = dict(status='PROVIDED_MASK', source=str(Path(mask_path).resolve()),
                            semantic_identity_verified=False)
    if mask.shape != rgb.shape[:2]:
        raise NormalGraspError('mask and image dimensions differ')
    Image.fromarray(mask.astype(np.uint8)*255).save(folder/'mask.png')
    save(folder/'segmentation.json', segmentation)
    g = geometry(mask, region_point, **options)
    plan = dict(decision='ACTION', grasp_px=list(g['grasp_px']), target_px=list(g['target_px']),
                motion_type='lift_pull_place', grasp_angle_deg=None, is_smoothing=False,
                action_purpose=f'沿轮廓法线拉动对侧可见布料，尝试间接展开{region_name}。',
                expected_stop_state=f'{region_name}的内收或遮挡减少。',
                short_reason='不可见区域侧轮廓的向内法线与对侧边缘相交；抓交点内侧并继续向外拉。')
    proposal = dict(status='OFFLINE_CANDIDATE', observation_image_sha256=digest(image_path),
                    source_image=str(image_path.resolve()), region_name=region_name,
                    visibility='USER_OR_AGENT_HYPOTHESIS', evidence=evidence,
                    geometry=g, plan=plan, host_validated=False, executed=False,
                    options=options, mask_sha256=digest(folder/'mask.png'))
    save(folder/'proposal.json', proposal)
    save(folder/'plan.json', plan)
    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    contour = [tuple(p) for p in g['contour_px']]
    draw.line(contour+[contour[0]], fill='#73f5c0', width=2)
    q, hit, pa, pb = [tuple(g[k]) for k in ('origin_px','intersection_px','grasp_px','target_px')]
    draw.line([q, hit], fill='#f6cf65', width=3)
    draw.line([pa, pb], fill='#ff586d', width=4)
    n = np.array(g['normal_unit_xy']);t = np.array([-n[1],n[0]])
    draw.polygon([pb, tuple(np.array(pb)-12*n+6*t), tuple(np.array(pb)-12*n-6*t)], fill='#ff586d')
    for p, label, color in [(q,'Q','#f6cf65'),(hit,'intersection','#73f5c0'),(pa,'Pa','#ff586d'),(pb,'Pb','#49d9ef')]:
        x,y=p; draw.ellipse((x-5,y-5,x+5,y+5),fill=color,outline='white')
        draw.text((x+8,y-18),label,fill=color,stroke_width=1,stroke_fill='black')
    draw.rectangle((0,0,overlay.width,24),fill='#111827')
    draw.text((8,5),'CONTOUR NORMAL | PREVIEW ONLY | Q -> opposite edge -> outward',fill='white')
    overlay.save(folder/'action.png')
    return proposal


def execute_once(runtime, before, proposal, folder, *, check_grasp_height_spread=True):
    """Use an already authorized runtime; fresh scene + all existing Host checks.

    A persisted dispatch marker prevents a second grasp from this output folder.
    This function never opens a controller connection or retries an action.
    """
    from cloth_agent.dual_arm.geometry import DualArmError
    from cloth_agent.preference_one_shot import verify_scene
    from cloth_agent.experiments.trajectory_memory_full_episode_v1 import prepare_action
    from cloth_agent.one_shot_orientation import resolve
    from cloth_agent.table_target_depth import project_target
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'dispatch_intent.json').exists():
        raise DualArmError('this action has already been dispatched; no retry')
    if proposal.get('status') != 'OFFLINE_CANDIDATE' or proposal.get('observation_image_sha256') != digest(before.image):
        raise DualArmError('proposal does not belong to this observation')
    before.validate_for(runtime.config, live=True, check_age=False)
    plan = proposal['plan']
    # Validate the schema before inspecting pixels or acquiring new observations.
    from cloth_agent.restricted_context_one_shot import validate
    validate(plan, Image.open(before.image).size, False)
    if plan['decision'] != 'ACTION' or plan['action_purpose'].startswith('HEM_REVEAL:'):
        raise DualArmError('expected a single ordinary contour-normal action')
    for key in ('grasp_px', 'target_px'):
        if plan[key] != proposal['geometry'][key]:
            raise DualArmError('plan coordinates changed after normal computation')
    delta = np.array(plan['target_px']) - plan['grasp_px']
    normal = np.array(proposal['geometry']['normal_unit_xy'])
    if normal.shape != (2,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1):
        raise DualArmError('invalid unit normal')
    if np.dot(delta, normal) <= 0 or abs(_cross(delta, normal)) > 1.5:
        raise DualArmError('action no longer follows the outward normal')
    fresh = runtime.capture_current(folder/'pre_execution_observation')
    verify_scene(before, fresh, plan, folder, automatic=True)
    operations, initial = prepare_action(runtime, fresh, plan, folder,
        target_sampler=project_target, orientation_resolver=resolve,
        check_grasp_height_spread=check_grasp_height_spread, smoothing=False)
    runtime.arrival(initial)
    with (folder/'dispatch_intent.json').open('x') as f:
        json.dump(dict(one_manipulation=True, plan=plan, observation_sha256=digest(fresh.image)), f)
    try:
        result = runtime._execute_operations(operations, folder)
        save(folder/'execution.json', result)
        if result.get('status') != 'EXECUTED':
            runtime.cancel.set()
            runtime.stop()
            return result
        runtime.move_to_perception(folder/'return_perception')
        runtime.capture_current(folder/'after')
        return result
    except BaseException:
        runtime.cancel.set()
        runtime.stop()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--mask', type=Path, help='optional registered binary visible-material mask')
    parser.add_argument('--region-point', type=float, nargs=2, required=True,
                        help='point near the VISIBLE contour adjacent to the invisible region')
    parser.add_argument('--region-name', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pull-px', type=float, default=80., help='preview distance; Host later enforces 10–150 mm')
    parser.add_argument('--inset-px', type=float, default=8.)
    parser.add_argument('--tangent-window-px', type=float, default=25.)
    parser.add_argument('--max-snap-px', type=float, default=80.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = build(args.image, args.region_point, args.region_name, args.evidence,
                       args.output, mask_path=args.mask, pull_px=args.pull_px,
                       inset_px=args.inset_px, tangent_window_px=args.tangent_window_px,
                       max_snap_px=args.max_snap_px)
    except NormalGraspError as exc:
        save(args.output/'result.json', dict(status='ABSTAIN', reason=str(exc), executed=False))
        print('ABSTAIN:',exc)
        return 2
    save(args.output/'result.json', dict(status=result['status'], executed=False))
    print(json.dumps(result['plan'], ensure_ascii=False, indent=2))
    print(args.output/'action.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
