"""Collar grasp with deterministic center-translation candidate in the existing RGB topdown frame."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cloth_agent.contour_normal_grasp import NormalGraspError
from cloth_agent.dual_arm.geometry import DualArmError

ROOT=Path(__file__).resolve().parents[2]


def polygon_center(points):
    p=np.asarray(points,float);q=np.roll(p,-1,axis=0)
    cross=p[:,0]*q[:,1]-q[:,0]*p[:,1]
    if not np.isfinite(p).all() or abs(cross.sum())<1e-6:
        raise NormalGraspError('invalid table polygon')
    return ((p+q)*cross[:,None]).sum(0)/(3*cross.sum())


def table_polygon(view):
    # Reuse the fixed-camera table ROI and current calibration, not canvas bounds.
    cfg=json.loads((ROOT/'config/topdown.unfold_zed.json').read_text())
    source=next((o for o in view.sources if str(o.meta.get('camera_serial'))==str(cfg['camera_serial'])),None)
    if source is None:raise NormalGraspError('recenter requires calibrated ZED table ROI')
    if list(Image.open(source.image).size)!=cfg['image_size_wh']:
        raise NormalGraspError('table ROI image size differs from current ZED')
    metadata=json.loads((view.directory/'view.json').read_text())
    z=float(metadata['plane_z_mm_display_only'])
    T=np.asarray(source.meta['world_from_camera_mm'],float)
    K=np.asarray(source.meta['intrinsics'],float)
    rays=(T[:3,:3]@np.linalg.solve(K,np.c_[cfg['table_polygon_px'],np.ones(len(cfg['table_polygon_px']))].T)).T
    if np.any(abs(rays[:,2])<1e-8):raise NormalGraspError('invalid table projection')
    scale=(z-T[2,3])/rays[:,2]
    if np.any(scale<=0):raise NormalGraspError('table ROI projects behind camera')
    return (T[:3,3]+scale[:,None]*rays)[:,:2]


def candidate(view, mask, grasp_px):
    meta=json.loads((view.directory/'view.json').read_text())
    if not meta.get('mode','').startswith('RGB_FIRST_'):
        raise NormalGraspError('recenter requires calibrated RGB topdown')
    mask=np.asarray(mask)>0
    if not mask.any() or mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        raise NormalGraspError('recenter needs an uncut garment silhouette')
    h,w=mask.shape;s=float(meta['mm_per_pixel']);origin=np.asarray(meta['center_world_xy_mm'])
    def world(p):return origin+(np.asarray(p)-[(w-1)/2,(h-1)/2])*[s,-s]
    yy,xx=np.nonzero(mask);center=world([xx.mean(),yy.mean()])
    polygon=table_polygon(view);target_center=polygon_center(polygon)
    delta=target_center-center;distance=float(np.linalg.norm(delta))
    if distance<10:raise NormalGraspError('garment already near table center; choose unfold or DONE')
    # Respect current 150 mm action limit, retaining full desired displacement.
    step=delta*min(1.,140./distance)
    # Keep the visible collar-area point selected by the single planner call.
    grasp_array=np.asarray(grasp_px,dtype=float)
    if grasp_array.shape!=(2,) or not np.isfinite(grasp_array).all():
        raise NormalGraspError('recenter requires a visible collar-area grasp_px')
    grasp=np.rint(grasp_array).astype(int)
    if not (0<=grasp[0]<w and 0<=grasp[1]<h and mask[grasp[1],grasp[0]]):
        raise NormalGraspError('recenter collar-area grasp must lie on visible garment')
    dest=np.rint(grasp+step/[s,-s]).astype(int)
    if not (0<=dest[0]<w and 0<=dest[1]<h):
        raise NormalGraspError('recenter destination outside image')
    if cv2.pointPolygonTest(polygon.astype('float32'),tuple(map(float,world(dest))),False)<0:
        raise NormalGraspError('recenter destination outside table')
    a=np.asarray(view.sample(grasp.tolist(),None));b=np.asarray(view.sample(dest.tolist(),None))
    measured=float(np.linalg.norm(b[:2]-a[:2]))
    if not 10<=measured<=150 or np.dot(b[:2]-a[:2],delta/distance)<=0:
        raise NormalGraspError('recenter measured translation outside limits or wrong direction')
    return dict(grasp_px=grasp.tolist(),target_px=dest.tolist(),grasp_source='planner_visible_collar_area',
                garment_center_world_xy_mm=center.tolist(),table_center_world_xy_mm=target_center.tolist(),
                table_polygon_world_xy_mm=polygon.tolist(),table_center_source='existing calibrated ZED table ROI',
                requested_center_delta_mm=delta.tolist(),requested_distance_mm=distance,
                planned_step_delta_mm=step.tolist(),measured_distance_mm=measured,
                segmentation_reason='existing 150 mm action limit; reobserve next round' if distance>140 else None)
