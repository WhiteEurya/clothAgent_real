"""Local measured geometry only. No semantic ranking or robot-safety override."""
import time
import cv2
import numpy as np
from .segmentation import segment

def check(observation,plan):
    start=time.perf_counter();result={'status':'UNVERIFIED','semantic_identity_checked':False,'robot_feasibility':'PENDING_EXISTING_HOST',
        'refinement':{'status':'DISABLED_UNCALIBRATED','action_changed':False},'checks':{}}
    try:
        pixel=plan.get('grasp_px');h,w=observation.xyz.shape[:2]
        if not isinstance(pixel,(list,tuple)) or len(pixel)!=2 or not all(isinstance(v,int) for v in pixel):
            result.update(status='REJECTED',reason='invalid image coordinate');return result
        x,y=pixel
        if not 2<=x<w-2 or not 2<=y<h-2:result.update(status='REJECTED',reason='pixel outside measured image interior');return result
        from PIL import Image
        mask,quality=segment(np.asarray(Image.open(observation.image).convert('RGB')))
        xyz=observation.xyz[y-2:y+3,x-2:x+3];valid=np.isfinite(xyz).all(2);values=xyz[valid]
        distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
        result['checks']={'cloth_membership':bool(mask[y,x]),'segmentation_quality':quality,'boundary_distance_px':float(distance[y,x]),
            'depth_valid_ratio':float(valid.mean()),'center_depth_valid':bool(valid[2,2]),'depth_z_range_mm':float(np.ptp(values[:,2])) if len(values) else None,
            'depth_z_median_mm':float(np.median(values[:,2])) if len(values) else None,'depth_spread_is_rejection_gate':False}
        if quality['status']!='HEURISTIC_OK':result['reason']='segmentation evidence insufficient';return result
        if not mask[y,x]:result['reason']='not supported by cloth mask; may be segmentation error';return result
        if len(values)<18 or not valid[2,2]:result['reason']='insufficient measured local depth';return result
        result.update(status='ACCEPTED',reason='mask membership and measured-depth checks passed; physical safety still pending')
        return result
    finally:result['wall_ms']=(time.perf_counter()-start)*1000
