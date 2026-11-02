import time
import numpy as np
from . import VERSION
from .segmentation import segment
from .motion_observer import register
from ..runtime.interfaces import Outcome,SCHEMA_VERSION

def execution_status(log):
    status=log.get('status','UNKNOWN');completed= status=='EXECUTED'
    actual='COMPLETED' if completed else ('PARTIAL_EXECUTION' if log.get('completed_commands',0)>0 else 'FAILED_OR_NOT_EXECUTED')
    # Motor completion/acquisition strings alone do not prove cloth contact or retention.
    return {'status':actual,'completed':completed,'grasp_status':'UNKNOWN','controller_status':status,
            'completed_commands':log.get('completed_commands'),'controller_error':log.get('error')}

def same_view(before,after):
    if before.camera_id!=after.camera_id:return False,'camera/calibration differs'
    a=(before.provenance or {}).get('joints_deg');b=(after.provenance or {}).get('joints_deg')
    if not a or not b or set(a)!=set(b):return False,'camera pose provenance missing'
    if any(np.max(np.abs(np.array(a[k])-np.array(b[k])))>.2 for k in a):return False,'camera pose changed'
    if before.rgb.shape!=after.rgb.shape:return False,'image size differs'
    return True,None

def observe(before,action,after,execution_log,area_noise_ratio=None):
    start=time.perf_counter();mb,qb=segment(before.rgb);ma,qa=segment(after.rgb)
    compatible,reason=same_view(before,after);valid=compatible and qb['status']=='HEURISTIC_OK' and qa['status']=='HEURISTIC_OK'
    coverage={'delta_ratio':None,'status':'UNKNOWN','area_before_px':int(mb.sum()),'area_after_px':int(ma.sum()),'units':'image pixels; not cloth surface area'}
    motion={'status':'UNKNOWN','reason':reason or 'segmentation quality insufficient'}
    if valid:
        delta=float((ma.sum()-float(mb.sum()))/max(1,mb.sum()))
        state='UNCALIBRATED' if area_noise_ratio is None else ('UNCHANGED' if abs(delta)<=area_noise_ratio else ('INCREASED' if delta>0 else 'DECREASED'))
        coverage.update(delta_ratio=delta,status=state,noise_threshold=area_noise_ratio)
        motion=register(before.rgb,after.rgb,mb,ma)
        # No automatic rigid-dominance label without independent noise/quality calibration.
        motion['interpretation']='UNVALIDATED_NUMERIC_EVIDENCE' if motion['status']=='OK' else 'UNKNOWN'
    out=Outcome(SCHEMA_VERSION,VERSION,execution_status(execution_log),coverage,motion,{'status':'UNKNOWN','reason':'material point tracking not implemented'},
                {'segmentation_before':qb,'segmentation_after':qa,'same_view':compatible,'view_reason':reason,'registration':motion['status'],'thresholds':'engineering defaults, not independent validation'},(time.perf_counter()-start)*1000)
    return out,mb,ma
