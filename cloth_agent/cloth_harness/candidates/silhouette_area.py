"""Separate outer projected footprint from color-support area; not a grasp mask.

Enclosed gaps may be print, shadow OR real holes. Filling them estimates an outer
silhouette footprint only; it must never certify a grasp or material occupancy.
"""
import time
from dataclasses import replace
import cv2
import numpy as np
from ..cv_harness.outcome_observer import observe,same_view
from ..cv_harness.segmentation import segment
from ..cv_harness.motion_observer import register


def footprint(mask):
    original=np.asarray(mask,dtype=bool)
    padded=np.pad(original.astype('uint8'),1)
    flooded=padded.copy();cv2.floodFill(flooded,None,(0,0),1)
    enclosed=(flooded[1:-1,1:-1]==0)
    return original|enclosed,enclosed


def observe_candidate(before,action,after,log):
    started=time.perf_counter()
    baseline,mb,ma=observe(before,action,after,log)
    fb,hb=footprint(mb);fa,ha=footprint(ma)
    c={'status':'UNKNOWN','delta_ratio':None,'area_before_px':int(fb.sum()),'area_after_px':int(fa.sum()),
       'units':'outer silhouette projected footprint px; not actual material area','enclosed_gap_before_px':int(hb.sum()),'enclosed_gap_after_px':int(ha.sum()),
       'raw_color_area_delta':baseline.coverage.get('delta_ratio'),'unfolding_inferred':False}
    valid=baseline.quality['same_view'] and all(baseline.quality[k]['status']=='HEURISTIC_OK' for k in ('segmentation_before','segmentation_after'))
    m={'status':'UNKNOWN','reason':baseline.motion.get('reason','quality insufficient')}
    if valid:
        c.update(status='UNCALIBRATED',delta_ratio=float((int(fa.sum())-int(fb.sum()))/max(1,int(fb.sum()))))
        m=register(before.rgb,after.rgb,fb,fa);m['interpretation']='UNVALIDATED_NUMERIC_EVIDENCE' if m['status']=='OK' else 'UNKNOWN'
    out=replace(baseline,harness_version='OFFLINE_CANDIDATE_OUTER_FOOTPRINT_01',coverage=c,motion=m,
                quality={**baseline.quality,'registration':m['status'],'candidate_not_deployed':True},latency_ms=(time.perf_counter()-started)*1000)
    return out,fb,fa,{'before_gap_mask':hb,'after_gap_mask':ha}
