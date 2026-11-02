"""Explicit purple-garment prior, reused from the current rig; not general CV."""
import cv2
import numpy as np
from cloth_agent.geometry_report import garment_mask

def segment(rgb):
    mask = garment_mask(rgb).astype(bool)
    area = int(mask.sum()); h,w = mask.shape
    crop_left,crop_right = int(.18*w),int(.84*w)-1
    clipped = bool(mask[0].any() or mask[-1].any() or mask[:,crop_left].any() or mask[:,crop_right].any())
    reasons = []
    if area < 500: reasons.append('insufficient cloth pixels')
    if area > .55*h*w: reasons.append('implausibly large foreground')
    if clipped: reasons.append('foreground touches image/segmentation ROI boundary')
    return mask, {'status':'UNKNOWN' if reasons else 'HEURISTIC_OK',
                  'reasons':reasons,'area_px':area,'clipped':clipped,
                  'prior':'purple garment on dark table; fixed side ROI',
                  'semantic_identity_verified':False}
