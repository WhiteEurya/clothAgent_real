"""Explicit purple-garment prior, reused from the current rig; not general CV."""
import json
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


def segment_light_garment(rgb, unknown_mask=None):
    """Independent CV seed + GrabCut for the current light cloth/yellow mat.

    Not a general clothing detector. No model points/polygon or depth filling.
    Small missing projection cells may bridge the silhouette only; callers
    retain the original unknown-depth mask for actual contact sampling.
    """
    rgb=np.asarray(rgb,dtype=np.uint8)
    if rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('expected RGB image')
    unknown=np.zeros(rgb.shape[:2],bool) if unknown_mask is None else np.asarray(unknown_mask,bool)
    if unknown.shape!=rgb.shape[:2]:raise ValueError('unknown mask size mismatch')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    neutral=(hsv[:,:,1]<=65)&(hsv[:,:,2]>=65)&~unknown
    seed=cv2.morphologyEx(neutral.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    seed=cv2.morphologyEx(seed,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    def largest(mask):
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        if count<2:raise ValueError('no light garment component found on yellow mat')
        return (labels==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    seed=largest(seed)
    inner=cv2.erode(seed,np.ones((5,5),np.uint8)).astype(bool)&neutral
    if not inner.any():raise ValueError('no interior color seeds for garment segmentation')
    outer=cv2.dilate(seed,np.ones((15,15),np.uint8))
    labels=np.full(seed.shape,cv2.GC_BGD,np.uint8)
    labels[outer>0]=cv2.GC_PR_BGD;labels[seed>0]=cv2.GC_PR_FGD
    labels[inner]=cv2.GC_FGD;labels[unknown]=cv2.GC_BGD
    cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),labels,None,
        np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),5,cv2.GC_INIT_WITH_MASK)
    mask=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
    closed=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    mask[unknown & (closed>0)]=1
    mask=largest(mask)
    return mask,dict(method='light_garment_color_seeds_grabcut',model_polygon_used=False,
        prior='light low-saturation garment on yellow mat',area_px=int(mask.sum()),
        seed_area_px=int(seed.sum()),unknown_silhouette_cells=int((unknown&(mask>0)).sum()),
        depth_filled=False)


def segment_dark_garment(rgb, unknown_mask=None, *, smooth=True):
    """Explicit black-cloth prior on a bright table, in uncorrected RGB.

    Unknown projection cells are never color seeds. Hole filling affects only
    the silhouette; the observation's measured-depth mask is not changed.
    """
    rgb=np.asarray(rgb,dtype=np.uint8)
    if rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('expected RGB image')
    unknown=np.zeros(rgb.shape[:2],bool) if unknown_mask is None else np.asarray(unknown_mask,bool)
    if unknown.shape!=rgb.shape[:2]:raise ValueError('unknown mask size mismatch')
    value=rgb.max(axis=2)
    dark=(value<65)&~unknown
    kernel=np.ones((3,3),np.uint8)
    seed=cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_CLOSE,kernel)
    seed=cv2.morphologyEx(seed,cv2.MORPH_OPEN,kernel)
    def largest(mask):
        n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        if n<2:raise ValueError('no dark garment component found')
        return (lab==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    seed=largest(seed)
    # Enclosed print belongs to the same silhouette; do not split at its color.
    contours,_=cv2.findContours(seed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    silhouette=np.zeros_like(seed);cv2.drawContours(silhouette,contours,-1,1,cv2.FILLED)
    # Learn nearby table colors outside the cloth seed, then include attached
    # print colors that differ from that background. Dark pixels locate cloth;
    # they are not the complete foreground color model.
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    near=cv2.dilate(silhouette,np.ones((31,31),np.uint8)).astype(bool)
    ring=cv2.dilate(silhouette,np.ones((81,81),np.uint8)).astype(bool)&~near&~unknown&(value>=65)
    if not ring.any():raise ValueError('no local table color samples')
    # Quantized chroma palette retains grey/yellow table colors; luminance
    # changes alone (cast shadows) must not become printed foreground.
    bins,counts=np.unique((lab[ring,1:]/4).astype(np.int16),axis=0,return_counts=True)
    palette=bins[counts>=max(3,int(ring.sum()*.005))]*4+2
    if not len(palette):palette=np.median(lab[ring,1:],axis=0)[None]
    distance=np.full(seed.shape,np.inf,np.float32)
    for color in palette:
        distance=np.minimum(distance,np.linalg.norm(lab[:,:,1:]-color,axis=2))
    print_color=near&~unknown&(distance>6)
    attached=largest((seed>0)|print_color)
    contours,_=cv2.findContours(attached,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    silhouette=np.zeros_like(seed);cv2.drawContours(silhouette,contours,-1,1,cv2.FILLED)
    inner=cv2.erode(seed,np.ones((5,5),np.uint8)).astype(bool)&dark
    if not inner.any():raise ValueError('no interior dark garment seeds')
    outer=cv2.dilate(silhouette,np.ones((15,15),np.uint8))
    labels=np.full(seed.shape,cv2.GC_BGD,np.uint8)
    labels[outer>0]=cv2.GC_PR_BGD;labels[silhouette>0]=cv2.GC_PR_FGD
    labels[inner|((attached>0)&print_color)]=cv2.GC_FGD;labels[unknown]=cv2.GC_BGD
    # Repeated observations must not depend on earlier GrabCut RNG calls.
    cv2.setRNGSeed(0)
    cv2.grabCut(rgb,labels,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),5,cv2.GC_INIT_WITH_MASK)
    mask=largest(np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask,contours,-1,1,cv2.FILLED)
    smoothing=dict(enabled=bool(smooth),kernel_size=3,operations=['close','open'],changed_pixels=0)
    if smooth:
        original=mask.copy()
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask=largest(mask)
        smoothing['changed_pixels']=int(np.count_nonzero(mask!=original))
    return mask,dict(cv_rng_seed=0,smoothing=smoothing,method='dark_garment_color_seeds_grabcut',prior='black cloth on bright table',
        area_px=int(mask.sum()),seed_area_px=int(seed.sum()),model_polygon_used=False,
        unknown_silhouette_cells=int((unknown&(mask>0)).sum()),depth_filled=False)


def _largest_component(mask, *, minimum_area=1):
    """Return the largest 8-connected component and its area."""
    mask=np.asarray(mask,dtype=np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<2: return np.zeros(mask.shape,np.uint8),0
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    area=int(stats[index,cv2.CC_STAT_AREA])
    if area<minimum_area:return np.zeros(mask.shape,np.uint8),area
    return (labels==index).astype(np.uint8),area


def external_contour_band(mask, *, thickness=2):
    """Return only the visible external contour as a narrow mask band."""
    mask=(np.asarray(mask)>0).astype(np.uint8)
    edge=np.zeros_like(mask)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if contours:cv2.drawContours(edge,contours,-1,1,max(1,int(thickness)))
    return edge


def segment_elevated_garment(rgb, xyz, unknown_mask=None, *, threshold_mm=12.0,
                             ransac_threshold_mm=4.0, iterations=300):
    """Segment the largest surface elevated above a robust tabletop plane.

    This uses measured XYZ only to estimate the support plane; it does not fill
    missing depth or use model-selected points.  It is intended for the
    top-down fused view, where the garment is a raised connected surface and
    the table is the dominant plane.
    """
    rgb=np.asarray(rgb)
    xyz=np.asarray(xyz,dtype=np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('expected RGB image')
    if xyz.shape!=(*rgb.shape[:2],3):raise ValueError('XYZ/RGB shape mismatch')
    unknown=np.zeros(rgb.shape[:2],bool) if unknown_mask is None else np.asarray(unknown_mask,bool)
    if unknown.shape!=rgb.shape[:2]:raise ValueError('unknown mask size mismatch')
    valid=np.isfinite(xyz).all(axis=2)&~unknown
    ys,xs=np.nonzero(valid)
    if len(xs)<1000:raise ValueError('insufficient measured XYZ for elevated garment segmentation')
    points=xyz[valid].astype(np.float64)
    xy=points[:,:2]; z=points[:,2]
    rng=np.random.default_rng(0)
    sample_count=min(len(points),100000)
    sample_ids=rng.choice(len(points),sample_count,replace=False) if len(points)>sample_count else np.arange(len(points))
    fit_xy=xy[sample_ids];fit_z=z[sample_ids]
    best=None
    for _ in range(max(20,int(iterations))):
        ids=rng.choice(len(fit_xy),3,replace=False)
        A=np.c_[fit_xy[ids],np.ones(3)]
        try:coef=np.linalg.solve(A,fit_z[ids])
        except np.linalg.LinAlgError:continue
        residual=np.abs(fit_z-(fit_xy@coef[:2]+coef[2]))
        inliers=residual<=ransac_threshold_mm
        score=(int(inliers.sum()),-float(np.median(residual[inliers])) if inliers.any() else -1e9)
        if best is None or score>best[0]:best=(score,coef)
    if best is None:raise ValueError('unable to fit tabletop plane')
    coef=best[1]
    # Refine against the dominant plane and reject gross sensor outliers.
    for _ in range(2):
        residual=z-(xy@coef[:2]+coef[2]);inliers=np.abs(residual)<=ransac_threshold_mm
        if int(inliers.sum())<500:break
        coef=np.linalg.lstsq(np.c_[xy[inliers],np.ones(int(inliers.sum()))],z[inliers],rcond=None)[0]
    residual_map=np.full(valid.shape,np.nan,np.float32)
    residual_map[valid]=(xyz[valid,2]-(coef[0]*xyz[valid,0]+coef[1]*xyz[valid,1]+coef[2])).astype(np.float32)
    candidates=[]
    for sign in (1,-1):
        raw=(residual_map*sign>=float(threshold_mm))&valid
        cleaned=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        cleaned=cv2.morphologyEx(cleaned,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        component,area=_largest_component(cleaned,minimum_area=500)
        if area:
            values=residual_map[component>0]*sign
            values=values[np.isfinite(values)]
            candidates.append((area,float(np.median(values)),component,sign))
    if not candidates:raise ValueError('no elevated garment component found')
    area,median_height,mask,sign=max(candidates,key=lambda row:(row[0],row[1]))
    area=int(mask.sum())
    h,w=mask.shape; ys,xs=np.where(mask>0)
    report=dict(method='elevated_surface_ransac',prior='measured tabletop plane plus raised garment',
        plane_coefficients_mm=[float(v) for v in coef],plane_inliers=int(best[0][0]),
        ransac_threshold_mm=float(ransac_threshold_mm),elevation_threshold_mm=float(threshold_mm),
        elevation_sign=int(sign),median_elevation_mm=round(median_height,3),area_px=int(area),
        bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
        unknown_silhouette_cells=int((unknown&(mask>0)).sum()),depth_filled=False)
    # Return residuals in the selected garment-above-plane orientation so the
    # common candidate scorer works when the world Z axis points downward.
    return mask.astype(np.uint8),report,residual_map*sign


def _candidate_quality(mask, *, residual=None, method=''):
    """Score a mask without using model points or semantic labels."""
    mask=np.asarray(mask,bool);h,w=mask.shape;area=int(mask.sum());fraction=area/(h*w)
    if not area:return -1e9,dict(area_px=0,area_fraction=0.0)
    # A useful garment occupies a meaningful connected region without filling
    # the whole tabletop.  The bounds are soft, so unusual garments remain
    # possible while obvious logo/table masks lose priority.
    if fraction<0.008:area_score=max(0.0,fraction/0.008)
    elif fraction<=0.35:area_score=1.0
    else:area_score=max(0.0,1.0-(fraction-0.35)/0.35)
    border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
    border_score=0.0 if border else 1.0
    score=2.0*area_score+0.5*border_score
    elevated_fraction=None;median_elevation=None
    if residual is not None:
        vals=np.asarray(residual)[mask];vals=vals[np.isfinite(vals)]
        if len(vals):
            elevated_fraction=float((vals>=8.0).mean());median_elevation=float(np.median(vals))
            score+=2.5*elevated_fraction+min(1.5,max(0.0,median_elevation/20.0))
    if method=='elevated_surface_ransac':score+=0.5
    return float(score),dict(area_px=area,area_fraction=round(float(fraction),6),touches_image_border=border,
        area_score=round(float(area_score),4),border_score=border_score,
        elevated_fraction=round(elevated_fraction,4) if elevated_fraction is not None else None,
        median_elevation_mm=round(median_elevation,3) if median_elevation is not None else None,
        score=round(float(score),4))


def segment_auto_garment(rgb, xyz=None, unknown_mask=None):
    """Choose a garment mask from depth and color candidates automatically."""
    rgb=np.asarray(rgb,dtype=np.uint8)
    candidates=[];errors={};residual=None
    if xyz is not None:
        try:
            mask,report,residual=segment_elevated_garment(rgb,xyz,unknown_mask)
            score,quality=_candidate_quality(mask,residual=residual,method=report['method'])
            report.update(profile='elevated',quality=quality)
            candidates.append((score,'elevated',mask,report))
        except (ValueError,cv2.error) as exc:errors['elevated']=str(exc)
    for name,fn in (('black',segment_dark_garment),('light',segment_light_garment)):
        try:
            mask,report=fn(rgb,unknown_mask)
            score,quality=_candidate_quality(mask,residual=residual,method=report.get('method',''))
            report.update(profile=name,quality=quality)
            candidates.append((score,name,mask,report))
        except (ValueError,cv2.error) as exc:errors[name]=str(exc)
    if not candidates:raise ValueError('automatic garment segmentation failed: '+json.dumps(errors,ensure_ascii=False))
    score,name,mask,report=max(candidates,key=lambda row:row[0])
    edge=external_contour_band(mask,thickness=2)
    report=dict(report,method='automatic_profile_selection',selected_profile=name,
                selected_score=round(float(score),4),candidate_scores={n:r['quality'] for _,n,_,r in candidates},
                candidate_errors=errors,edge_mask_area_px=int(edge.sum()),edge_mask_thickness_px=2)
    return mask,report


def refine_mask_with_rgb(rgb, coarse_mask, unknown_mask=None, *, core_radius_px=8,
                         search_radius_px=10):
    """Refine a coarse silhouette with RGB; opt-in, no depth modification.

    Thick interior pixels seed foreground. Thin depth-noise branches remain
    uncertain, so color evidence can remove them. This cannot recover cloth
    outside the local search band or guarantee correctness for similar colors.
    """
    rgb=np.asarray(rgb,dtype=np.uint8);coarse=np.asarray(coarse_mask)>0
    if rgb.ndim!=3 or rgb.shape[2]!=3 or coarse.shape!=rgb.shape[:2]:
        raise ValueError('RGB/coarse mask shape mismatch')
    if core_radius_px<=0 or search_radius_px<1:raise ValueError('invalid refinement radius')
    unknown=np.zeros(coarse.shape,bool) if unknown_mask is None else np.asarray(unknown_mask,bool)
    if unknown.shape!=coarse.shape:raise ValueError('unknown mask size mismatch')
    distance=cv2.distanceTransform(coarse.astype(np.uint8),cv2.DIST_L2,5)
    core=(distance>=core_radius_px)&~unknown
    if not core.any():raise ValueError('no foreground interior for RGB refinement')
    size=2*int(search_radius_px)+1
    support=cv2.dilate(coarse.astype(np.uint8),np.ones((size,size),np.uint8))>0
    labels=np.full(coarse.shape,cv2.GC_BGD,np.uint8)
    labels[support]=cv2.GC_PR_BGD;labels[coarse]=cv2.GC_PR_FGD
    labels[core]=cv2.GC_FGD;labels[unknown]=cv2.GC_BGD
    if not (labels==cv2.GC_BGD).any():raise ValueError('no background for RGB refinement')
    cv2.setRNGSeed(0)
    cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),labels,None,
        np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),5,cv2.GC_INIT_WITH_MASK)
    mask=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
    return mask,dict(method='coarse_interior_seeded_rgb_grabcut',
        core_radius_px=core_radius_px,search_radius_px=search_radius_px,
        before_area_px=int(coarse.sum()),area_px=int(mask.sum()),
        removed_pixels=int((coarse&~mask.astype(bool)).sum()),
        added_pixels=int((~coarse&mask.astype(bool)).sum()),depth_modified=False)
