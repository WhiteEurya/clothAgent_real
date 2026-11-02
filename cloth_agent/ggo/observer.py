"""Metric upper-body SE(2) registration and independent regional mask deficits.

Input XYZ must already be aligned to RGB in a common calibrated world-mm frame.
All thresholds are explicit provisional engineering settings, not learned truth.
"""
from dataclasses import dataclass, asdict
from itertools import combinations
import cv2
import numpy as np
from scipy.spatial import cKDTree
from cloth_agent.geometry_report import garment_mask

LANDMARKS = ('N','AL','AR','SL','SR','HL','HR')
REGIONS = ('LEFT_SLEEVE','RIGHT_SLEEVE','LEFT_LOWER_BODY','RIGHT_LOWER_BODY')

@dataclass(frozen=True)
class Settings:
    grid_mm: float = 2.0
    tolerance_mm: float = 6.0
    ransac_mm: float = 12.0
    max_rmse_mm: float = 12.0
    min_anchor_span_mm: float = 80.0
    min_anchor_minor_mm: float = 12.0
    min_inlier_fraction: float = .65
    max_upper_contour_mm: float = 20.0
    max_table_rmse_mm: float = 3.0
    min_depth_fraction: float = .85
    min_observed_fraction: float = .95
    max_deviation_mm: float = 80.0
    contraction_ratio: float = .12
    contraction_area_mm2: float = 800.0
    contraction_component_mm2: float = 600.0
    contraction_contour_mm: float = 10.0
    perturb_translation_mm: float = 3.0
    perturb_angle_deg: float = 2.0
    section_step_mm: float = 10.0
    boundary_support_mm: float = 6.0
    asymmetry_margin: float = .05
    anchor_boundary_tolerance_px: float = 3.

    def __post_init__(self):
        for key,value in asdict(self).items():
            if not np.isfinite(value) or value<=0:raise ValueError('settings must be finite and positive: '+key)
        for key in ('min_inlier_fraction','min_depth_fraction','min_observed_fraction','contraction_ratio'):
            if getattr(self,key)>1:raise ValueError('fraction must be <=1: '+key)


def fit_table(xyz, cloth, roi=None):
    """RANSAC world z=ax+by+c on bare background; deterministic, millimetres."""
    h,w=cloth.shape
    if roi is None:
        roi=np.zeros((h,w),bool);roi[int(.08*h):int(.95*h),int(.22*w):int(.80*w)]=True
    excluded=cv2.dilate(cloth.astype('uint8'),np.ones((21,21),np.uint8))>0
    pts=np.asarray(xyz)[roi & ~excluded & np.isfinite(xyz).all(2)]
    if len(pts)<300:raise ValueError('insufficient measured bare-table support')
    pts=pts[::max(1,len(pts)//6000)]
    A=np.c_[pts[:,:2],np.ones(len(pts))];rng=np.random.default_rng(241024);best=None
    for _ in range(160):
        idx=rng.choice(len(pts),3,replace=False)
        if np.linalg.cond(A[idx])>1e7:continue
        coef=np.linalg.solve(A[idx],pts[idx,2]);err=np.abs(A@coef-pts[:,2]);inside=err<3.
        if best is None or inside.sum()>best.sum():best=inside
    if best is None or best.mean()<.6:raise ValueError('bare-table plane has insufficient consensus')
    coef=np.linalg.lstsq(A[best],pts[best,2],rcond=None)[0]
    normal=np.r_[-coef[:2],1.];normal/=np.linalg.norm(normal)
    ex=np.array([1.,0.,0.]);ex-=normal*ex.dot(normal);ex/=np.linalg.norm(ex)
    ey=np.cross(normal,ex);basis=np.stack([ex,ey,normal],axis=1)
    origin=np.array([0.,0.,coef[2]])
    return dict(origin_world_mm=origin.tolist(),basis_columns=basis.tolist(),
                plane_z_equals_ax_by_c=coef.tolist(),inlier_fraction=float(best.mean()),
                rmse_mm=float(np.sqrt(np.mean((A[best]@coef-pts[best,2])**2))),
                support_span_mm=np.ptp(pts[best,:2],axis=0).tolist(),source='reference bare-table RANSAC')


def table_cloud(xyz, table):
    return (np.asarray(xyz)-np.array(table['origin_world_mm']))@np.array(table['basis_columns'])


def sample_anchor(xyz, mask, pixel, *, tolerance_px=0, audit=None):
    """Use measured cloth only; bounded boundary correction never fills depth holes."""
    info={} if audit is None else audit
    info.update(requested_px=pixel,status='REJECTED',tolerance_px=tolerance_px)
    def reject(reason):
        info['reason']=reason
        return None
    if pixel is None:return reject('MODEL_UNOBSERVED')
    p=np.asarray(pixel)
    if p.shape!=(2,) or not np.isfinite(p).all() or not np.equal(p,np.round(p)).all():return reject('INVALID_PIXEL')
    x,y=map(int,p);h,w=mask.shape
    if not (2<=x<w-2 and 2<=y<h-2):return reject('OUTSIDE_IMAGE')
    if not np.isfinite(xyz[y,x]).all():return reject('CENTER_DEPTH_INVALID')
    corrected=False
    if not mask[y,x]:
        r=int(tolerance_px)
        if r<=0:return reject('OUTSIDE_CLOTH_MASK')
        yy,xx=np.mgrid[max(2,y-r):min(h-2,y+r+1),max(2,x-r):min(w-2,x+r+1)]
        distance=np.hypot(xx-x,yy-y)
        valid=(distance<=r)&mask[yy,xx]&np.isfinite(xyz[yy,xx]).all(2)
        if not valid.any():return reject('NO_CLOTH_WITHIN_TOLERANCE')
        candidates=np.argwhere(valid)
        candidates=sorted(candidates,key=lambda ij:distance[tuple(ij)])
        selected=None
        for i,j in candidates:
            u,v=int(xx[i,j]),int(yy[i,j])
            ok=mask[v-2:v+3,u-2:u+3]&np.isfinite(xyz[v-2:v+3,u-2:u+3]).all(2)
            if ok.sum()>=3:selected=(u,v);break
        if selected is None:return reject('INSUFFICIENT_LOCAL_CLOTH_SUPPORT')
        x,y=selected;corrected=True
    patch=xyz[y-2:y+3,x-2:x+3];ok=mask[y-2:y+3,x-2:x+3]&np.isfinite(patch).all(2)
    if ok.sum()<3:return reject('INSUFFICIENT_LOCAL_CLOTH_SUPPORT')
    point=np.median(patch[ok],axis=0)
    info.update(status='BOUNDARY_CORRECTED' if corrected else 'MEASURED',reason=None,
                sampled_px=[x,y],offset_px=float(np.linalg.norm(np.array([x,y])-p)),
                boundary_offset_xy_mm=float(np.linalg.norm(point[:2]-xyz[int(p[1]),int(p[0]),:2])) if corrected else 0.,
                support_pixels=int(ok.sum()),support_xy_radius_mm=float(np.max(np.linalg.norm(patch[ok,:2]-point[:2],axis=1))))
    return point


def rigid_fit(a,b,weights=None):
    a,b=np.asarray(a,float),np.asarray(b,float)
    weights=np.ones(len(a)) if weights is None else np.asarray(weights,float)
    weights=weights/weights.sum();ca=np.sum(a*weights[:,None],axis=0);cb=np.sum(b*weights[:,None],axis=0)
    u,_,vt=np.linalg.svd((a-ca).T@((b-cb)*weights[:,None]))
    fix=np.eye(2);fix[-1,-1]=np.linalg.det(vt.T@u.T)
    R=vt.T@fix@u.T;t=cb-R@ca
    return R,t


def transform(points, R, t):return np.asarray(points)@np.asarray(R).T+np.asarray(t)


def register(a,b,cfg=Settings()):
    a,b=np.asarray(a,float),np.asarray(b,float)
    result=dict(status='UNKNOWN',confidence='LOW',scale=1.,reason='insufficient or invalid upper-body anchors')
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1:]!=(2,) or not np.isfinite([a,b]).all():
        result['reason']='invalid anchor array or nonfinite coordinates'
        return result
    if len(a)<3:
        result.update(reason='fewer than three measured upper-body anchors',measured_anchor_count=len(a),required_anchor_count=3)
        return result
    best=None;hypotheses=[]
    for ids in list(combinations(range(len(a)),2))[:1000]:
        if np.linalg.norm(a[ids[0]]-a[ids[1]])<20:continue
        R,t=rigid_fit(a[list(ids)],b[list(ids)])
        err=np.linalg.norm(transform(a,R,t)-b,axis=1);inside=err<=cfg.ransac_mm
        hypotheses.append((inside.sum(),R,t))
        score=(int(inside.sum()),-float(np.minimum(err,cfg.ransac_mm).sum()))
        if best is None or score>best[0]:best=(score,R,t,inside)
    # A full-set fit is a candidate, not a replacement for robust hypotheses.
    R,t=rigid_fit(a,b)
    err=np.linalg.norm(transform(a,R,t)-b,axis=1);inside=err<=cfg.ransac_mm
    score=(int(inside.sum()),-float(np.minimum(err,cfg.ransac_mm).sum()))
    if best is None or score>best[0]:best=(score,R,t,inside)
    _,R,t,inside=best
    if inside.sum()<3:
        result.update(reason='fewer than three geometrically consistent anchors',residuals_mm=np.linalg.norm(transform(a,R,t)-b,axis=1).tolist(),inlier_count=int(inside.sum()))
        return result
    initial_R,initial_t=R.copy(),t.copy()
    for _ in range(8):
        err=np.linalg.norm(transform(a[inside],R,t)-b[inside],axis=1)
        R,t=rigid_fit(a[inside],b[inside],np.minimum(1.,5./np.maximum(err,1e-6)))
    err=np.linalg.norm(transform(a,R,t)-b,axis=1);inside=err<=cfg.ransac_mm
    refined_score=(int(inside.sum()),-float(np.minimum(err,cfg.ransac_mm).sum()))
    retained=refined_score<best[0]
    if retained:
        R,t=initial_R,initial_t
        err=np.linalg.norm(transform(a,R,t)-b,axis=1);inside=err<=cfg.ransac_mm
    result.update(refinement_retained_initial=retained,residuals_mm=err.tolist())
    spread=np.linalg.svd(a[inside]-a[inside].mean(0),compute_uv=False)/np.sqrt(inside.sum())
    span=float(np.linalg.norm(np.ptp(a[inside],axis=0)));rmse=float(np.sqrt(np.mean(err[inside]**2)))
    ambiguous=False
    center=a[inside].mean(0)
    for count,r2,t2 in hypotheses:
        angle=abs(np.arctan2((r2@R.T)[1,0],(r2@R.T)[0,0]))
        if count>=inside.sum() and (angle>np.deg2rad(10) or np.linalg.norm(transform(center,r2,t2)-transform(center,R,t))>15):ambiguous=True
    reasons=[]
    if rmse>cfg.max_rmse_mm:reasons.append('registration residual too large')
    if span<cfg.min_anchor_span_mm or spread[-1]<cfg.min_anchor_minor_mm:reasons.append('anchors have insufficient spatial support')
    if inside.mean()<cfg.min_inlier_fraction:reasons.append('insufficient anchor consensus')
    if ambiguous:reasons.append('multiple incompatible pose hypotheses')
    result.update(status='UNCERTAIN' if reasons else 'OK',confidence='LOW' if reasons else 'CONDITIONAL',
        reason='; '.join(reasons) if reasons else 'upper-body metric anchors pass provisional checks',
        rotation=R.tolist(),translation_mm=t.tolist(),inlier_count=int(inside.sum()),inlier_indices=np.flatnonzero(inside).tolist(),
        rmse_mm=rmse,spatial_support=dict(span_mm=span,minor_std_mm=float(spread[-1])),ambiguous=ambiguous)
    return result


def disk(radius):
    r=max(0,int(radius));y,x=np.ogrid[-r:r+1,-r:r+1];return (x*x+y*y<=r*r).astype('uint8')


def raster(points, origin, shape, resolution):
    p=np.asarray(points);out=np.zeros(shape,np.uint8)
    if len(p):
        ij=np.floor((p[:,:2]-origin)/resolution).astype(int)
        ok=(ij[:,0]>=0)&(ij[:,0]<shape[1])&(ij[:,1]>=0)&(ij[:,1]<shape[0]);ij=ij[ok]
        out[ij[:,1],ij[:,0]]=1
    return out


def contour_points(mask,origin,resolution):
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return np.empty((0,2))
    return max(contours,key=cv2.contourArea)[:,0,:]*resolution+origin+resolution/2


def reference_regions(mask, origin, cfg, points):
    """Partition in fixed garment coordinates, not screen/world left and right."""
    al,ar,hl,hr=(points[k][:2] for k in ('AL','AR','HL','HR'))
    center=(al+ar)/2;ex=(ar-al)/np.linalg.norm(ar-al);ey=np.array([-ex[1],ex[0]])
    if ey@((hl+hr)/2-center)<0:ey=-ey
    y,x=np.indices(mask.shape);xy=np.stack([x,y],axis=-1)*cfg.grid_mm+origin+cfg.grid_mm/2
    down=(xy-center)@ey;side=(xy-center)@ex
    # Lower body is below the fixed axillary line. Sleeves occupy upper outer sides.
    upper=(down<0);left=side<0
    sleeves=upper & (np.abs(side)>np.linalg.norm(ar-al)*.38)
    return dict(LEFT_LOWER_BODY=mask.astype(bool)&~upper&left,
                RIGHT_LOWER_BODY=mask.astype(bool)&~upper&~left,
                LEFT_SLEEVE=mask.astype(bool)&sleeves&left,
                RIGHT_SLEEVE=mask.astype(bool)&sleeves&~left),dict(center=center,ex=ex,ey=ey)


def boundary_arc(contour,start,end):
    i=int(np.argmin(np.linalg.norm(contour-start,axis=1)));j=int(np.argmin(np.linalg.norm(contour-end,axis=1)))
    forward=contour[(np.arange(i,j+len(contour)+1) if j<i else np.arange(i,j+1))%len(contour)]
    backward=contour[(np.arange(j,i+len(contour)+1) if i<j else np.arange(j,i+1))%len(contour)][::-1]
    length=lambda p:np.linalg.norm(np.diff(p,axis=0),axis=1).sum()
    return forward if length(forward)<=length(backward) else backward


def warp(mask,origin,cfg,R,t):
    # grid cell centers to table coordinates and back
    c=origin+cfg.grid_mm/2
    affine=np.c_[R,(R@c+t-c)/cfg.grid_mm]
    return cv2.warpAffine(mask.astype('uint8'),affine,(mask.shape[1],mask.shape[0]),flags=cv2.INTER_NEAREST)>0


def state_for(m,cfg):
    if m['reliability']!='OK':return 'UNKNOWN'
    if (m['missing_ratio']>=cfg.contraction_ratio and m['missing_area_mm2']>=cfg.contraction_area_mm2
        and m['largest_missing_component_mm2']>=cfg.contraction_component_mm2 and m['contour_deviation_mm']>=cfg.contraction_contour_mm):return 'SUSPECTED_CONTRACTION'
    if m['missing_ratio']<.05 and m['contour_deviation_mm']<cfg.tolerance_mm:return 'MOSTLY_EXTENDED'
    return 'NO_CLEAR_CONTRACTION'


def compare(regions,paths,current,observed,origin,cfg,R,t):
    grown=cv2.dilate(current.astype('uint8'),disk(np.ceil(cfg.tolerance_mm/cfg.grid_mm)))>0
    distance=cv2.distanceTransform((~current).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)*cfg.grid_mm
    results={};missing_all=np.zeros(current.shape,bool)
    for name,region in regions.items():
        moved=warp(region,origin,cfg,R,t);interior=cv2.erode(moved.astype('uint8'),disk(np.ceil(cfg.tolerance_mm/cfg.grid_mm)))>0
        area=int(interior.sum());coverage=float((interior&observed).sum()/area) if area else 0.
        missing=interior&observed&~grown;missing_all|=missing
        n,_,stats,_=cv2.connectedComponentsWithStats(missing.astype('uint8'),8)
        path=transform(paths[name],R,t);ij=np.floor((path-origin)/cfg.grid_mm).astype(int)
        inbounds=(ij[:,0]>=0)&(ij[:,0]<current.shape[1])&(ij[:,1]>=0)&(ij[:,1]<current.shape[0])
        valid=np.zeros(len(path),bool);valid[inbounds]=observed[ij[inbounds,1],ij[inbounds,0]]
        weights=np.r_[0.,np.linalg.norm(np.diff(path,axis=0),axis=1)]
        path_coverage=float(weights[valid].sum()/max(weights.sum(),1e-6))
        deviations=distance[ij[valid,1],ij[valid,0]]
        mean=float(np.average(np.minimum(deviations,cfg.max_deviation_mm),weights=weights[valid])) if weights[valid].sum()>0 else None
        reliable=area>0 and coverage>=cfg.min_observed_fraction and path_coverage>=cfg.min_observed_fraction
        metric=dict(reference_interior_area_mm2=area*cfg.grid_mm**2,observed_fraction=coverage,path_observed_fraction=path_coverage,
                    reliability='OK' if reliable else 'UNKNOWN',missing_ratio=float(missing.sum()/area) if reliable else None,
                    missing_area_mm2=float(missing.sum()*cfg.grid_mm**2) if reliable else None,
                    largest_missing_component_mm2=float(stats[1:,cv2.CC_STAT_AREA].max()*cfg.grid_mm**2) if reliable and n>1 else (0. if reliable else None),
                    contour_deviation_mm=mean if reliable else None)
        metric['state']=state_for(metric,cfg);results[name]=metric
    suspects=[k for k,v in results.items() if v['state']=='SUSPECTED_CONTRACTION']
    # Primary region follows absolute contiguous missing area, never percent alone.
    primary=max(suspects,key=lambda k:results[k]['largest_missing_component_mm2']) if suspects else None
    return dict(status='OK' if all(v['reliability']=='OK' for v in results.values()) else 'PARTIAL',
                primary_region=primary,regions=results),missing_all


def silhouette(rgb):
    # Printed letters are not holes in garment occupancy. Fill only enclosed RGB
    # segmentation holes; measured-depth validity is still tracked separately.
    mask=garment_mask(rgb)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(mask)
    if contours:cv2.drawContours(filled,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    return filled.astype(bool)


def observe(reference_rgb,reference_xyz,current_rgb,current_xyz,annotations,cfg=Settings()):
    """V0/V1 only. Manual anchors are explicit inputs; no corner verification claim."""
    refmask=silhouette(reference_rgb);curmask=silhouette(current_rgb)
    report=dict(version='GGO_V0_V1',model_calls=0,robot_execution=False,settings=asdict(cfg),
                annotation_provenance=annotations.get('provenance','unspecified manual input'),
                limitations=['Region deficits are projected geometry, not missing material or proof of folding.',
                             'Manual upper-body correspondence is an input, not automatically verified semantics.',
                             'Thresholds are provisional; no calibrated DONE rule or action success claim.',
                             'V2 corner verification and V3 automatic matching are not implemented.'],
                corners={k:dict(status='UNVERIFIED',point=None) for k in ('left_hem','right_hem')})
    quality={}
    for name,mask,xyz in [('reference',refmask,reference_xyz),('current',curmask,current_xyz)]:
        fraction=float(np.isfinite(xyz[mask]).all(1).mean()) if mask.any() else 0.
        yy,xx=np.where(mask);h,w=mask.shape
        clipped=bool(len(xx) and (xx.min()<=int(.18*w)+2 or xx.max()>=int(.84*w)-3 or yy.min()<=2 or yy.max()>=h-3))
        quality[name]=dict(pixels=int(mask.sum()),depth_valid_fraction=fraction,clipped=clipped)
    report['segmentation']=quality
    if any(q['pixels']<500 or q['depth_valid_fraction']<cfg.min_depth_fraction for q in quality.values()):
        report.update(registration=dict(status='UNKNOWN',reason='segmentation or measured depth insufficient'),deformation=dict(status='UNKNOWN',primary_region=None))
        return report,dict(reference_mask=refmask,current_mask=curmask)
    try:table=fit_table(reference_xyz,refmask)
    except ValueError as exc:
        report.update(registration=dict(status='UNKNOWN',reason=str(exc)),deformation=dict(status='UNKNOWN',primary_region=None))
        return report,dict(reference_mask=refmask,current_mask=curmask)
    report['table']=table
    if table['rmse_mm']>cfg.max_table_rmse_mm:
        report.update(registration=dict(status='UNKNOWN',reason='table fit residual too large'),deformation=dict(status='UNKNOWN',primary_region=None))
        return report,dict(reference_mask=refmask,current_mask=curmask)
    refcloud=table_cloud(reference_xyz,table);curcloud=table_cloud(current_xyz,table)
    report['anchor_diagnostics']={'reference':{},'current':{}}
    refpts={}
    for k in LANDMARKS:
        audit={};report['anchor_diagnostics']['reference'][k]=audit
        refpts[k]=sample_anchor(refcloud,refmask,annotations['reference'].get(k),audit=audit)
    missing=[k for k,v in refpts.items() if v is None]
    if missing:raise ValueError('reference landmarks lack measured cloth depth: '+','.join(missing))
    # Fixed reference includes all semantic landmarks, but only N/AL/AR align the pose.
    currentpts={}
    for k in ('N','AL','AR'):
        audit={};report['anchor_diagnostics']['current'][k]=audit
        currentpts[k]=sample_anchor(curcloud,curmask,annotations['current'].get(k),tolerance_px=int(cfg.anchor_boundary_tolerance_px),audit=audit)
    valid=[k for k,v in currentpts.items() if v is not None]
    reg=register(np.array([refpts[k][:2] for k in valid]).reshape(-1,2),np.array([currentpts[k][:2] for k in valid]).reshape(-1,2),cfg)
    reg['anchor_names']=valid;reg['anchor_source']=annotations.get('provenance','unspecified upper-body anchors');report['registration']=reg
    refvalid=refmask&np.isfinite(refcloud).all(2);curvalid=curmask&np.isfinite(curcloud).all(2)
    refxy=refcloud[refvalid,:2];curxy=curcloud[curvalid,:2]
    bounds=np.vstack([refxy,curxy])
    if 'rotation' in reg:bounds=np.vstack([bounds,transform(refxy,reg['rotation'],reg['translation_mm'])])
    origin=np.floor((bounds.min(0)-40)/cfg.grid_mm)*cfg.grid_mm
    shape=tuple((np.ceil((bounds.max(0)+40-origin)/cfg.grid_mm).astype(int)+1)[::-1])
    if max(shape)>2000:raise ValueError('metric grid exceeds bounded workspace; check calibration')
    refgrid=raster(refxy,origin,shape,cfg.grid_mm).astype(bool)
    curgrid=raster(curxy,origin,shape,cfg.grid_mm).astype(bool)
    # Bin all measured surface points for coverage. Missing depth never counts as bare table.
    coverage=raster(curcloud[np.isfinite(curcloud).all(2),:2],origin,shape,cfg.grid_mm).astype(bool)
    regions,frame=reference_regions(refgrid,origin,cfg,refpts)
    contour=contour_points(refgrid,origin,cfg.grid_mm)
    pairs={'LEFT_LOWER_BODY':('AL','HL'),'RIGHT_LOWER_BODY':('AR','HR'),'LEFT_SLEEVE':('AL','N'),'RIGHT_SLEEVE':('N','AR'),'HEM':('HL','HR')}
    paths={k:boundary_arc(contour,refpts[a][:2],refpts[b][:2]) for k,(a,b) in pairs.items()}
    report['reference']={'landmarks_table_mm':{k:v.tolist() for k,v in refpts.items()},'boundary_paths':{},'contour_xy_mm':contour.tolist()}
    tree=cKDTree(refcloud[refvalid,:2]);cloud=refcloud[refvalid]
    for name,path in paths.items():
        distances,indices=tree.query(path);points=cloud[indices];lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
        ok=bool(np.all(distances<=cfg.grid_mm*2) and np.all(lengths<=12))
        report['reference']['boundary_paths'][name]=dict(start_landmark=pairs[name][0],end_landmark=pairs[name][1],
            points_3d=points.tolist(),projected_points_xy=path.tolist(),length_3d_mm=float(lengths.sum()) if ok else None,
            projected_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
            status='MEASURED_VISIBLE_REFERENCE' if ok else 'INCOMPLETE_DEPTH',not_material_geodesic=True)
    artifacts=dict(reference_mask=refmask,current_mask=curmask,reference_grid=refgrid,current_grid=curgrid,observed_grid=coverage,origin_mm=origin)
    artifacts.update({'region_'+k:v for k,v in regions.items()})
    # Height grid records only measured samples; no imputation across holes.
    ij=np.floor((curcloud[curvalid,:2]-origin)/cfg.grid_mm).astype(int)
    flat=ij[:,1]*shape[1]+ij[:,0];count=np.bincount(flat,minlength=np.prod(shape));total=np.bincount(flat,weights=curcloud[curvalid,2],minlength=np.prod(shape))
    height=np.full(np.prod(shape),np.nan);np.divide(total,count,out=height,where=count>0);artifacts['height_mm']=height.reshape(shape)
    if reg['status']=='OK':
        R,t=np.array(reg['rotation']),np.array(reg['translation_mm'])
        upper=contour[((contour-frame['center'])@frame['ey'])<0]
        transformed=transform(upper,R,t)
        dist,_=cKDTree(curxy).query(transformed)
        upper_error=float(np.mean(np.minimum(dist,cfg.max_deviation_mm))) if len(dist) else float('inf')
        reg['upper_contour_mean_deviation_mm']=upper_error
        if upper_error>cfg.max_upper_contour_mm:reg.update(status='UNCERTAIN',reason='upper-body contour inconsistent with anchor pose')
        if quality['reference']['clipped']:reg.update(status='UNCERTAIN',reason='reference silhouette is clipped')
    if reg['status']!='OK':
        report['deformation']=dict(status='UNKNOWN',primary_region=None,reason='Unreliable upper-body registration')
        return report,artifacts
    deform,missing=compare(regions,paths,curgrid,coverage,origin,cfg,R,t)
    from .assessment import boundary_profiles, summarize_asymmetry
    report['body_cross_sections']=boundary_profiles(refgrid,curgrid,coverage,origin,frame,refpts,cfg,R,t)
    baseline_primary=deform['primary_region'];trials=[];per_region={k:[] for k in REGIONS}
    center=np.mean([refpts[k][:2] for k in ('N','AL','AR')],axis=0)
    # Residual-aware sensitivity box, not a calibrated statistical confidence region.
    boundary_uncertainty=max((v.get('boundary_offset_xy_mm',0.)+v.get('support_xy_radius_mm',0.)
                             for v in report['anchor_diagnostics']['current'].values()
                             if v['status']=='BOUNDARY_CORRECTED'),default=0.)
    translation=max(cfg.perturb_translation_mm,reg['rmse_mm'],boundary_uncertainty)
    angle=max(cfg.perturb_angle_deg,float(np.rad2deg(np.arctan2(reg['rmse_mm'],reg['spatial_support']['span_mm']))))
    audited=[];trial_metrics=[]
    from itertools import product
    for dx,dy,ang in product((-translation,0.,translation),(-translation,0.,translation),(-angle,0.,angle)):
        a=np.deg2rad(ang);rr=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@R
        tt=t+R@center-rr@center+np.array([dx,dy])
        trial,_=compare(regions,paths,curgrid,coverage,origin,cfg,rr,tt)
        trials.append(trial['primary_region']);trial_metrics.append(trial)
        audited.append(dict(dx_mm=dx,dy_mm=dy,angle_deg=ang,rotation=rr.tolist(),translation_mm=tt.tolist(),
                            regions=trial['regions'],primary_region=trial['primary_region']))
        for k in REGIONS:per_region[k].append(trial['regions'][k]['state'])
    stable=all(x==baseline_primary for x in trials)
    ranges={}
    for k in REGIONS:
        ranges[k]={}
        for metric in ('missing_ratio','missing_area_mm2','largest_missing_component_mm2','contour_deviation_mm'):
            values=[trial['regions'][k][metric] for trial in trial_metrics if trial['regions'][k][metric] is not None]
            ranges[k][metric]=dict(min=min(values) if values else None,max=max(values) if values else None,valid_trials=len(values),total_trials=len(trial_metrics))
    deform['registration_uncertainty']=dict(
        bounds=dict(translation_each_axis_mm=translation,rotation_deg=angle,rotation_center_reference_mm=center.tolist(),
                    boundary_sampling_uncertainty_mm=boundary_uncertainty,
                    rule='translation=max(configured floor, anchor RMSE, boundary sampling uncertainty); angle=max(configured floor, atan(RMSE/span))',
                    interpretation='finite sensitivity stress test, not calibrated confidence bounds'),
        sample_count=len(audited),trials=audited,regional_ranges=ranges,
        asymmetry=summarize_asymmetry(trial_metrics,cfg.asymmetry_margin))
    deform['pose_sensitivity']=dict(primary_stable=stable,perturbed_primary_regions=trials,translation_mm=translation,angle_deg=angle)
    for k,v in deform['regions'].items():
        v['perturbed_states']=per_region[k]
        if any(state!=v['state'] for state in per_region[k]):v.update(state='UNCERTAIN',reliability='POSE_SENSITIVE')
    if any(v['state']=='UNCERTAIN' for v in deform['regions'].values()):deform['status']='PARTIAL'
    if not stable:deform.update(status='UNCERTAIN',primary_region=None)
    elif baseline_primary and deform['regions'][baseline_primary]['state']=='UNCERTAIN':deform.update(status='UNCERTAIN',primary_region=None)
    aligned=warp(refgrid,origin,cfg,R,t)
    artifacts.update(aligned_reference=aligned,missing=missing,excess=curgrid&~aligned)
    report['deformation']=deform
    report['grid']=dict(origin_table_mm=origin.tolist(),resolution_mm=cfg.grid_mm,shape=list(map(int,shape)))
    return report,artifacts
