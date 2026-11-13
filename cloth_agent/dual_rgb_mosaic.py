"""RGB-first planning canvas with explicit original-color and measured-depth maps.

The virtual plane defines display coordinates only. It is never grasp geometry.
"""
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def _feature_align_secondary(primary, secondary):
    """Align a secondary virtual-plane RGB map and all of its provenance.

    The affine is accepted only when image evidence is strong and the change
    stays small.  UV values move with the RGB pixels, so downstream lookup
    still resolves to the original camera pixel.
    """
    a,av,_=primary;b,bv,buv=secondary
    report=dict(method='sift_partial_affine',status='NOT_APPLIED')
    try:
        sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.008,edgeThreshold=16)
        gray=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in (a,b)]
        ka,da=sift.detectAndCompute(gray[0],(av*255).astype(np.uint8))
        kb,db=sift.detectAndCompute(gray[1],(bv*255).astype(np.uint8))
        if da is None or db is None:raise ValueError('insufficient features')
        matches=cv2.BFMatcher(cv2.NORM_L2).knnMatch(db,da,k=2)
        good=[first for first,second in matches if first.distance<.72*second.distance]
        if len(good)<12:raise ValueError(f'only {len(good)} reliable matches')
        source=np.float32([kb[m.queryIdx].pt for m in good]);target=np.float32([ka[m.trainIdx].pt for m in good])
        affine,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,
            ransacReprojThreshold=2.5,maxIters=5000,confidence=.999)
        if affine is None or inliers is None or int(inliers.sum())<8:raise ValueError('robust fit failed')
        keep=inliers.ravel().astype(bool);predicted=cv2.transform(source.reshape(-1,1,2),affine).reshape(-1,2)
        residual=np.linalg.norm(predicted-target,axis=1)[keep]
        scale=float(np.hypot(affine[0,0],affine[0,1]));angle=float(np.degrees(np.arctan2(affine[1,0],affine[0,0])))
        translation=float(np.linalg.norm(affine[:,2]))
        if not .94<=scale<=1.06:raise ValueError(f'scale {scale:.4f} outside limit')
        if abs(angle)>5:raise ValueError(f'rotation {angle:.3f} deg outside limit')
        if translation>45:raise ValueError(f'translation {translation:.2f} px outside limit')
        if np.percentile(residual,95)>4:raise ValueError('feature residual outside limit')
        h,w=av.shape
        warped_rgb=cv2.warpAffine(b,affine,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(24,30,39))
        warped_valid=cv2.warpAffine((bv*255).astype(np.uint8),affine,(w,h),flags=cv2.INTER_NEAREST)>0
        warped_uv=cv2.warpAffine(buv.astype(np.float32),affine,(w,h),flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,borderValue=(-1.,-1.))
        warped_uv[~warped_valid]=-1
        report.update(status='APPLIED',matches=len(good),inliers=int(keep.sum()),
                      median_residual_px=float(np.median(residual)),p95_residual_px=float(np.percentile(residual,95)),
                      scale=scale,rotation_deg=angle,translation_px=translation,affine=affine.tolist())
        return (warped_rgb,warped_valid,warped_uv),report
    except (ValueError,cv2.error) as exc:
        report['reason']=str(exc);return secondary,report


def _robust_color_affine(primary,secondary,support):
    corrected=secondary.astype(float);parameters=[]
    support=np.asarray(support,bool)&(primary.min(2)>15)&(primary.max(2)<245)&(secondary.min(2)>15)&(secondary.max(2)<245)
    for channel in range(3):
        x=secondary[...,channel][support].astype(float);y=primary[...,channel][support].astype(float)
        if len(x)<500:slope,offset=1.,0.
        else:
            stride=max(1,len(x)//20000);x=x[::stride];y=y[::stride]
            fit=least_squares(lambda p:p[0]*x+p[1]-y,[1.,0.],bounds=([.75,-45.],[1.3,45.]),loss='soft_l1',f_scale=8.)
            slope,offset=fit.x
        corrected[...,channel]=corrected[...,channel]*slope+offset;parameters.append([float(slope),float(offset)])
    return np.clip(corrected,0,255).astype(np.uint8),parameters


def render(cameras, *, bounds_xy_mm, plane_z_mm, size=640, primary=0,
           seam_mode='priority', color_reference=None):
    """Each camera supplies rgb, K, world_from_color and native world xyz.

    Executable XYZ always comes from one original measured source.  The
    optional feature-feather mode blends only the planning appearance while
    retaining a dominant original source and original color UV per pixel.
    """
    if len(cameras)<2 or primary not in range(len(cameras)):
        raise ValueError('RGB mosaic requires at least two cameras and a valid primary')
    if seam_mode not in ('priority', 'graphcut', 'feature_feather', 'feature_graphcut'):
        raise ValueError('unknown RGB mosaic seam mode')
    reference=primary if color_reference is None else color_reference
    if reference not in range(len(cameras)):
        raise ValueError('invalid RGB color reference')
    bounds=np.asarray(bounds_xy_mm,float)
    if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[1]<=bounds[0]):
        raise ValueError('bounds must be finite [[xmin,ymin],[xmax,ymax]]')
    if not np.isfinite(plane_z_mm) or size<2:raise ValueError('invalid display plane/grid')
    center=bounds.mean(0);mpp=float(np.max(bounds[1]-bounds[0])/(size-1))
    yy,xx=np.indices((size,size));world=np.stack([center[0]+(xx-(size-1)/2)*mpp,center[1]-(yy-(size-1)/2)*mpp,np.full_like(xx,plane_z_mm,dtype=float)],-1)
    roi=(world[...,:2]>=bounds[0]).all(2)&(world[...,:2]<=bounds[1]).all(2)
    owner=np.full((size,size),-1,np.int8);color_xy=np.full((size,size,2),np.nan,np.float32);image=np.full((size,size,3),[235,235,235],np.uint8)
    from .dual_cloud_render import texture_surface
    maps=[]
    for c in cameras:
        rgb,valid,uv=texture_surface(world,c['rgb'],c['world_from_color'],c['K']);maps.append((rgb,valid&roi,uv))
    feature_report={'status':'NOT_REQUESTED'}
    if seam_mode=='feature_feather':
        maps[1],feature_report=_feature_align_secondary(maps[0],maps[1])
    if seam_mode=='feature_graphcut':
        alignment={}
        for i in range(len(maps)):
            if i!=primary:
                maps[i],alignment[str(i)]=_feature_align_secondary(maps[primary],maps[i])
        feature_report={'status':'PER_SOURCE','sources':alignment}
    # Photometric matching uses bright, non-clipped overlap in the RGB views.
    # It changes color gain, never pixel ownership, coordinates or depth.
    from .dual_cloud_render import align_secondary_color
    color_reports={}
    a,av,_=maps[reference]
    for secondary in range(len(cameras)):
        if secondary==reference:continue
        b,bv,buv=maps[secondary]
        # The yellow mat is not low-saturation. Use bright overlap with the
        # existing held-out gain validation, not a "neutral means table" test.
        table=av&bv&(a.min(2)>90)&(b.min(2)>90)&(a.max(2)<245)&(b.max(2)<245)
        color_support=world.copy();color_support[~table]=np.nan
        adjusted,color_report=align_secondary_color(a,b,color_support,color_support)
        maps[secondary]=(adjusted,bv,buv);color_reports[str(secondary)]=color_report
    color_report=color_reports[str(1-reference)] if len(cameras)==2 else color_reports
    # Build the priority fallback first; extra sources fill wrist coverage holes.
    priority=[primary]+[i for i in range(len(cameras)) if i!=primary]
    for i in reversed(priority):
        rgb,valid,uv=maps[i];image[valid]=rgb[valid];owner[valid]=i;color_xy[valid]=uv[valid]
    seam_report=dict(mode=seam_mode,status='PRIORITY',feature_alignment=feature_report)
    if seam_mode=='feature_graphcut':
        masks=[cv2.UMat((m[1]*255).astype(np.uint8)) for m in maps]
        try:
            cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD').find(
                [m[0].astype(np.float32) for m in maps],[(0,0)]*len(maps),masks)
            for i in reversed(priority):
                rgb,valid,uv=maps[i];chosen=(masks[i].get()>0)&valid
                image[chosen]=rgb[chosen];owner[chosen]=i;color_xy[chosen]=uv[chosen]
            seam_report.update(status='APPLIED',optimized_sources=list(range(len(maps))),
                               blending=False,geometric_warp='bounded_partial_affine')
        except cv2.error as exc:
            seam_report.update(status='PRIORITY_FALLBACK',reason=str(exc))
    elif seam_mode=='graphcut' and np.count_nonzero(maps[0][1]&maps[1][1])>=300:
        # Optimize only the two wrist sources. ZED continues to fill holes.
        # Every selected pixel retains exactly one original RGB/depth source.
        masks=[cv2.UMat((m[1]*255).astype(np.uint8)) for m in maps[:2]]
        try:
            cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD').find(
                [m[0].astype(np.float32) for m in maps[:2]],[(0,0),(0,0)],masks)
            selected=[(mask.get()>0)&maps[i][1] for i,mask in enumerate(masks)]
            # Keep original ownership in any hole in the optimized masks.
            for i in reversed([primary,1-primary] if primary in (0,1) else [0,1]):
                rgb,_,uv=maps[i];valid=selected[i]
                image[valid]=rgb[valid];owner[valid]=i;color_xy[valid]=uv[valid]
            seam_report.update(status='APPLIED',optimized_sources=[0,1],
                               blending=False,geometric_warp=False)
        except cv2.error as exc:
            seam_report.update(status='PRIORITY_FALLBACK',reason=str(exc))
    elif seam_mode=='feature_feather':
        if feature_report.get('status')!='APPLIED':
            seam_report.update(status='PRIORITY_FALLBACK',reason=feature_report.get('reason','feature alignment unavailable'))
        else:
            a,av,auv=maps[0];b,bv,buv=maps[1];overlap=av&bv
            b,affine_color=_robust_color_affine(a,b,overlap);maps[1]=(b,bv,buv)
            image[bv&~av]=b[bv&~av]
            def edge_distance(mask):
                padded=np.pad(mask.astype(np.uint8),1,constant_values=0)
                return cv2.distanceTransform(padded,cv2.DIST_L2,5)[1:-1,1:-1]
            distance_a=edge_distance(av);distance_b=edge_distance(bv)
            denominator=distance_a+distance_b
            weight_a=np.divide(distance_a,denominator,out=np.zeros_like(distance_a),where=denominator>0)
            image[overlap]=np.clip(a[overlap].astype(float)*weight_a[overlap,None]+b[overlap].astype(float)*(1-weight_a[overlap,None]),0,255).astype(np.uint8)
            choose_a=overlap&(weight_a>=.5);choose_b=overlap&~choose_a
            owner[choose_a]=0;color_xy[choose_a]=auv[choose_a]
            owner[choose_b]=1;color_xy[choose_b]=buv[choose_b]
            seam_report.update(status='APPLIED',blending=True,blend_scope='RGB_APPEARANCE_ONLY',
                               executable_source_policy='dominant_distance_weight',robust_color_affine=affine_color)
    native_xy=np.full((size,size,2),-1,np.int32);measured=np.full((size,size,3),np.nan,np.float32)
    for i,c in enumerate(cameras):
        points=np.asarray(c['xyz']);ny,nx=np.nonzero(np.isfinite(points).all(2));pts=points[ny,nx];T=np.asarray(c['world_from_color']);cam=(pts-T[:3,3])@T[:3,:3];good=cam[:,2]>0;cam=cam[good];pts=pts[good];nx=nx[good];ny=ny[good]
        if not len(cam):continue
        projected=cam@np.asarray(c['K']).T;uv=projected[:,:2]/projected[:,2:];pix=np.rint(uv).astype(int);h,w=c['rgb'].shape[:2];inside=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h);uv=uv[inside];pix=pix[inside];cam=cam[inside];pts=pts[inside];nx=nx[inside];ny=ny[inside]
        if not len(cam):continue
        # At each original RGB pixel keep the nearest observed surface, so a
        # background measurement cannot win behind an observed foreground.
        order=np.argsort(cam[:,2],kind='stable');_,first=np.unique((pix[order,1]*w+pix[order,0]),return_index=True);keep=order[first];uv=uv[keep];pts=pts[keep];nx=nx[keep];ny=ny[keep]
        y,x=np.nonzero(owner==i)
        if not len(x):continue
        distance,index=cKDTree(uv).query(color_xy[y,x]);ok=distance<=1.0;y=y[ok];x=x[ok];index=index[ok]
        measured[y,x]=pts[index];native_xy[y,x]=np.c_[nx[index],ny[index]]
    return dict(rgb=image,source_camera=owner,source_color_xy=color_xy,source_xy=native_xy,world_xyz_mm=measured,appearance_valid=owner>=0,depth_valid=np.isfinite(measured).all(2),center=center,mm_per_pixel=mpp,
                report={'method':'original_rgb_virtual_plane_mosaic','color_alignment':color_report,'color_reference':reference,'seam':seam_report,'plane_z_mm_display_only':float(plane_z_mm),'rgb_primary_camera':primary,'depth_modified':False,'depth_holes_affect_rgb':False,'color_inpainting':False,'native_depth_association_max_color_px':1.0,'limitation':'Virtual-plane projection and feature feathering are appearance-only. Executable XYZ and source UV remain original measured camera data.'})
