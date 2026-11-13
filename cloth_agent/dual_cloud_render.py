"""Continuous appearance rendering of measured RGB-D surfaces.

Triangle interpolation is for appearance ONLY. It never creates executable
measurements; the measured topdown/source maps remain a separate artifact.
"""
import cv2
import numpy as np


def surface_raster(xyz, valid, center, span, *, size=640, supersample=2,
                   max_edge_mm=12., max_z_step_mm=10.):
    """Rasterize adjacent native depth triangles, skipping discontinuities.

    No triangulation across missing source pixels, no extrapolation. Returns
    interpolated world positions for texture lookup, not measured grasp XYZ.
    """
    scale=supersample
    h,w=valid.shape
    ids=np.arange(h*w).reshape(h,w)
    a,b,c,d=ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
    triangles=np.concatenate([np.stack([a,b,c],-1).reshape(-1,3),
                              np.stack([b,d,c],-1).reshape(-1,3)])
    triangles=triangles[np.asarray(valid).ravel()[triangles].all(1)]
    points=np.asarray(xyz).reshape(-1,3)[triangles].astype(np.float32)
    good=np.ptp(points[:,:,2],axis=1)<=max_z_step_mm
    for i,j in [(0,1),(1,2),(2,0)]:
        good&=np.linalg.norm(points[:,i]-points[:,j],axis=1)<=max_edge_mm
    points=points[good]
    uv=np.empty(points.shape[:2]+(2,),np.float32)
    uv[:,:,0]=(320+(points[:,:,0]-center[0])/span*520)*scale
    uv[:,:,1]=(320-(points[:,:,1]-center[1])/span*520)*scale
    lo=np.ceil(uv.min(1)).astype(int);hi=np.floor(uv.max(1)).astype(int)
    denom=(uv[:,1,1]-uv[:,2,1])*(uv[:,0,0]-uv[:,2,0])+(uv[:,2,0]-uv[:,1,0])*(uv[:,0,1]-uv[:,2,1])
    n=size*scale
    keep=(np.abs(denom)>1e-7)&(lo<=hi).all(1)&(lo>=0).all(1)&(hi<n).all(1)&((hi-lo)<=16).all(1)
    points,uv,lo,hi,denom=(x[keep] for x in (points,uv,lo,hi,denom))
    rendered=np.full((n*n,3),np.nan,np.float32);zbuf=np.full(n*n,-np.inf,np.float32)
    if not len(points):return rendered.reshape(n,n,3)
    max_dx,max_dy=(hi-lo).max(0)
    for dy in range(max_dy+1):
        for dx in range(max_dx+1):
            pick=(lo[:,0]+dx<=hi[:,0])&(lo[:,1]+dy<=hi[:,1])
            t=np.flatnonzero(pick);u=uv[t];x=lo[t,0]+dx;y=lo[t,1]+dy
            aa=((u[:,1,1]-u[:,2,1])*(x-u[:,2,0])+(u[:,2,0]-u[:,1,0])*(y-u[:,2,1]))/denom[t]
            bb=((u[:,2,1]-u[:,0,1])*(x-u[:,2,0])+(u[:,0,0]-u[:,2,0])*(y-u[:,2,1]))/denom[t]
            cc=1-aa-bb;inside=(aa>=-1e-5)&(bb>=-1e-5)&(cc>=-1e-5)
            t=t[inside];x=x[inside];y=y[inside]
            p=points[t,0]*aa[inside,None]+points[t,1]*bb[inside,None]+points[t,2]*cc[inside,None]
            idx=y*n+x
            # Resolve all triangles in this batch deterministically before write.
            order=np.argsort(p[:,2],kind='stable');idx=idx[order];p=p[order]
            _,rev=np.unique(idx[::-1],return_index=True);win=len(idx)-1-rev
            idx=idx[win];p=p[win];near=p[:,2]>zbuf[idx];idx=idx[near];p=p[near]
            rendered[idx]=p;zbuf[idx]=p[:,2]
    return rendered.reshape(n,n,3)


def texture_surface(world, rgb, world_from_camera, intrinsics, *, scale=1., scale_origin=None):
    """Sample original color image at rendered surface positions (bilinear)."""
    valid=np.isfinite(world).all(2)
    T=np.asarray(world_from_camera);K=np.asarray(intrinsics)
    raw=world if scale==1 else (world-np.asarray(scale_origin))/scale+np.asarray(scale_origin)
    cam=(raw-T[:3,3])@T[:3,:3]
    valid&=cam[:,:,2]>0
    projected=cam@K.T
    with np.errstate(invalid='ignore',divide='ignore'):
        uv=(projected[:,:,:2]/projected[:,:,2:]).astype(np.float32)
    valid&=(uv[:,:,0]>=0)&(uv[:,:,0]<=rgb.shape[1]-1)&(uv[:,:,1]>=0)&(uv[:,:,1]<=rgb.shape[0]-1)
    uv[~valid]=-1
    color=cv2.remap(rgb,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    color[~valid]=[24,30,39]
    return color,valid,uv


def align_secondary_color(primary, secondary, primary_xyz, secondary_xyz):
    """Gain-only correction fitted on table overlap, checked on all colors.

    Zero remains zero: never use an additive table offset to brighten black cloth.
    """
    overlap=np.isfinite(primary_xyz).all(2)&np.isfinite(secondary_xyz).all(2)
    overlap&=np.abs(primary_xyz[:,:,2]-secondary_xyz[:,:,2])<=5
    yy,xx=np.indices(overlap.shape);train=overlap&((xx//48+yy//48)%2==0)
    test=overlap&~train
    bright=train&(primary.min(2)>90)&(secondary.min(2)>90)&(primary.max(2)<245)&(secondary.max(2)<245)
    info=dict(method='gain_only_current_overlap',fit_pixels=int(bright.sum()),validation_pixels=int(test.sum()),offset_rgb=[0.,0.,0.])
    gain=np.ones(3)
    if bright.sum()>=300 and test.sum()>=300:
        proposed=np.median(primary[bright].astype(float)/secondary[bright],axis=0)
        a=secondary[test].astype(float);b=primary[test].astype(float)
        corrected=np.clip(a*proposed,0,255)
        old=float(np.mean(np.abs(a-b)));new=float(np.mean(np.abs(corrected-b)))
        dark=(a.max(1)<65)&(b.max(1)<65)
        dark_old=float(np.mean(np.abs(a[dark]-b[dark]))) if dark.any() else 0.
        dark_new=float(np.mean(np.abs(corrected[dark]-b[dark]))) if dark.any() else 0.
        accept=bool(((proposed>=.5)&(proposed<=2)).all() and new<old and (dark.sum()<100 or dark_new<=dark_old+2.))
        info.update(before_mae=old,after_mae=new,dark_before_mae=dark_old,dark_after_mae=dark_new,accepted=accept)
        if accept:gain=proposed
    else:info['accepted']=False
    info['gain_rgb']=gain.tolist()
    return np.clip(np.rint(secondary*gain),0,255).astype(np.uint8),info


def small_enclosed_holes(coverage, max_area):
    n,labels,stats,_=cv2.connectedComponentsWithStats((~coverage).astype(np.uint8),8)
    accept=np.zeros(n,bool);h,w=coverage.shape
    for i in range(1,n):
        x,y,ww,hh,area=stats[i]
        accept[i]=area<=max_area and x>0 and y>0 and x+ww<w and y+hh<h
    return accept[labels]



def upper_surface_patch(coverage,worlds,silhouettes):
    """Select coherent upper cloth, not a noisy fringe beside shared cloth."""
    difference=worlds[0][:,:,2]-worlds[1][:,:,2]
    candidate=coverage[0]&coverage[1]&silhouettes[0]&~silhouettes[1]&(difference>5.)
    primary_border=cv2.dilate(silhouettes[1].astype(np.uint8),np.ones((17,17),np.uint8))>0
    core=candidate&~primary_border&(difference>10.)
    n,labels,_,_=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8)
    core_counts=np.bincount(labels[core],minlength=n)
    accept=core_counts>=64;accept[0]=False
    return accept[labels]

def render_appearance(observations, native_valid, center, span, config, *, garment_profile=None, seam_mode=None):
    """Render coherent RGB; preserve a separate, explicit appearance provenance.

    RS is the primary color view. Secondary fills only missing primary surface.
    Interpolation/inpaint metadata must never be used as executable depth.
    """
    from PIL import Image
    images=[];worlds=[];coverage=[]
    for i,(source,valid) in enumerate(zip(observations,native_valid)):
        world=surface_raster(source.xyz,valid,center,span)
        # A native triangle may be unavailable even when this output cell has
        # an actual point (e.g. sparse valid depth beside a fold). Keep that
        # measurement as a color fallback instead of turning it into a hole.
        points=source.xyz[valid]
        px=np.rint(320+(points[:,0]-center[0])/span*520).astype(int)
        py=np.rint(320-(points[:,1]-center[1])/span*520).astype(int)
        inside=(px>=0)&(px<640)&(py>=0)&(py<640)
        points=points[inside];flat=py[inside]*640+px[inside]
        order=np.argsort(points[:,2],kind='stable');flat=flat[order];points=points[order]
        _,rev=np.unique(flat[::-1],return_index=True);win=len(flat)-1-rev
        fallback=np.full((640*640,3),np.nan,np.float32);fallback[flat[win]]=points[win]
        fallback=np.repeat(np.repeat(fallback.reshape(640,640,3),2,axis=0),2,axis=1)
        missing=~np.isfinite(world).all(2);world[missing]=fallback[missing]
        if i==1:
            rgb=np.asarray(Image.open(source.directory/'color_rgb.png').convert('RGB'))
            transform=np.asarray(config['calibration']['world_from_realsense_color_mm'])
            intrinsics=np.asarray(config['calibration']['realsense_K'])
        else:
            rgb=np.asarray(Image.open(source.image).convert('RGB'))
            transform=np.asarray(source.meta['world_from_camera_mm'])
            intrinsics=np.asarray(source.meta['intrinsics'])
        scale=config['depth_scale']['zed' if i==0 else 'realsense']
        color,valid_image,_=texture_surface(world,rgb,transform,intrinsics,scale=scale,
                    scale_origin=np.asarray(source.meta['world_from_camera_mm'])[:3,3])
        world[~valid_image]=np.nan
        images.append(color);worlds.append(world);coverage.append(valid_image)
    secondary,color_info=align_secondary_color(images[1],images[0],worlds[1],worlds[0])
    fill=~coverage[1]&coverage[0]
    upper_cloth=np.zeros(fill.shape,bool)
    protected_primary=np.zeros(fill.shape,bool)
    # ``auto`` keeps the existing appearance protection during rendering;
    # execution-time mask selection is performed later from the fused view.
    if garment_profile in ('black','auto'):
        from cloth_agent.cloth_harness.cv_harness.segmentation import segment_dark_garment
        silhouettes=[]
        for color,known in zip(images,coverage):
            small=cv2.resize(color,(640,640),interpolation=cv2.INTER_AREA)
            unknown=cv2.resize(known.astype(np.uint8),(640,640),interpolation=cv2.INTER_NEAREST)==0
            try:mask,_=segment_dark_garment(small,unknown)
            except (ValueError,cv2.error):mask=np.zeros((640,640),np.uint8)
            silhouettes.append(cv2.resize(mask,fill.shape[::-1],interpolation=cv2.INTER_NEAREST)>0)
        # If the primary sees table while the other camera sees measured
        # raised cloth, retain that upper cloth. Never switch colors inside
        # the shared garment merely because of small depth disagreement.
        upper_cloth=upper_surface_patch(coverage,worlds,silhouettes)
        protected_primary=coverage[1]&silhouettes[1]&~silhouettes[0]
        fill|=upper_cloth
    seam_report={'method':'primary_hole_fill'}
    color_corrected=np.zeros(fill.shape,bool)
    if seam_mode=='local_gain':
        from cloth_agent.dual_cloud_seam import local_appearance_seam
        fill,secondary,color_corrected,seam_report=local_appearance_seam(images[1],secondary,coverage,upper_cloth,protected_primary)
    elif seam_mode is not None:
        raise ValueError(f'unknown appearance seam mode: {seam_mode}')
    rgb=images[1].copy();rgb[fill]=secondary[fill]
    world=worlds[1].copy();world[fill]=worlds[0][fill]
    observed=coverage[0]|coverage[1]
    owner=np.full(observed.shape,-1,np.int8);owner[coverage[1]]=1;owner[fill]=0
    # Limit texture completion to enclosed holes <=512 output pixels. This is
    # color-only reconstruction, logged separately from measured support.
    inferred=small_enclosed_holes(observed,2048)
    rgb=cv2.inpaint(rgb,inferred.astype(np.uint8),3,cv2.INPAINT_TELEA)
    out_rgb=cv2.resize(rgb,(640,640),interpolation=cv2.INTER_AREA)
    out_coverage=cv2.resize((observed|inferred).astype(np.float32),(640,640),interpolation=cv2.INTER_AREA)>=.99
    out_inferred=cv2.resize(inferred.astype(np.float32),(640,640),interpolation=cv2.INTER_AREA)>0
    # Center subpixel defines provenance and estimated display surface. Mixed
    # subpixel colors are antialiasing, not averaged execution XYZ.
    out_owner=owner[::2,::2];out_z=world[::2,::2,2]
    conflict=coverage[0]&coverage[1]&(worlds[0][:,:,2]-worlds[1][:,:,2]>5.)
    report=dict(method='native_surface_triangles_original_rgb_rs_primary',supersample=2,
        triangle_max_edge_mm=12.,triangle_max_z_step_mm=10.,primary='realsense',
        upper_cloth_min_core_subpixels=64,upper_cloth_core_height_mm=10.,upper_cloth_primary_exclusion_radius_output_px=4,
        color_alignment=color_info,seam=seam_report,texture_inpaint_output_pixels=int(out_inferred.sum()),
        max_texture_hole_area_output_pixels=512,measured_depth_modified=False,
        display_surface_is_interpolated=True,secondary_upper_cloth_subpixels=int(upper_cloth.sum()),secondary_higher_conflict_subpixels=int(conflict.sum()))
    return dict(rgb=out_rgb,coverage=out_coverage,texture_inferred=out_inferred,
                color_corrected=cv2.resize(color_corrected.astype(np.float32),(640,640),interpolation=cv2.INTER_AREA)>0,
                source_camera=out_owner,display_z_mm=out_z,report=report)
