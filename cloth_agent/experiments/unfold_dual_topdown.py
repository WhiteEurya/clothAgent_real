"""Dual measured top-down with per-pixel source ownership, not an RGB-only overlay."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from cloth_agent.pixel_mapping_visualization import PixelMappingView
from cloth_agent.dual_arm.observation import Observation,DepthUnavailableError,sha
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_cloud_color import match_colors
from cloth_agent.depth_stability import table_depth_mask


def sources(before):
    info=before.meta['dual_cloud'];directory=before.directory/info['secondary_directory']
    if sha(directory/'observation.json')!=info['secondary_observation_sha256']:
        raise DualArmError('secondary observation metadata changed')
    result=[before,Observation.load(directory)]
    for entry in info.get('additional_sources',[]):
        extra=before.directory/entry['directory']
        if sha(extra/'observation.json')!=entry['observation_sha256']:
            raise DualArmError('additional observation metadata changed')
        result.append(Observation.load(extra))
    return result


def table_projection(source, obs, cfg, roi):
    """Crop each source in its own camera, never by the other camera's FOV.

    Old saved configurations without per-camera regions retain their original
    projection for reproducible offline comparisons.
    """
    regions=roi.get('table_rois_by_serial')
    if regions is None:
        return (table_depth_mask(obs[0].xyz.shape[:2],roi['table_polygon_px']),
                np.asarray(obs[0].meta['world_from_camera_mm']),
                np.asarray(obs[0].meta['intrinsics']))
    serial=str(source.meta['camera_serial'])
    if serial not in regions:
        raise DualArmError('missing camera-specific table ROI for '+serial)
    region=regions[serial];width,height=region['image_size_wh']
    mask=table_depth_mask((height,width),region['polygon_px'])
    if region['frame']=='native':
        if source.xyz.shape[:2]!=(height,width):
            raise DualArmError('table ROI native resolution mismatch for '+serial)
        return mask,np.asarray(source.meta['world_from_camera_mm']),np.asarray(source.meta['intrinsics'])
    if region['frame']=='realsense_color':
        cal=cfg['calibration']
        if serial!=str(cal['realsense_serial']) or [width,height]!=cfg['realsense_color_size_wh']:
            raise DualArmError('table ROI color camera mismatch for '+serial)
        return mask,np.asarray(cal['world_from_realsense_color_mm']),np.asarray(cal['realsense_K'])
    raise DualArmError('unknown table ROI projection frame for '+serial)


def reconstruct(before,folder,roi):
    if roi.get('appearance_render')=='rgb_mosaic':
        return reconstruct_rgb_mosaic(before,folder,roi)
    folder=Path(folder)/'topdown';folder.mkdir(parents=True,exist_ok=True)
    obs=sources(before);cfg=before.meta['dual_cloud']['config'];clouds=[];colors=[];pixels=[];native_valid=[]
    for source in obs:
        mask,T,K=table_projection(source,obs,cfg,roi)
        rgb=np.asarray(Image.open(source.image).convert('RGB'));xyz=source.xyz
        valid=np.isfinite(xyz).all(2)&(xyz[:,:,2]>=roi.get('display_min_world_z_mm',-20))&(xyz[:,:,2]<=400)
        y,x=np.nonzero(valid);pts=xyz[y,x];cam=(pts-T[:3,3])@T[:3,:3];front=cam[:,2]>0
        uv=np.zeros((len(cam),2),int);uv[front]=np.rint((cam[front]@K.T)[:,:2]/cam[front,2:]).astype(int)
        inside=front&(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
        ids=np.flatnonzero(inside);inside[ids]&=mask[uv[ids,1],uv[ids,0]]
        y,x=y[inside],x[inside]
        kept=np.zeros(valid.shape,bool);kept[y,x]=True;native_valid.append(kept)
        clouds.append(xyz[y,x]);colors.append(rgb[y,x]);pixels.append(np.c_[x,y])
    if not sum(map(len,clouds)):raise DualArmError('no measured table points from either fusion camera')
    def make(x,c):return PixelMappingView(c[None],x[None],np.ones((1,len(x)),bool),show_xy_grid=False,max_points=None,frame_label='WORLD XY TOP-DOWN')
    union=make(np.concatenate(clouds),np.concatenate(colors));views=[]
    for x,c in zip(clouds,colors):
        if len(x):
            v=make(x,c);v.center=union.center;v.span=union.span;v._xy_image()
        else:
            from types import SimpleNamespace
            v=SimpleNamespace(measured_image=Image.fromarray(np.full((640,640,3),[24,30,39],np.uint8)),projected_xyz=np.full((640,640,3),np.nan))
        views.append(v)
    gain,offset,color_report=match_colors([np.asarray(v.measured_image) for v in views],[v.projected_xyz for v in views],cfg['color_fallback'])
    view=union
    # Both views use exactly the union grid. Within the existing 5 mm
    # same-surface tolerance prefer one camera consistently, instead of
    # letting tiny depth noise switch color/source on every cell.
    valid=[np.isfinite(v.projected_xyz).all(2) for v in views]
    overlap=valid[0]&valid[1]
    dz=views[1].projected_xyz[:,:,2]-views[0].projected_xyz[:,:,2]
    use_rs=valid[1]&(~valid[0]|(dz>5.))
    known=valid[0]|valid[1]
    cameras=np.full(known.shape,-1,np.int8)
    cameras[valid[0]]=0;cameras[use_rs]=1
    xy=np.full((*known.shape,2),-1,np.int32)
    fused=np.full((*known.shape,3),[24,30,39],np.uint8)
    measured=np.full((*known.shape,3),np.nan,np.float32)
    unmatched=fused.copy()
    for i,v in enumerate(views):
        chosen=cameras==i
        if not chosen.any():continue
        indices=v.source_xy[:,:,0][chosen]
        xy[chosen]=pixels[i][indices]
        measured[chosen]=v.projected_xyz[chosen]
        original=np.asarray(v.measured_image)[chosen]
        unmatched[chosen]=original
        fused[chosen]=original if i==0 else np.clip(np.rint(original*gain+offset),0,255).astype(np.uint8)
    appearance=None
    if roi.get('appearance_render')=='surface_rgb':
        from cloth_agent.dual_cloud_render import render_appearance
        appearance=render_appearance(obs,native_valid,view.center,view.span,cfg,garment_profile=roi.get('appearance_garment_profile',roi.get('segmentation_profile')),seam_mode=roi.get('appearance_seam'))
        Image.fromarray(fused).save(folder/'point_rgb.png')
        Image.fromarray(appearance['rgb']).save(folder/'rendered_rgb.png')
        if 'color_corrected' in appearance:
            np.save(folder/'appearance_color_corrected.npy',appearance['color_corrected'],allow_pickle=False)
        for name,key in [('appearance_valid','coverage'),('appearance_texture_inferred','texture_inferred'),
                         ('appearance_source_camera','source_camera'),('appearance_display_z_mm','display_z_mm')]:
            np.save(folder/(name+'.npy'),appearance[key],allow_pickle=False)
        # Display interpolation is not a depth source. Resolve only original
        # measured cells on the same displayed surface (existing 5 mm tolerance).
        cameras[:]=-1
        matches=[valid[i]&np.isfinite(appearance['display_z_mm'])&
                 (np.abs(views[i].projected_xyz[:,:,2]-appearance['display_z_mm'])<=5.) for i in (0,1)]
        for i in (0,1):
            cameras[matches[i]&(appearance['source_camera']==i)]=i
        for i in (1,0):cameras[(cameras<0)&matches[i]]=i
        xy[:]=-1;measured[:]=np.nan
        for i in (0,1):
            chosen=cameras==i
            if not chosen.any():continue
            indices=views[i].source_xy[:,:,0][chosen]
            xy[chosen]=pixels[i][indices];measured[chosen]=views[i].projected_xyz[chosen]
        fused=appearance['rgb']
    view.measured_image=Image.fromarray(fused);view.projected_xyz=measured
    view.measured_image.save(folder/'planning_rgb.png');view.measured_image.save(folder/'rgb.png')
    Image.fromarray(unmatched).save(folder/'unmatched_rgb.png')
    np.save(folder/'source_camera.npy',cameras);np.save(folder/'source_xy.npy',xy);np.save(folder/'world_xyz_mm.npy',view.projected_xyz)
    for name,v in zip(['zed','realsense'],views):v.measured_image.save(folder/f'{name}_rgb.png')
    z_delta=dz[overlap]
    report={'fusion_quality':{'overlap_cells':int(overlap.sum()),'same_surface_cells':int((overlap&(np.abs(dz)<=5)).sum()),'abs_z_difference_mm_p50_p90_p95':np.percentile(np.abs(z_delta),[50,90,95]).tolist() if len(z_delta) else []},'status':'AVAILABLE','mode':'DUAL_CAMERA_MEASURED_FUSION','role':'primary','coordinate_output':'topdown_pixels',
        'observation_id':before.meta['observation_id'],'source_observation_ids':[o.meta['observation_id'] for o in obs],
        'color_alignment':color_report,'center_world_xy_mm':view.center.tolist(),'span_mm':view.span,'mm_per_pixel':view.span/view.plot_size,
        'source_camera_ids':{'0':'zed','1':'realsense_native_depth'},'depth_scales':cfg['depth_scale'],
        'valid_points_per_camera':[len(x) for x in clouds],'holes_filled':False,'global_depth_coverage_gate':False,'projection':'measured ZED preferred within 5 mm same-surface tolerance; otherwise higher measured surface; no depth averaging or filling',
        'limitation':'Residual inter-camera depth disagreement remains; selected measured points require local same-surface support from the fused clouds.'}
    report['table_crop']={'mode':'independent_camera_roi_union' if 'table_rois_by_serial' in roi else 'legacy_shared_zed_roi',
        'regions':roi.get('table_rois_by_serial',{}),
        'zed_only_cells':int((valid[0]&~valid[1]).sum()),
        'realsense_only_cells':int((valid[1]&~valid[0]).sum()),
        'overlap_cells':int(overlap.sum())}
    if 'segmentation_profile' in roi:report['segmentation_profile']=roi['segmentation_profile']
    if appearance is not None:
        report['appearance']=appearance['report']
        report['point_projection_color_alignment']=report['color_alignment']
        report['color_alignment']=appearance['report'].get('color_alignment',{})
        report['depth_holes_filled']=False
        report['color_holes_completed']=bool(appearance['texture_inferred'].any())
        report['projection']='continuous RGB appearance with separate original measured XYZ/source maps; no executable depth interpolation'
        report['measured_cells']=int(np.isfinite(measured).all(2).sum())
    (folder/'view.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return DualPrimaryView(before,folder,obs)


class DualPrimaryView:
    plan_coordinates='topdown'
    def __init__(self,before,folder,obs):
        self._resolved={};self._local_matches={};self._samples={};self._trees={};self._depth_failures={};self._audit={}
        self.source=before;self.sources=obs;self.directory=Path(folder);self.image=self.directory/'planning_rgb.png'
        metadata=json.loads((self.directory/'view.json').read_text(encoding='utf-8')) if (self.directory/'view.json').is_file() else {}
        self._mm_per_pixel=metadata.get('mm_per_pixel')
        if 'segmentation_profile' in metadata:self.segmentation_profile=metadata['segmentation_profile']
        self.xyz=np.load(self.directory/'world_xyz_mm.npy');self.source_xy=np.load(self.directory/'source_xy.npy');self.camera_ids=np.load(self.directory/'source_camera.npy');self.unknown_mask=~np.isfinite(self.xyz).all(2)
    def _associate_hole(self, key):
        """Resolve only small appearance holes to original, supported measurements.

        Never fill XYZ or average layers. Ambiguous local surfaces stay invalid.
        """
        if key in self._local_matches:return self._local_matches[key]
        x,y=key
        inferred_path=self.directory/'appearance_texture_inferred.npy'
        if not inferred_path.exists() or not self._mm_per_pixel or not self.image.exists():
            raise DepthUnavailableError('selected topdown point has no measured source depth')
        inferred=np.load(inferred_path)
        if not inferred[y,x]:
            raise DepthUnavailableError('selected topdown point has no measured source depth; not a small appearance hole')
        rgb=np.asarray(Image.open(self.image).convert('RGB'),dtype=float)
        radius=min(2.,5./self._mm_per_pixel)
        candidates=[]
        for yy in range(max(0,y-2),min(self.xyz.shape[0],y+3)):
            for xx in range(max(0,x-2),min(self.xyz.shape[1],x+3)):
                distance=float(np.hypot(xx-x,yy-y))
                if distance>radius or self.unknown_mask[yy,xx]:continue
                # Avoid snapping cloth onto visibly different tabletop pixels.
                if np.max(np.abs(rgb[yy,xx]-rgb[y,x]))>30:continue
                candidates.append((distance,xx,yy,float(self.xyz[yy,xx,2])))
        record=dict(status='LOCAL_ASSOCIATION_REJECTED',requested_topdown_pixel=list(key),
                    radius_px=radius,max_shift_mm=5.,candidate_count=len(candidates),
                    depth_filled=False)
        def reject(reason):
            record['reason']=reason;self._audit[str(key)]=record;self._save_depth_audit()
            raise DepthUnavailableError('local measured-point association failed: '+reason)
        if len(candidates)<3:reject('insufficient nearby measured surface')
        zs=np.array([c[3] for c in candidates]);median=float(np.median(zs))
        same=np.abs(zs-median)<=5.
        record.update(same_surface_count=int(same.sum()),candidate_z_mm=zs.tolist())
        if same.sum()<3 or same.mean()<.75:
            reject('ambiguous neighboring cloth layers')
        # Validate each candidate using the unchanged source identity and fused
        # support checks. source_ref/sample/reference all resolve consistently.
        for distance,xx,yy,z in sorted(c for c,ok in zip(candidates,same) if ok):
            try:anchor=self.sample([xx,yy],None)
            except DepthUnavailableError:continue
            self._local_matches[key]=(xx,yy)
            record.update(status='LOCAL_MEASURED_POINT_ASSOCIATION',
                          resolved_topdown_pixel=[xx,yy],shift_px=distance,
                          shift_mm=distance*self._mm_per_pixel,world_mm=anchor.tolist(),
                          measured_point_audit_key=str((xx,yy)))
            self._audit[str(key)]=record;self._save_depth_audit()
            print(f'[局部实测点关联] {list(key)} -> {[xx,yy]}；偏移{record["shift_mm"]:.2f} mm；未填补深度',flush=True)
            return (xx,yy)
        reject('nearby measurements failed existing fused support checks')

    def source_ref(self,pixel):
        x,y=map(int,pixel)
        if not (0<=x<self.xyz.shape[1] and 0<=y<self.xyz.shape[0]):
            raise DepthUnavailableError('selected topdown point has no measured source depth')
        if self.unknown_mask[y,x]:x,y=self._associate_hole((x,y))
        i,xy=self._resolved.get((x,y),(int(self.camera_ids[y,x]),self.source_xy[y,x].tolist()))
        return self.sources[i],xy,i
    def source_pixel(self,pixel):return self.source_ref(pixel)[1]
    def sample(self,pixel,spread):
        """Use the selected measured fused point, supported in world-space.

        No native-camera 5x5/18-pixel gate and no pixel/depth filling. Reuse
        the existing 3 mm XY / 5 mm Z same-surface tolerances across cameras.
        Three distinct local XY measurements support a point; duplicates from
        the two cameras cannot turn an isolated point into a supported surface.
        """
        key=tuple(map(int,pixel))
        if key in self._depth_failures:
            raise DepthUnavailableError(self._depth_failures[key])
        if key in self._samples:return self._samples[key].copy()
        source,xy,owner=self.source_ref(pixel)
        if key in self._local_matches:
            return self.sample(self._local_matches[key],spread)
        anchor=np.asarray(self.xyz[key[1],key[0]],float)
        original=np.asarray(source.xyz[xy[1],xy[0]],float)
        if not np.isfinite(original).all() or not np.allclose(original,anchor,atol=1e-5,rtol=0):
            raise DepthUnavailableError('fused point does not match its measured source')
        from scipy.spatial import cKDTree
        support=[];counts={}
        for i,observation in enumerate(self.sources):
            if i not in self._trees:
                points=observation.xyz[np.isfinite(observation.xyz).all(2)]
                self._trees[i]=(cKDTree(points[:,:2]),points)
            tree,points=self._trees[i]
            ids=tree.query_ball_point(anchor[:2],3.)
            local=points[ids]
            local=local[np.abs(local[:,2]-anchor[2])<=5.]
            counts[str(i)]=len(local)
            support.extend(local)
        points=np.asarray(support,float).reshape(-1,3)
        unique=len(np.unique(np.round(points[:,:2],1),axis=0))
        record=dict(status='FUSED_MEASURED_POINT',world_mm=anchor.tolist(),
                    source_camera_id=owner,source_pixel_xy=xy,
                    support_per_camera=counts,distinct_xy_support=unique,
                    xy_radius_mm=3.,same_surface_z_tolerance_mm=5.,minimum_distinct_xy_support=3,
                    native_patch_gate=False,depth_filled=False)
        if unique<3:
            record['status']='INSUFFICIENT_FUSED_SUPPORT'
            message=f'fused point {list(key)} has only {unique} distinct same-surface measurements within 3 mm XY'
            self._depth_failures[key]=message
            self._audit[str(key)]=record;self._save_depth_audit()
            raise DepthUnavailableError(message)
        self._audit[str(key)]=record;self._save_depth_audit()
        self._samples[key]=anchor.copy()
        return anchor.copy()
    def _save_depth_audit(self):
        (self.directory/'depth_resolution.json').write_text(json.dumps(self._audit,indent=2),encoding='utf-8')
    def raw_plan(self,plan):
        # Keep top-down coordinates; prepare_action resolves each pixel independently.
        self.source.planning_view=self
        return dict(plan)
    def reference(self,pixel):
        source,xy,i=self.source_ref(pixel)
        ref={'camera_id':i,'camera_serial':source.meta['camera_serial'],'pixel_xy':xy,'observation_id':source.meta['observation_id']}
        key=tuple(map(int,pixel))
        if key in self._local_matches:
            ref.update(requested_topdown_pixel=list(key),resolved_topdown_pixel=list(self._local_matches[key]))
        return ref
    def orientation_inputs(self,pixel,angle):
        a=self.sample(pixel,None);source,xy,_=self.source_ref(pixel);rad=np.radians(angle)
        b=a+10*np.array([np.cos(rad),-np.sin(rad),0.]);T=np.asarray(source.meta['world_from_camera_mm']);K=np.asarray(source.meta['intrinsics'])
        cam=(np.array([a,b])-T[:3,3])@T[:3,:3]
        if np.any(cam[:,2]<=0):raise DualArmError('grasp direction behind source camera')
        uv=cam@K.T;uv=uv[:,:2]/uv[:,2:];direction=uv[1]-uv[0]
        if np.linalg.norm(direction)<1e-6:raise DualArmError('degenerate grasp direction')
        return float(np.degrees(np.arctan2(direction[1],direction[0]))%180),xy,K,T


class RGBMosaicView(DualPrimaryView):
    """Visible RGB is independent of measured depth; selected actions are not."""
    def __init__(self,before,folder,obs):
        super().__init__(before,folder,obs)
        self.source_color_xy=np.load(self.directory/'source_color_xy.npy')
        self.depth_unknown_mask=~np.isfinite(self.xyz).all(2)
        self.unknown_mask=~np.load(self.directory/'appearance_valid.npy')

    def source_ref(self,pixel):
        x,y=map(int,pixel)
        if not (0<=x<self.xyz.shape[1] and 0<=y<self.xyz.shape[0]) or self.unknown_mask[y,x]:
            raise DepthUnavailableError('selected RGB mosaic pixel is outside observed color coverage')
        if self.depth_unknown_mask[y,x]:
            raise DepthUnavailableError('selected original RGB pixel has no measured source depth; display plane is not executable depth')
        i=int(self.camera_ids[y,x]);return self.sources[i],self.source_xy[y,x].tolist(),i

    def reference(self,pixel):
        result=super().reference(pixel);x,y=map(int,pixel)
        result['original_color_pixel_xy']=self.source_color_xy[y,x].tolist()
        result['mapping']='RGB canvas -> original color -> original measured native depth'
        return result


def reconstruct_rgb_mosaic(before,folder,roi):
    from cloth_agent.dual_rgb_mosaic import render
    obs=sources(before);cameras=[];source_files=[]
    for source in obs:
        path=source.directory/'color_rgb.png'
        if path.is_file():
            transform=source.meta.get('world_from_color_mm')
            intrinsics=source.meta.get('color_intrinsics')
            if transform is None or intrinsics is None:
                raise DualArmError('RGB mosaic requires explicit original color intrinsics and transform')
        else:
            path=source.image;transform=source.meta['world_from_camera_mm'];intrinsics=source.meta['intrinsics']
        cameras.append(dict(rgb=np.asarray(Image.open(path).convert('RGB')),K=np.asarray(intrinsics),world_from_color=np.asarray(transform),xyz=source.xyz))
        source_files.append(dict(path=str(path),sha256=sha(path),camera_serial=source.meta['camera_serial']))
    settings=roi['rgb_mosaic'];result=render(cameras,bounds_xy_mm=settings['bounds_xy_mm'],plane_z_mm=settings['plane_z_mm'],size=settings.get('size',640),primary=settings.get('primary',0),seam_mode=settings.get('seam_mode','priority'),color_reference=settings.get('color_reference'))
    folder=Path(folder)/'topdown';folder.mkdir(parents=True,exist_ok=True)
    for name in ('planning_rgb.png','rgb.png'):Image.fromarray(result['rgb']).save(folder/name)
    for name in ('source_camera','source_color_xy','source_xy','world_xyz_mm','appearance_valid','depth_valid'):np.save(folder/(name+'.npy'),result[name],allow_pickle=False)
    report=dict(result['report'],status='AVAILABLE',mode='RGB_FIRST_DUAL_CAMERA' if len(obs)==2 else 'RGB_FIRST_MULTI_CAMERA',role='primary',coordinate_output='topdown_pixels',observation_id=before.meta['observation_id'],source_images=source_files,center_world_xy_mm=result['center'].tolist(),mm_per_pixel=result['mm_per_pixel'],segmentation_profile=roi.get('segmentation_profile','auto'),rgb_coverage_pixels=int(result['appearance_valid'].sum()),measured_depth_pixels=int(result['depth_valid'].sum()),global_depth_coverage_gate=False)
    (folder/'view.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return RGBMosaicView(before,folder,obs)
