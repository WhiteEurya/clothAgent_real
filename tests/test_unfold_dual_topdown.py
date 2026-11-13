from types import SimpleNamespace
import json
import numpy as np
import pytest
from PIL import Image
from cloth_agent.dual_arm.observation import save_observation
from cloth_agent.dual_cloud_observation import attach_secondary,scale_world
from cloth_agent.experiments.unfold_topdown import primary
from cloth_agent.dual_arm.geometry import DualArmError


def dual_fixture(tmp_path):
    n=80;y,x=np.indices((n,n));xyz=np.stack([x-40,y-40,np.full_like(x,100)],axis=-1).astype(float)
    zed=xyz.copy();zed[:,40:]=np.nan
    rs=xyz.copy();rs[:,:40]=np.nan
    cfg=SimpleNamespace(raw={'calibration_id':'test','camera':{}})
    meta={'camera_serial':'zed','intrinsics':[[100,0,40],[0,100,40],[0,0,1]],'world_from_camera_mm':np.eye(4).tolist()}
    im=np.full((n,n,3),150,np.uint8)
    before=save_observation(tmp_path/'observation',im,zed,cfg,{},synthetic=True,extra=meta)
    fusion={'depth_scale':{'zed':1.,'realsense':1.},'color_fallback':{'gain_rgb':[1,1,1],'offset_rgb':[10,10,10]}}
    second={'rgb':im,'xyz':rs,'raw_xyz':rs,'color_rgb':im,'meta':dict(meta,camera_serial='rs',synthetic=True)}
    before=attach_secondary(before,second,fusion,cfg)
    roi={'table_polygon_px':[[0,0],[79,0],[79,79],[0,79]],'display_min_world_z_mm':-20}
    return before,roi,xyz


def test_owning_camera_depth_and_color_only_fallback(tmp_path):
    before,roi,xyz=dual_fixture(tmp_path);view=primary(before,tmp_path/'view',config=roi)
    for source_id,point in [(0,[20,40]),(1,[60,40])]:
        match=(view.camera_ids==source_id)&np.all(view.source_xy==point,axis=2)
        y,x=np.argwhere(match)[0];p=[int(x),int(y)]
        assert view.reference(p)['camera_serial']==('zed' if source_id==0 else 'rs')
        assert np.allclose(view.sample(p,None),xyz[point[1],point[0]])
        assert np.allclose(view.xyz[y,x],xyz[point[1],point[0]])
        image=np.asarray(Image.open(view.image));assert np.all(image[y,x]==150)
        angle,pixel,K,T=view.orientation_inputs(p,0)
        assert angle==pytest.approx(0);assert pixel==point
    plan={'decision':'ACTION','grasp_px':p,'target_px':p,'grasp_angle_deg':0}
    assert view.raw_plan(plan)==plan and before.planning_view is view
    with pytest.raises(DualArmError,match='no measured source'):view.sample([1,1],None)
    assert json.loads((view.directory/'view.json').read_text())['color_alignment']['status']=='IDENTITY_UNVALIDATED_COLOR'


def test_independent_rois_keep_secondary_points_outside_primary_crop(tmp_path):
    before,roi,xyz=dual_fixture(tmp_path)
    roi['table_polygon_px']=[[0,0],[39,0],[39,79],[0,79]]
    old=primary(before,tmp_path/'old',config=roi)
    assert not np.any(old.camera_ids==1)
    roi['table_rois_by_serial']={
        'zed':dict(frame='native',image_size_wh=[80,80],polygon_px=roi['table_polygon_px']),
        'rs':dict(frame='native',image_size_wh=[80,80],polygon_px=[[45,0],[70,0],[70,79],[45,79]])}
    view=primary(before,tmp_path/'union',config=roi)
    match=(view.camera_ids==1)&np.all(view.source_xy==[60,40],axis=2)
    y,x=np.argwhere(match)[0]
    np.testing.assert_allclose(view.sample([int(x),int(y)],None),xyz[40,60])
    assert not np.any((view.camera_ids==1)&(view.source_xy[:,:,0]>70))
    assert np.any(view.camera_ids==0)
    assert json.loads((view.directory/'view.json').read_text())['table_crop']['mode']=='independent_camera_roi_union'


def test_secondary_roi_projects_into_color_not_depth_grid(tmp_path):
    from cloth_agent.experiments.unfold_dual_topdown import table_projection,sources
    before,roi,_=dual_fixture(tmp_path);obs=sources(before)
    T=np.eye(4);T[0,3]=10
    cfg=dict(calibration=dict(realsense_serial='rs',world_from_realsense_color_mm=T.tolist(),
                             realsense_K=obs[1].meta['intrinsics']),realsense_color_size_wh=[80,80])
    roi['table_rois_by_serial']={'rs':dict(frame='realsense_color',image_size_wh=[80,80],
        polygon_px=[[45,0],[55,0],[55,79],[45,79]])}
    mask,t,k=table_projection(obs[1],obs,cfg,roi)
    point=obs[1].xyz[40,60];cam=(point-t[:3,3])@t[:3,:3];uv=np.rint((k@cam)[:2]/cam[2]).astype(int)
    assert uv.tolist()==[50,40] and mask[uv[1],uv[0]]
    assert not mask[40,60]


def test_independent_roi_requires_region_for_each_camera(tmp_path):
    before,roi,_=dual_fixture(tmp_path)
    roi['table_rois_by_serial']={}
    with pytest.raises(DualArmError,match='missing camera-specific table ROI'):
        primary(before,tmp_path/'view',config=roi)


def test_modified_secondary_provenance_rejected(tmp_path):
    before,roi,_=dual_fixture(tmp_path)
    path=before.directory/'realsense/observation.json';path.write_text(path.read_text()+' ')
    with pytest.raises(DualArmError,match='metadata changed'):primary(before,tmp_path/'view',config=roi)


def test_native_origin_scale_preserves_extrinsics_and_unknown():
    T=np.eye(4);T[:3,3]=[100,20,300]
    points=np.array([[120,50,900],[np.nan,np.nan,np.nan]])
    result=scale_world(points,T,1.01)
    assert np.allclose(result[0],[120.2,50.3,906]);assert np.isnan(result[1]).all()
    assert np.array_equal(T[:3,3],[100,20,300])


def test_prepare_action_uses_secondary_samples_and_orientation(tmp_path,monkeypatch):
    from cloth_agent.experiments import partitioned_unfold as unfold
    from cloth_agent.dual_arm.geometry import pose_matrix
    from cloth_agent.codex_motion import parse_code
    runtime=unfold.configure_runtime(unfold.PartitionedReplayRuntime(unfold.DEFAULT_CONFIG,planning_only=True))
    joints={k:v.home_joints.copy() for k,v in runtime.config.arms.items()}
    arm=runtime.config.arms['right'];a=(arm.world_from_base@pose_matrix(runtime.models['right'].forward(joints['right'])))[:3,3];b=a+[20,0,0]
    im=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(im)
    secondary_k=np.diag([700.,700.,1.]);secondary_t=np.eye(4);seen=[]
    measurement=SimpleNamespace(image=im,sample=lambda p,s:a if p[0]==40 else b,
        reference=lambda p:{'camera_id':1,'pixel_xy':[10,20]},
        orientation_inputs=lambda p,angle:(30.,[10,20],secondary_k,secondary_t))
    def orientation(angle,pixel,xyz,rpy,k,t,axis):
        assert angle==30 and pixel==[10,20]
        assert np.array_equal(k,secondary_k) and np.array_equal(t,secondary_t)
        seen.append(True);return rpy,0
    monkeypatch.setattr('cloth_agent.grasp_angle.image_angle_to_rpy',orientation)
    before=SimpleNamespace(image=im,planning_view=measurement,meta={'joints_deg':joints,'intrinsics':np.eye(3).tolist(),'world_from_camera_mm':np.eye(4).tolist()},sample=lambda *args:pytest.fail('must not sample ZED for a secondary point'),validate_for=lambda *a,**k:None)
    plan=dict(decision='ACTION',grasp_px=[40,40],target_px=[60,40],motion_type='lift_pull_place',short_reason='test',grasp_angle_deg=0.,is_smoothing=False,action_purpose='test',expected_stop_state='test')
    try:
        source=unfold.prepare_action(runtime,before,plan,tmp_path)
        assert seen
        calls=parse_code(source,allow_grasp=True);assert np.allclose(np.array(calls[0][1]['right'][:3])*1000,a)
        assert json.loads((tmp_path/'geometry.json').read_text())['source_references']['grasp_px']['camera_id']==1
    finally:runtime.close()


def test_empty_secondary_depth_keeps_primary_without_global_gate(tmp_path):
    from cloth_agent.dual_arm.observation import sha
    before,roi,_=dual_fixture(tmp_path);folder=before.directory/'realsense'
    data=np.load(folder/'world_xyz_mm.npy');data[:]=np.nan;np.save(folder/'world_xyz_mm.npy',data)
    meta=json.loads((folder/'observation.json').read_text());meta['sha256']['world_xyz_mm.npy']=sha(folder/'world_xyz_mm.npy');(folder/'observation.json').write_text(json.dumps(meta))
    before.meta['dual_cloud']['secondary_observation_sha256']=sha(folder/'observation.json')
    view=primary(before,tmp_path/'view',config=roi)
    assert (view.camera_ids==0).any() and not (view.camera_ids==1).any()
    report=json.loads((view.directory/'view.json').read_text())
    assert report['valid_points_per_camera'][1]==0 and not report['global_depth_coverage_gate']


def fallback_view(tmp_path,secondary_offset=(0,0,0)):
    from cloth_agent.dual_arm.observation import Observation
    from cloth_agent.experiments.unfold_dual_topdown import DualPrimaryView
    y,x=np.indices((11,11));xyz=np.stack([x-5,y-5,np.full_like(x,100)],axis=-1).astype(float)
    sparse=np.full_like(xyz,np.nan);sparse[4:7,3:8]=xyz[4:7,3:8]
    obs=[]
    for i,data in enumerate([sparse,xyz+secondary_offset]):
        o=SimpleNamespace(xyz=data,meta={'camera_serial':str(i),'observation_id':str(i),
            'intrinsics':np.eye(3).tolist(),'world_from_camera_mm':np.eye(4).tolist()})
        o.sample=lambda p,s,o=o:Observation.sample(o,p,s)
        obs.append(o)
    np.save(tmp_path/'world_xyz_mm.npy',xyz)
    np.save(tmp_path/'source_xy.npy',np.stack([x,y],axis=-1))
    np.save(tmp_path/'source_camera.npy',np.zeros((11,11),np.int8))
    return DualPrimaryView(obs[0],tmp_path,obs)


def test_fused_depth_uses_measured_point_without_native_patch_gate(tmp_path):
    v=fallback_view(tmp_path)
    for source in v.sources:
        source.sample=lambda *args:pytest.fail('native 5x5 sampling must not be used')
    angle,pixel,_,_=v.orientation_inputs([5,5],0)
    assert angle==pytest.approx(0) and pixel==[5,5]
    assert v.reference([5,5])['camera_id']==0
    assert np.allclose(v.sample([5,5],None),[0,0,100])
    report=json.loads((tmp_path/'depth_resolution.json').read_text())['(5, 5)']
    assert report['status']=='FUSED_MEASURED_POINT' and report['native_patch_gate'] is False


@pytest.mark.parametrize('offset',[(35,0,0),(0,0,-66)])
def test_fused_support_does_not_include_remote_or_occluded_surface(tmp_path,offset):
    v=fallback_view(tmp_path,offset)
    assert np.allclose(v.sample([5,5],None),[0,0,100])
    report=json.loads((tmp_path/'depth_resolution.json').read_text())['(5, 5)']
    assert report['support_per_camera']['1']==0


def test_isolated_fused_point_rejected_even_if_duplicated_by_two_cameras(tmp_path):
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    v=fallback_view(tmp_path)
    for source in v.sources:
        source.xyz[:]=np.nan;source.xyz[5,5]=[0,0,100]
    with pytest.raises(DepthUnavailableError,match='only 1 distinct'):
        v.sample([5,5],None)


def test_unknown_point_not_filled_from_neighbors(tmp_path):
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    v=fallback_view(tmp_path);v.unknown_mask[5,5]=True;v.xyz[5,5]=np.nan
    with pytest.raises(DepthUnavailableError,match='no measured source'):
        v.sample([5,5],None)


def test_measured_anchor_preserved_when_other_surface_is_nearby(tmp_path):
    v=fallback_view(tmp_path,(0,0,4))
    assert np.array_equal(v.sample([5,5],None),[0,0,100])


@pytest.mark.parametrize('delta,owner',[(4.,0),(6.,1),(-6.,0)])
def test_same_surface_ownership_and_measured_provenance(tmp_path,monkeypatch,delta,owner):
    from cloth_agent.experiments import unfold_dual_topdown as fusion
    before,roi,xyz=dual_fixture(tmp_path)
    obs=fusion.sources(before)
    obs[0].xyz=xyz.copy();obs[1].xyz=xyz.copy();obs[1].xyz[:,:,2]+=delta
    monkeypatch.setattr(fusion,'sources',lambda before:obs)
    view=fusion.reconstruct(before,tmp_path/'fused',roi)
    both=np.isfinite(view.xyz).all(2)
    assert (view.camera_ids[both]==owner).all()
    uv=view.source_xy[both]
    assert np.allclose(view.xyz[both],obs[owner].xyz[uv[:,1],uv[:,0]])


def test_appearance_fill_does_not_create_execution_depth(tmp_path,monkeypatch):
    from cloth_agent.experiments import unfold_dual_topdown as fusion
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    before,roi,xyz=dual_fixture(tmp_path)
    baseline=fusion.reconstruct(before,tmp_path/'old',roi)
    known=~baseline.unknown_mask
    def render(*args,**kwargs):
        z=baseline.xyz[:,:,2].copy();z[~known]=100
        return dict(rgb=np.full((640,640,3),150,np.uint8),coverage=np.ones((640,640),bool),
            texture_inferred=~known,source_camera=np.ones((640,640),np.int8),display_z_mm=z,report={})
    monkeypatch.setattr('cloth_agent.dual_cloud_render.render_appearance',render)
    new=fusion.reconstruct(before,tmp_path/'new',dict(roi,appearance_render='surface_rgb',segmentation_profile='black'))
    assert new.segmentation_profile=='black'
    assert np.array_equal(new.unknown_mask,baseline.unknown_mask)
    assert np.allclose(new.xyz,baseline.xyz,equal_nan=True)
    with pytest.raises(DepthUnavailableError,match='insufficient nearby measured surface'):new.sample([1,1],None)


def test_appearance_other_surface_cannot_relabel_old_depth(tmp_path,monkeypatch):
    from cloth_agent.experiments import unfold_dual_topdown as fusion
    before,roi,_=dual_fixture(tmp_path)
    def render(*args,**kwargs):
        return dict(rgb=np.full((640,640,3),150,np.uint8),coverage=np.ones((640,640),bool),
            texture_inferred=np.zeros((640,640),bool),source_camera=np.ones((640,640),np.int8),
            display_z_mm=np.full((640,640),200.),report={})
    monkeypatch.setattr('cloth_agent.dual_cloud_render.render_appearance',render)
    new=fusion.reconstruct(before,tmp_path/'new',dict(roi,appearance_render='surface_rgb'))
    assert new.unknown_mask.all() and np.isnan(new.xyz).all()


def local_hole_view(tmp_path):
    v=fallback_view(tmp_path)
    v._mm_per_pixel=1.
    v.xyz[5,5]=np.nan;v.unknown_mask[5,5]=True
    inferred=np.zeros((11,11),bool);inferred[5,5]=True
    np.save(tmp_path/'appearance_texture_inferred.npy',inferred)
    Image.new('RGB',(11,11),(20,20,20)).save(v.image)
    return v


def test_small_hole_associates_real_point_without_depth_fill(tmp_path):
    v=local_hole_view(tmp_path)
    result=v.sample([5,5],None)
    ref=v.reference([5,5]);xy=ref['pixel_xy']
    assert np.array_equal(result,v.sources[ref['camera_id']].xyz[xy[1],xy[0]])
    assert ref['requested_topdown_pixel']==[5,5]
    assert ref['resolved_topdown_pixel']!=[5,5]
    assert np.isnan(v.xyz[5,5]).all() and v.unknown_mask[5,5]
    assert np.array_equal(v.sample([5,5],None),result)
    report=json.loads((tmp_path/'depth_resolution.json').read_text())['(5, 5)']
    assert report['status']=='LOCAL_MEASURED_POINT_ASSOCIATION'
    assert report['shift_mm']<=5 and report['depth_filled'] is False


def test_hole_does_not_mix_two_layers(tmp_path):
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    v=local_hole_view(tmp_path);v.xyz[:5,:,2]=70
    with pytest.raises(DepthUnavailableError,match='ambiguous neighboring'):
        v.sample([5,5],None)
    assert not v._local_matches


def test_hole_never_uses_unsupported_neighbor(tmp_path):
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    v=local_hole_view(tmp_path)
    for source in v.sources:source.xyz[:]=np.nan
    with pytest.raises(DepthUnavailableError,match='support checks'):
        v.sample([5,5],None)
    assert not v._local_matches


def test_hole_does_not_snap_to_different_color_table(tmp_path):
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    v=local_hole_view(tmp_path)
    rgb=np.full((11,11,3),200,np.uint8);rgb[5,5]=20
    Image.fromarray(rgb).save(v.image)
    with pytest.raises(DepthUnavailableError,match='insufficient nearby'):
        v.sample([5,5],None)


def test_large_hole_not_filled(tmp_path):
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    v=local_hole_view(tmp_path);v.xyz[3:8,3:8]=np.nan;v.unknown_mask[3:8,3:8]=True
    with pytest.raises(DepthUnavailableError,match='insufficient nearby'):
        v.sample([5,5],None)
