from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from cloth_agent.dual_wrist_observation import load_fixed,assert_capture_pose,capture_pair
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.experiments.unfold_topdown import primary

FIXED=Path('config/calibration/dual_wrist_relaxed_20261110/candidate.json')
OVERLAP=Path('config/calibration/dual_wrist_topdown_overlap_20261112/extrinsics.json')

def test_wide_overlap_default_is_operational_and_transform_consistent():
 fixed=load_fixed(OVERLAP)
 assert fixed['pose_name']=='dual_topdown_wide_overlap'
 assert 'world_from_right_color_mm' in fixed
 assert 'candidate_world_from_right_color_mm' not in fixed
 assert fixed['validation']['current_pose_depth_cross_check']['status']=='PASSED_ROBUST_OVERLAP_CHECK'

def test_new_default_pose_cannot_use_old_fixed_extrinsics():
 new=load_fixed(FIXED)
 old=load_fixed('config/calibration/dual_wrist_fixed_20261110/candidate.json')
 state={k:{'joints':v} for k,v in new['capture_joints_deg'].items()}
 assert_capture_pose(new,state)
 with pytest.raises(DualArmError,match='recorded perception'):
  assert_capture_pose(old,state)
 assert abs(new['capture_joints_deg']['right'][6])<10

def test_fixed_calibration_requires_consistent_pair_and_capture_pose(tmp_path):
 c=load_fixed(FIXED);state={k:{'joints':list(v)} for k,v in c['capture_joints_deg'].items()};assert_capture_pose(c,state)
 state['right']['joints'][0]+=.2
 with pytest.raises(DualArmError,match='recorded perception'):assert_capture_pose(c,state)
 c['candidate_world_from_right_color_mm'][0][3]+=10;p=tmp_path/'bad.json';p.write_text(json.dumps(c))
 with pytest.raises(ValueError,match='Inconsistent'):load_fixed(p)

@pytest.mark.parametrize('operational',[False,True])
def test_full_capture_to_rgb_view_uses_both_cameras_and_original_depth(tmp_path,monkeypatch,operational):
 fixed=load_fixed(FIXED);fixed['world_from_left_color_mm']=np.eye(4).tolist();fixed['candidate_world_from_right_color_mm']=np.eye(4).tolist();K=[[10,0,10],[0,10,10],[0,0,1]];fixed['color_intrinsics']={'left':K,'right':K};events=[]
 if operational:fixed['world_from_right_color_mm']=fixed.pop('candidate_world_from_right_color_mm')
 y,x=np.indices((21,21));xyz=np.stack([x-10,y-10,np.full_like(x,10)],-1).astype(float);im=np.full((21,21,3),140,np.uint8)
 def capture(cfg,**kwargs):
  serial=cfg['calibration']['realsense_serial'];events.append(('capture',serial));return {'rgb':im,'color_rgb':im,'xyz':xyz.copy(),'meta':{'camera_serial':serial,'intrinsics':K,'color_intrinsics':K,'world_from_camera_mm':np.eye(4).tolist(),'world_from_color_mm':np.eye(4).tolist()}}
 monkeypatch.setattr('cloth_agent.dual_cloud_observation.capture_secondary',capture)
 def move(folder,before,snapshots,*,arm,saved):events.append(('move',arm));return before
 state={k:{'joints':v} for k,v in fixed['capture_joints_deg'].items()}
 config=SimpleNamespace(raw={'calibration_id':'test','camera':{}},limits={'stationary_tolerance_deg':.1})
 runtime=SimpleNamespace(dual_wrist_config=fixed,config=config,models={k:SimpleNamespace(forward=lambda q:np.zeros(6)) for k in state},_perception_before_capture=move,topdown_config={'appearance_render':'rgb_mosaic','rgb_mosaic':{'bounds_xy_mm':[[-5,-5],[5,5]],'plane_z_mm':10,'size':11}})
 before=capture_pair(runtime,tmp_path/'capture',state,lambda:state)
 assert events==[('move','left'),('move','right'),('capture',fixed['camera_serials']['left']),('capture',fixed['camera_serials']['right'])]
 if operational:
  before.validate_for(config,live=True,check_age=False)
  before.secondary.validate_for(config,live=True,check_age=False)
  assert 'world_from_right_color_mm' in before.meta['fixed_camera_mapping']
 else:
  assert before.meta['execution_allowed'] is False
  with pytest.raises(DualArmError,match='observation-only'):before.validate_for(config,live=True,check_age=False)
 view=primary(before,tmp_path/'view');assert view.image.exists();np.testing.assert_array_equal(view.sample([5,5],None),[0,0,10]);assert view.reference([5,5])['camera_serial']==fixed['camera_serials']['left']

def test_operational_mapping_applies_to_native_capture_and_retains_measured_depth_checks(tmp_path,monkeypatch):
 fixed=load_fixed('config/calibration/dual_wrist_relaxed_20261110/extrinsics.json')
 calls=[];y,x=np.indices((21,21));native=np.stack([x-10,y-10,np.full_like(x,100)],-1).astype(float)
 def capture(cfg,**kwargs):
  cal=cfg['calibration'];T=np.asarray(cal['world_from_realsense_color_mm']);calls.append(T)
  xyz=native@T[:3,:3].T+T[:3,3]
  return {'rgb':np.zeros((21,21,3),np.uint8),'color_rgb':np.zeros((21,21,3),np.uint8),'xyz':xyz,'meta':{}}
 monkeypatch.setattr('cloth_agent.dual_cloud_observation.capture_secondary',capture)
 state={k:{'joints':v} for k,v in fixed['capture_joints_deg'].items()}
 config=SimpleNamespace(raw={'calibration_id':'offline-test','camera':{}},limits={'stationary_tolerance_deg':.1})
 runtime=SimpleNamespace(dual_wrist_config=fixed,config=config,models={k:SimpleNamespace(forward=lambda q:np.zeros(6)) for k in state},_perception_before_capture=lambda folder,before,snapshots,**kwargs:before,topdown_config={})
 ob=capture_pair(runtime,tmp_path/'capture',state,lambda:state)
 T=np.asarray(fixed['world_from_right_color_mm'])
 np.testing.assert_allclose(calls[1],T)
 np.testing.assert_allclose(ob.secondary.xyz[10,10],T[:3,:3]@[0,0,100]+T[:3,3],atol=1e-4)
 ob.validate_for(config,live=True,check_age=False)
 ob.secondary.xyz[8:13,8:13]=np.nan
 with pytest.raises(DualArmError,match='insufficient measured depth'):ob.secondary.sample([10,10],None)

def test_capture_drift_is_not_saved(tmp_path,monkeypatch):
 fixed=load_fixed(FIXED);state={k:{'joints':list(v)} for k,v in fixed['capture_joints_deg'].items()};after={k:{'joints':list(v)} for k,v in fixed['capture_joints_deg'].items()};after['right']['joints'][0]+=.3
 monkeypatch.setattr('cloth_agent.dual_cloud_observation.capture_secondary',lambda *a,**kw:{})
 runtime=SimpleNamespace(dual_wrist_config=fixed,models={k:SimpleNamespace(forward=lambda q:np.zeros(6)) for k in state},_perception_before_capture=lambda folder,before,snapshots,**kw:before)
 with pytest.raises(DualArmError,match='recorded perception'):capture_pair(runtime,tmp_path/'capture',state,lambda:after)
 assert not (tmp_path/'capture').exists()

def test_guidance_describes_rgb_display_not_depth_surface():
 from cloth_agent.experiments.unfold_topdown import guidance_for
 before=SimpleNamespace(meta={'topdown_config':{'appearance_render':'rgb_mosaic'}})
 text=guidance_for(before);assert '显示不依赖深度完整性' in text and '最高已测表面' not in text

def test_wrist_video_fresh_frame_wait_never_returns_moving_frame():
 import threading,time
 from cloth_agent.wrist_video import WristVideo
 v=WristVideo.__new__(WristVideo);v.condition=threading.Condition();v.stop_event=threading.Event();v.seq=1;v.error=None;v.latest_native_frames='old'
 def supply():
  time.sleep(.02)
  with v.condition:v.latest_native_frames='new';v.seq+=1;v.condition.notify_all()
 thread=threading.Thread(target=supply);thread.start();assert v.read_native_frames()=='new';thread.join()


def test_zed_supplement_saved_and_loaded_as_third_source(tmp_path,monkeypatch):
 from cloth_agent.experiments.unfold_dual_topdown import sources
 fixed=load_fixed('config/calibration/dual_wrist_relaxed_20261110/extrinsics.json')
 state={k:{'joints':v} for k,v in fixed['capture_joints_deg'].items()}
 rgb=np.zeros((21,21,3),np.uint8);xyz=np.ones((21,21,3))*10
 monkeypatch.setattr('cloth_agent.dual_cloud_observation.capture_secondary',lambda *a,**k:dict(rgb=rgb,color_rgb=rgb,xyz=xyz,meta={}))
 frame=SimpleNamespace(rgb=rgb,serial='zed-test',X_base_camera=np.eye(4),intrinsics=np.eye(3))
 calls=[]
 def capture(*args,**kwargs):calls.append(kwargs);return [frame]
 monkeypatch.setattr('cloth_agent.perception.capture_two_view_rgbd',capture)
 monkeypatch.setattr('cloth_agent.site_observation.world_xyz_mm',lambda frame:xyz)
 cfg=SimpleNamespace(raw={'calibration_id':'test','camera':{'serial':'zed-test','world_from_camera_mm':np.eye(4).tolist()}},limits={'stationary_tolerance_deg':.1})
 runtime=SimpleNamespace(dual_wrist_config=fixed,config=cfg,topdown_config={'zed_supplement':{'enabled':True}},
  models={k:SimpleNamespace(forward=lambda q:np.zeros(6)) for k in state},
  _perception_before_capture=lambda folder,before,snapshots,**kw:before)
 before=capture_pair(runtime,tmp_path/'capture',state,lambda:state)
 obs=sources(before)
 assert len(calls)==1 and len(obs)==3
 assert obs[2].meta['camera_serial']=='zed-test'
 np.testing.assert_allclose(obs[2].xyz,xyz)
