import numpy as np
import pytest
from types import SimpleNamespace
from cloth_agent import depth_image_input as d


def frames():
 a=np.ones((20,20,3));b=a.copy();a[10,10]=np.nan;a[2,2]=np.nan;b[2,2]=np.nan
 rgb=np.full((20,20,3),100,np.uint8)
 meta={'calibration_id':'test','world_from_camera_mm':np.eye(4).tolist()}
 return a,b,rgb,meta


def test_only_measured_holes_filled_and_existing_depth_unchanged():
 a,b,rgb,m=frames();b[5,5]=99
 fused,filled,report=d.fuse(a,b,rgb,rgb,m,m)
 assert filled.sum()==1 and filled[10,10]
 assert np.isnan(fused[2,2]).all()
 assert np.array_equal(fused[5,5],a[5,5])
 assert report['filled_pixels']==1


def test_changed_local_rgb_is_not_fused():
 a,b,rgb,m=frames();other=rgb.copy();other[10,11]=200
 fused,filled,_=d.fuse(a,b,rgb,other,m,m)
 assert not filled.any() and np.isnan(fused[10,10]).all()


def test_changed_camera_pose_falls_back_to_first_frame():
 a,b,rgb,m=frames();other={**m,'world_from_camera_mm':np.eye(4).tolist()};other['world_from_camera_mm'][0][3]=20
 fused,filled,report=d.fuse(a,b,rgb,rgb,m,other)
 assert not filled.any() and report['status']=='SINGLE_FRAME_POSE_MISMATCH'


def test_failed_second_capture_restores_open_gripper(tmp_path):
 calls=[]
 class Connection:
  def gripper(self,action):calls.append(action);return {'action':action}
 def fail(path):raise RuntimeError('camera failed')
 runtime=SimpleNamespace(arm_key='left',perception_target=lambda:('left',np.zeros(6)),config=SimpleNamespace(limits={'stationary_tolerance_deg':.2}),cancel=SimpleNamespace(is_set=lambda:False),coordinator=SimpleNamespace(connections={'left':Connection()}),capture_current=fail)
 before=SimpleNamespace(directory=tmp_path/'before',meta={'joints_deg':{'left':[0]*6},'snapshot_after':{'left':{'gripper':{'usable_for_completion':True,'position_pulse':841}}}})
 with pytest.raises(RuntimeError,match='camera failed'):d.capture_pair(runtime,before,tmp_path)
 assert calls==['close','open']
 assert (tmp_path/'depth_pair/capture.json').exists()
