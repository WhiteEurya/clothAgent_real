from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from PIL import Image

from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode


def mock_scene(tmp_path, height=20., sponge=True):
    image=tmp_path/'rgb.png';Image.new('RGB',(200,100),'purple').save(image)
    arm=SimpleNamespace(world_from_base=np.eye(4),home_joints=np.zeros(6),
                        validate_point=lambda p: (_ for _ in ()).throw(ValueError('below table')) if p[2]<0 else None)
    config=SimpleNamespace(raw=dict(frame_id='world_mm',environment_confirmation=dict(
        support_layer=dict(type='sponge' if sponge else 'none'))),
        obstacles=[dict(name='table_z0_user_confirmed',max_mm=[1000,1000,0])],
        limits=dict(max_surface_spread_mm=8),arms=dict(left=arm))
    observation=SimpleNamespace(image=image,meta=dict(joints_deg=dict(left=[0.]*6,right=[0.]*7)),
        sample=lambda pixel,spread:np.array([300.,0.,height]),validate_for=lambda *a,**kw:None)
    runtime=SimpleNamespace(config=config,cancel=threading.Event(),coordinator=SimpleNamespace(
        snapshots=lambda:dict(left=dict(joints=[0.]*6),right=dict(joints=[0.]*7))),
        arrival=lambda *a:None,plan=lambda source,*a,**kw:[source],
        _execute_operations=lambda ops,folder:dict(status='EXECUTED'))
    recorder=SimpleNamespace(check=lambda:None,stage=lambda name:None)
    return runtime,observation,recorder


def test_press_uses_new_observation_center_and_zero_table_then_lifts(tmp_path,monkeypatch):
    runtime,observation,recorder=mock_scene(tmp_path)
    seen=[]
    def request(folder,body,*a):
        images=[v for v in body['input'][0]['content'] if v['type']=='input_image']
        assert len(images)==1 and body['text']['format']['schema']==episode.reset.REGION
        return dict(center_px=[100,50],radius_px=10,confidence=.9,reason='中心可见'),dict(wall_latency_s=2)
    monkeypatch.setattr(episode,'direct_once',request)
    monkeypatch.setattr(episode,'approach_program',lambda rt,pose:'motion.gripper(arm="left", action="open")\n')
    def plan(source,*a,**kw):
        seen.append(source);assert source.count('motion.move_tcp')==2
        return [source]
    runtime.plan=plan
    folder=tmp_path/'press';folder.mkdir();report=dict(physical_actions=1,robot_execution_s=0.)
    result=episode.press_dropped_cloth(runtime,folder,observation,{},5,[0,0,0],recorder,report)
    assert result['status']=='EXECUTED' and report['physical_actions']==2
    assert episode.read(folder/'press_plan.json')['press_world_pose_mm_deg'][2]==0
    assert episode.read(folder/'press_plan.json')['above_world_pose_mm_deg'][2]>=60
    assert 'action="close"' in seen[0]
    assert seen[0].endswith('motion.gripper(arm="left", action="open")\n')


@pytest.mark.parametrize('height,sponge',[(1.,True),(40.,True),(20.,False)])
def test_press_rejects_unsupported_contact_before_motion(tmp_path,monkeypatch,height,sponge):
    runtime,observation,recorder=mock_scene(tmp_path,height,sponge)
    calls=[]
    def request(*a):
        calls.append('request');return dict(center_px=[100,50],radius_px=10,confidence=.9,reason='中心可见'),dict(wall_latency_s=1)
    monkeypatch.setattr(episode,'direct_once',request)
    runtime.plan=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('unsafe plan'))
    with pytest.raises((episode.DualArmError,ValueError)):
        episode.press_dropped_cloth(runtime,tmp_path,observation,{},5,[0,0,0],recorder,
                                    dict(physical_actions=1,robot_execution_s=0.))
    assert not (tmp_path/'post_drop_press/result.json').exists()


def test_initialization_center_radius_and_press_stage(tmp_path,monkeypatch):
    runtime,observation,recorder=mock_scene(tmp_path)
    observation.meta={'joints_deg':dict(left=[0.]*6,right=[0.]*7)}
    runtime.observe=lambda folder:observation
    runtime.capture_current=lambda folder:observation
    runtime.models=dict(left=SimpleNamespace(forward=lambda q:[0,0,50,0,0,0]))
    runtime.move_to_perception=lambda folder:None
    runtime._execute_operations=lambda ops,folder:dict(status='EXECUTED')
    monkeypatch.setattr(episode,'TrajectoryRecorder',lambda *a:SimpleNamespace(start=lambda:None,stage=lambda name:None,check=lambda:None,close=lambda:None))
    monkeypatch.setattr(episode,'direct_once',lambda *a:(dict(center_px=[100,50],radius_px=10,confidence=.9,reason='中心可见'),dict(wall_latency_s=1)))
    def sample(obs,region,spread,radius,seed):
        assert radius==20
        return dict(pixel_xy=[100,50],world_xyz_mm=[300,0,20],center_world_xyz_mm=[300,0,20])
    monkeypatch.setattr(episode.reset,'sample_grasp',sample)
    monkeypatch.setattr(episode.reset,'reset_program',lambda *a,**kw:'motion.gripper(arm="left", action="open")\n')
    monkeypatch.setattr(episode,'press_dropped_cloth',lambda rt,folder,obs,headers,timeout,rpy,rec,report:
        (report.update(physical_actions=2) or dict(status='EXECUTED')))
    runtime.plan=lambda *a,**kw:[]
    folder=tmp_path/'initialization'
    result=episode.initialize(runtime,folder,{},5,center_radius_mm=20,press_after_drop=True)
    assert result['physical_actions']==2 and result['press_status']=='EXECUTED'
    assert result['center_radius_mm']==20 and result['press_after_drop']
