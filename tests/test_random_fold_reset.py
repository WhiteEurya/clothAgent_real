from types import SimpleNamespace
import threading
import json
import numpy as np
import pytest
from PIL import Image
from cloth_agent import random_fold_reset as reset
from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import DualArmError


def observation(tmp_path):
    image=tmp_path/'rgb.png';Image.new('RGB',(300,300),'purple').save(image)
    return SimpleNamespace(image=image,sample=lambda xy,spread:np.array([xy[0],xy[1],40.]))


def test_sampling_stays_in_visual_and_metric_regions_and_reproduces_seed(tmp_path):
    obs=observation(tmp_path)
    region=dict(center_px=[150,150],radius_px=50,confidence=.95)
    points=[reset.sample_grasp(obs,region,8,30,i) for i in range(30)]
    assert len({tuple(p['pixel_xy']) for p in points})>20
    assert all(np.linalg.norm(np.array(p['pixel_xy'])-[150,150])<=30 for p in points)
    assert points[0]==reset.sample_grasp(obs,region,8,30,0)


def test_low_confidence_and_missing_depth_never_guess(tmp_path):
    obs=observation(tmp_path)
    region=dict(center_px=[150,150],radius_px=40,confidence=.7)
    with pytest.raises(DualArmError):reset.sample_grasp(obs,region,8,30,0)
    region['confidence']=.9
    def invalid(*args):raise DualArmError('no depth')
    obs.sample=invalid
    with pytest.raises(DualArmError,match='no depth'):reset.sample_grasp(obs,region,8,30,0)
    with pytest.raises(DualArmError):reset.validate_region(dict(center_px=[5,5],radius_px=10,confidence=.99),(300,300))


def test_drop_opens_only_after_vertical_lift_and_home_xy_transfer(monkeypatch):
    monkeypatch.setattr(reset,'approach_program',lambda *a:'motion.gripper(arm="left", action="open")\n')
    base=np.eye(4);base[:3,3]=[10,20,30]
    runtime=SimpleNamespace(config=SimpleNamespace(arms={'left':SimpleNamespace(
        world_from_base=base,home_joints=np.zeros(6))}),
        models={'left':SimpleNamespace(forward=lambda q:[400,50,500,180,0,0])})
    source=reset.reset_program(runtime,[100,200,40],[180,0,0],lift_height=200)
    calls=parse_code(source)
    assert [x[0] for x in calls]==['gripper','move_tcp','gripper','move_tcp','move_tcp','gripper']
    assert calls[2][1]['action']=='close' and calls[-1][1]['action']=='open'
    assert calls[1][1]['left'][:3]==pytest.approx([.1,.2,.025])
    assert calls[3][1]['left'][:3]==pytest.approx([.1,.2,.240])
    assert calls[4][1]['left'][:3]==pytest.approx([.410,.070,.240])
    assert calls[4][1]['left'][3:]==pytest.approx(calls[3][1]['left'][3:])
    with pytest.raises(ValueError):reset.reset_program(None,[100,200,40],[180,0,0],lift_height=400)


@pytest.mark.parametrize('comparison,expected_attempts', [
    (dict(change='CHANGED',confidence=.95,reason='new folds'),1),
    (dict(change='TOO_SIMILAR',confidence=.95,reason='same folds'),2),
    (dict(change='TOO_SIMILAR',confidence=.6,reason='unclear'),1),
    (dict(change='UNCERTAIN',confidence=.95,reason='occluded'),1),
])
@pytest.mark.parametrize('fail',[False,True])
@pytest.mark.parametrize('external',[False,True])
def test_reset_records_separately_and_stops_on_planning_failure(tmp_path,monkeypatch,fail,external,comparison,expected_attempts):
    seen=[];obs=observation(tmp_path)
    obs.validate_for=lambda *a,**kw:None
    class Recorder:
        error=None;samples=5
        def __init__(self,*a):pass
        def start(self):pass
        def check(self):pass
        def close(self):seen.append('close_recorder')
        def event(self,*a,**kw):pass
        def stage(self,s):seen.append(s)
    monkeypatch.setattr(reset,'TrajectoryRecorder',Recorder)
    monkeypatch.setattr(reset,'compare_reset',lambda *a,**kw:comparison)
    monkeypatch.setattr(reset,'locate_region',lambda *a,**kw:dict(center_px=[150,150],radius_px=20,confidence=.95))
    monkeypatch.setattr(reset,'reset_program',lambda *a,**kw:'program')
    def execute(*args):
        seen.append('plan_and_execute')
        if fail:raise DualArmError('IK failed')
    monkeypatch.setattr(reset,'execute_program',execute)
    class Cancel:
        def is_set(self):return False
        def set(self):seen.append('cancel')
        def wait(self,t):seen.append('settle_wait');return False
    runtime=SimpleNamespace(cancel=Cancel(),config=SimpleNamespace(limits={'max_surface_spread_mm':8},
        arms={'left':SimpleNamespace(world_from_base=np.eye(4),home_joints=np.zeros(6))}),
        models={'left':SimpleNamespace(forward=lambda q:[0,0,100,180,0,0])},
        observe=lambda p:obs,move_to_perception=lambda p:seen.append('home'),
        capture_current=lambda p:(seen.append('after') or obs),stop=lambda:seen.append('stop'))
    path=tmp_path/'reset'
    recorder=Recorder() if external else None
    if fail:
        with pytest.raises(DualArmError):reset.run_reset(runtime,path,seed=1,recorder=recorder)
        assert 'home' not in seen and 'after' not in seen and 'stop' in seen
    else:
        reset.run_reset(runtime,path,seed=1,recorder=recorder)
        assert seen.index('plan_and_execute')<seen.index('settle_wait')<seen.index('home')<seen.index('after')
    assert ('close_recorder' in seen)==(not external)
    report=json.loads((path/'result.json').read_text())
    assert report['not_a_demonstration']
    assert report['status']==('FAILED' if fail else 'EXECUTED')
    if not fail:
        assert seen.count('plan_and_execute')==expected_attempts
        assert len(report['attempts'])==expected_attempts
        if expected_attempts==2:
            assert report['outcome']=='TOO_SIMILAR'
            assert report['attempts'][0]['release']!=report['attempts'][1]['release']
            assert report['final_after']=='retry_01/after/rgb.png'
    assert (path/'grasp.json').exists() and (path/'grasp_preview.png').exists()


def test_random_release_reproducible_and_bounded():
    reset.validate_randomization(300,50,45,40,1)
    samples=[reset.sample_release(i,300,50,45,40) for i in range(100)]
    assert samples[0]==reset.sample_release(0,300,50,45,40)
    assert all(250 <= p['lift_height'] <= 300 and abs(p['yaw_delta'])<=45
               and np.linalg.norm(p['drop_offset_xy'])<=40 for p in samples)
    assert len({round(p['lift_height']) for p in samples})>20
    for values in [(300,float('nan'),45,40,1),(300,300,45,40,1),
                   (300,50,91,40,1),(300,50,45,61,1),(300,50,45,40,3)]:
        with pytest.raises(ValueError):reset.validate_randomization(*values)


def test_rotation_occurs_only_after_lift_and_before_offset_release(monkeypatch):
    monkeypatch.setattr(reset,'approach_program',lambda *a:'motion.gripper(arm="left", action="open")\n')
    runtime=SimpleNamespace(config=SimpleNamespace(arms={'left':SimpleNamespace(
        world_from_base=np.eye(4),home_joints=np.zeros(6))}),
        models={'left':SimpleNamespace(forward=lambda q:[400,50,500,180,0,0])})
    calls=parse_code(reset.reset_program(runtime,[100,200,40],[180,0,0],
        lift_height=275,yaw_delta=30,drop_offset_xy=[10,-20]))
    assert calls[3][1]['left'][:3]==pytest.approx([.1,.2,.315])
    assert calls[4][1]['left'][:3]==pytest.approx([.1,.2,.315])
    assert calls[4][1]['left'][5]==pytest.approx(np.pi/6)
    assert calls[5][1]['left'][:3]==pytest.approx([.410,.030,.315])
    assert calls[-1][1]['action']=='open'
