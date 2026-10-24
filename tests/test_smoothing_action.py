from types import SimpleNamespace
import json
import numpy as np
import pytest
from cloth_agent.experiments import trajectory_memory_full_episode_v1 as shared
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent import restricted_context_one_shot as r


@pytest.mark.parametrize('smoothing,distance,lift,limit',[(True,20,10,30),(True,30,10,30),(False,100,50,150)])
def test_profile_reaches_motion_generator(monkeypatch,tmp_path,smoothing,distance,lift,limit):
    from cloth_agent import gamepad_collect as motion
    calls=[]
    monkeypatch.setattr(motion,'automatic_program',lambda *a,**kw:calls.append(kw) or 'program')
    monkeypatch.setattr(motion,'staged_place_release',lambda ops:ops)
    arm=SimpleNamespace(world_from_base=np.eye(4),home_joints=np.zeros(6))
    runtime=SimpleNamespace(config=SimpleNamespace(limits={'max_surface_spread_mm':8},arms={'left':arm},
        path_validation_enabled=True,collision_envelopes_disabled_arms=[]),
        models={'left':SimpleNamespace(forward=lambda q:np.zeros(6))},
        coordinator=SimpleNamespace(snapshots=lambda:{'left':{'joints':np.zeros(6)}}),
        arrival=lambda q:None,plan=lambda *a,**kw:[('gripper',{'action':'open'}),('gripper',{'action':'open'})],plan_stats={})
    obs=SimpleNamespace(validate_for=lambda *a,**kw:None,sample=lambda px,spread:np.array([px[0],0.,30.]),meta={'joints_deg':{'left':[0]*6}})
    shared.prepare_action(runtime,obs,dict(grasp_px=[0,0],target_px=[distance,0]),tmp_path,smoothing=smoothing)
    assert calls[0]['lift_height']==lift
    assert calls[0]['approach_height']==50 and calls[0]['grasp_offset']==-15
    profile=json.loads((tmp_path/'motion_profile.json').read_text())
    assert profile['max_translation_mm']==limit and profile['transfer_z_mm']==30+lift


@pytest.mark.parametrize('distance',[9,31,100])
def test_smoothing_rejects_distance_before_motion(tmp_path,distance):
    runtime=SimpleNamespace(config=SimpleNamespace(limits={'max_surface_spread_mm':8}))
    obs=SimpleNamespace(validate_for=lambda *a,**kw:None,sample=lambda px,spread:np.array([px[0],0.,30.]))
    with pytest.raises(DualArmError,match='smoothing action rejected'):
        shared.prepare_action(runtime,obs,dict(grasp_px=[0,0],target_px=[distance,0]),tmp_path,smoothing=True)
    assert not (tmp_path/'motion.py').exists()


def test_smoothing_required_boolean_and_nonaction_false():
    plan=dict(decision='ACTION',grasp_px=[20,20],target_px=[40,20],motion_type='lift_pull_place',grasp_angle_deg=None,short_reason='整理下摆',is_smoothing=True)
    assert r.validate(plan,(100,100),False)['is_smoothing']
    for value in ['true',1,None]:
        with pytest.raises(Exception):r.validate({**plan,'is_smoothing':value},(100,100),False)
    done={**plan,'decision':'DONE','grasp_px':None,'target_px':None,'motion_type':None}
    with pytest.raises(ValueError,match='non-ACTION'):r.validate(done,(100,100),False)


def test_low_lift_actual_generated_transfer(monkeypatch):
    from cloth_agent import gamepad_collect as motion
    from cloth_agent.codex_motion import parse_code
    monkeypatch.setattr(motion,'approach_program',lambda *a:'motion.gripper(arm="left", action="open")\n')
    program=motion.automatic_program(None,[400,0,35],[420,0,25],[180,0,0],approach_height=50,grasp_offset=-15,lift_height=10,clearance=30,place_offset=5)
    poses=[data['left'] for kind,data in parse_code(program) if kind=='move_tcp']
    assert poses[0][2]==pytest.approx(.020)
    assert poses[1][2]==pytest.approx(.045) and poses[2][2]==pytest.approx(.045)
