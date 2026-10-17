import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.codex_dual_runtime import DualArmRuntime
from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene
from .test_codex_native_runtime import runtime_for


def test_dual_lowering_applies_to_both_grasps_before_planning(scene,tmp_path,monkeypatch):
    r=runtime_for(scene);r.__class__=DualArmRuntime
    seen=[]
    def plan(source,initial):
        seen.append(parse_code(source))
        raise DualArmError('stop before dispatch')
    monkeypatch.setattr(r,'plan',plan)
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},
                        validate_for=lambda *a,**kw:None)
    source=('motion.move_tcp(left=[.4,.2,.13,0,0,0],right=[.7,.2,.135,0,0,0])\n'
            'motion.gripper(arm="left",action="close")\n'
            'motion.gripper(arm="right",action="close")\n'
            'motion.move_tcp(left=[.4,.2,.23,0,0,0],right=[.7,.2,.235,0,0,0])')
    try:
        assert r.execute(source,obs,tmp_path)['status']=='PLANNING_FAILED'
        assert seen[0][0][1]['left'][2]==pytest.approx(.12)
        assert seen[0][0][1]['right'][2]==pytest.approx(.125)
        assert seen[0][3][1]['left'][2]==.23
        assert seen[0][3][1]['right'][2]==.235
        assert parse_code((tmp_path/'effective_motion.py').read_text())==seen[0]
        report=json.loads((tmp_path/'grasp_height_policy.json').read_text())
        assert {c['arm'] for c in report['changes']}=={'left','right'}
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_dual_observation_uses_camera_b_without_single_arm_restrictions(monkeypatch):
    from cloth_agent import perception
    from cloth_agent.dual_arm.observation import create_camera
    path=Path('config/dual_arm.local.json');original=path.read_bytes()
    r=DualArmRuntime(path)
    try:
        camera=r.config.raw['camera']
        assert camera['serial']=='233622079809' and camera['mount_arm']=='right'
        assert camera['depth_exposure_us']==4000 and camera['depth_gain']==16
        assert camera['support_height_patch']['enabled'] is True
        assert not getattr(r.config,'static_arm_obstacles',{})
        assert not getattr(r.config,'collision_envelopes_disabled_arms',set())
        from cloth_agent.codex_arm7_runtime import Arm7Runtime
        other=Arm7Runtime(path)
        try:assert camera==other.config.raw['camera']
        finally:other.close()
        assert 'extrinsic_tilt_correction' not in camera
        owner,q=r.perception_target()
        assert owner=='right'
        assert np.allclose(q,json.loads(Path('data/robot/xarm7_perception_pose.json').read_text())['joint_angles_deg'][:7])
        assert 'Camera B' in r.planning_policy and '10 mm' in r.planning_policy
        opened=[]
        monkeypatch.setattr(perception,'RealSenseRGBD',lambda spec,*args:opened.append(spec.serial))
        create_camera(r.config)
        assert opened==['233622079809']
        initial={k:a.home_joints.copy() for k,a in r.config.arms.items()}
        final,source=r._preparation_program(initial,owner,q)
        assert np.allclose(final['right'],q)
        assert np.allclose(final['left'],r.config.arms['left'].home_joints)
        assert set(parse_code(source)[2][1])=={'left','right'}
        assert path.read_bytes()==original
    finally:r.close()

@pytest.mark.parametrize('arms',[('left',),('right',),('left','right')])
def test_dual_allows_independent_grasp_targets(tmp_path,arms):
    r=DualArmRuntime.__new__(DualArmRuntime)
    kw=','.join(f'{arm}={[.4,.1,.15,3.14,0,0]}' for arm in arms)
    move=','.join(f'{arm}={[.5,.2,.2,3.14,0,0]}' for arm in arms)
    calls=parse_code(r.adjust_motion(f'motion.grasp({kw})\nmotion.move_tcp({move})',tmp_path),allow_grasp=True)
    assert set(calls[0][1])==set(arms)
    for arm in arms:
        assert calls[0][1][arm][2]==pytest.approx(.14)
        assert calls[1][1][arm][2]==.2


def test_dual_explicit_independent_gripper_sequence(tmp_path):
    r=DualArmRuntime.__new__(DualArmRuntime)
    source=('motion.move_tcp(left=[.4,.1,.15,3.14,0,0])\n'
            'motion.gripper(arm="left",action="close")\n'
            'motion.move_tcp(left=[.5,.1,.2,3.14,0,0])\n'
            'motion.gripper(arm="left",action="open")\n'
            'motion.move_tcp(right=[.6,-.1,.12,3.14,0,0])\n'
            'motion.gripper(arm="right",action="close")\n'
            'motion.move_tcp(right=[.5,-.1,.2,3.14,0,0])\n'
            'motion.gripper(arm="right",action="open")')
    calls=parse_code(r.adjust_motion(source,tmp_path))
    assert calls[0][1]['left'][2]==pytest.approx(.14)
    assert calls[4][1]['right'][2]==pytest.approx(.11)
    assert [kw['action'] for n,kw in calls if n=='gripper']==['close','open','close','open']


def test_dual_exact_ik_cache_keeps_seed_and_target(scene,monkeypatch):
    r=runtime_for(scene);r.__class__=DualArmRuntime;r.plan_stats={};r.planning_context={}
    calls=[];q=scene.config.arms['left'].home_joints.copy()
    def inverse(target,seed):calls.append((target,seed));return q.copy()
    monkeypatch.setattr(r.models['left'],'inverse',inverse)
    try:
        first=r._inverse_target('left',[300,0,200,0,0,0],q,'test');first[0]+=1
        second=r._inverse_target('left',[300,0,200,0,0,0],q,'test')
        assert np.array_equal(second,q) and len(calls)==1
        r._inverse_target('left',[301,0,200,0,0,0],q,'test')
        r._inverse_target('left',[300,0,200,0,0,0],q+1,'test')
        assert len(calls)==3
    finally:r.close()
