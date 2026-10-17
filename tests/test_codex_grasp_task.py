import json
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.codex_motion import parse_code
from cloth_agent.codex_grasp_task import lower_task_grasps
from cloth_agent.codex_dual_runtime import DualArmRuntime
from cloth_agent.codex_arm7_runtime import Arm7Runtime
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene
from .test_codex_native_runtime import runtime_for


@pytest.mark.parametrize('source',[
 'motion.move_tcp(left=[.25,-.12,.2,0,0,0])\nmotion.grasp(left=[.25,-.12,.18,0,0,0])',
 'motion.grasp(left=[.25,-.12,.18,0,0,0])',
 'motion.grasp(left=[.25,-.12,.18,0,0,0])\nmotion.gripper(arm="left",action="close")',
 'motion.grasp(left=[.25,-.12,.18,0,0,0])\nmotion.move_tcp(right=[.25,.12,.2,0,0,0])',
])
def test_invalid_task_boundaries_rejected(source):
    with pytest.raises(ValueError):parse_code(source,allow_grasp=True)


def test_grasp_offset_only_changes_declaration(tmp_path):
    source='motion.grasp(left=[.25,-.12,.18,0,0,0])\nmotion.move_tcp(left=[.3,-.12,.22,0,0,0])'
    code=lower_task_grasps(source,tmp_path)
    calls=parse_code(code,allow_grasp=True)
    assert calls[0][1]['left'][2]==pytest.approx(.17)
    assert calls[1][1]['left'][2]==.22
    assert json.loads((tmp_path/'grasp_height_policy.json').read_text())['offset_mm']==10


@pytest.mark.parametrize('single',[False,True])
def test_host_pick_fold_release_home_and_adaptive_approach(scene,tmp_path,monkeypatch,single):
    r=runtime_for(scene);r.__class__=Arm7Runtime if single else DualArmRuntime
    initial={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    # Fake scene uses joint xyz as TCP; reject the highest host hover to verify
    # the host tries a lower height instead of asking GPT for an auxiliary point.
    original=scene.models['right'].inverse;heights=[]
    def ik(p,seed):
        heights.append(p[2])
        if p[2]>225:raise DualArmError('test unreachable high hover')
        return original(p,seed)
    monkeypatch.setattr(scene.models['right'],'inverse',ik)
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    arms='right=[.25,.12,.18,0,0,0]' if single else 'left=[.25,-.12,.18,0,0,0],right=[.25,.12,.18,0,0,0]'
    fold='right=[.26,.12,.185,0,0,0]' if single else 'left=[.26,-.12,.185,0,0,0],right=[.26,.12,.185,0,0,0]'
    source=f'motion.grasp({arms})\nmotion.move_tcp({fold})'
    obs=SimpleNamespace(meta={'joints_deg':initial},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='EXECUTED',result
        assert any(h>225 for h in heights) and any(205<=h<=225 for h in heights)
        active=['right'] if single else ['left','right']
        for arm in active:
            events=[c for c in scene.connections[arm].commands if c[0]=='gripper']
            expected=['open','close','release_partial','open'] if single else ['open','close','open']
            assert events==[('gripper',action) for action in expected]
            assert np.allclose(scene.connections[arm].joints,initial[arm])
        if single:assert scene.connections['left'].commands==[]
        rows=json.loads((tmp_path/'native_plan.json').read_text())
        closing=[i for i,row in enumerate(rows) if row['kind']=='gripper' and row['data']['action']=='close']
        for index in closing:
            assert all(rows[index]['data']['joints'][k][2]==pytest.approx(180-r.grasp_offset_mm if single else 170) for k in active)
        if single:
            partial=next(i for i,row in enumerate(rows) if row['kind']=='gripper' and row['data']['action']=='release_partial')
            opened=next(i for i in range(partial+1,len(rows)) if rows[i]['kind']=='gripper')
            a=rows[partial]['data'];b=rows[opened]['data']
            assert a['opening_delta_pulse']==120
            assert b['action']=='open' and opened>partial+1
            np.testing.assert_allclose(a['joints']['right'][:2],b['joints']['right'][:2])
            assert b['joints']['right'][2]-a['joints']['right'][2]==pytest.approx(40.)
            assert rows[opened+1]['kind']=='move'  # Home only after full opening.
    finally:r.close()


@pytest.mark.parametrize('failed_phase',['gpt_cloth_motion','host_retreat'])
def test_failed_middle_or_return_never_dispatches_host_open_or_grasp(scene,tmp_path,monkeypatch,failed_phase):
    r=runtime_for(scene);r.__class__=DualArmRuntime
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    original=r._tcp_segments
    def fail_middle(*a,**kw):
        if r.planning_context.get('task_phase')==failed_phase:
            raise DualArmError('fold blocked')
        return original(*a,**kw)
    monkeypatch.setattr(r,'_tcp_segments',fail_middle)
    source=('motion.grasp(left=[.25,-.12,.18,0,0,0],right=[.25,.12,.18,0,0,0])\n'
            'motion.move_tcp(left=[.26,-.12,.2,0,0,0],right=[.26,.12,.2,0,0,0])')
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert result['failure_context']['task_phase']==failed_phase
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_host_retries_other_arm_order_from_original_configuration(scene):
    from cloth_agent.codex_grasp_task import _ordered_phase
    r=runtime_for(scene);q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    r.planning_context={}
    tried=[]
    def step(arm,state):
        tried.append(arm)
        if arm=='left' and state['right'][0]==q['right'][0]:
            raise DualArmError('other arm must move first')
        state={k:v.copy() for k,v in state.items()};state[arm][0]+=1
        return [],state
    try:
        _,result=_ordered_phase(r,q,('left','right'),step)
        assert tried==['left','right','left']
        assert all(result[k][0]==q[k][0]+1 for k in q)
        assert all(q[k][0]==scene.config.arms[k].home_joints[0] for k in q)
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_transfer_requires_task_and_preserves_targets(tmp_path):
    transfer='motion.move_transfer(right=[.26,.12,.2,0,0,0])'
    with pytest.raises(ValueError):parse_code(transfer,allow_grasp=True)
    source='motion.grasp(right=[.25,.12,.18,0,0,0])\n'+transfer
    with pytest.raises(ValueError):parse_code(source)
    calls=parse_code(lower_task_grasps(source,tmp_path),allow_grasp=True)
    assert calls[1]==('move_transfer',{'right':[.26,.12,.2,0,0,0]})


def test_inward_approach_when_all_vertical_hovers_unreachable(scene,monkeypatch):
    from cloth_agent.codex_grasp_task import _approach
    r=runtime_for(scene);r.planning_context={}
    q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    pose=[.25,.12,.18,0,0,0];attempts=[]
    def inverse(arm,target,seed,phase):
        attempts.append(target.copy())
        world=scene.config.arms[arm].world_from_base
        from cloth_agent.dual_arm.geometry import pose_matrix
        target_world=(world@pose_matrix(target))[:3,3]
        if np.allclose(target_world[:2],np.array(pose[:2])*1000):
            raise DualArmError('vertical hover unreachable')
        return q[arm].copy()
    monkeypatch.setattr(r,'_inverse_target',inverse)
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    descents=[]
    def descend(arm,target,state,**kw):
        descents.append((target,kw));return [],state
    monkeypatch.setattr(r,'_tcp_segments',descend)
    try:
        _approach(r,'right',pose,q)
        assert len(attempts)==5
        assert descents==[(pose,{'allow_detour':False})]
        assert r.planning_context['inward_mm']==20
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


@pytest.mark.parametrize('single',[False,True])
def test_transfer_uses_joint_route_without_cartesian_intermediates(scene,tmp_path,monkeypatch,single):
    r=runtime_for(scene);r.__class__=Arm7Runtime if single else DualArmRuntime
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    original=r._tcp_segments
    def tcp(*a,**kw):
        assert r.planning_context.get('task_phase')!='gpt_cloth_transfer'
        return original(*a,**kw)
    monkeypatch.setattr(r,'_tcp_segments',tcp)
    arms='right=[.25,.12,.18,0,0,0]' if single else 'left=[.25,-.12,.18,0,0,0],right=[.25,.12,.18,0,0,0]'
    targets='right=[.26,.12,.185,0,0,0]' if single else 'left=[.26,-.12,.185,0,0,0],right=[.26,.12,.185,0,0,0]'
    source=f'motion.grasp({arms})\nmotion.move_transfer({targets})'
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='EXECUTED',result
        if single:assert scene.connections['left'].commands==[]
    finally:r.close()


def test_transfer_failure_does_not_dispatch_grasp(scene,tmp_path,monkeypatch):
    r=runtime_for(scene);r.__class__=Arm7Runtime
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    original=r._inverse_target
    def reject(arm,pose,seed,phase):
        if phase=='transfer_goal_ik':raise DualArmError('transfer endpoint unreachable')
        return original(arm,pose,seed,phase)
    monkeypatch.setattr(r,'_inverse_target',reject)
    source='motion.grasp(right=[.25,.12,.18,0,0,0])\nmotion.move_transfer(right=[.26,.12,.185,0,0,0])'
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert result['failure_context']['command']=='move_transfer'
        assert result['failure_context']['task_phase']=='gpt_cloth_transfer'
        assert all(not c.commands for c in scene.connections.values())
        preview=json.loads((tmp_path/'trajectory_preview.json').read_text())
        assert any(t['command_index']==2 for t in preview['targets'])
    finally:r.close()


def test_partial_release_failure_prevents_lift_and_full_open(scene,tmp_path,monkeypatch):
    r=runtime_for(scene);r.__class__=Arm7Runtime
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    connection=scene.connections['right'];original=connection.gripper;at_failure=[]
    def fail(action,**kwargs):
        if action=='release_partial':
            at_failure.extend(connection.commands)
            raise DualArmError('partial opening did not reach measured target')
        return original(action,**kwargs)
    monkeypatch.setattr(connection,'gripper',fail)
    source='motion.grasp(right=[.25,.12,.18,0,0,0])\nmotion.move_tcp(right=[.26,.12,.185,0,0,0])'
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='EXECUTION_FAILED'
        assert connection.commands==at_failure
        assert [x for x in connection.commands if x[0]=='gripper']==[('gripper','open'),('gripper','close')]
        assert any('partial opening' in p.read_text() for p in tmp_path.glob('gripper_*.json'))
    finally:r.close()


def test_staged_release_requires_validated_vertical_lift_before_any_dispatch(scene,tmp_path,monkeypatch):
    r=runtime_for(scene);r.__class__=Arm7Runtime
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    original=r._tcp_segments;tried=[]
    def reject_lift(*args,**kwargs):
        if r.planning_context.get('task_phase')=='host_retreat':
            tried.append(dict(r.planning_context))
            raise DualArmError('vertical release lift unavailable')
        return original(*args,**kwargs)
    monkeypatch.setattr(r,'_tcp_segments',reject_lift)
    source='motion.grasp(right=[.25,.12,.18,0,0,0])\nmotion.move_tcp(right=[.26,.12,.185,0,0,0])'
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='PLANNING_FAILED' and result['physical_execution'] is False
        assert [x['retreat_height_mm'] for x in tried]==[60.,40.,20.]
        assert all(x['inward_mm']==0 for x in tried)
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_release_lift_execution_failure_never_fully_opens_or_returns_home(scene,tmp_path,monkeypatch):
    r=runtime_for(scene);r.__class__=Arm7Runtime
    monkeypatch.setattr(r.paths,'joint_path',lambda a,b,**kw:[a,b])
    connection=scene.connections['right'];original=connection.move_joint_position;stopped_at=[]
    def move(*args,**kwargs):
        if ('gripper','release_partial') in connection.commands:
            stopped_at.extend(connection.commands)
            raise DualArmError('injected lift execution failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(connection,'move_joint_position',move)
    source='motion.grasp(right=[.25,.12,.18,0,0,0])\nmotion.move_tcp(right=[.26,.12,.185,0,0,0])'
    obs=SimpleNamespace(meta={'joints_deg':{k:a.home_joints.copy() for k,a in scene.config.arms.items()}},validate_for=lambda *a,**kw:None)
    try:
        result=r.execute(source,obs,tmp_path)
        assert result['status']=='EXECUTION_FAILED'
        assert connection.commands==stopped_at
        assert [x for x in connection.commands if x[0]=='gripper']==[
            ('gripper','open'),('gripper','close'),('gripper','release_partial')]
        assert scene.connections['left'].commands==[]
    finally:r.close()
