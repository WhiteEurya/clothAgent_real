from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.codex_arm6_runtime import Arm6Runtime
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene
from .test_codex_native_runtime import runtime_for
from .test_codex_loop import FakeRuntime,model
from cloth_agent.codex_loop import loop


def single(scene):
    runtime=runtime_for(scene);runtime.__class__=Arm6Runtime
    return runtime


@pytest.mark.parametrize('source',[
 'motion.gripper(arm="right",action="open")',
 'motion.move_tcp(right=[.2,0,.2,0,0,0])',
 'motion.move_joints(right=[0,0,0,0,0,0,0],left=[0,0,0,0,0,0])',
 'motion.gripper(arm="left",action="close")\nmotion.gripper(arm="right",action="open")',
])
def test_right_commands_rejected_before_any_execution(scene,tmp_path,source):
    r=single(scene);scene.obs.meta['synthetic']=False
    try:
        result=r.execute(source,scene.obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert result['failure_context']['stage']=='arm6_only_command_check'
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_preparation_reaches_perception_then_only_left_folds(scene,tmp_path,monkeypatch):
    r=single(scene)
    scene.connections['right'].joints[0]+=2
    parked=scene.connections['right'].joints.copy()
    target=scene.config.arms['left'].home_joints.copy();target[0]+=2
    monkeypatch.setattr(r,'perception_target',lambda:('left',target))
    monkeypatch.setattr(scene.connections['right'],'prepare_position',lambda:pytest.fail('preparation must not enable arm7'))
    scene.connections['left'].joints[0]+=1
    try:
        r.move_to_perception(tmp_path/'before')
        assert scene.connections['left'].commands[0]==('gripper','open')
        assert np.allclose(scene.connections['left'].joints,target)
        assert np.allclose(scene.connections['right'].joints,parked)
        parked=scene.connections['right'].joints.copy()
        assert scene.connections['right'].commands==[]
        assert r.active_arms==('left',)
        scene.connections['right'].commands.clear()
        monkeypatch.setattr(scene.connections['right'],'prepare_position',lambda:pytest.fail('folding must not prepare arm7'))
        q={k:c.joints.copy() for k,c in scene.connections.items()}
        obs=SimpleNamespace(meta={'joints_deg':q},validate_for=lambda *a,**kw:None)
        result=r.execute('motion.move_tcp(left=[.255,-.12,.18,0,0,0])\nmotion.gripper(arm="left",action="close")',obs,tmp_path)
        assert result['status']=='EXECUTED',result
        assert scene.connections['right'].commands==[]
        assert np.array_equal(scene.connections['right'].joints,parked)
    finally:r.close()


def test_dispatch_rejects_right_even_if_planner_is_bypassed(scene,tmp_path):
    r=single(scene)
    try:
        with pytest.raises(DualArmError,match='rejects arm7'):
            r._execute_operations([('gripper',dict(arm='right',action='open'))],tmp_path)
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_single_arm_prompt_replaces_dual_arm_policy(tmp_path):
    runtime=FakeRuntime();prompts=[]
    for name in ('planning_policy','preparation_description','motion_contract'):
        setattr(runtime,name,getattr(Arm6Runtime,name))
    loop('single arm folding',runtime,tmp_path/'run',max_rounds=1,
         model_call=model(runtime,prompts,['motion.complete()']))
    assert '只使用 arm6' in prompts[0]
    assert '尽可能和human demonstration相似' in prompts[0]
    assert '当前步骤尚未平整' not in prompts[0]
    assert '两爪抓住' not in prompts[0]
    assert 'motion.grasp(left=[x,y,z,roll,pitch,yaw])' in prompts[0]
    assert 'motion.move_tcp(right=' not in prompts[0]


def test_capture_occurs_after_saved_perception_and_failed_preparation_blocks_capture(scene,tmp_path,monkeypatch):
    from cloth_agent.dual_arm import observation
    r=single(scene)
    target=scene.config.arms['left'].home_joints.copy();target[0]+=1
    monkeypatch.setattr(r,'perception_target',lambda:('left',target))
    captures=[]
    monkeypatch.setattr(observation,"create_camera",lambda config:SimpleNamespace(stop=lambda:None))
    def capture(directory,config,models,snapshot,*,camera):
        assert np.allclose(scene.connections['left'].joints,target)
        captures.append(directory);directory.mkdir()
        return SimpleNamespace(meta={'joints_deg':{k:v['joints'] for k,v in snapshot().items()}})
    monkeypatch.setattr(observation,'capture',capture)
    try:
        r.observe(tmp_path/'obs')
        assert len(captures)==1
        count=len(scene.connections['left'].commands)
        r.observe(tmp_path/'same_pose')
        assert len(scene.connections['left'].commands)==count
        assert scene.connections['right'].commands==[]
        def failed(*args):raise DualArmError('injected preparation failure')
        monkeypatch.setattr(r,'move_to_perception',failed)
        with pytest.raises(DualArmError,match='preparation failure'):r.observe(tmp_path/'failed')
        assert len(captures)==2
    finally:r.close()


@pytest.mark.parametrize('drift,allowed',[(1e-13,True),(1e-5,False)])
def test_stationary_arm_roundtrip_noise_is_not_motion(scene,tmp_path,monkeypatch,drift,allowed):
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    r=single(scene)
    start={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    end={k:q.copy() for k,q in start.items()};end['right'][0]+=drift;end['left'][0]+=.1
    dispatched=[]
    monkeypatch.setattr(NativeRuntime,'_execute_operations',lambda *args:dispatched.append(True))
    try:
        operation=[('move',PositionSegment('left',start,end,5,10,30))]
        if allowed:
            r._execute_operations(operation,tmp_path);assert dispatched==[True]
        else:
            with pytest.raises(DualArmError,match='keep arm7 stationary'):r._execute_operations(operation,tmp_path)
            assert not dispatched
    finally:r.close()



def test_grasp_lowering_reaches_planner(scene,tmp_path,monkeypatch):
    from cloth_agent.codex_motion import parse_code
    r=single(scene);scene.obs.meta['synthetic']=False;seen=[]
    def inspect_plan(source,initial):
        seen.append(parse_code(source));raise DualArmError('test ends at planner')
    monkeypatch.setattr(r,'plan',inspect_plan)
    try:
        result=r.execute('motion.move_tcp(left=[.4,0,.043,0,0,0])\nmotion.gripper(arm="left",action="close")',scene.obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert np.isclose(seen[0][0][1]['left'][2],.028)
        assert (tmp_path/'grasp_height_policy.json').exists()
        assert parse_code((tmp_path/'effective_motion.py').read_text())==seen[0]
        assert not any(c.commands for c in scene.connections.values())
    finally:r.close()


def test_arm6_uses_saved_camera_a_and_perception():
    from pathlib import Path
    import json
    path=Path('config/dual_arm.local.json');original=path.read_bytes()
    r=Arm6Runtime(path)
    try:
        owner,q=r.perception_target()
        pose=json.loads(Path('data/robot/xarm_perception_pose.json').read_text())
        assert owner=='left' and np.allclose(q,pose['joint_angles_deg'][:6])
        assert r.config.raw['camera']['serial']=='317222073552'
        assert r.config.raw['camera']['mount_arm']=='left'
        assert r.config.collision_envelopes_disabled_arms==frozenset({'left'})
        assert r.config.raw['camera']['depth_exposure_us']==8500.
        assert r.config.raw['camera']['depth_gain']==16.
        assert r.config.raw['camera']['depth_stability']['warmup_frames']==30
        assert path.read_bytes()==original
    finally:r.close()


def test_declared_grasp_lowered_15mm_once(tmp_path):
    from cloth_agent.codex_motion import parse_code
    r=object.__new__(Arm6Runtime)
    source='motion.grasp(left=[.4,0,.043,0,0,0])\nmotion.move_tcp(left=[.4,0,.08,0,0,0])'
    calls=parse_code(r.adjust_motion(source,tmp_path),allow_grasp=True)
    assert calls[0][1]['left'][2]==pytest.approx(.028)
    assert calls[1][1]['left'][2]==.08
    assert '额外降低 15 mm' in r.motion_contract
    assert '15 mm' in r.planning_policy
