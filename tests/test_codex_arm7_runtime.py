from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.codex_arm7_runtime import Arm7Runtime,wrist_equivalent
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene
from .test_codex_native_runtime import runtime_for
from .test_codex_loop import FakeRuntime,model
from cloth_agent.codex_loop import loop


def single(scene):
    runtime=runtime_for(scene);runtime.__class__=Arm7Runtime
    return runtime


@pytest.mark.parametrize('angle,seed,prefer,expected',[
    (185.9,.3,True,-174.1),
    (120.,0.,True,120.),
    (20.,0.,True,20.),
    (20.,0.,False,20.),
    (0.,0.,True,0.),
    (-20.,0.,True,-20.),
    (185.9,-174.,False,-174.1),
    (-179.,179.,False,181.),
    (0.,-174.,False,0.),
])
def test_wrist_winding_direction_and_continuity(angle,seed,prefer,expected):
    assert wrist_equivalent(angle,seed,-360.,360.,prefer_ccw=prefer)==pytest.approx(expected)


def test_wrist_preference_respects_limits():
    assert wrist_equivalent(120.,0.,-180.,180.,prefer_ccw=True)==120.
    assert wrist_equivalent(20.,0.,-180.,180.,prefer_ccw=True)==20.


def test_arm7_queues_fold_moves_but_preserves_preparation_boundaries(scene,monkeypatch):
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    runtime=single(scene)
    q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    b={k:v.copy() for k,v in q.items()};b['right'][0]+=.1
    c={k:v.copy() for k,v in b.items()};c['right'][0]+=.1
    original=[('move',PositionSegment('right',q,b,5,10,30)),
              ('move',PositionSegment('right',b,c,5,10,30))]
    def plan(*args,**kwargs):
        runtime.plan_stats={}
        return original
    monkeypatch.setattr(NativeRuntime,'plan',plan)
    source='motion.move_tcp(right=[.4,0,.1,0,0,0])'
    try:
        batch=runtime.plan(source,q)
        assert len(batch)==1 and batch[0][1].segments==tuple(v for _,v in original)
        assert runtime.plan(source,q,preserve_command_boundaries=True)==original
    finally:runtime.close()


def test_arm7_rejects_hidden_arm6_motion_inside_queue(scene,tmp_path):
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    from cloth_agent.queued_motion import queue_moves
    runtime=single(scene)
    q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    b={k:v.copy() for k,v in q.items()};b['left'][0]+=1.;b['right'][0]+=.1
    c={k:v.copy() for k,v in q.items()};c['right'][0]+=.2
    batch=queue_moves([('move',PositionSegment('right',q,b,5,10,30)),
                       ('move',PositionSegment('right',b,c,5,10,30))])
    try:
        with pytest.raises(DualArmError,match='keep arm6 stationary'):
            runtime._execute_operations(batch,tmp_path)
        assert not any(c.commands for c in scene.connections.values())
    finally:runtime.close()


@pytest.mark.parametrize('angle,expected',[(185.9,-174.1),(20.,20.)])
def test_wrist_equivalent_preserves_real_arm7_tcp(monkeypatch,angle,expected):
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm.geometry import pose_error
    runtime=Arm7Runtime('config/dual_arm.local.json')
    runtime.planning_context={}
    model=runtime.models['right']
    seed=model.config.home_joints.copy();seed[-1]=.3
    q=seed.copy();q[-1]=angle
    target=model.forward(q)
    monkeypatch.setattr(NativeRuntime,'_inverse_target',lambda *args:q.copy())
    try:
        selected=runtime._inverse_target('right',target,seed,'host_approach_ik')
        assert selected[-1]==pytest.approx(expected)
        position,rotation=pose_error(model.forward(selected),target)
        assert position<1e-6 and rotation<1e-6
        continuation=runtime._inverse_target('right',target,selected,'tcp_direct_ik')
        np.testing.assert_allclose(continuation,selected)
    finally:runtime.close()


@pytest.mark.parametrize('source',[
 'motion.gripper(arm="left",action="open")',
 'motion.move_tcp(left=[.2,0,.2,0,0,0])',
 'motion.move_joints(left=[0,0,0,0,0,0],right=[0,0,0,0,0,0,0])',
 'motion.gripper(arm="right",action="close")\nmotion.gripper(arm="left",action="open")',
])
def test_left_commands_rejected_before_any_execution(scene,tmp_path,source):
    r=single(scene);scene.obs.meta['synthetic']=False
    try:
        result=r.execute(source,scene.obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert result['failure_context']['stage']=='arm7_only_command_check'
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_preparation_reaches_perception_then_only_right_folds(scene,tmp_path,monkeypatch):
    r=single(scene)
    scene.connections['left'].joints[0]+=2
    parked=scene.connections['left'].joints.copy()
    target=scene.config.arms['right'].home_joints.copy();target[0]+=2
    monkeypatch.setattr(r,'perception_target',lambda:('right',target))
    monkeypatch.setattr(scene.connections['left'],'prepare_position',lambda:pytest.fail('preparation must not enable arm6'))
    scene.connections['right'].joints[0]+=1
    try:
        r.move_to_perception(tmp_path/'before')
        assert scene.connections['right'].commands[0]==('gripper','open')
        assert np.allclose(scene.connections['right'].joints,target)
        assert np.allclose(scene.connections['left'].joints,parked)
        parked=scene.connections['left'].joints.copy()
        assert scene.connections['left'].commands==[]
        assert r.active_arms==('right',)
        scene.connections['left'].commands.clear()
        monkeypatch.setattr(scene.connections['left'],'prepare_position',lambda:pytest.fail('folding must not prepare arm6'))
        q={k:c.joints.copy() for k,c in scene.connections.items()}
        obs=SimpleNamespace(meta={'joints_deg':q},validate_for=lambda *a,**kw:None)
        result=r.execute('motion.move_tcp(right=[.255,.12,.18,0,0,0])\nmotion.gripper(arm="right",action="close")',obs,tmp_path)
        assert result['status']=='EXECUTED',result
        assert scene.connections['left'].commands==[]
        assert np.array_equal(scene.connections['left'].joints,parked)
    finally:r.close()


def test_dispatch_rejects_left_even_if_planner_is_bypassed(scene,tmp_path):
    r=single(scene)
    try:
        with pytest.raises(DualArmError,match='rejects arm6'):
            r._execute_operations([('gripper',dict(arm='left',action='open'))],tmp_path)
        assert all(not c.commands for c in scene.connections.values())
    finally:r.close()


def test_single_arm_prompt_replaces_dual_arm_policy(tmp_path):
    runtime=FakeRuntime();prompts=[]
    for name in ('planning_policy','preparation_description','motion_contract'):
        setattr(runtime,name,getattr(Arm7Runtime,name))
    loop('single arm folding',runtime,tmp_path/'run',max_rounds=1,
         model_call=model(runtime,prompts,['motion.complete()']))
    assert '只使用 arm7' in prompts[0]
    assert '尽可能和human demonstration相似' in prompts[0]
    assert '当前步骤尚未平整' not in prompts[0]
    assert '两爪抓住' not in prompts[0]
    assert 'motion.grasp(right=[x,y,z,roll,pitch,yaw])' in prompts[0]
    assert 'motion.move_tcp(left=' not in prompts[0]


def test_capture_occurs_after_saved_perception_and_failed_preparation_blocks_capture(scene,tmp_path,monkeypatch):
    from cloth_agent.dual_arm import observation
    r=single(scene)
    target=scene.config.arms['right'].home_joints.copy();target[0]+=1
    monkeypatch.setattr(r,'perception_target',lambda:('right',target))
    captures=[]
    monkeypatch.setattr(observation,"create_camera",lambda config:SimpleNamespace(stop=lambda:None))
    def capture(directory,config,models,snapshot,*,camera):
        assert np.allclose(scene.connections['right'].joints,target)
        captures.append(directory);directory.mkdir()
        return SimpleNamespace(meta={'joints_deg':{k:v['joints'] for k,v in snapshot().items()}})
    monkeypatch.setattr(observation,'capture',capture)
    try:
        r.observe(tmp_path/'obs')
        assert len(captures)==1
        count=len(scene.connections['right'].commands)
        r.observe(tmp_path/'same_pose')
        assert len(scene.connections['right'].commands)==count
        assert scene.connections['left'].commands==[]
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
    end={k:q.copy() for k,q in start.items()};end['left'][0]+=drift;end['right'][0]+=.1
    dispatched=[]
    monkeypatch.setattr(NativeRuntime,'_execute_operations',lambda *args:dispatched.append(True))
    try:
        operation=[('move',PositionSegment('right',start,end,5,10,30))]
        if allowed:
            r._execute_operations(operation,tmp_path);assert dispatched==[True]
        else:
            with pytest.raises(DualArmError,match='keep arm6 stationary'):r._execute_operations(operation,tmp_path)
            assert not dispatched
    finally:r.close()


def test_arm7_loads_its_own_pose_and_camera_without_mutating_dual_config():
    import json
    from pathlib import Path
    path=Path('config/dual_arm.local.json');original=path.read_bytes()
    r=Arm7Runtime(path)
    try:
        pose=json.loads(Path('data/robot/xarm7_perception_pose.json').read_text())
        owner,q=r.perception_target()
        assert owner=='right' and np.allclose(q,pose['joint_angles_deg'])
        camera=r.config.raw['camera']
        assert camera['serial']==pose['camera_serial']=='233622079809'
        assert camera['mount_arm']=='right'
        assert camera['depth_exposure_us']==4000.
        assert camera['depth_gain']==16.
        assert r.config.collision_envelopes_disabled_arms==frozenset({'right'})
        assert set(r.config.static_arm_obstacles)=={'left'}
        assert len(r.config.static_arm_obstacles['left']['joint_angles_deg'])==6
        assert 'extrinsic_tilt_correction' not in camera
        import yaml
        extrinsic_path=Path('config/calibration/dual_arm_working_20261008/camB_extrinsics.yaml')
        extrinsic=yaml.safe_load(extrinsic_path.read_text())
        expected=np.asarray(extrinsic['X_CammountCam'],dtype=float).copy()
        assert extrinsic['translation_units']=='m'
        expected[:3,3]*=1000
        assert np.allclose(camera['mount_from_camera_mm'],expected)
        assert path.read_bytes()==original
    finally:r.close()


def test_grasp_lowering_reaches_planner_and_saved_effective_code(scene,tmp_path,monkeypatch):
    from cloth_agent.codex_motion import parse_code
    r=single(scene);scene.obs.meta['synthetic']=False;seen=[]
    def inspect_plan(source,initial):
        seen.append(parse_code(source));raise DualArmError('test ends at planner')
    monkeypatch.setattr(r,'plan',inspect_plan)
    try:
        result=r.execute('motion.move_tcp(right=[.4,0,.043,0,0,0])\nmotion.gripper(arm="right",action="close")',scene.obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        assert np.isclose(seen[0][0][1]['right'][2],.043)
        assert parse_code((tmp_path/'effective_motion.py').read_text())==seen[0]
        assert not any(c.commands for c in scene.connections.values())
    finally:r.close()


def test_declared_arm7_grasp_has_no_extra_lowering(tmp_path):
    import json
    from cloth_agent.codex_motion import parse_code
    runtime=object.__new__(Arm7Runtime)
    source='motion.grasp(right=[.4,0,.043,0,0,0])\nmotion.move_tcp(right=[.4,0,.08,0,0,0])'
    effective=runtime.adjust_motion(source,tmp_path)
    calls=parse_code(effective,allow_grasp=True)
    assert calls[0][1]['right'][2]==pytest.approx(.043)
    assert calls[1][1]['right'][2]==.08
    assert json.loads((tmp_path/'grasp_height_policy.json').read_text())['offset_mm']==0.
    assert '自主选择抓取及后续动作的 TCP 高度' in runtime.planning_policy
    prompt=runtime.planning_policy+runtime.motion_contract
    assert '观测表面' not in prompt
    assert '原始观测 Z' not in prompt


@pytest.mark.parametrize('unfold', [False, True])
def test_fold_and_unfold_share_parallel_jaw_symmetry(tmp_path, monkeypatch, unfold):
    import json
    from scipy.spatial.transform import Rotation as R
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.codex_motion import parse_code
    runtime=object.__new__(Arm7Runtime)
    runtime.unfold_grasp_symmetry=unfold
    runtime.config=SimpleNamespace(arms={'right':SimpleNamespace(world_from_base=np.eye(4))})
    runtime.models={'right':SimpleNamespace(forward=lambda q: np.array([400,200,100,180,0,180]))}
    observation=SimpleNamespace(meta={'joints_deg':{'right':[0]*7}})
    captured=[]
    def execute(self,source,obs,folder):
        captured.append(self.adjust_motion(source,folder))
        return {'status':'EXECUTED','physical_execution':False}
    monkeypatch.setattr(NativeRuntime,'execute',execute)
    source=('motion.grasp(right=[.4,.2,.07,3.141592653589793,0,0])\n'
            'motion.move_transfer(right=[.5,.1,.12,2.8,.2,.4])\n'
            'motion.move_tcp(right=[.5,0,.08,2.9,.1,.3])')
    runtime.execute(source,observation,tmp_path)
    report=json.loads((tmp_path/'grasp_orientation_policy.json').read_text())
    assert report['applied'] and report['equivalent_rotation_deg']<1e-6
    old=parse_code(source,allow_grasp=True); new=parse_code(captured[0],allow_grasp=True)
    assert [n for n,_ in old]==[n for n,_ in new]
    for i,((_,a),(_,b)) in enumerate(zip(old,new)):
        expected=np.array(a['right'][:3])
        assert np.allclose(b['right'][:3],expected)
    original=[R.from_euler('xyz',kw['right'][3:]) for _,kw in old]
    actual=[R.from_euler('xyz',kw['right'][3:]) for _,kw in new]
    for a,b in zip(original,actual):
        assert np.allclose((a*original[0].inv()).as_matrix(),(b*actual[0].inv()).as_matrix())


def test_non_grasp_program_does_not_require_orientation_observation(tmp_path,monkeypatch):
    from cloth_agent.codex_native_runtime import NativeRuntime
    runtime=object.__new__(Arm7Runtime)
    source='motion.move_joints(right=[0,0,0,0,0,0,0])'
    monkeypatch.setattr(NativeRuntime,'execute',lambda self,code,*args: code)
    assert runtime.execute(source,SimpleNamespace(),tmp_path)==source
    assert not (tmp_path/'grasp_orientation_policy.json').exists()


def test_initial_wrist_never_forces_full_turn_at_limits():
    with pytest.raises(DualArmError,match='180 degrees'):
        wrist_equivalent(170.,-170.,-180.,180.,prefer_ccw=True)
    assert wrist_equivalent(180.,0.,-360.,360.,prefer_ccw=True)==-180.
