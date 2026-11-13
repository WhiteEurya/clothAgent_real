import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime, route_grasp


def source(x, y=0.):
    return (f'motion.grasp(right={[x, y, .02, 0., 0., 0.]})\n'
            'motion.move_tcp(right=[0.9, 0.2, 0.03, 0., 0., 1.])\n')


@pytest.mark.parametrize('x,arm', [(-.2, 'left'), (.2, 'right'), (0., 'right')])
def test_route_and_crossing_target(x, arm):
    routed, audit = route_grasp(source(x), {'left': [-300, 0, 150], 'right': [300, 0, 100]})
    assert audit['active_arm'] == arm
    original, actual = parse_code(source(x), allow_grasp=True), parse_code(routed, allow_grasp=True)
    assert all(set(args) == {arm} for _, args in actual)
    assert [args[arm] for _, args in actual] == [args['right'] for _, args in original]


def test_rotated_and_translated_tcp_boundary():
    # Physical left/right are along Y here, not image X or world X.
    _, audit = route_grasp(source(.8, .1), {'left': [800, 0, 100], 'right': [800, 400, 100]})
    assert audit['active_arm'] == 'left'
    _, moved = route_grasp(source(.8, .1), {'left': [800, -400, 100], 'right': [800, 0, 100]})
    assert moved['active_arm'] == 'right'


def test_reject_ambiguous_tcp_and_dual_grasp():
    with pytest.raises(DualArmError, match='indistinguishable'):
        route_grasp(source(0), {'left': [0, 0, 100], 'right': [0, 0, 200]})
    with pytest.raises(DualArmError, match='logical right'):
        route_grasp(source(0).replace('right=', 'left='), {'left': [-300, 0, 100], 'right': [300, 0, 100]})


def test_current_config_loads_both_live_models_without_old_arm6_pose(tmp_path):
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json', planning_only=True)
    try:
        assert runtime.active_arms == ('left', 'right')
        assert not hasattr(runtime.config, "static_arm_obstacles")
        assert runtime.config.execution_mode == "geometry_only"
        assert all(a.axis == 7 for a in runtime.config.arms.values())
        assert runtime.config.raw['camera']['backend'] == 'zed'
        with pytest.raises(DualArmError, match='cannot execute'):
            runtime.execute('', None, tmp_path)
    finally:
        runtime.close()


def test_failed_snapshot_preserves_error_without_checks(tmp_path, monkeypatch):
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json')
    runtime._lease = (tmp_path / 'offline.lock').open('w')
    monkeypatch.setattr('cloth_agent.site_robot_check.read_arm', lambda *a, **kw:
        dict(status='READ_FAILED', error='connect socket failed'))
    try:
        with pytest.raises(DualArmError, match='left.*READ_FAILED.*connect socket failed'):
            runtime.observe(tmp_path / 'capture')
    finally:
        runtime.close()


def test_factory_selects_partitioned_runtime_for_current_site(monkeypatch):
    from cloth_agent.experiments.graph_policy import make_runtime
    calls = []
    runtime = SimpleNamespace(config=object())
    def create(config):
        calls.append(config)
        return runtime
    monkeypatch.setattr('cloth_agent.replay_dual_arm_runtime.PartitionedReplayRuntime', create)
    monkeypatch.setattr('cloth_agent.trajectory_preview.TrajectoryPreview', lambda *a, **k: 'preview')
    path = 'config/dual_arm.zed_dual_xarm7_20261103.json'
    assert make_runtime(path, 8088) is runtime
    assert calls == [path]


@pytest.mark.parametrize('moved', [False, True])
def test_fixed_camera_reads_arms_only_before_and_after_capture(tmp_path, monkeypatch, moved):
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json')
    # Mock controllers/camera must not compete with a live robot process lease.
    runtime._lease = (tmp_path / 'offline.lock').open('w')
    calls = []
    monkeypatch.setattr(runtime, '_open_startup', lambda directory,before,snapshots: before)
    def perception(directory,before,snapshots):
        directory.mkdir(parents=True)
        (directory/'result.json').write_text('{}')
        return before
    monkeypatch.setattr(runtime, '_perception_before_capture', perception)
    class Camera:
        started = False
        intrinsics = np.diag([1000., 1000., 1.])
        frames = 0
        def start(self): self.started = True
        def read(self):
            self.frames += 1
            depth = np.full((16, 16), .7)
            depth[:5, :5] = np.nan  # Missing background must not reject capture.
            if self.frames == 13:
                depth[8, 8] = 1.5  # Existing temporal median removes an outlier.
            if self.frames == 14:
                depth[8, 8] = 0.  # Existing invalid-depth filtering excludes zero.
            return np.zeros((16, 16, 3), dtype=np.uint8), depth
        def stop(self): self.started = False
    camera = Camera()
    def read(profile, key, **kwargs):
        assert kwargs == ({'recover_stopped': True} if len(calls) < 2 else {})
        calls.append(key)
        q = runtime.config.arms[key].home_joints.copy()
        if moved and len(calls) > 2:
            q[0] += 1.
        return dict(status='READ_ONLY_CHECKS_PASS', joints_deg=q.tolist(),
                    control_box_sn=runtime.config.arms[key].serial)
    monkeypatch.setattr('cloth_agent.site_robot_check.read_arm', read)
    def create(spec, *args):
        camera.spec = spec
        return camera
    monkeypatch.setattr('cloth_agent.perception.create_rgbd_camera', create)
    try:
        if moved:
            with pytest.raises(DualArmError, match='arm moved'):
                runtime.observe(tmp_path / 'capture')
        else:
            observation = runtime.observe(tmp_path / 'capture')
            matrix = np.asarray(runtime.config.raw['camera']['world_from_camera_mm'])
            expected = matrix[:3, :3] @ np.array([5.6, 5.6, 700.]) + matrix[:3, 3]
            np.testing.assert_allclose(observation.xyz[8, 8], expected, atol=1e-4)
            assert np.isfinite(observation.sample([8, 8], 20.)).all()
            with pytest.raises(DualArmError, match='insufficient measured depth'):
                observation.sample([2, 2], 20.)
            observation.validate_for(runtime.config, live=True)
        assert calls == ['left', 'right', 'left', 'right']
        assert camera.frames == 15
        assert not camera.started
    finally:
        runtime.close()


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('contact_offset', [0., -5., -10., -15.])
def test_graph_compiles_to_original_experiment_api_for_either_arm(tmp_path, arm, contact_offset):
    from cloth_agent.dual_arm.geometry import matrix_pose, pose_matrix
    from cloth_agent.experiment import ExperimentRunner
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json', planning_only=True)
    runtime.task_policy['grasp_surface_offset_mm'] = contact_offset
    joints = {k: a.home_joints.copy() for k, a in runtime.config.arms.items()}
    pose = matrix_pose(runtime.config.arms[arm].world_from_base @ pose_matrix(runtime.models[arm].forward(joints[arm])))
    grasp = np.r_[pose[:3] / 1000, np.radians(pose[3:])].tolist()
    target = grasp.copy(); target[2] += .02
    source = f'motion.grasp(right={grasp!r})\nmotion.move_tcp(right={target!r})\n'
    config, program, assignment = runtime.compile(source, SimpleNamespace(meta={'joints_deg': joints}))
    assert assignment['active_arm'] == arm
    assert config.robot_ip == runtime.config.arms[arm].ip
    assert config.collision_scene_path
    runner = ExperimentRunner(tmp_path / arm, config)
    path = runner.workspace / 'action.py'; path.write_text(program)
    preflight = runner.preflight(path)
    assert preflight.error is None
    assert [a['name'] for a in preflight.actions] == ['home', 'open_gripper', 'move', 'move', 'close_gripper',
                                                    'move', 'open_gripper', 'move', 'home']
    up = runtime.config.arms[arm].world_from_base[:3, :3].T @ np.array([0., 0., 1.])
    base = runtime.models[arm].forward(joints[arm])[:3]
    np.testing.assert_allclose(preflight.actions[3]['actual_ee_pose'][:3], base + contact_offset * up)
    np.testing.assert_allclose(preflight.actions[2]['actual_ee_pose'][:3], base + runtime.task_policy['approach_clearance_mm'] * up)
    np.testing.assert_allclose(preflight.actions[5]['actual_ee_pose'][:3], base + 20. * up)
    runtime.close()


@pytest.mark.parametrize('failure', [None, 'plan_right', 'execute_left'])
def test_startup_home_checks_both_legs_and_runs_sequentially(tmp_path, monkeypatch, failure):
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json')
    actual = {k: a.home_joints.copy() + .5 for k, a in runtime.config.arms.items()}
    original_right = actual['right'].copy()
    events = []
    class Checker:
        def __init__(self, path, initial, key, **kw):
            self.key = key
            self.initial = {k: np.asarray(q).copy() for k, q in initial.items()}
        def require_execution(self, config, actions):
            events.append(('plan', self.key))
            if self.key == 'right':
                np.testing.assert_array_equal(self.initial['left'], runtime.config.arms['left'].home_joints)
            if failure == 'plan_right' and self.key == 'right':
                raise DualArmError('blocked right path')
    class Runner:
        def __init__(self, folder, config, collision_checker):
            self.workspace = folder
            folder.mkdir(parents=True)
            self.checker = collision_checker
        def run_experiment(self, path, **kw):
            assert path.read_text() == 'def run():\n    home()\n'
            assert kw == dict(real=True, confirmed=True)
            key = self.checker.key
            events.append(('execute', key))
            for k in actual:
                np.testing.assert_array_equal(self.checker.initial[k], actual[k])
            if key == 'left':
                np.testing.assert_array_equal(actual['right'], original_right)
            if failure == 'execute_left':
                return dict(execution_completed=False)
            actual[key] = runtime.config.arms[key].home_joints.copy()
            return dict(execution_completed=True)
    def snapshots():
        events.append(('snapshot', None))
        return {k: {'joints': q.tolist()} for k, q in actual.items()}
    before = snapshots(); events.clear()
    monkeypatch.setattr('cloth_agent.robot_collision.RobotCollision', Checker)
    monkeypatch.setattr('cloth_agent.experiment.ExperimentRunner', Runner)
    try:
        if failure:
            with pytest.raises(DualArmError):
                runtime._home_startup(tmp_path / 'home', before, snapshots)
            assert ('execute', 'right') not in events
            if failure == 'plan_right':
                assert not any(kind == 'execute' for kind, _ in events)
        else:
            after = runtime._home_startup(tmp_path / 'home', before, snapshots)
            assert events == [('plan', 'left'), ('plan', 'right'), ('execute', 'left'),
                              ('snapshot', None), ('execute', 'right'), ('snapshot', None)]
            for key in actual:
                np.testing.assert_array_equal(after[key]['joints'], runtime.config.arms[key].home_joints)
    finally:
        runtime.close()


@pytest.mark.parametrize('failure',[None,'left','right'])
def test_startup_opens_both_even_at_home_and_blocks_on_failure(tmp_path,monkeypatch,failure):
    runtime=PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json')
    before={k:{'joints':a.home_joints.tolist()} for k,a in runtime.config.arms.items()}
    events=[]
    class Checker:
        def __init__(self,path,initial,key,**kwargs):self.key=key
        def require_execution(self,config,actions):
            assert actions==[{'name':'open_gripper','args':{}}]
    class Runner:
        def __init__(self,folder,config,collision_checker):
            self.workspace=folder;folder.mkdir(parents=True);self.key=collision_checker.key
        def run_experiment(self,path,**kwargs):
            assert path.read_text()=='def run():\n    open_gripper()\n'
            events.append(self.key)
            return {'execution_completed':self.key!=failure}
    monkeypatch.setattr('cloth_agent.robot_collision.RobotCollision',Checker)
    monkeypatch.setattr('cloth_agent.experiment.ExperimentRunner',Runner)
    try:
        if failure:
            with pytest.raises(DualArmError,match='Home not started'):
                runtime._open_startup(tmp_path,before,lambda:before)
        else:assert runtime._open_startup(tmp_path,before,lambda:before)==before
        assert events==(['left'] if failure=='left' else ['left','right'])
    finally:runtime.close()


@pytest.mark.parametrize('arm',['left','right'])
@pytest.mark.parametrize('clearance',[5.,40.])
def test_grasp_descent_clamped_to_configured_floor(monkeypatch,arm,clearance):
    from dataclasses import replace
    from cloth_agent.dual_arm.geometry import matrix_pose,pose_matrix
    from cloth_agent.experiments import full_grasp_adapter as adapter
    runtime=PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json',planning_only=True)
    try:
        runtime.task_policy['grasp_surface_offset_mm']=-20.
        joints={k:a.home_joints.copy() for k,a in runtime.config.arms.items()}
        base=runtime.models[arm].forward(joints[arm])
        pose=matrix_pose(runtime.config.arms[arm].world_from_base@pose_matrix(base))
        original=adapter.build_robot_config
        def build(*args,**kwargs):
            cfg=original(*args,**kwargs)
            return replace(cfg,boundaries=replace(cfg.boundaries,z_min=float(base[2]-clearance)),lower_z_margin_mm=0.)
        monkeypatch.setattr(adapter,'build_robot_config',build)
        grasp=np.r_[pose[:3]/1000,np.radians(pose[3:])].tolist()
        target=grasp.copy();target[2]+=.02
        cfg,program,assignment=runtime.compile(f'motion.grasp(right={grasp!r})\nmotion.move_tcp(right={target!r})\n',SimpleNamespace(meta={'joints_deg':joints}))
        up=runtime.config.arms[arm].world_from_base[:3,:3].T@np.array([0.,0.,1.])
        descent=min(20.,clearance/up[2])
        np.testing.assert_allclose(assignment['contact_base_mm_deg'][:3],base[:3]-descent*up,atol=1e-6)
        assert assignment['contact_base_mm_deg'][2]>=cfg.boundaries.z_min-1e-6
        assert assignment['grasp_height_limit']['applied_descent_mm']==pytest.approx(descent)
    finally:runtime.close()


@pytest.mark.parametrize('case',['success','already','plan_failure','motion_failure','wrong_target','right_moved'])
def test_perception_position_uses_checked_motion_and_measured_arrival(tmp_path,monkeypatch,case):
    runtime=PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json')
    target=np.asarray(runtime.perception_position['joints_deg'])
    before={k:{'joints':a.home_joints.tolist()} for k,a in runtime.config.arms.items()}
    if case=='already':before['left']['joints']=target.tolist()
    events=[]
    class Checker:
        def __init__(self,path,initial,active,**kw):
            assert active=='left'
            assert initial=={k:v['joints'] for k,v in before.items()}
        def require_execution(self,config,actions):
            events.append('validate')
            assert actions==[{'name':'perception_position','args':{}}]
            np.testing.assert_array_equal(config.perception_joints_deg,target)
            if case=='plan_failure':raise DualArmError('path blocked')
            return []
    class Backend:
        def __init__(self,config,*,collision_checker):events.append('connect')
        def perception_position(self,config):
            events.append('move')
            if case=='motion_failure':raise DualArmError('motion failed')
            return list(config.perception_pose_mm_deg),{}
        def stop(self):events.append('stop')
        def close(self):events.append('close')
    def snapshots():
        events.append('snapshot')
        after={k:{'joints':list(v['joints'])} for k,v in before.items()}
        after['left']['joints']=(target+(1 if case=='wrong_target' else 0)).tolist()
        if case=='right_moved':after['right']['joints'][0]+=1
        return after
    monkeypatch.setattr('cloth_agent.robot_collision.RobotCollision',Checker)
    monkeypatch.setattr('cloth_agent.robot_api.XArmBackend',Backend)
    try:
        if case in ('success','already'):
            after=runtime._perception_before_capture(tmp_path,before,snapshots)
            np.testing.assert_array_equal(after['left']['joints'],target)
            assert events==([] if case=='already' else ['validate','connect','move','snapshot','close'])
        else:
            with pytest.raises(DualArmError):runtime._perception_before_capture(tmp_path,before,snapshots)
            assert events==(['validate'] if case=='plan_failure' else
                ['validate','connect','move']+([] if case=='motion_failure' else ['snapshot'])+['stop','close'])
        report=json.loads((tmp_path/'result.json').read_text())
        assert report['status']=={'success':'REACHED_PERCEPTION','already':'ALREADY_AT_PERCEPTION'}.get(case,'FAILED')
    finally:runtime.close()


@pytest.mark.parametrize('failure',[False,True])
def test_observe_moves_before_camera_every_time(tmp_path,monkeypatch,failure):
    runtime=PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json')
    runtime._lease=(tmp_path/'offline.lock').open('w')
    runtime._startup_checked=runtime._startup_home_complete=True
    events=[]
    def read(profile,key,**kwargs):
        return dict(status='READ_ONLY_CHECKS_PASS',joints_deg=runtime.config.arms[key].home_joints.tolist(),
                    control_box_sn=runtime.config.arms[key].serial)
    def position(directory,before,snapshots):
        events.append('perception')
        if failure:raise DualArmError('position failed')
        return before
    class Captured(Exception):pass
    def capture(*args,**kwargs):
        events.append('capture')
        raise Captured()
    monkeypatch.setattr('cloth_agent.site_robot_check.read_arm',read)
    monkeypatch.setattr(runtime,'_perception_before_capture',position)
    monkeypatch.setattr('cloth_agent.perception.capture_two_view_rgbd',capture)
    try:
        for i in range(2):
            with pytest.raises(DualArmError if failure else Captured):runtime.observe(tmp_path/f'obs_{i}')
        assert events==(['perception','perception'] if failure else ['perception','capture']*2)
    finally:runtime.close()


@pytest.mark.parametrize('left_ok,right_ok,expected',[(True,True,'left'),(False,True,'right'),(False,False,None)])
def test_workspace_fallback_checks_complete_program_before_switch(left_ok,right_ok,expected):
    from cloth_agent.config import SafetyError
    compiled=[]
    def candidate(source,observation,active_arm=None,**kwargs):
        arm=active_arm or 'left';compiled.append(arm)
        def validate(x,y,z,yaw):
            # Only the transport endpoint is outside; grasp itself fits.
            if x==2 and not {'left':left_ok,'right':right_ok}[arm]:raise SafetyError('outside workspace')
        cfg=SimpleNamespace(validate_workspace_pose=validate)
        return cfg,'def run():\n    move(1,0,0,0)\n    move(2,0,0,0)\n',{'active_arm':arm,'inactive_arm':'right' if arm=='left' else 'left'}
    robot=SimpleNamespace(workspace_arm_fallback=True,_compile_for_arm=candidate,
        config=SimpleNamespace(arms={k:SimpleNamespace(world_from_base=np.eye(4),home_joints=np.zeros(7)) for k in ('left','right')}),
        models={k:SimpleNamespace(forward=lambda q:np.zeros(6)) for k in ('left','right')})
    source='motion.grasp(right=[0,0,0,0,0,0])\nmotion.move_tcp(right=[0,0,0,0,0,0])\n'
    if expected is None:
        with pytest.raises(DualArmError,match='Neither arm'):PartitionedReplayRuntime.compile(robot,source,None)
    else:
        _,_,assignment=PartitionedReplayRuntime.compile(robot,source,None)
        assert assignment['active_arm']==expected
        assert assignment['workspace_candidates'][expected]['status']=='PASS'
    assert compiled==(['left'] if left_ok else ['left','right'])


@pytest.mark.parametrize('arm', ['left', 'right'])
@pytest.mark.parametrize('world_y,reset,expected', [(-450.,False,2.),(-400.,False,2.),(-399.,False,0.),(-450.,True,0.)])
def test_far_grasp_raise_shared_world_frame(arm, world_y, reset, expected):
    from cloth_agent.dual_arm.geometry import matrix_pose, pose_matrix
    runtime=PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json', planning_only=True)
    try:
        runtime.task_policy['grasp_surface_offset_mm']=-20.
        joints={k:a.home_joints.copy() for k,a in runtime.config.arms.items()}
        pose=matrix_pose(runtime.config.arms[arm].world_from_base@pose_matrix(runtime.models[arm].forward(joints[arm])))
        pose[1]=world_y
        grasp=np.r_[pose[:3]/1000,np.radians(pose[3:])].tolist()
        target=grasp.copy();target[2]+=.02
        code=f'motion.grasp(right={grasp!r})\nmotion.move_tcp(right={target!r})\n'
        obs=SimpleNamespace(meta={'joints_deg':joints})
        kwargs=dict(active_arm=arm,post_release_descent_mm=50. if reset else 0.)
        _,_,before=runtime._compile_for_arm(code,obs,**kwargs)
        runtime.task_policy.update(far_grasp_world_y_max_mm=-400.,far_grasp_raise_mm=2.)
        _,_,after=runtime._compile_for_arm(code,obs,**kwargs)
        up=runtime.config.arms[arm].world_from_base[:3,:3].T@np.array([0.,0.,1.])
        np.testing.assert_allclose(np.array(after['contact_base_mm_deg'][:3])-before['contact_base_mm_deg'][:3],expected*up,atol=1e-6)
        assert after['grasp_surface_offset_mm']==pytest.approx(-20.+expected)
    finally:
        runtime.close()

@pytest.mark.parametrize('rejected', [(), ('left',), ('left','right')])
def test_complete_path_fallback_before_any_dispatch(tmp_path, monkeypatch, rejected):
    from cloth_agent import experiment, robot_collision
    from cloth_agent.config import SafetyError
    events=[]
    class Checker:
        def __init__(self, config, joints, arm, **kwargs):self.arm=arm
        def require_execution(self, config, actions):
            events.append(('check',self.arm))
            if self.arm in rejected:
                error=SafetyError('action 7: self-collision' if self.arm=='left' else 'TCP projection 882 exceeds halfspace')
                if self.arm=='left':
                    error.planning_failure=dict(kind='IK_REJECTED',action_index=7,active_arm='left',
                        diagnostics={'target_base_mm_deg':[119,-96,71,178,0,0]})
                raise error
    class Runner:
        def __init__(self, folder, config, collision_checker):
            self.workspace=folder/'workspace';self.workspace.mkdir(parents=True)
            self.arm=collision_checker.arm
        def preflight(self, path):return SimpleNamespace(error=None,actions=[])
        def run_experiment(self,*args,**kwargs):
            events.append(('dispatch',self.arm))
            return dict(execution_completed=True,physical_execution=True)
    monkeypatch.setattr(robot_collision,'RobotCollision',Checker)
    monkeypatch.setattr(experiment,'ExperimentRunner',Runner)
    def candidate(arm):return None,'def run(): pass',dict(active_arm=arm,inactive_arm='right' if arm=='left' else 'left')
    runtime=SimpleNamespace(planning_only=False,_routing_lock=threading.Lock(),workspace_arm_fallback=True,
        config_path=tmp_path/'config',config=SimpleNamespace(root=tmp_path),
        compile=lambda *a:candidate('left'),_alternate_candidate=lambda *a:candidate('right'))
    obs=SimpleNamespace(validate_for=lambda *a,**kw:None,meta={'joints_deg':{}})
    result=PartitionedReplayRuntime.execute(runtime,'source',obs,tmp_path/'execution')
    if len(rejected)==2:
        assert result['physical_execution'] is False
        assert not any(e[0]=='dispatch' for e in events)
        assert 'self-collision' in result['error'] and '882' in result['error']
        candidates=result['planning_failure']['arm_candidates']
        assert candidates['left']['planning_failure']['action_index']==7
        assert candidates['left']['planning_failure']['diagnostics']['target_base_mm_deg'][0]==119
        assert '882' in candidates['right']['error']
        assert result['arm_path_checks']==candidates
    else:
        selected='right' if rejected else 'left'
        assert events[-1]==('dispatch',selected)
        assert sum(e[0]=='dispatch' for e in events)==1
        assert result['arm_assignment']['active_arm']==selected
