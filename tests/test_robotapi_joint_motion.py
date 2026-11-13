"""Offline SDK doubles: no sockets, no physical motion."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cloth_agent.config import SafetyError
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.experiments.full_grasp_adapter import build_robot_config
from cloth_agent.robot_api import RobotAPI, RobotExecutionError, XArmBackend
from cloth_agent.robot_collision import RobotCollision

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'config/dual_arm.zed_dual_xarm7_20261103.json'


@pytest.fixture
def hardware(monkeypatch):
    scene = DualConfig.load_geometry(PATH, root=ROOT)
    initial = {k: a.home_joints.copy() for k, a in scene.arms.items()}
    initial['right'][0] += 1
    checker = RobotCollision(PATH, initial, 'right', root=ROOT)
    pose = checker.models['right'].forward(initial['right'])
    config = build_robot_config(scene.raw, {}, 'right', *pose[3:5], config_path=PATH)

    class Controller:
        def __init__(self, key):
            self.key = key
            self.q = initial[key].copy()
            arm = scene.arms[key]
            self.connected, self.axis, self.mode, self.state = True, 7, 0, 2
            self.version_number = (2, 8, 2)
            self.control_box_sn, self.sn = arm.serial, arm.raw['robot_sn']
            self.tcp_offset, self.world_offset = arm.tcp_offset.tolist(), [0.] * 6
            self.commands = []
            self.on_move = None
        def get_robot_sn(self): return 0, self.sn
        def get_servo_angle(self, **kwargs): return 0, self.q.tolist()
        def get_err_warn_code(self): return 0, [0, 0]
        def get_state(self): return 0, self.state
        def get_position(self, **kwargs): return 0, checker.models[self.key].forward(self.q).tolist()
        def motion_enable(self, **kwargs): self.commands.append('enable'); return 0
        def set_mode(self, mode): self.mode = mode; return 0
        def set_state(self, state): self.state = 2 if state == 0 else state; self.commands.append(('state', state)); return 0
        def set_gripper_mode(self, *args): return 0
        def set_gripper_enable(self, *args): return 0
        def set_gripper_speed(self, *args): return 0
        def get_gripper_position(self): return 0, 850
        def get_gripper_err_code(self): return 0, 0
        def get_gripper_status(self): return 0, 0
        def set_servo_angle(self, **kwargs):
            self.commands.append(kwargs)
            if self.on_move:
                self.on_move()
            else:
                self.q = np.array(kwargs['angle'])
            return 0
        def set_position(self, **kwargs): pytest.fail('Cartesian re-IK must not be used')
        def connect(self): self.connected = True
        def disconnect(self): self.connected = False

    controllers = {k: Controller(k) for k in initial}
    import xarm.wrapper
    monkeypatch.setattr(xarm.wrapper, 'XArmAPI', lambda ip, **kwargs: controllers[next(k for k, a in scene.arms.items() if a.ip == ip)])
    return config, checker, controllers


def test_home_dispatches_exact_checked_joints_without_moving_other_arm(hardware):
    config, checker, controllers = hardware
    backend = XArmBackend(config, collision_checker=checker)
    try:
        pose, state = backend.home(config)
        commands = [c for c in controllers['right'].commands if isinstance(c, dict)]
        assert len(commands) == 1
        assert commands[0]['angle'] == list(config.init_joints_deg)
        assert commands[0]['radius'] == -1 and commands[0]['wait'] is True
        assert commands[0]['relative'] is False
        assert commands[0]['speed'] <= config.home_speed_deg_s
        assert commands[0]['mvacc'] <= config.home_acceleration_deg_s2
        assert not controllers['left'].commands
        assert state['checked_joint_path']['status'] == 'JOINT_RANGES_CLEAR'
    finally:
        backend.close()


def test_perception_dispatches_recorded_joints_through_checked_queue(hardware):
    config,checker,controllers=hardware
    target=np.array(config.init_joints_deg);target[0]+=.4
    config=replace(config,perception_joints_deg=tuple(target),
                   perception_pose_mm_deg=tuple(checker.models['right'].forward(target)))
    backend=XArmBackend(config,collision_checker=checker)
    try:
        _,state=backend.perception_position(config)
        commands=[c for c in controllers['right'].commands if isinstance(c,dict)]
        assert commands
        np.testing.assert_allclose(commands[-1]['angle'],target)
        assert commands[-1]['wait'] is True and commands[-1]['radius']==-1
        assert state['checked_joint_path']['source_actions']==[{'name':'perception_position','args':{}}]
        assert not controllers['left'].commands
    finally:backend.close()


def test_cartesian_target_is_sent_as_the_solved_joint_target(hardware, monkeypatch):
    config, checker, controllers = hardware
    model = checker.models['right']
    goal = np.array(config.init_joints_deg)
    pose = model.forward(goal)
    calls = []
    def inverse(target, seed):
        calls.append((target.copy(), seed.copy()))
        return goal.copy()
    monkeypatch.setattr(model, 'inverse', inverse)
    backend = XArmBackend(config, collision_checker=checker)
    try:
        backend.move(*pose[:3], config.relative_yaw_from_absolute_deg(pose[5]), config)
        assert calls
        expected = pose.copy()
        expected[3:5] = [config.orientation_roll_deg, config.orientation_pitch_deg]
        np.testing.assert_allclose(calls[-1][0], expected)
        commands = [c for c in controllers['right'].commands if isinstance(c, dict)]
        np.testing.assert_allclose(commands[-1]['angle'], goal)
        for index, command in enumerate(commands):
            assert command['speed'] == checker.config.limits['joint_speed_deg_s']
            assert command['mvacc'] == checker.config.limits['joint_accel_deg_s2']
            assert command['radius'] == (-1 if index == len(commands)-1 else 0)
            assert command['wait'] is (index == len(commands) - 1)
        assert not controllers['left'].commands
    finally:
        backend.close()


def test_static_obstacle_movement_stops_both_and_blocks_next_command(hardware):
    config, checker, controllers = hardware
    backend = XArmBackend(config, collision_checker=checker)
    robot = RobotAPI(config, backend)
    try:
        controllers['left'].q[0] += 1
        with pytest.raises(RobotExecutionError, match='moved|range|monitor'):
            robot.home()
        assert all(('state', 4) in c.commands for c in controllers.values())
        assert not any(isinstance(c, dict) for c in controllers['right'].commands)
        with pytest.raises(RobotExecutionError, match='halted'):
            robot.home()
    finally:
        backend.close()


def test_success_response_without_arrival_is_rejected(hardware):
    config, checker, controllers = hardware
    backend = XArmBackend(config, collision_checker=checker)
    try:
        controllers['right'].on_move = lambda: None
        with pytest.raises(RobotExecutionError, match='not reached'):
            RobotAPI(config, backend).home()
        assert ('state', 4) in controllers['right'].commands
    finally:
        backend.close()


def test_changed_live_start_rejected_before_enable(hardware):
    config, checker, controllers = hardware
    controllers['right'].q[0] += 1
    with pytest.raises(RobotExecutionError, match='moved since'):
        XArmBackend(config, collision_checker=checker)
    assert not any(c.commands for c in controllers.values())


def test_intermediate_workspace_crossing_and_unknown_motion_rejected(hardware):
    config, checker, controllers = hardware
    config = replace(config, boundaries=replace(config.boundaries, x_max=1.))
    with pytest.raises(SafetyError, match='bound'):
        checker.plan_action(config, 'home', {})
    with pytest.raises(SafetyError, match='no checked joint path'):
        checker.plan_action(config, 'shake', {})


def test_gripper_needs_no_ik_and_home_uses_pyroki(hardware, monkeypatch):
    config, checker, controllers = hardware
    model = checker.models['right']; inverse = model.inverse; calls = []
    def solve(*args):
        calls.append(args)
        return inverse(*args)
    monkeypatch.setattr(model, 'inverse', solve)
    opened = checker.plan_action(config, 'open_gripper', {})
    assert not calls and len(opened['path']) == 1
    home = checker.plan_action(config, 'home', {})
    assert calls
    np.testing.assert_array_equal(home['path'][-1]['right'], config.init_joints_deg)
    assert all(row['left'] == checker.initial['left'].tolist() for row in home['path'])
    assert not controllers['right'].commands


def test_large_replay_return_compacts_and_dispatches_checked_home_waypoints(hardware):
    config, checker, controllers = hardware
    # Recorded end of the first sleeve fold, before returning to Home.
    start = np.array([-35.47830402251608, -4.05068316208708, 22.996835459506574,
                      38.96433846085815, 3.063172689608528, 42.952760766342415,
                      63.248069670726764])
    checker.initial['right'] = start.copy()
    controllers['right'].q = start.copy()
    plan = checker.plan_action(config, 'home', {})
    assert plan['original_waypoints'] >= len(plan['path']) >= 2
    assert plan['removed_stops'] == plan['original_waypoints']-len(plan['path'])
    assert plan['checked_segments'] == len(plan['path'])-1
    joints = np.array([row['right'] for row in plan['path']])
    checker.validate_joint_path(plan['path'],robot_config=config,
                                tolerance_deg=checker.config.limits['stationary_tolerance_deg'])
    np.testing.assert_array_equal(joints[-1], config.init_joints_deg)
    backend = XArmBackend(config, collision_checker=checker)
    try:
        backend.home(config)
        commands = [c for c in controllers['right'].commands if isinstance(c, dict)]
        assert len(commands) == len(plan['path']) - 1
        for index, (command, row) in enumerate(zip(commands, plan['path'][1:])):
            np.testing.assert_array_equal(command['angle'], row['right'])
            assert command['radius'] == (-1 if index == len(commands)-1 else 0)
            assert command['wait'] is (index == len(commands) - 1)
            np.testing.assert_array_equal(row['left'], checker.initial['left'])
        assert not controllers['left'].commands
    finally:
        backend.close()


def test_monitor_stops_during_blocking_joint_command(hardware):
    config, checker, controllers = hardware
    backend = XArmBackend(config, collision_checker=checker)
    def motion():
        controllers['left'].q[0] += 1
        backend._guard_thread.join(timeout=2)
        assert not backend._guard_thread.is_alive()
    controllers['right'].on_move = motion
    try:
        with pytest.raises(RobotExecutionError, match='monitor'):
            RobotAPI(config, backend).home()
        assert all(('state', 4) in c.commands for c in controllers.values())
    finally:
        backend.close()


def test_tcp_motion_bound_covers_calibrated_fk(hardware):
    _, checker, _ = hardware
    rng = np.random.default_rng(17)
    for key, model in checker.models.items():
        q = checker.initial[key]
        origin = model.forward(q)[:3]
        for delta in rng.uniform(-2, 2, (32, 7)):
            displacement = np.linalg.norm(model.forward(q + delta)[:3] - origin)
            assert displacement <= model.tcp_motion_bound(delta) + 1e-8


def test_runner_uses_joint_chain_without_legacy_controller_ik(hardware, monkeypatch, tmp_path):
    from cloth_agent.experiment import ExperimentRunner
    config, checker, controllers = hardware
    monkeypatch.setattr('cloth_agent.experiment.validate_controller_trajectory',
                        lambda *a: pytest.fail('legacy Cartesian preview should not run'))
    runner = ExperimentRunner(tmp_path, config, collision_checker=checker)
    source = runner.workspace / 'home.py'
    source.write_text('def run():\n    home()\n')
    result = runner.run_experiment(source, real=True, confirmed=True)
    assert result['execution_completed']
    assert (runner.results_dir / 'joint_plan.json').is_file()
    assert not controllers['left'].commands
    assert all(not controller.connected for controller in controllers.values())


def test_queue_buffers_intermediate_points_and_verifies_final_arrival(hardware, monkeypatch):
    config, checker, controllers = hardware
    backend = XArmBackend(config, collision_checker=checker)
    start = {k: q.copy() for k, q in checker.initial.items()}
    middle = {k: q.copy() for k, q in start.items()}
    end = {k: q.copy() for k, q in start.items()}
    middle['right'][0] += .5
    end['right'][0] += 1.
    path = [{k: q.tolist() for k, q in s.items()} for s in (start, middle, end)]
    checker.validate_joint_path([start, middle, end], robot_config=config,
                               tolerance_deg=checker.config.limits['stationary_tolerance_deg'])
    monkeypatch.setattr(checker, 'plan_action', lambda *a, **kw: {'path': path})
    sent = []
    def queued(**kw):
        sent.append(kw)
        if not kw['wait']:
            # SDK has accepted the point; hardware has not reached it yet.
            backend._guard_check()
        else:
            controllers['right'].q = np.array(kw['angle'])
        return 0
    monkeypatch.setattr(controllers['right'], 'set_servo_angle', queued)
    try:
        backend.home(config)
        assert [c['wait'] for c in sent] == [False, True]
        assert backend._guard_ranges is None
        np.testing.assert_array_equal(controllers['right'].q, end['right'])
    finally:
        backend.close()


def test_queue_guard_does_not_accept_gaps_between_checked_boxes(hardware):
    config, checker, controllers = hardware
    backend = XArmBackend(config, collision_checker=checker)
    try:
        initial = {k: q.copy() for k, q in checker.initial.items()}
        far = {k: q.copy() for k, q in initial.items()}
        far['right'][0] += 2.
        with backend._guard_lock:
            backend._guard_ranges = [({k: q-.1 for k,q in row.items()},
                                      {k: q+.1 for k,q in row.items()}) for row in (initial, far)]
            controllers['right'].q[0] += 1.
        with pytest.raises(RobotExecutionError, match='range'):
            backend._guard_check()
    finally:
        backend.close()


def test_old_firmware_cannot_silently_turn_continuous_queue_into_stop_go(hardware, monkeypatch):
    config, checker, controllers = hardware
    start={k:q.copy() for k,q in checker.initial.items()}
    middle={k:q.copy() for k,q in start.items()}
    end={k:q.copy() for k,q in start.items()}
    middle['right'][0]-=.5;end['right'][0]-=1.
    monkeypatch.setattr(checker,'plan_action',lambda *a,**k:dict(path=[start,middle,end]))
    controllers['right'].version_number=(1,5,19)
    backend=XArmBackend(config,collision_checker=checker)
    try:
        with pytest.raises(RobotExecutionError,match='firmware'):
            RobotAPI(config,backend).home()
        assert not any(isinstance(c,dict) for c in controllers['right'].commands)
        assert all(('state',4) in c.commands for c in controllers.values())
    finally:backend.close()


def test_multi_action_sequence_planned_before_any_dispatch(hardware,monkeypatch):
    config,checker,controllers=hardware
    config=replace(config,home_speed_deg_s=20,home_acceleration_deg_s2=40,joint_queue_prefill_s=.15)
    prefills=[]
    def prefill(seconds,wait=False):
        assert not any(isinstance(c,dict) for c in controllers['right'].commands)
        prefills.append((seconds,wait))
        return 0
    controllers['right'].set_pause_time=prefill
    events=[]
    def plan(config,name,args,*,initial):
        events.append(('plan',args['x']))
        assert not any(isinstance(c,dict) for c in controllers['right'].commands)
        end={k:np.array(q).copy() for k,q in initial.items()};end['right'][0]-=.25
        return dict(path=[initial,end],status='JOINT_RANGES_CLEAR')
    monkeypatch.setattr(checker,'plan_action',plan)
    backend=XArmBackend(config,collision_checker=checker)
    try:
        actions=[dict(name='move',args=dict(x=x,y=0,z=100,yaw=0)) for x in (1,2,3)]
        _,state=backend.joint_sequence(config,actions)
        commands=[c for c in controllers['right'].commands if isinstance(c,dict)]
        assert len(commands)==3
        assert [c['radius'] for c in commands]==[0,0,-1]
        assert [c['wait'] for c in commands]==[False,False,True]
        assert state['checked_joint_path']['source_actions']==actions
        assert prefills==[(.15,False)]
        assert len(state['joint_dispatch_timing'])==3
        assert all(c['speed']==20 and c['mvacc']==40 for c in commands)
        timing=state['motion_timing_s']
        assert 0<=timing['planning_and_start_check']<=timing['total']
        assert 0<=timing['queue_and_completion']<=timing['total']
        assert not controllers['left'].commands
    finally:backend.close()


def test_later_action_planning_failure_dispatches_nothing(hardware,monkeypatch):
    config,checker,controllers=hardware
    def plan(config,name,args,*,initial):
        if args['x']==2:raise SafetyError('later segment blocked')
        end={k:np.array(q).copy() for k,q in initial.items()};end['right'][0]-=.25
        return dict(path=[initial,end])
    monkeypatch.setattr(checker,'plan_action',plan)
    backend=XArmBackend(config,collision_checker=checker)
    try:
        with pytest.raises(SafetyError,match='later segment'):
            RobotAPI(config,backend).motion_sequence([dict(name='move',args=dict(x=x,y=0,z=100,yaw=0)) for x in (1,2)])
        assert not any(isinstance(c,dict) for c in controllers['right'].commands)
        assert all(('state',4) in c.commands for c in controllers.values())
    finally:backend.close()


def test_real_validation_accepts_dual_joint_speed_but_preserves_caps(hardware):
    from cloth_agent.config import ConfigError
    config,_,_=hardware
    fast=replace(config,home_speed_deg_s=20,home_acceleration_deg_s2=40)
    fast.validate_for_real()
    for invalid in (replace(fast,home_speed_deg_s=21),
                    replace(fast,home_acceleration_deg_s2=61),
                    replace(fast,collision_scene_path=None)):
        with pytest.raises(ConfigError):invalid.validate_for_real()


def test_continuous_stream_uses_checked_path_and_restores_position_mode(hardware):
    config,checker,controllers=hardware
    config=replace(config,continuous_joint_stream=True)
    right=controllers['right'];stream=[]
    def servo(q, **kwargs):
        stream.append(q);right.q=np.array(q);return 0
    right.set_servo_angle_j=servo
    backend=XArmBackend(config,collision_checker=checker)
    try:
        _,state=backend.home(config)
        assert len(stream)>10
        assert right.mode==0
        assert state['joint_queue_mode']=='time_parameterized_servoj'
        np.testing.assert_allclose(right.q,config.init_joints_deg,atol=1e-8)
        assert not controllers['left'].commands
        assert not [c for c in right.commands if isinstance(c,dict)]
    finally:backend.close()


def test_continuous_stream_sdk_failure_stops_both_arms(hardware):
    config,checker,controllers=hardware
    config=replace(config,continuous_joint_stream=True)
    controllers['right'].set_servo_angle_j=lambda *a,**k: 1
    backend=XArmBackend(config,collision_checker=checker)
    try:
        with pytest.raises(RobotExecutionError,match='set_servo_angle_j'):
            backend.home(config)
        for controller in controllers.values():assert ('state',4) in controller.commands
    finally:backend.close()


def test_stream_initializes_mode_changed_state_before_first_servo(hardware):
    config,checker,controllers=hardware
    config=replace(config,continuous_joint_stream=True)
    backend=XArmBackend(config,collision_checker=checker)
    right=controllers['right'];events=[]
    def mode(value):
        events.append(('mode',value));right.mode=value;right.state=5;return 0
    original_state=right.set_state
    def state(value):
        events.append(('state',value));return original_state(value)
    def servo(q,**kw):
        assert right.state==2
        events.append(('servo',None));right.q=np.asarray(q);return 0
    right.set_mode=mode;right.set_state=state;right.set_servo_angle_j=servo
    try:
        backend.home(config)
        assert events[:3]==[('mode',1),('state',0),('servo',None)]
        assert events[-2:]==[('mode',0),('state',0)]
        assert right.mode==0 and right.state==2
    finally:backend.close()


@pytest.mark.parametrize('problem', ['paused','stopped','fault','stop_requested','ready_failure'])
def test_mode_transition_does_not_resume_abnormal_controller(hardware,problem):
    config,checker,controllers=hardware
    config=replace(config,continuous_joint_stream=True)
    backend=XArmBackend(config,collision_checker=checker)
    right=controllers['right'];ready=[];sent=[]
    def mode(value):
        right.mode=value;right.state=5
        if problem=='paused':right.state=3
        if problem=='stopped':right.state=4
        if problem=='fault':right.get_err_warn_code=lambda:(0,[1,0])
        if problem=='stop_requested':backend._stop_requested.set()
        return 0
    original_state=right.set_state
    def state(value):
        if value==0:
            ready.append(value)
            if problem=='ready_failure':return 9
        return original_state(value)
    right.set_mode=mode;right.set_state=state
    right.set_servo_angle_j=lambda *a,**k:sent.append(a) or 0
    try:
        with pytest.raises(RobotExecutionError):backend.home(config)
        assert not sent
        assert ready==([0] if problem=='ready_failure' else [])
        assert all(('state',4) in c.commands for c in controllers.values())
    finally:backend.close()


def test_stop_latch_prevents_later_mode_reactivation(hardware):
    config,checker,controllers=hardware
    backend=XArmBackend(config,collision_checker=checker)
    try:
        backend.stop()
        # Even if someone externally changes reported state, this backend stays stopped.
        controllers['right'].state=2
        with pytest.raises(RobotExecutionError,match='cancelled'):
            with backend._guard_lock:backend._change_stream_mode(1)
        assert controllers['right'].commands[-1]==('state',4)
    finally:backend.close()


def test_continuous_noop_does_not_switch_modes_or_send_targets(hardware,monkeypatch):
    config,checker,controllers=hardware
    config=replace(config,continuous_joint_stream=True)
    backend=XArmBackend(config,collision_checker=checker)
    def plan(config,name,args,*,initial):return dict(path=[initial,initial])
    monkeypatch.setattr(checker,'plan_action',plan)
    for controller in controllers.values():
        controller.commands.clear()
        controller.set_mode=lambda *a:pytest.fail('No-op must not change mode')
        controller.set_servo_angle_j=lambda *a,**k:pytest.fail('No-op must not send targets')
    try:
        backend.joint_sequence(config,[dict(name='home',args={})])
        assert all(not c.commands for c in controllers.values())
    finally:backend.close()


def test_stream_preserves_monitor_reason_after_stop_latched(hardware):
    config,checker,controllers=hardware
    config=replace(config,continuous_joint_stream=True)
    backend=XArmBackend(config,collision_checker=checker)
    sent=[]
    reason='joint monitor: RobotExecutionError: simulated checked range violation'
    def servo(q,**kw):
        sent.append(q)
        backend._guard_error=reason
        backend._stop_requested.set()
        return 0
    controllers['right'].set_servo_angle_j=servo
    try:
        with pytest.raises(RobotExecutionError,match='simulated checked range violation'):
            backend.home(config)
        assert len(sent)==1
        assert all(('state',4) in c.commands for c in controllers.values())
    finally:backend.close()
