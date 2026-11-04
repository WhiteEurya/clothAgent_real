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
        def disconnect(self): self.connected = False

    controllers = {k: Controller(k) for k in initial}
    import xarm.wrapper
    monkeypatch.setattr(xarm.wrapper, 'XArmAPI', lambda ip: controllers[next(k for k, a in scene.arms.items() if a.ip == ip)])
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


def test_gripper_and_home_need_no_ik_solver(hardware, monkeypatch):
    config, checker, controllers = hardware
    monkeypatch.setattr(checker.models['right'], 'inverse', lambda *a: pytest.fail('unneeded IK'))
    plans = checker.require_execution(config, [{'name': 'open_gripper'}, {'name': 'home'}])
    assert [len(p['path']) for p in plans] == [1, 2]
    for plan in plans:
        assert all(row['left'] == checker.initial['left'].tolist() for row in plan['path'])


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
