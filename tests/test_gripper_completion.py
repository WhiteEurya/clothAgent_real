"""Controller feedback sequences; no robot, SDK, network or wall-clock waits."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloth_agent.config import ConfigError, RobotConfig, WorkspaceBounds
from cloth_agent.robot_api import RobotAPI, RobotExecutionError, XArmBackend


@pytest.fixture
def config():
    return RobotConfig(robot_ip='test', boundaries=WorkspaceBounds(x_min=0, x_max=600,
        y_min=-300, y_max=300, z_min=0, z_max=500), init_joints_deg=(0,)*6,
        init_pose_mm_deg=(300, 0, 200, 180, 0, 0), orientation_roll_deg=180,
        orientation_pitch_deg=0, gripper_completion_timeout_s=2.)


class Arm:
    connected = True

    def __init__(self, samples):
        self.samples = samples
        self.index = 0
        self.commands = []
        self.moves = []
        self.result = 0

    def get_gripper_position(self):
        self.sample = self.samples[min(self.index, len(self.samples)-1)]
        self.index += 1
        if isinstance(self.sample, BaseException):
            raise self.sample
        return self.sample.get('position_result', (0, self.sample['position']))

    def get_gripper_status(self):
        return self.sample.get('status_result', (0, self.sample['status']))

    def get_gripper_err_code(self):
        return 0, self.sample.get('error', 0)

    def set_gripper_position(self, position, **kwargs):
        self.commands.append((position, kwargs))
        return self.result

    def set_position(self, **kwargs):
        self.moves.append({'kwargs': kwargs, 'samples_read': self.index})
        return 0


def sample(position, status=0, **kwargs):
    return dict(position=position, status=status, **kwargs)


@pytest.fixture
def backend(monkeypatch):
    now = [0.]
    monkeypatch.setattr('cloth_agent.robot_api.time.monotonic', lambda: now[0])
    monkeypatch.setattr('cloth_agent.robot_api.time.sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    monkeypatch.setattr('cloth_agent.robot_api._gripper_log', lambda message: None)
    def build(samples):
        result = XArmBackend.__new__(XArmBackend)
        result.arm = Arm(samples)
        result._state = lambda: ([300, 0, 100, 180, 0, 0], {})
        return result
    return build


def test_old_stop_at_open_position_does_not_allow_lift(config, backend):
    b = backend([sample(850), sample(850), sample(850), sample(600, 1),
                 sample(200, 1), sample(0, 1), sample(0)])
    robot = RobotAPI(config, b)
    robot.close_gripper()
    robot.move(300, 0, 100, 0)
    assert b.arm.moves[0]['samples_read'] >= 66
    assert b.arm.commands == [(0., {'speed': 500., 'wait': False})]
    completion = robot.actions[0].gripper_result['completion']
    assert completion['reason'] == 'measured_close_stable'
    assert len(completion['samples']) >= 65
    assert all(row['completion_reason'] is None for row in completion['samples'][:-1])


def test_observed_early_grasp_at_834_never_releases_lift(config, backend):
    b = backend([sample(840), sample(840, 2), sample(834, 2),
                 sample(500, 1), sample(35, 2), sample(35, 2), sample(3, 1), sample(3)])
    robot = RobotAPI(config, b)
    robot.close_gripper()
    robot.move(300, 0, 100, 0)
    completion = robot.actions[0].gripper_result['completion']
    assert completion['reason'] == 'measured_close_stable'
    assert len(completion['samples']) >= 66
    assert all(row['completion_reason'] is None for row in completion['samples'][:-1])
    assert b.arm.moves[0]['samples_read'] >= 67
    assert robot.actions[0].gripper_result['feedback']['position_pulse'] == 3


@pytest.mark.parametrize('position,status', [(13, 2), (35, 2), (100, 0)])
def test_stable_closure_above_target_allows_next_action(config, backend, position, status):
    b = backend([sample(840), sample(500, 1)] + [sample(position, status)]*20)
    robot = RobotAPI(replace(config, gripper_close=5), b)
    robot.close_gripper()
    robot.move(300, 0, 100, 0)
    completion = robot.actions[0].gripper_result['completion']
    assert completion['reason'] == 'measured_close_stable'
    assert completion['duration_s'] >= .2
    assert len(b.arm.moves) == 1
    assert len(b.arm.commands) == 1


def test_invalid_feedback_resets_close_stability(config, backend):
    b = backend([sample(840)] + [sample(13, 2)]*4 +
                [sample(13, 2, position_result=(9, 13))] + [sample(13, 2)]*20)
    result, _ = b.close_gripper(config)
    assert result['completion']['reason'] == 'measured_close_stable'
    assert result['completion']['duration_s'] >= .4


@pytest.mark.parametrize('samples', [
    [sample(850), sample(850)],
    [sample(850), sample(850, 2)],
    [sample(500), sample(700, 2)],
    [sample(850), sample(0, 1)],
])
def test_stale_stopped_wrong_direction_or_still_moving_blocks_next_action(config, backend, samples):
    # Last unconfirmed sample persists beyond the former timeout; then the
    # operator interrupts. No completion is invented and no command is resent.
    b = backend(samples + [samples[-1]]*60 + [KeyboardInterrupt()])
    robot = RobotAPI(config, b)
    with pytest.raises(KeyboardInterrupt):
        robot.close_gripper()
    with pytest.raises(RobotExecutionError, match='halted'):
        robot.move(300, 0, 100, 0)
    assert not b.arm.moves
    trace = robot.action_dicts()[0]['gripper_result']['completion']
    assert trace['status'] == 'FAILED'
    assert trace['samples']
    assert trace['duration_s'] >= 2.
    assert len(b.arm.commands) == 1
    assert trace['timeout_s'] is None


@pytest.mark.parametrize('bad', [
    sample(850, status_result=(7, 0)), sample(850, position_result=(9, 0)),
    sample(850, position_result=0), sample(850, position_result=(0, None)),
    sample(850, position_result=(0, float('nan'))),
    sample(850, status_result=('bad_code', 0)),
])
def test_bad_reads_retry_then_complete_without_resending(config, backend, bad):
    b = backend([sample(850)] + [bad]*60 + [sample(300, 1), sample(0)])
    robot = RobotAPI(config, b)
    robot.close_gripper()
    robot.move(300, 0, 100, 0)
    assert not robot.halted
    assert b.arm.moves[0]['samples_read'] >= 122
    assert len(b.arm.commands) == 1
    assert robot.actions[0].gripper_result['completion']['duration_s'] >= 3.


@pytest.mark.parametrize('bad', [sample(850, error=3), sample(850, status=3)])
def test_hardware_fault_still_blocks_motion(config, backend, bad):
    robot = RobotAPI(config, backend([sample(850), bad]))
    with pytest.raises(RobotExecutionError, match='hardware fault'):
        robot.close_gripper()
    assert robot.halted
    assert robot.action_dicts()[0]['gripper_result']['completion']['failed_feedback']


def test_open_needs_target_position_and_terminal_state(config, backend):
    b = backend([sample(20, 2), sample(20), sample(600, 1), sample(850, 1), sample(850)])
    result, _ = b.open_gripper(config)
    assert len(result['completion']['samples']) == 4
    assert result['completion']['reason'] == 'measured_target_reached'


def test_no_closing_trend_does_not_complete_even_at_target(config, backend):
    b = backend([sample(0)]*10 + [KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        b.close_gripper(config)


def test_open_840_stopped_accepts_850_target(config, backend):
    b = backend([sample(840), sample(840)])
    result, _ = b.open_gripper(config)
    assert result['completion']['reason'] == 'measured_target_reached'
    assert result['completion']['position_tolerance_pulse'] == 15.
    assert b.arm.commands[0][0] == 850.


def test_open_tolerance_never_accepts_moving_or_far_stopped_position(config, backend):
    b = backend([sample(0), sample(830), sample(840, 1), sample(840)])
    result, _ = b.open_gripper(config)
    assert [r['completion_reason'] for r in result['completion']['samples']] == [
        None, None, 'measured_target_reached']


def test_open_tolerance_does_not_relax_close(config, backend):
    b = backend([sample(850), sample(10), sample(4)])
    result, _ = b.close_gripper(replace(config, gripper_open_tolerance_pulse=25))
    assert result['completion']['position_tolerance_pulse'] == 5.
    assert result['completion']['samples'][0]['completion_reason'] is None
    assert result['feedback']['position_pulse'] == 4


def test_configured_open_tolerance_is_used(config, backend):
    b = backend([sample(840), sample(840), sample(848)])
    result, _ = b.open_gripper(replace(config, gripper_open_tolerance_pulse=3))
    assert result['completion']['samples'][0]['completion_reason'] is None
    assert result['completion']['position_tolerance_pulse'] == 3.


@pytest.mark.parametrize('tolerance', [-1, 26, float('nan'), float('inf')])
def test_invalid_open_tolerance_rejected(config, tolerance):
    with pytest.raises(ConfigError, match='open tolerance'):
        replace(config, gripper_open_tolerance_pulse=tolerance).validate_for_real()


def test_failed_command_blocks_following_action(config, backend):
    b = backend([sample(850)])
    b.arm.result = 23
    robot = RobotAPI(config, b)
    with pytest.raises(RobotExecutionError, match='code=23'):
        robot.close_gripper()
    assert robot.halted
    assert not robot.actions[0].gripper_result['completion']['samples']


def test_interrupt_is_not_swallowed_as_telemetry(config, backend):
    b = backend([sample(850), KeyboardInterrupt()])
    robot = RobotAPI(config, b)
    with pytest.raises(KeyboardInterrupt):
        robot.close_gripper()
    assert robot.halted
    assert robot.actions[0].gripper_result['completion']['status'] == 'FAILED'


def test_read_failure_before_command_does_not_send_command(config, backend):
    b = backend([sample(850, error=5)])
    with pytest.raises(RobotExecutionError):
        b.close_gripper(config)
    assert not b.arm.commands


def test_already_moving_does_not_send_another_command(config, backend):
    b = backend([sample(500, 1)]*60 + [sample(850), sample(0)])
    b.close_gripper(config)
    assert len(b.arm.commands) == 1
    assert b.arm.index >= 121


def test_initial_read_failure_waits_before_sending_command(config, backend):
    b = backend([sample(850, status_result=(7, 0))]*60 + [sample(850), sample(0)])
    result, _ = b.close_gripper(config)
    assert result['completion']['close_stable_elapsed_s'] >= 3.0
    assert b.arm.index >= 121
    assert len(b.arm.commands) == 1


@pytest.mark.parametrize('target,initial,final', [('open', 0, 850), ('close', 850, 0)])
def test_long_motion_waits_until_feedback_confirms_completion(config, backend, target, initial, final):
    b = backend([sample(initial)] + [sample(400, 1)]*260 + [sample(final)])
    result, _ = getattr(b, target + '_gripper')(config)
    trace = result['completion']
    assert trace['duration_s'] >= 13.
    assert trace['status'] == 'COMPLETED'
    assert trace['reason'] == ('measured_close_stable' if target == 'close' else 'measured_target_reached')
    assert len(trace['samples']) == 200
    assert trace['dropped_sample_count'] >= (120 if target == 'close' else 61)
    assert trace['sample_count'] >= (320 if target == 'close' else 261)
    assert len(b.arm.commands) == 1


def test_permanently_unavailable_feedback_waits_for_interrupt(config, backend):
    b = backend([sample(850)] + [sample(850, status_result=(7, 0))]*260 + [KeyboardInterrupt()])
    robot = RobotAPI(config, b)
    with pytest.raises(KeyboardInterrupt):
        robot.close_gripper()
    assert robot.halted
    assert not b.arm.moves
    assert robot.actions[0].gripper_result['completion']['duration_s'] >= 13.


def test_wait_log_is_visible_and_captured(monkeypatch):
    import io
    import sys
    from cloth_agent.robot_api import _gripper_log
    captured, terminal = io.StringIO(), io.StringIO()
    with monkeypatch.context() as context:
        context.setattr(sys, 'stdout', captured)
        context.setattr(sys, '__stdout__', terminal)
        _gripper_log('[gripper] close: READ_RETRY; arm stays still')
    assert captured.getvalue() == terminal.getvalue()
    assert 'READ_RETRY' in terminal.getvalue()


def test_failed_gripper_result_suppresses_session_home(config, backend):
    from cloth_agent.experiment import ExperimentRunner
    from cloth_agent.session import AgentSession
    robot = RobotAPI(config, backend([sample(850), sample(850, error=3)]))
    with pytest.raises(RobotExecutionError):
        robot.close_gripper()
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.config, runner.run_dir = config, Path('/tmp/test-run')
    preflight = SimpleNamespace(source_path=runner.run_dir / 'test.py', source='', actions=[], error=None, stdout='')
    result = runner._result(preflight, physical=True, completed=False, notes='', error='failed',
                            actual_actions=robot.action_dicts())
    assert result['gripper_completion_failed']
    session = AgentSession.__new__(AgentSession)
    session.runner = runner  # no workspace/backend: Home must not touch either
    outcome = session._attempt_return_home(notes='test')
    assert not outcome['attempted']
    assert not outcome['completed']


def test_completion_timeout_loads_with_backward_compatible_default(tmp_path):
    import json
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / 'config' / 'robot.example.json').read_text())
    raw['gripper'].pop('completion_timeout_s', None)
    raw['gripper'].pop('open_tolerance_pulse', None)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(raw))
    assert RobotConfig.load(root, path).gripper_completion_timeout_s == 10.
    assert RobotConfig.load(root, path).gripper_open_tolerance_pulse == 15.
    raw['gripper']['completion_timeout_s'] = 15.
    raw['gripper']['open_tolerance_pulse'] = 12.
    path.write_text(json.dumps(raw))
    assert RobotConfig.load(root, path).gripper_completion_timeout_s == 15.
    assert RobotConfig.load(root, path).gripper_open_tolerance_pulse == 12.


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_completion_timeout_rejected(config, timeout):
    with pytest.raises(ConfigError, match='completion timeout'):
        replace(config, gripper_completion_timeout_s=timeout).validate_for_real()


@pytest.mark.parametrize('checkpointed', [False, True])
def test_runner_does_not_lift_release_or_home_after_unconfirmed_close(
        tmp_path, config, backend, monkeypatch, checkpointed):
    from cloth_agent.experiment import ExperimentRunner
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'test.py').write_text(
        'def run():\n    close_gripper()\n    move(300, 0, 100, 0)\n    open_gripper()\n    home()\n')
    b = backend([sample(850), sample(850, error=3)])
    b.close = lambda: None
    b.open_gripper = lambda *args: pytest.fail('no automatic release after unconfirmed gripper')
    b.home = lambda *args: pytest.fail('no automatic Home after unconfirmed gripper')
    monkeypatch.setattr('cloth_agent.experiment.XArmBackend', lambda config: b)
    monkeypatch.setattr('cloth_agent.experiment.validate_controller_trajectory', lambda *args: None)
    runner = ExperimentRunner(tmp_path, config)
    if checkpointed:
        result = runner.run_checkpointed_experiment('test.py', real=True, confirmed=True,
            checkpoint_action_index=1, abort_actions=[{'name': 'open_gripper', 'args': {}},
                                                     {'name': 'home', 'args': {}}],
            checkpoint_callback=lambda: pytest.fail('checkpoint must not be reached'))
        assert not result['emergency_cleanup']['attempted']
    else:
        result = runner.run_experiment('test.py', real=True, confirmed=True)
    assert not result['execution_completed']
    assert result['gripper_completion_failed']
    assert not b.arm.moves
    assert [a['name'] for a in result['actual_robot_actions']] == ['close_gripper']
    assert (tmp_path / 'results' / 'test.trace.json').is_file()


def test_failed_emergency_release_also_blocks_home(tmp_path, config, backend, monkeypatch):
    from cloth_agent.experiment import ExperimentRunner
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'test.py').write_text(
        'def run():\n    move(300, 0, 100, 0)\n    close_gripper()\n    home()\n')
    b = backend([sample(0), sample(0, error=3)])  # release reports hardware fault
    b.close = lambda: None
    b.move = lambda *args: (_ for _ in ()).throw(RobotExecutionError('motion failed'))
    b.home = lambda *args: pytest.fail('no Home after failed emergency release')
    monkeypatch.setattr('cloth_agent.experiment.XArmBackend', lambda config: b)
    monkeypatch.setattr('cloth_agent.experiment.validate_controller_trajectory', lambda *args: None)
    runner = ExperimentRunner(tmp_path, config)
    result = runner.run_checkpointed_experiment('test.py', real=True, confirmed=True,
        checkpoint_action_index=1, abort_actions=[{'name': 'open_gripper', 'args': {}},
                                                 {'name': 'home', 'args': {}}],
        checkpoint_callback=lambda: pytest.fail('checkpoint must not be reached'))
    assert result['gripper_completion_failed']
    assert runner.gripper_completion_failed
    assert not result['emergency_cleanup']['home_completed']
    assert result['emergency_cleanup']['gripper_completion']['status'] == 'FAILED'


@pytest.mark.parametrize('route', ['fold_pipeline', 'replay_script'])
@pytest.mark.parametrize('reaches_closed_target', [True, False])
def test_both_entrypoints_use_strict_close_gate(
        tmp_path, config, backend, monkeypatch, route, reaches_closed_target):
    import json
    from scripts import replay_gripper_test as replay
    from cloth_agent.config import ExperimentConfig
    from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline
    from cloth_agent.session import AgentSession

    samples = [sample(840), sample(840),  # open already reached
               sample(840), sample(840, 2), sample(834, 2)]
    if reaches_closed_target:
        samples += [sample(3, 1)] + [sample(3)]*65 + [  # stable window before lift
                    sample(3), sample(39, 2), sample(840)]  # release open
    else:
        samples += [sample(840, 2)]*60 + [KeyboardInterrupt()]
    b = backend(samples)
    b.close = lambda: None
    home_calls = []
    capture_samples = []
    def snapshot(*args):
        capture_samples.append(b.arm.index)
        return {'status': 'UNAVAILABLE', 'reason': 'camera mocked in gripper test'}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb', snapshot)
    def home(*args):
        home_calls.append(b.arm.index)
        return [300, 0, 200, 180, 0, 0], {}
    b.home = home
    monkeypatch.setattr('cloth_agent.experiment.XArmBackend', lambda config: b)
    monkeypatch.setattr('cloth_agent.experiment.validate_controller_trajectory', lambda *args: None)
    if route == 'replay_script':
        monkeypatch.setattr(replay, 'PROJECT_ROOT', tmp_path)
        monkeypatch.setattr(replay.RobotConfig, 'load', lambda *args: config)
        assert replay.main(['--real', '--confirm-real']) == (0 if reaches_closed_target else 130)
        run = next((tmp_path / 'runs').iterdir())
        result = json.loads((run / 'results' / 'recorded_gripper_test.json').read_text())
    else:
        run = tmp_path / 'runs' / 'fold_test'
        session = AgentSession(tmp_path, run, config, ExperimentConfig())
        (run / 'run_metadata.json').write_text(json.dumps({
            'last_perception_mode': 'single_camera_rgbd', 'last_active_cameras': ['A']}))
        source = session.workspace / 'fold_test.py'
        source.write_text(replay.RECORDED_SOURCE)
        pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
        pipeline.real = pipeline.confirm_real = True
        pipeline.record_video = False
        pipeline.observer_camera_serial = None
        pipeline.session = session
        pipeline._debug = lambda *args, **kwargs: None
        def execute():
            return pipeline._execute(source, SimpleNamespace(active_camera_labels=('A',)), run,
                                     label='strict_close_test')
        if reaches_closed_target:
            result, _ = execute()
        else:
            with pytest.raises(KeyboardInterrupt):
                execute()
            result = json.loads((run / 'results' / 'fold_test.json').read_text())
            assert session.last_return_home_outcome['attempted'] is False
    assert result['execution_completed'] is reaches_closed_target
    if reaches_closed_target:
        # Third Cartesian call is the lift. It must follow the seventh feedback
        # read (position=3, stop), never the fifth (position=834, grasp).
        assert len(b.arm.moves) == 6
        assert b.arm.moves[2]['samples_read'] >= 66
        if route == 'fold_pipeline':
            assert capture_samples == [b.arm.moves[2]['samples_read']]  # photo only after measured closure
        assert home_calls
    else:
        assert len(b.arm.moves) == 2  # approach + descend only
        assert not home_calls
        assert result['gripper_completion_failed'] is True
        assert capture_samples == []


@pytest.mark.parametrize('interrupt_sample', [sample(722, 1), sample(722, 2, position_result=(9, 722))])
def test_twenty_stopped_samples_restart_after_motion_or_bad_read(config, backend, interrupt_sample):
    b = backend([sample(839), sample(723, 1)] + [sample(722, 2)]*19 +
                [interrupt_sample] + [sample(722, 2)]*19 + [sample(13, 2)]*20)
    result, _ = b.close_gripper(config)
    trace = result['completion']
    assert trace['close_stability_samples'] == 20
    assert result['feedback']['position_pulse'] == 13
    assert b.arm.index >= 101
    assert trace['close_recent_positions_pulse'] == [13]*20
