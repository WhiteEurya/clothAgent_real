import json
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.config import RobotConfig, WorkspaceBounds, ExperimentConfig
from cloth_agent.free_exploration import ExplorationPlanningError
from cloth_agent.grasp_checkpoint import compile_grasp_checkpoint, compile_grasp_capture, validate_grasp_decision, GraspCheckpointRejected
from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline
from cloth_agent.robot_api import SimulatedBackend
from cloth_agent.session import AgentSession


@dataclass
class Proposal:
    actions: tuple


def proposal(lift=80):
    def move(x, z):
        return {'name': 'move', 'args': {'x': x, 'y': 0., 'z': z, 'yaw': 0.}}
    return Proposal((move(300, 100), {'name': 'open_gripper', 'args': {}}, move(300, 20),
                     {'name': 'close_gripper', 'args': {}}, move(300, lift), move(400, lift),
                     move(400, 20), {'name': 'open_gripper', 'args': {}}, move(400, 100),
                     {'name': 'home', 'args': {}}))


def decision(kind='GRASP_CONFIRMED', confidence=.9):
    return {'classification': kind, 'confidence': confidence,
            'evidence': ['Fabric visibly retained above the support.'], 'reason': 'Visible contact and lifted cloth.'}


def test_micro_lift_preserves_model_continuation_and_builds_reverse_release():
    original = proposal()
    compiled, gate = compile_grasp_checkpoint(original)
    assert len(original.actions) == 10
    assert gate['checkpoint_action_index'] == 4
    assert compiled.actions[4]['args']['z'] == 30
    assert compiled.actions[5:] == original.actions[4:]
    assert gate['abort_actions'][0]['args']['z'] == 20
    assert [a['name'] for a in gate['abort_actions']] == ['move', 'open_gripper', 'move', 'home']
    short, plan = compile_grasp_checkpoint(proposal(lift=25))
    assert plan['lift_mm'] == 5 and not plan['inserted_micro_lift']
    assert len(short.actions) == 10
    invalid = proposal()
    invalid.actions[4]['args']['x'] += 1
    with pytest.raises(ExplorationPlanningError, match='vertical'):
        compile_grasp_checkpoint(invalid)


@pytest.mark.parametrize('value', [True, float('nan'), -1, 1.1, '0.9'])
def test_invalid_confidence_cannot_authorize_motion(value):
    with pytest.raises(ValueError):
        validate_grasp_decision(decision(confidence=value))


@pytest.mark.parametrize('lift,expected,extended', [(25, 50, True), (50, 50, False), (80, 80, False)])
def test_capture_lift_minimum_preserves_other_actions(lift, expected, extended):
    original = proposal(lift=lift)
    compiled, plan = compile_grasp_capture(original)
    assert compiled.actions[4]['args']['z'] == expected
    assert original.actions[4]['args']['z'] == lift
    assert compiled.actions[:4] == original.actions[:4]
    assert compiled.actions[5:] == original.actions[5:]
    assert plan['lift_mm'] >= 30
    assert plan['lift_extended'] is extended
    assert plan['blocking'] is False and plan['evaluation_stage'] == 'final'


@pytest.mark.parametrize('bad', ['lateral', 'down', 'nan'])
def test_capture_lift_rejects_invalid_geometry(bad):
    original = proposal()
    original.actions[4]['args']['x' if bad == 'lateral' else 'z'] = (
        310 if bad == 'lateral' else 10 if bad == 'down' else float('nan'))
    with pytest.raises(ExplorationPlanningError):
        compile_grasp_capture(original)


def test_raised_capture_pose_still_fails_workspace_preflight(tmp_path):
    config = RobotConfig(robot_ip='test', boundaries=WorkspaceBounds(x_min=0, x_max=600,
        y_min=-300, y_max=300, z_min=0, z_max=500), init_joints_deg=(0,)*6,
        init_pose_mm_deg=(300, 0, 200, 180, 0, 0), orientation_roll_deg=180, orientation_pitch_deg=0)
    session = AgentSession(tmp_path, tmp_path / 'runs' / 'run', config, ExperimentConfig())
    original = proposal(lift=490)
    original.actions[2]['args']['z'] = 480
    compiled, _ = compile_grasp_capture(original)
    assert compiled.actions[4]['args']['z'] == 510
    source = session.workspace / 'fold.py'
    lines = ['def run():']
    for action in compiled.actions:
        args = action['args']
        lines.append('    ' + (f"move({args['x']}, {args['y']}, {args['z']}, {args['yaw']})"
                              if action['name'] == 'move' else action['name'] + '()'))
    source.write_text('\n'.join(lines) + '\n')
    assert session.runner.preflight(source).error


@pytest.mark.parametrize('outcome', ['captured', 'capture_failure', 'late'])
def test_production_pipeline_continues_motion_while_lift_photo_is_pending(tmp_path, monkeypatch, outcome):
    robot_config = RobotConfig(robot_ip='test', boundaries=WorkspaceBounds(x_min=0, x_max=600,
        y_min=-300, y_max=300, z_min=0, z_max=500), init_joints_deg=(0,)*6,
        init_pose_mm_deg=(300, 0, 200, 180, 0, 0), orientation_roll_deg=180, orientation_pitch_deg=0)
    log = []
    transport = threading.Event()
    trajectory_finished = threading.Event()
    class Arm(SimulatedBackend):
        def move(self, x, y, z, yaw, config):
            log.append(('move', x, z))
            if x == 400:
                transport.set()
            return super().move(x, y, z, yaw, config)
        def close_gripper(self, config):
            log.append(('close',))
            return super().close_gripper(config)
        def open_gripper(self, config):
            log.append(('open',))
            return super().open_gripper(config)
        def home(self, config):
            log.append(('home',))
            return super().home(config)
    monkeypatch.setattr('cloth_agent.experiment.XArmBackend', Arm)
    checked_paths = []
    monkeypatch.setattr('cloth_agent.experiment.validate_controller_trajectory',
                        lambda config, actions: checked_paths.append(actions))
    run = tmp_path / 'runs' / 'run'
    session = AgentSession(tmp_path, run, robot_config, ExperimentConfig())
    (run / 'run_metadata.json').write_text(json.dumps({
        'last_perception_mode': 'single_camera_rgbd', 'last_active_cameras': ['A']}))
    compiled, capture_plan = compile_grasp_capture(proposal(lift=25))
    # Use the restricted runtime's normal program format.
    source = session.workspace / 'fold.py'
    lines = ['def run():']
    for action in compiled.actions:
        args = action['args']
        lines.append('    ' + (f"move({args['x']}, {args['y']}, {args['z']}, {args['yaw']})"
                              if action['name'] == 'move' else action['name'] + '()'))
    source.write_text('\n'.join(lines) + '\n')
    def snapshot(config, recorder, path, after_ns):
        # A synchronous callback would time out here and fail this test.
        assert transport.wait(2), 'camera blocked transport'
        if outcome == 'late':
            assert trajectory_finished.wait(2)
        log.append(('snapshot', path.name))
        if outcome == 'capture_failure':
            raise TimeoutError('no fresh frame')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12, 12), (180, 180, 180)).save(path)
        return {'status': 'CAPTURED', 'image': str(path),
                'frame_monotonic_ns': time.monotonic_ns() if outcome == 'late' else after_ns + 1}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb', snapshot)
    class Backend:
        def invoke(self, **kwargs):
            pytest.fail('No mid-motion Claude call is permitted')
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.real = pipeline.confirm_real = True
    pipeline.record_video = False
    pipeline.observer_camera_serial = None
    pipeline.session = session
    pipeline.client = SimpleNamespace(backend=Backend())
    pipeline._debug = lambda *args, **kwargs: None
    pipeline._debug_exception = lambda *args, **kwargs: None
    run_experiment = session.run_experiment
    def execute_trajectory(*args, **kwargs):
        try:
            return run_experiment(*args, **kwargs)
        finally:
            trajectory_finished.set()
    session.run_experiment = execute_trajectory
    result, recording = pipeline._execute(source, SimpleNamespace(active_camera_labels=('A',)),
        run / 'iteration_001', label='fold', grasp_capture=capture_plan)
    assert checked_paths  # compiled full trajectory still goes through controller checks
    assert result['execution_completed'] is True
    assert 'checkpoint' not in result
    assert ('move', 400, 25) in log
    assert ('move', 300, 50) in log
    assert not (run / 'iteration_001/hold_check/grasp_decision.json').exists()
    saved = recording['grasp_snapshots']['after_lift']
    assert saved['requested_lift_mm'] == 30
    assert saved['asynchronous'] is True
    assert saved['status'] == {'captured': 'CAPTURED', 'capture_failure': 'FAILED', 'late': 'MISSED_WINDOW'}[outcome]


def test_checkpoint_rejection_never_unattended_restarts():
    assert not FoldExplorationPipeline._unattended_error_is_retriable(
        GraspCheckpointRejected('visual assessment failed'), 'evaluation')
