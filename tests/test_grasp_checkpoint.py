import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.config import RobotConfig, WorkspaceBounds, ExperimentConfig
from cloth_agent.free_exploration import ExplorationPlanningError, exploration_source
from cloth_agent.grasp_checkpoint import compile_grasp_checkpoint, validate_grasp_decision, GraspCheckpointRejected
from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline
from cloth_agent.planner_backend import BackendResult, PlannerBackendError
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


@pytest.mark.parametrize('outcome', ['positive', 'empty', 'uncertain', 'low_confidence',
                                   'invalid', 'timeout', 'refusal', 'capture_failure', 'interrupt'])
def test_production_pipeline_gates_transport_and_handles_abort(tmp_path, monkeypatch, outcome):
    robot_config = RobotConfig(robot_ip='test', boundaries=WorkspaceBounds(x_min=0, x_max=600,
        y_min=-300, y_max=300, z_min=0, z_max=500), init_joints_deg=(0,)*6,
        init_pose_mm_deg=(300, 0, 200, 180, 0, 0), orientation_roll_deg=180, orientation_pitch_deg=0)
    log = []
    class Arm(SimulatedBackend):
        def move(self, x, y, z, yaw, config):
            log.append(('move', x, z))
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
    compiled, gate = compile_grasp_checkpoint(proposal())
    # Use the restricted runtime's normal program format.
    source = session.workspace / 'fold.py'
    lines = ['def run():']
    for action in compiled.actions:
        args = action['args']
        lines.append('    ' + (f"move({args['x']}, {args['y']}, {args['z']}, {args['yaw']})"
                              if action['name'] == 'move' else action['name'] + '()'))
    source.write_text('\n'.join(lines) + '\n')
    def snapshot(config, recorder, path, after_ns):
        log.append(('snapshot', path.name))
        if outcome == 'capture_failure':
            raise TimeoutError('no fresh frame')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (12, 12)).save(path)
        return {'status': 'CAPTURED', 'image': str(path)}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb', snapshot)
    calls = []
    class Backend:
        def invoke(self, **kwargs):
            calls.append(kwargs)
            log.append(('judge',))
            assert log.index(('move', 300, 30)) < len(log) - 1
            assert ('move', 300, 80) not in log and ('move', 400, 80) not in log
            assert kwargs['max_turns'] == 4 and kwargs['image_edit_limit'] == 0
            assert kwargs['overall_timeout_s'] == 90
            assert len(kwargs['image_paths']) == 2
            if outcome == 'interrupt':
                raise KeyboardInterrupt()
            if outcome == 'timeout':
                raise PlannerBackendError('timed out')
            payload = decision()
            if outcome == 'empty': payload = decision('EMPTY')
            if outcome == 'uncertain': payload = decision('UNKNOWN')
            if outcome == 'low_confidence': payload = decision(confidence=.5)
            if outcome == 'invalid': payload = {'classification': 'GRASP_CONFIRMED'}
            envelope = {'result': json.dumps(payload)}
            if outcome == 'refusal': envelope = {'is_error': True, 'result': 'API refusal'}
            return BackendResult(json.dumps(envelope), '', 0, ())
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.real = pipeline.confirm_real = True
    pipeline.record_video = False
    pipeline.observer_camera_serial = None
    pipeline.session = session
    pipeline.client = SimpleNamespace(backend=Backend())
    pipeline._debug = lambda *args, **kwargs: None
    pipeline._debug_exception = lambda *args, **kwargs: None
    execute = lambda: pipeline._execute(source, SimpleNamespace(active_camera_labels=('A',)),
        run / 'iteration_001', label='fold', grasp_gate=gate)
    if outcome == 'interrupt':
        with pytest.raises(KeyboardInterrupt): execute()
        assert log[-1] == ('judge',)
        assert session.last_return_home_outcome['attempted'] is False
        return
    result, recording = execute()
    assert len(checked_paths) >= 2  # both full continuation and alternate path checked
    assert result['checkpoint']['executed_branch'] == ('CONTINUATION' if outcome == 'positive' else 'ABORT_RELEASE')
    assert (('move', 400, 80) in log) == (outcome == 'positive')
    assert (('move', 300, 80) in log) == (outcome == 'positive')
    assert len(calls) == (0 if outcome == 'capture_failure' else 1)
    assert log.index(('close',)) < log.index(('snapshot', 'camera_A_grasp_after_close.png')) < log.index(('move', 300, 30))
    saved = json.loads((run / 'iteration_001/hold_check/grasp_decision.json').read_text())
    assert saved['continue_transport'] is (outcome == 'positive')


def test_checkpoint_rejection_never_unattended_restarts():
    assert not FoldExplorationPipeline._unattended_error_is_retriable(
        GraspCheckpointRejected('visual assessment failed'), 'evaluation')
