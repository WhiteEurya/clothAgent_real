import json
from unittest.mock import Mock

import numpy as np
import pytest

from cloth_agent.config import RobotConfig, SafetyError
from cloth_agent.perception import PerceptionConfig, RGBDFrame, camera_base_xyz_map_mm
from cloth_agent.robot_api import ControllerTrajectoryValidation, SimulatedBackend
from scripts import manual_pixel_move as script
from scripts.check_camera_calibration import load_capture, measure_point


@pytest.fixture
def config():
    return RobotConfig.load(script.PROJECT_ROOT, script.PROJECT_ROOT / 'config/robot.example.json')


@pytest.fixture
def scene():
    # Camera looking down: centre pixel maps to base (400, 0, 50) mm.
    transform = np.diag([1., -1., -1., 1.])
    transform[:3, 3] = [.4, 0, .55]
    frame = RGBDFrame('A', 'test', np.zeros((6, 8, 3), dtype=np.uint8),
                     np.full((6, 8), .5), np.array([[100., 0, 4], [0, 100, 3], [0, 0, 1]]),
                     transform)
    xyz, valid = camera_base_xyz_map_mm(frame, PerceptionConfig(cameras=()))
    return frame, xyz, valid


def test_display_scaling_uses_raw_coordinates():
    assert script.raw_pixel(179.5, 275.5, (640, 360), (1280, 720)) == (359, 551)
    assert script.raw_pixel(639, 359, (640, 360), (1280, 720)) == (1278, 718)
    with pytest.raises(ValueError):
        script.raw_pixel(640, 0, (640, 360), (1280, 720))


def test_exact_pixel_target_and_high_approach(config, scene):
    selection = script.select_target(*scene, (4, 3), config, 30)
    assert selection['measured_base_xyz_mm'] == pytest.approx([400, 0, 50])
    assert selection['target_tcp_xyz_mm'] == pytest.approx([400, 0, 80])
    actions = script.build_actions(selection, config)
    assert [a['name'] for a in actions] == ['home', 'move', 'move']
    assert actions[1]['args']['z'] >= config.init_pose_mm_deg[2]
    assert actions[1]['args']['x'] == actions[2]['args']['x'] == 400
    assert actions[1]['args']['y'] == actions[2]['args']['y'] == 0
    assert actions[2]['args']['z'] == 80
    assert script.select_target(*scene, (4, 3), config, 0)['target_tcp_xyz_mm'][2] == 50


def test_invalid_depth_never_snaps_to_valid_neighbor(config, scene):
    frame, xyz, valid = scene
    valid[3, 4] = False
    with pytest.raises(ValueError, match='No valid depth'):
        script.select_target(frame, xyz, valid, (4, 3), config, 30)


def test_invalid_target_and_clearance_rejected(config, scene):
    for clearance in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            script.select_target(*scene, (4, 3), config, clearance)
    frame, xyz, valid = scene
    xyz[3, 4, 1] = 500
    with pytest.raises(SafetyError):
        script.select_target(frame, xyz, valid, (4, 3), config, 30)


def test_capture_round_trip_with_offline_checker(config, scene, tmp_path):
    frame, xyz, _ = scene
    script.save_capture(tmp_path, frame, xyz, {'actual_tcp_pose_mm_deg': [1, 2, 3, 0, 0, 0]}, config)
    capture = load_capture(tmp_path / 'result.json')
    measured = measure_point(capture, (4, 3))
    assert measured['base_xyz_mm'] == pytest.approx([400, 0, 50])
    assert measured['map_error_mm'] < 1e-4


def test_ik_failure_prevents_motion_backend_creation(config, scene, monkeypatch, tmp_path):
    selection = script.select_target(*scene, (4, 3), config, 30)
    validate = Mock(side_effect=SafetyError('unreachable'))
    backend = Mock()
    monkeypatch.setattr(script, 'validate_controller_trajectory', validate)
    monkeypatch.setattr(script, 'XArmBackend', backend)
    with pytest.raises(SafetyError):
        script.execute_target(config, script.build_actions(selection, config), {}, tmp_path / 'execution.json')
    backend.assert_not_called()


def test_move_once_without_gripper_or_automatic_return(config, scene, monkeypatch, tmp_path):
    selection = script.select_target(*scene, (4, 3), config, 30)
    backend = SimulatedBackend(config)
    backend.close = Mock()
    validate = Mock(return_value=ControllerTrajectoryValidation({}, 0, (0, 0, 172, 0, 0, 0), 42))
    monkeypatch.setattr(script, 'validate_controller_trajectory', validate)
    monkeypatch.setattr(script, 'XArmBackend', lambda cfg: backend)
    report = {}
    path = tmp_path / 'execution.json'
    script.execute_target(config, script.build_actions(selection, config), report, path)
    assert report['status'] == 'AT_TARGET'
    assert [a['name'] for a in report['actual_robot_actions']] == ['home', 'move', 'move']
    assert report['actual_robot_actions'][-1]['actual_ee_pose'][:3] == [400, 0, 80]
    assert json.loads(path.read_text())['controller_ik']['validated_sample_count'] == 42
    backend.close.assert_called_once()


def test_approach_failure_saves_feedback_and_does_not_descend(config, scene, monkeypatch, tmp_path):
    selection = script.select_target(*scene, (4, 3), config, 30)
    backend = SimulatedBackend(config)
    backend.move = Mock(side_effect=RuntimeError('controller failure'))
    backend.close = Mock()
    monkeypatch.setattr(script, 'validate_controller_trajectory',
                        lambda *args: ControllerTrajectoryValidation({}, 0, (0, 0, 172, 0, 0, 0), 42))
    monkeypatch.setattr(script, 'XArmBackend', lambda cfg: backend)
    report = {}
    path = tmp_path / 'execution.json'
    with pytest.raises(RuntimeError, match='controller failure'):
        script.execute_target(config, script.build_actions(selection, config), report, path)
    backend.move.assert_called_once()
    backend.close.assert_called_once()
    saved = json.loads(path.read_text())
    assert len(saved['actual_robot_actions']) == 2
    assert not saved['actual_robot_actions'][-1]['success']
    assert 'controller failure' in saved['actual_robot_actions'][-1]['error']


@pytest.mark.parametrize('argv', [[], ['--real'], ['--confirm-real'],
                                ['--real', '--confirm-real', '--clearance-mm', 'nan']])
def test_invalid_invocation_cannot_connect_or_move(argv, monkeypatch):
    observation = Mock()
    monkeypatch.setattr(script, 'move_robot_to_perception_position', observation)
    with pytest.raises(SystemExit) as exc:
        script.main(argv)
    assert exc.value.code == 2
    observation.assert_not_called()
