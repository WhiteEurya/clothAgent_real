import json
from pathlib import Path

import pytest

from scripts import replay_gripper_test as replay
from cloth_agent.config import RobotConfig, WorkspaceBounds


@pytest.fixture
def replay_root(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, 'PROJECT_ROOT', tmp_path)
    config = RobotConfig(robot_ip='test', boundaries=WorkspaceBounds(x_min=0, x_max=600,
        y_min=-300, y_max=300, z_min=0, z_max=500), init_joints_deg=(0,)*6,
        init_pose_mm_deg=(300, 0, 200, 180, 0, 0), orientation_roll_deg=180, orientation_pitch_deg=0)
    monkeypatch.setattr(replay.RobotConfig, 'load', lambda *args: config)
    return tmp_path


def test_default_replays_exact_ten_actions_without_robot(replay_root, monkeypatch):
    monkeypatch.setattr('cloth_agent.experiment.XArmBackend',
                        lambda *args: pytest.fail('mock must never connect'))
    assert replay.main([]) == 0
    run = next((replay_root / 'runs').iterdir())
    result = json.loads((run / 'results' / 'recorded_gripper_test.json').read_text())
    assert not result['physical_execution']
    actions = result['actual_robot_actions']
    assert [a['name'] for a in actions] == [
        'move', 'open_gripper', 'move', 'close_gripper', 'move',
        'move', 'move', 'open_gripper', 'move', 'home']
    assert [list(a['args'].values()) for a in actions if a['name'] == 'move'] == [
        [410.527, -149.384, 116.690, 0.], [410.527, -149.384, 36.690, 0.],
        [410.527, -149.384, 106.690, 0.], [382.736, -79.166, 106.690, 0.],
        [382.736, -79.166, 42.690, 0.], [382.736, -79.166, 131.690, 0.]]
    assert len((run / 'action_events.jsonl').read_text().splitlines()) == 10
    assert replay.main([]) == 0
    assert len(list((replay_root / 'runs').iterdir())) == 2


@pytest.mark.parametrize('args', [['--real'], ['--confirm-real']])
def test_real_requires_explicit_flags_before_any_run(replay_root, args):
    with pytest.raises(SystemExit) as error:
        replay.main(args)
    assert error.value.code == 2
    assert not (replay_root / 'runs').exists()


def test_real_uses_controller_preflight_before_connecting(replay_root, monkeypatch):
    def blocked(*args):
        raise ValueError('IK rejected')
    monkeypatch.setattr('cloth_agent.experiment.validate_controller_trajectory', blocked)
    monkeypatch.setattr('cloth_agent.experiment.XArmBackend',
                        lambda *args: pytest.fail('no execution after IK failure'))
    assert replay.main(['--real', '--confirm-real']) == 1
    run = next((replay_root / 'runs').iterdir())
    assert json.loads((run / 'summary.json').read_text())['status'] == 'FAILED'
    result = json.loads((run / 'results' / 'recorded_gripper_test.json').read_text())
    assert result['actual_robot_actions'] == []
