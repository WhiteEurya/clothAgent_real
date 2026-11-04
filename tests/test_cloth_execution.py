import pytest

from cloth_agent import experiment
from cloth_agent.experiments.cloth_execution import main, validate_home


def test_home_uses_current_state_and_checked_planner_without_hardware(tmp_path, monkeypatch):
    import json
    from cloth_agent.experiments import cloth_execution
    from cloth_agent.dual_arm.geometry import DualArmError
    def forbidden(*args, **kwargs):
        pytest.fail('Hardware access during software preflight')
    monkeypatch.setattr(experiment, 'XArmBackend', forbidden)
    monkeypatch.setattr(experiment, 'validate_controller_trajectory', forbidden)
    source = 'outputs/home_request_20261104_retry_01/robot_state.json'
    recorded = json.load(open(source))
    def reject(initial, config, models):
        assert initial['right'].tolist() == recorded['arms']['right']['joints_deg']
        assert initial['right'].tolist() != config.arms['right'].home_joints.tolist()
        assert set(models) == {'left', 'right'}
        raise DualArmError('test collision refusal')
    monkeypatch.setattr(cloth_execution, 'plan_home', reject)
    report = validate_home(source, tmp_path / 'home')
    assert report['status'] == 'BLOCKED'
    assert 'test collision refusal' in report['error']
    assert report['execution_allowed'] is False
    assert report['robot_connections'] == 0
    assert report['controller_path_check'] == 'NOT_RUN'


def test_no_unchecked_real_option():
    with pytest.raises(SystemExit):
        main(['home', '--home-state', 'unused', '--output', 'unused', '--real'])
