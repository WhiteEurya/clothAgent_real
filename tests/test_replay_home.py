from types import SimpleNamespace

from cloth_agent.dual_arm import homing, pyroki_home
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.experiments import graph_policy_replay as replay


def test_seven_axis_home_routes_to_pyroki(monkeypatch):
    config = SimpleNamespace(arms={key: SimpleNamespace(axis=7) for key in ('left', 'right')})
    initial, models = object(), object()
    def plan(actual, configured, geometry):
        assert actual is initial and configured is config and geometry is models
        return ['checked program'], ['checked segment'], ['attempt']
    monkeypatch.setattr(pyroki_home, 'plan_home', plan)
    program, attempts, planner = homing.plan_configured_home(initial, config, models)
    assert program == ['checked program'] and attempts == ['attempt']
    assert 'PyRoki' in planner


def test_replay_home_does_not_build_historical_runtime(monkeypatch, tmp_path):
    calls = []
    def run(config, **kwargs):
        calls.append((config, kwargs))
        return {'status': 'COMPLETED', 'physical_execution': False}
    monkeypatch.setattr(homing, 'gripper_home', run)
    assert replay.main(['--home-only', '--config', 'current.json', '--output', str(tmp_path)]) == 0
    assert str(calls[0][0]) == 'current.json'
    assert calls[0][1]['simulated'] is True


def test_replay_home_preserves_rejection_and_explicit_execution(monkeypatch, tmp_path):
    def reject(config, **kwargs):
        assert kwargs['simulated'] is False
        raise DualArmError('existing path check rejected')
    monkeypatch.setattr(homing, 'gripper_home', reject)
    assert replay.main(['--home-only', '--execute', '--config', 'current.json', '--output', str(tmp_path)]) == 1
    assert 'existing path check rejected' in (tmp_path / 'home_failure.json').read_text()
