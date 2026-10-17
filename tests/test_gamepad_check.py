from collections import namedtuple
from types import SimpleNamespace

import pytest

from cloth_agent import gamepad_check as check
from cloth_agent.gamepad_teleop import Intent


class Pad:
    keys = {304: 'A', 305: 'B', 307: 'X', 308: 'Y', 310: 'LB', 315: 'START'}

    def __init__(self, samples):
        self.samples = iter(samples)
        self.last = (0., 0., 0., set())
        self.device = SimpleNamespace(path='fake', name='synthetic gamepad',
            info=namedtuple('Info', 'vendor product')(1, 2),
            active_keys=lambda: [code for code, name in self.keys.items() if name in self.last[3]])

    def sample(self):
        self.last = next(self.samples, self.last)
        return self.last


@pytest.fixture
def clock(monkeypatch):
    now = [100.]
    monkeypatch.setattr(check.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(check.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    return now


def sample_for(step):
    axes = [0., 0., 0.]
    if step.axis is not None:
        axes[step.axis] = float(step.sign)
    return (*axes, set(step.keys))


def test_full_guided_session_without_keyboard_or_robot(tmp_path, clock):
    neutral = (0., 0., 0., set())
    samples = [neutral] * 25
    for step in check.STEPS:
        samples += [sample_for(step)] * 25
        if step.name != 'release_prepare':
            samples += [neutral] * 25
    runner = check.Check(Pad(samples), tmp_path, timeout=3)
    runner.run()
    runner.finish()
    assert runner.report['status'] == 'PASS'
    assert len(runner.report['steps']) == len(check.STEPS)
    assert all(row['status'] == 'PASS' for row in runner.report['steps'])
    assert runner.report['robot_connected'] is False
    assert (tmp_path / 'report.json').exists()
    assert (tmp_path / 'samples.jsonl').stat().st_size > 0


def test_wrong_direction_and_button_do_not_pass():
    step = next(s for s in check.STEPS if s.name == 'forward')
    assert not check.matches(step, (0, 1, 0), {'LB'})
    assert not check.matches(step, (0, -1, 0), {'RB'})
    assert not check.matches(step, (0, -1, 0), {'LB', 'KEY_311'})
    assert not check.matches(step, (1, -1, 0), {'LB'})
    assert check.matches(step, (0, -1, 0), {'LB'})


def test_unmoved_controller_times_out_instead_of_passing(tmp_path, clock):
    runner = check.Check(Pad([]), tmp_path, timeout=.5)
    result = runner.step(check.STEPS[1])
    runner.finish()
    assert result['status'] == 'FAIL'
    assert runner.report['status'] == 'INCOMPLETE'


def test_disable_requires_all_motion_and_gripper_zero():
    step = check.STEPS[0]
    assert check.effect_matches(step, Intent())
    assert not check.effect_matches(step, Intent(forward=1))
    assert not check.effect_matches(step, Intent(gripper='open'))


def test_gripper_requires_actual_rising_edge(tmp_path, clock):
    step = next(s for s in check.STEPS if s.name == 'open')
    runner = check.Check(Pad([sample_for(step)]*100), tmp_path, timeout=.5)
    runner.mapper.update(0, 0, 0, set())
    # X was already held while disabled: adding LB must not pass open test.
    runner.mapper.update(0, 0, 0, {'X'})
    result = runner.step(step)
    runner.finish()
    assert result['status'] == 'FAIL'


def test_disconnect_preserves_error_report(tmp_path, monkeypatch):
    import json
    class BrokenPad(Pad):
        def __init__(self, *args, **kwargs):
            super().__init__([])
        def sample(self):
            raise OSError('USB disconnected')
        def close(self):
            pass
    monkeypatch.setattr(check, 'Gamepad', BrokenPad)
    output = tmp_path / 'report'
    assert check.main(['--device', 'fake', '--output', str(output), '--probe-seconds', '.1']) == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'ERROR'
    assert 'USB disconnected' in report['reason']
