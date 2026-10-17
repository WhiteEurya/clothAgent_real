from types import SimpleNamespace
import sys

import pytest

from cloth_agent.gamepad_calibrate import Learner, axis_candidate, button_candidate
from cloth_agent.gamepad_profile import identity, validate_profile, save_profile, load_profile
from cloth_agent.gamepad_teleop import Gamepad, Mapper, list_gamepads


def device():
    return SimpleNamespace(
        info=SimpleNamespace(bustype=3, vendor=0x810, product=1, version=0x110),
        name='USB Gamepad', capabilities=lambda **kw: {3: [0, 1, 2, 5], 1: list(range(288, 300))},
        close=lambda: None)


def profile(dev):
    return dict(version=1, device=identity(dev), axes=[{'code':0,'sign':1},
        {'code':1,'sign':-1}, {'code':5,'sign':1}],
        buttons=dict(A=290, B=289, X=291, Y=288, LB=292, START=297))


def test_candidate_requires_one_distinct_control():
    assert axis_candidate({0:1, 1:0}, set()) == (0, 1)
    assert axis_candidate({0:-1, 1:0}, set()) == (0, -1)
    assert axis_candidate({0:1, 1:1}, set()) is None
    assert axis_candidate({0:1}, set(), [0]) is None
    assert axis_candidate({0:1}, {288}) is None
    assert button_candidate({0:0}, {288}) == 288
    assert button_candidate({0:0}, {288, 289}) is None
    assert button_candidate({0:0}, {288}, [288]) is None


def test_profile_round_trip_identity_and_duplicates(tmp_path, monkeypatch):
    from cloth_agent import gamepad_profile
    monkeypatch.setattr(gamepad_profile, 'PROFILE_ROOT', tmp_path)
    dev = device()
    p = profile(dev)
    save_profile(p, dev)
    assert load_profile(dev) == p
    p['device']['product'] = 2
    with pytest.raises(ValueError, match='identity'):
        validate_profile(p, dev)
    p = profile(dev)
    p['buttons']['X'] = p['buttons']['Y']
    with pytest.raises(ValueError, match='distinct'):
        validate_profile(p, dev)


def test_generic_device_discovery_and_learned_mapping(monkeypatch):
    dev = device()
    held = []
    dev.active_keys = lambda: held
    dev.absinfo = lambda code: SimpleNamespace(value=0 if code == 1 else 128, min=0, max=255)
    codes = SimpleNamespace(ABS_X=0, ABS_Y=1, ABS_RX=3, BTN_SOUTH=304, BTN_EAST=305,
        BTN_X=307, BTN_Y=308, BTN_TL=310, BTN_START=315, BTN_GAMEPAD=304, BTN_JOYSTICK=288,
        EV_ABS=3, EV_KEY=1)
    monkeypatch.setitem(sys.modules, 'evdev', SimpleNamespace(InputDevice=lambda path:dev,
                        list_devices=lambda:['fake'], ecodes=codes))
    assert list_gamepads()[0]['vendor'] == '0x810'
    pad = Gamepad('fake', profile=profile(dev))
    m = Mapper()
    m.update(0, 0, 0, set())
    held[:] = [292, 288]  # learned LB + Y
    result = m.update(*pad.sample())
    assert result.up == 1 and result.gripper is None
    assert result.forward == -1  # learned inverted left-Y direction
    held[:] = [292, 291]  # learned LB + X
    assert m.update(*pad.sample()).gripper == 'open'


def test_complete_learning_from_synthetic_operator(monkeypatch):
    from cloth_agent import gamepad_calibrate
    now = [100.]
    monkeypatch.setattr(gamepad_calibrate.time, 'monotonic', lambda:now[0])
    monkeypatch.setattr(gamepad_calibrate.time, 'sleep', lambda dt:now.__setitem__(0, now[0]+dt))
    neutral = ({0:0, 1:0, 2:0, 5:0}, set())
    samples = [neutral]*25
    for code, sign in [(0, 1), (1, -1), (5, 1)]:
        for direction in [sign, -sign]:
            axes = dict(neutral[0]); axes[code] = direction
            samples += [(axes, set())]*25 + [neutral]*25
    for code in (290, 289, 291, 288, 292, 297):
        samples += [(neutral[0], {code})]*25 + [neutral]*25
    learner = object.__new__(Learner)
    learner.device, learner.timeout = device(), 3
    stream = iter(samples)
    learner.sample = lambda:next(stream, neutral)
    result = learner.learn()
    assert result == dict(profile(learner.device), name='USB Gamepad')
