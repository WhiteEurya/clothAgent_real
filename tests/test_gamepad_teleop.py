from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from cloth_agent.gamepad_teleop import Intent, Mapper, InputGuard, normalize, motion_source
from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import pose_matrix, DualArmError


def ready_mapper():
    m = Mapper()
    m.update(0, 0, 0, set())
    return m


def test_physical_x_y_codes_reach_correct_actions(monkeypatch):
    import sys
    from cloth_agent.gamepad_teleop import Gamepad
    # Actual Xbox report codes, independent of potentially confusing aliases.
    codes = SimpleNamespace(ABS_X=0, ABS_Y=1, ABS_RX=3, EV_ABS=3, EV_KEY=1,
                           BTN_SOUTH=304, BTN_EAST=305, BTN_X=307, BTN_Y=308,
                           BTN_TL=310, BTN_START=315)
    held = []
    device = SimpleNamespace(
        capabilities=lambda **kw: {3: [0, 1, 3], 1: [304, 305, 307, 308, 310, 315]},
        absinfo=lambda code: SimpleNamespace(value=0, min=-32768, max=32767),
        active_keys=lambda: held, close=lambda: None)
    monkeypatch.setitem(sys.modules, 'evdev', SimpleNamespace(ecodes=codes, InputDevice=lambda path: device))
    pad, mapper = Gamepad('fake'), Mapper()
    mapper.update(*pad.sample())
    held[:] = [310, 307]  # LB + physical X
    intent = mapper.update(*pad.sample())
    assert intent.gripper == 'open' and intent.up == 0
    held[:] = [310, 308]  # LB + physical Y
    intent = mapper.update(*pad.sample())
    assert intent.up == 1 and intent.gripper is None


def test_layout_and_deadman():
    m = ready_mapper()
    assert m.update(1, -1, 1, {'Y', 'X'}) == Intent()
    actual = m.update(1, -1, 1, {'LB', 'Y'})
    assert (actual.forward, actual.right, actual.up, actual.yaw) == (1, 1, 1, -1)
    assert m.update(0, 0, 0, {'LB', 'A'}).up == -1
    assert m.update(0, 0, 0, {'LB', 'X'}).gripper == 'open'
    assert m.update(0, 0, 0, {'LB', 'B'}).gripper == 'close'
    assert m.update(0, 0, 0, {'LB', 'START'}).stop
    assert not m.update(0, 0, 0, {'LB', 'START'}).enabled


def test_initial_held_controls_do_not_arm():
    m = Mapper()
    assert not m.update(1, 0, 0, {'LB'}).enabled
    assert not m.update(0, 0, 0, {'LB'}).enabled
    m.update(0, 0, 0, set())
    assert m.update(0, 0, 0, {'LB'}).enabled


def test_gripper_hold_release_and_conflicts():
    m = ready_mapper()
    assert m.update(0, 0, 0, {'X'}).gripper is None
    assert m.update(0, 0, 0, {'LB', 'X'}).gripper is None
    m.update(0, 0, 0, {'LB'})
    assert m.update(0, 0, 0, {'LB', 'X'}).gripper == 'open'
    assert m.update(0, 0, 0, {'LB', 'X'}).gripper == 'open'
    assert m.update(0, 0, 0, {'LB'}).gripper is None
    conflict = m.update(0, 0, 0, {'LB', 'X', 'B', 'Y', 'A'})
    assert conflict.gripper is None and conflict.up == 0


def test_deadzone_and_bad_ranges():
    assert normalize(32768, 0, 65535) == 0
    assert normalize(-32768, -32768, 32767) == -1
    assert normalize(32767, -32768, 32767) == 1
    assert normalize(.1, -1, 1) == 0
    assert normalize(.575, -1, 1) == pytest.approx(.5)
    with pytest.raises(ValueError):
        normalize(0, 0, 0)
    with pytest.raises(ValueError):
        normalize(float('nan'), -1, 1)


def test_world_jog_and_units():
    initial = np.array([500., 20., 100., 180., 20., 35.])
    source = motion_source(Intent(forward=1, right=1, up=1, yaw=1, enabled=True), initial)
    name, arguments = parse_code(source)[0]
    assert name == 'move_tcp' and set(arguments) == {'right'}
    pose = np.array(arguments['right'])
    target = np.r_[pose[:3]*1000, np.degrees(pose[3:])]
    assert np.linalg.norm(target[:3] - initial[:3]) == pytest.approx(1)
    assert np.sign(target[:3] - initial[:3]).tolist() == [1, -1, 1]
    from scipy.spatial.transform import Rotation
    relative = pose_matrix(target)[:3, :3] @ pose_matrix(initial)[:3, :3].T
    assert Rotation.from_matrix(relative).as_rotvec() == pytest.approx([0, 0, np.deg2rad(2.)])


@pytest.mark.parametrize('step,stick', [(.5, 1), (2, .25), (5, -1)])
def test_yaw_step_adjustment_preserves_tcp_and_scales_with_stick(step, stick):
    from scipy.spatial.transform import Rotation
    initial = [500, 20, 100, 180, 20, 35]
    source = motion_source(Intent(yaw=stick, enabled=True), initial, yaw_step_deg=step)
    target = np.array(parse_code(source)[0][1]['right'])
    target = np.r_[target[:3]*1000, np.degrees(target[3:])]
    assert target[:3] == pytest.approx(initial[:3])
    relative = pose_matrix(target)[:3, :3] @ pose_matrix(initial)[:3, :3].T
    assert Rotation.from_matrix(relative).as_rotvec() == pytest.approx([0, 0, np.deg2rad(step*stick)])


def test_continuous_cli_defaults_and_limits(monkeypatch):
    from cloth_agent import gamepad_teleop
    seen = []
    monkeypatch.setattr(gamepad_teleop, 'run', lambda args: seen.append(args.yaw_speed_deg_s) or 0)
    assert gamepad_teleop.main([]) == 0
    assert seen == [5.]
    assert gamepad_teleop.main(['--yaw-speed-deg-s', '3']) == 0
    for value in ('0', '5.1', 'nan', 'inf'):
        with pytest.raises(SystemExit):
            gamepad_teleop.main(['--yaw-speed-deg-s', value])
        with pytest.raises(ValueError):
            motion_source(Intent(yaw=1, enabled=True), [0]*6, yaw_step_deg=float(value))


def test_gripper_priority_and_no_disabled_commands():
    assert motion_source(Intent(forward=1), [0]*6) is None
    source = motion_source(Intent(forward=1, gripper='open', enabled=True), [0]*6)
    assert parse_code(source) == [('gripper', {'arm': 'right', 'action': 'open'})]
    with pytest.raises(ValueError):
        motion_source(Intent(enabled=True), [0]*6, step_mm=10)


def test_guard_expires_and_invalidates_release_repress():
    guard = InputGuard()
    candidate = Intent(forward=1, enabled=True)
    assert not guard.permits()
    guard.update(candidate)
    token = guard.token()
    assert guard.permits(candidate, token)
    guard.update(Intent(enabled=True))
    assert not guard.permits(candidate, token)
    guard.update(Intent())
    guard.update(candidate)
    assert not guard.permits(candidate, token)
    guard.updated = time.monotonic() - 1
    assert not guard.permits()


def test_runtime_refuses_detour_and_left(monkeypatch):
    from cloth_agent.gamepad_arm7 import JogRuntime
    from cloth_agent.codex_native_runtime import NativeRuntime
    seen = []
    monkeypatch.setattr(NativeRuntime, '_tcp_segments', lambda *a, **kw: seen.append(kw))
    runtime = object.__new__(JogRuntime)
    runtime._tcp_segments('right', [], {}, allow_detour=True)
    assert seen == [{'allow_detour': False}]
    with pytest.raises(DualArmError):
        runtime._tcp_segments('left', [], {})


def test_watchdog_stops_on_release():
    from cloth_agent.gamepad_arm7 import Arm7Jogger
    jogger = object.__new__(Arm7Jogger)
    jogger.guard = InputGuard()
    jogger.executing = threading.Event()
    jogger.execution_generation = None
    jogger.closed = threading.Event()
    stopped = threading.Event()
    jogger.runtime = SimpleNamespace(stop=stopped.set)
    jogger.executing.set()
    worker = threading.Thread(target=jogger._watch)
    worker.start()
    try:
        assert stopped.wait(1)
    finally:
        jogger.closed.set()
        worker.join(1)


def fake_jogger(tmp_path):
    from cloth_agent.gamepad_arm7 import Arm7Jogger
    jogger = object.__new__(Arm7Jogger)
    jogger.guard = InputGuard()
    jogger.output = tmp_path
    jogger.step_mm, jogger.yaw_step_deg = 1., .5
    jogger.index = 0
    jogger.executing = threading.Event()
    jogger.execution_generation = None
    calls = []
    q = {'left': np.zeros(6), 'right': np.zeros(7)}
    operation = ('move', SimpleNamespace(arm='right', start=q, end=q))
    runtime = SimpleNamespace(
        cancel=threading.Event(),
        coordinator=SimpleNamespace(snapshots=lambda: {k: {'joints': v} for k, v in q.items()}),
        config=SimpleNamespace(arms={'right': SimpleNamespace(world_from_base=np.eye(4))}),
        models={'right': SimpleNamespace(forward=lambda q: [500, 0, 100, 180, 0, 0])},
        plan=lambda *a, **kw: [operation], arrival=lambda q: None,
        _execute_operations=lambda *a: calls.append(a) or {'status': 'EXECUTED'})
    jogger.runtime = runtime
    return jogger, calls


def test_dispatch_after_validation_and_expired_input(tmp_path):
    jogger, calls = fake_jogger(tmp_path)
    intent = Intent(forward=1, enabled=True)
    jogger.guard.update(intent)
    assert jogger.step(intent, jogger.guard.token())['status'] == 'EXECUTED'
    assert len(calls) == 1
    plan = jogger.runtime.plan
    def disarm(*a, **kw):
        jogger.guard.update(Intent())
        return plan(*a, **kw)
    jogger.runtime.plan = disarm
    jogger.guard.update(intent)
    assert jogger.step(intent, jogger.guard.token())['status'] == 'INPUT_EXPIRED'
    assert len(calls) == 1


def test_validation_failure_never_executes(tmp_path):
    jogger, calls = fake_jogger(tmp_path)
    intent = Intent(forward=1, enabled=True)
    jogger.guard.update(intent)
    def fail(*a, **kw):
        raise DualArmError('collision')
    jogger.runtime.plan = fail
    assert jogger.step(intent, jogger.guard.token())['status'] == 'REJECTED'
    assert not calls


def test_dispatch_rejects_left_movement(tmp_path):
    jogger, calls = fake_jogger(tmp_path)
    intent = Intent(forward=1, enabled=True)
    jogger.guard.update(intent)
    jogger.runtime.plan = lambda *a, **kw: [
        ('move', SimpleNamespace(arm='left'))]
    with pytest.raises(DualArmError):
        jogger.step(intent, jogger.guard.token())
    assert not calls


def test_jog_connects_only_arm7_with_parked_arm6(tmp_path, monkeypatch):
    import json
    from cloth_agent.gamepad_arm7 import JogRuntime
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm import cli
    from cloth_agent.dual_arm.static_arm import StaticArmObstacle
    folder = tmp_path / 'data/robot'
    folder.mkdir(parents=True)
    reference = folder / 'arm6_parked_for_arm7.json'
    reference.write_text(json.dumps({'arm': 'left', 'joint_angles_deg': [1, 2, 3, 4, 5, 6]}))
    config = SimpleNamespace(root=tmp_path, arms={
        'left': SimpleNamespace(axis=6), 'right': SimpleNamespace(axis=7)},
        collision_envelopes_disabled_arms=frozenset())
    monkeypatch.setattr(NativeRuntime, 'load_config', lambda *a: config)
    runtime = object.__new__(JogRuntime)
    assert runtime.load_config('fake') is config
    assert not config.collision_envelopes_disabled_arms
    opened = []
    def connection(arm, *a, **kw):
        assert arm is config.arms['right']
        opened.append('right')
        return SimpleNamespace(disconnect=lambda: None)
    monkeypatch.setattr(cli, 'XArmConnection', connection)
    models = {'left': SimpleNamespace(forward=lambda q: np.asarray(q))}
    connections = cli.connect(config, models, threading.Event(), True)
    assert opened == ['right']
    assert isinstance(connections['left'], StaticArmObstacle)
    assert connections['left'].snapshot()['joints'] == [1, 2, 3, 4, 5, 6]
    assert connections['left'].stop()['status'] == 'NOT_CONNECTED'
    reference.write_text(json.dumps({'arm': 'left', 'joint_angles_deg': [1, 2]}))
    with pytest.raises(DualArmError, match='invalid arm6'):
        runtime.load_config('fake')
