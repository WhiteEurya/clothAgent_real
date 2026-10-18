import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent import gamepad_collect as collect
from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.gamepad_continuous import ContinuousArm6
from cloth_agent.gamepad_teleop import InputGuard, Intent


def test_transfer_preserves_orientation_and_places_before_release():
    source = collect.transfer_program([100, 200, 150, 175, 0, 25], [100, 200, 100],
                                      [300, 250, 140], clearance=30, place_offset=5)
    calls = parse_code(source)
    assert [name for name, _ in calls] == ['move_tcp']*2 + ['gripper', 'move_tcp']
    poses = [np.r_[np.asarray(args['left'][:3])*1000, np.degrees(args['left'][3:])]
             for name, args in calls if name == 'move_tcp']
    assert [p[2] for p in poses] == pytest.approx([150, 145, 175])
    assert poses[0][:2] == pytest.approx([300, 250])
    for pose in poses:
        assert pose[3:] == pytest.approx([175, 0, 25])
    assert calls[2][1] == dict(arm='left', action='open')


@pytest.mark.parametrize('height', [100, 104.9])
def test_insufficient_manual_lift_blocks_transfer(height):
    with pytest.raises(DualArmError, match='低于放置高度'):
        collect.transfer_program([0, 0, height, 180, 0, 0], [0, 0, 100], [100, 0, 100],
                                 clearance=30, place_offset=5)


def handoff_backend(monkeypatch):
    monkeypatch.setattr('cloth_agent.gamepad_continuous.time.sleep', lambda _: None)
    obj = object.__new__(ContinuousArm6)
    obj.guard = InputGuard(); obj.guard.update(Intent())
    obj.closed = threading.Event(); obj.workers = []; obj.failure = None
    obj.runtime = SimpleNamespace(cancel=threading.Event(), prepared=True,
        coordinator=SimpleNamespace(event=lambda *a, **kw: None))
    obj.mode_entered = True; obj.latency = .05
    obj.fast = SimpleNamespace(mode=4, state=0)
    calls = []
    def mode(value):
        calls.append(('mode', value)); obj.fast.mode = value; return 0
    obj.arm = SimpleNamespace(vc_set_joint_velocity=lambda *a, **kw: calls.append(('zero', a[0])) or 0,
                              set_mode=mode, set_state=lambda x: calls.append(('state', x)) or 0)
    obj.gripper = SimpleNamespace(active=True)
    obj.gripper_api = SimpleNamespace(get_gripper_position=lambda: (0, 321),
        set_gripper_position=lambda value, **kw: calls.append(('hold', value)) or 0,
        disconnect=lambda: calls.append(('disconnect',)))
    obj._feedback = lambda: (np.zeros(6), None, 0)
    obj.stop = lambda: obj.runtime.cancel.set()
    return obj, calls


def test_handoff_holds_gripper_restores_mode_without_stop(monkeypatch):
    obj, calls = handoff_backend(monkeypatch)
    obj.handoff()
    assert ('zero', [0.]*6) in calls and ('hold', 321) in calls
    assert ('mode', 0) in calls and ('state', 4) not in calls
    assert not obj.runtime.cancel.is_set()
    assert not obj.runtime.prepared
    assert obj.gripper_api is None


def test_handoff_active_input_cancels(monkeypatch):
    obj, calls = handoff_backend(monkeypatch)
    obj.guard.update(Intent(enabled=True))
    with pytest.raises(DualArmError, match='neutral'):
        obj.handoff()
    assert obj.runtime.cancel.is_set()
    assert not calls


def test_handoff_moving_feedback_blocks_position_mode(monkeypatch):
    obj, calls = handoff_backend(monkeypatch)
    feedback = iter([np.zeros(6), np.ones(6)])
    obj._feedback = lambda: (next(feedback), None, 0)
    with pytest.raises(DualArmError, match='still moving'):
        obj.handoff()
    assert obj.runtime.cancel.is_set()
    assert ('mode', 0) not in calls


@pytest.mark.parametrize('fail', [False, True])
def test_iteration_order_and_manifest_on_failure(tmp_path, monkeypatch, fail):
    stages = []
    class Recorder:
        def __init__(self, *args):
            self.started = collect.time.monotonic(); self.error = None; self.samples = 10
        def start(self): pass
        def check(self): pass
        def close(self): pass
        def stage(self, stage): stages.append(stage)
    monkeypatch.setattr(collect, 'TrajectoryRecorder', Recorder)
    obs = SimpleNamespace(validate_for=lambda *a, **kw: None)
    from cloth_agent.dual_arm.execution import DualArmCoordinator
    coordinator = object.__new__(DualArmCoordinator)
    coordinator.directory = tmp_path/'old_removed_iteration'/'controller'
    coordinator.log_lock = threading.Lock(); coordinator.sequence = 0
    def observe(folder):
        coordinator.event('new_iteration_observation')
        stages.append('capture'); folder.mkdir(); (folder/'rgb.png').write_bytes(b'saved')
        return obs
    runtime = SimpleNamespace(calibration_id='test', cancel=threading.Event(),
        config=SimpleNamespace(limits={'max_surface_spread_mm': 8}), observe=observe,
        coordinator=coordinator,
        stop=lambda: stages.append('stop'), move_to_perception=lambda p: stages.append('perception'))
    monkeypatch.setattr(collect, 'select_points', lambda *a: [
        dict(name='start', world_xyz_mm=[0, 0, 100]), dict(name='target', world_xyz_mm=[100, 0, 100])])
    monkeypatch.setattr(collect, 'approach_program', lambda *a: 'approach')
    monkeypatch.setattr(collect, 'world_pose', lambda r: np.array([0, 0, 200, 180, 0, 0.]))
    def execute(r, recorder, folder, stage, source):
        stages.append(stage)
        if fail and stage == 'transfer_place': raise DualArmError('blocked')
    monkeypatch.setattr(collect, 'execute_program', execute)
    monkeypatch.setattr(collect, 'manual_stage', lambda *a: stages.append('manual'))
    args = SimpleNamespace(approach_rpy_deg=[180, 0, 0], approach_height_mm=50,
                           clearance_mm=30, place_offset_mm=5)
    folder = tmp_path/'traj'
    if fail:
        with pytest.raises(DualArmError): collect.run_iteration(runtime, None, args, folder)
    else:
        collect.run_iteration(runtime, None, args, folder)
    assert stages[:5] == ['capture', 'select_points', 'approach', 'manual', 'transfer_place']
    report = json.loads((folder/'traj.json').read_text())
    assert report['arm'] == 'left' and report['camera'] == 'A'
    assert report['status'] == ('FAILED' if fail else 'PENDING_CONFIRMATION')
    assert (folder/'observation/rgb.png').exists()
    assert (folder/'points.json').exists()
    assert coordinator.directory == folder/'controller'
    assert 'new_iteration_observation' in (folder/'controller/events.jsonl').read_text()
    if fail:
        assert 'perception' not in stages and runtime.cancel.is_set()
    else:
        assert stages[-1] == 'perception'


def test_handoff_height_is_operator_selected_without_start_clearance():
    source = collect.transfer_program([0, 0, 110, 180, 0, 0], [0, 0, 100],
                                      [100, 0, 100], clearance=30, place_offset=5)
    calls = parse_code(source)
    assert calls[0][1]['left'][:3] == pytest.approx([.1, 0, .110])


def test_approach_solves_endpoint_in_base_and_uses_joint_path():
    transform = np.eye(4); transform[0, 3] = 100
    seed = np.arange(6.)
    captured = {}
    def inverse(arm, base, initial, phase):
        captured.update(arm=arm, base=base, seed=initial, phase=phase)
        return np.arange(6.) + 10
    runtime = SimpleNamespace(
        config=SimpleNamespace(arms={'left': SimpleNamespace(world_from_base=transform)}),
        coordinator=SimpleNamespace(snapshots=lambda: {'left': {'joints': seed}}),
        _inverse_target=inverse)
    calls = parse_code(collect.approach_program(runtime, [500, 200, 114, 180, 0, 0]))
    assert captured['arm'] == 'left'
    assert captured['base'][:3] == pytest.approx([400, 200, 114])
    assert captured['seed'] == pytest.approx(seed)
    assert [name for name, _ in calls] == ['gripper', 'move_joints']
    assert calls[1][1]['left'] == pytest.approx(np.radians(np.arange(6.) + 10))


@pytest.mark.parametrize('accepted', [True, False])
def test_confirmation_publishes_only_accepted_iteration(tmp_path, monkeypatch, accepted):
    import tkinter as tk
    from tkinter import messagebox
    folder = tmp_path/'pending'/'iter_0000'
    folder.mkdir(parents=True)
    collect.save_json(folder/'traj.json', dict(status='PENDING_CONFIRMATION', samples=10))
    (folder/'rgb.png').write_bytes(b'image')
    destination = tmp_path/'traj_0000'
    destroyed = []
    monkeypatch.setattr(tk, 'Tk', lambda: SimpleNamespace(withdraw=lambda: None,
        destroy=lambda: destroyed.append(True)))
    def confirm(*args, **kwargs):
        assert not destination.exists()
        assert kwargs['default'] == messagebox.NO
        return accepted
    monkeypatch.setattr(messagebox, 'askyesno', confirm)
    from cloth_agent.dual_arm.execution import DualArmCoordinator
    coordinator = object.__new__(DualArmCoordinator)
    coordinator.directory = folder/'controller'; coordinator.directory.mkdir()
    coordinator.log_lock = threading.Lock(); coordinator.sequence = 0
    coordinator.event('before_confirmation')
    assert collect.confirm_save_iteration(folder, destination, coordinator) is accepted
    coordinator.event('after_confirmation')
    assert destination.exists() is accepted
    saved = destination if accepted else folder
    manifest = json.loads((saved/'traj.json').read_text())
    assert manifest['status'] == ('COMPLETE' if accepted else 'REJECTED')
    assert manifest['operator_accepted'] is accepted
    assert (saved/'rgb.png').read_bytes() == b'image'
    assert destroyed == [True]
    assert coordinator.directory == saved/'controller'
    assert 'after_confirmation' in (saved/'controller/events.jsonl').read_text()


def test_place_release_stays_partial_through_retreat():
    moves = [('move', object()), ('move', object())]
    opening = ('gripper', dict(arm='left', action='open', joints={'left':[0]*6}))
    closing = ('gripper', dict(arm='left', action='close', joints={'left':[0]*6}))
    operations = moves + [opening, ('move', object())]
    result = collect.staged_place_release(operations)
    assert result[2][1]['action'] == 'release_partial'
    assert result[2][1]['opening_delta_pulse'] == 180
    assert result[2][1]['settle_s'] == .5
    assert result[3][0] == 'move'
    assert len(result) == 4
    assert opening[1]['action'] == 'open'
    with pytest.raises(DualArmError):
        collect.staged_place_release(moves + [opening, closing])


def test_recorder_timestamps_after_acquiring_feedback_lock(monkeypatch):
    recorder = object.__new__(collect.TrajectoryRecorder)
    recorder.runtime = SimpleNamespace(config=SimpleNamespace(raw={
        'safety': {'max_feedback_age_s': .1}}))
    clock = [10.]
    connection = SimpleNamespace(_latest_report=None)
    class CallbackDuringLockAcquisition:
        def __enter__(self):
            clock[0] = 10.002
            connection._latest_report = ([0.]*6, [0.]*6, 10.001)
        def __exit__(self, *args): pass
    connection._report_lock = CallbackDuringLockAcquisition()
    monkeypatch.setattr(collect.time, 'monotonic', lambda: clock[0])
    q, pose, sampled, observed = recorder._read_feedback(connection)
    assert observed - sampled == pytest.approx(.001)


@pytest.mark.parametrize('sampled', [9.8, 10.1, float('nan')])
def test_recorder_rejects_bad_feedback_age_with_diagnostics(monkeypatch, sampled):
    recorder = object.__new__(collect.TrajectoryRecorder)
    recorder.runtime = SimpleNamespace(config=SimpleNamespace(raw={
        'safety': {'max_feedback_age_s': .1}}))
    connection = SimpleNamespace(_report_lock=threading.Lock(),
        _latest_report=([0.]*6,[0.]*6,sampled))
    monkeypatch.setattr(collect.time, 'monotonic', lambda: 10.)
    with pytest.raises(DualArmError, match='age_s=.*limit_s='):
        recorder._read_feedback(connection)
