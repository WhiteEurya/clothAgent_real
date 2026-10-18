import threading
from collections import deque
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.gamepad_continuous import (
    CertifiedBox, ContinuousArm7, HeldGripper, VELOCITY_DURATION,
    slew, velocity_ik, world_twist,
    certify_adaptive_box,
)
from cloth_agent.gamepad_teleop import InputGuard, Intent
from cloth_agent.dual_arm.geometry import DualArmError


def test_velocity_ik_world_mapping_and_speed_limit():
    model = SimpleNamespace(forward=lambda q: np.asarray(q[:6]))
    twist = world_twist(Intent(forward=1, right=1, up=1, yaw=-1, enabled=True), 10, 5)
    assert np.linalg.norm(twist[:3]) == pytest.approx(10)
    assert np.sign(twist).tolist() == [1, -1, 1, 0, 0, -1]
    qdot = velocity_ik(model, np.zeros(7), np.eye(4), twist, 5)
    assert np.max(np.abs(qdot)) <= 5
    assert np.sign(qdot[:6]).tolist() == [1, -1, 1, 0, 0, -1]
    assert not world_twist(Intent(forward=1), 10, 5).any()
    assert slew(np.zeros(7), np.ones(7)*5, .02, 10) == pytest.approx(np.ones(7)*.2)
    with pytest.raises(DualArmError):
        slew(np.zeros(7), np.ones(7), 1, 10)


def test_joint_box_contains_entire_lease_not_only_current_position():
    box = CertifiedBox(np.full(7, -2.), np.full(7, 2.), 10.)
    assert box.permits(np.zeros(7), 10., 10.01, 5, .1, .05)
    assert not box.permits(np.ones(7)*1.5, 10., 10.01, 5, .1, .05)
    assert not box.permits(np.zeros(7), 10., 10.11, 5, .1, .05)
    assert not box.permits(np.zeros(7), 10.6, 10.6, 5, .1, .05)


def test_unachievable_twist_is_rejected_instead_of_rotating_wrong_direction():
    model = SimpleNamespace(forward=lambda q: np.zeros(6))
    with pytest.raises(DualArmError, match='singularity'):
        velocity_ik(model, np.zeros(7), np.eye(4), np.array([10, 0, 0, 0, 0, 0]), 5)


def test_gripper_held_short_targets_and_hold_on_release():
    grip = HeldGripper(5, 850, 100)
    assert grip.target(400, None) is None
    assert grip.target(400, 'open') == 412
    assert grip.target(405, 'open') == 417
    assert grip.target(410, None) == 410
    assert grip.target(410, None) is None
    assert grip.target(410, 'close') == 398
    assert grip.target(6, 'close') == 5
    assert grip.target(845, 'open') == 850
    with pytest.raises(DualArmError):
        grip.target(float('nan'), 'open')


def test_display_reports_bounds_block_instead_of_input_running():
    obj = object.__new__(ContinuousArm7)
    obj.failure = None
    obj.status = '等待可验证的运动范围：right: target outside measured workspace'
    obj.bounds_available = False
    assert 'outside measured workspace' in obj.control_status('连续控制中')
    obj.status = None
    assert '运动暂停' in obj.control_status('连续控制中')
    obj.bounds_available = True
    assert obj.control_status('连续控制中') == '连续控制中'
    obj.failure = 'feedback expired'
    assert 'feedback expired' in obj.control_status('连续控制中')


def controller():
    obj = object.__new__(ContinuousArm7)
    obj.owns_runtime = True
    obj.closed = threading.Event()
    obj.guard = InputGuard()
    obj.guard.update(Intent(forward=1, enabled=True))
    obj.runtime = SimpleNamespace(cancel=threading.Event(),
        config=SimpleNamespace(arms={'right': SimpleNamespace(world_from_base=np.eye(4),
                gripper=SimpleNamespace(gripper_speed=500))}),
        coordinator=SimpleNamespace(event=lambda *a, **kw: None))
    obj.mode_entered = True
    obj.fast = SimpleNamespace(mode=4)
    obj.model = SimpleNamespace(forward=lambda q: np.asarray(q[:6]))
    obj._feedback = lambda: (np.zeros(7), np.zeros(6), time.monotonic())
    obj.zone_lock = threading.Lock()
    obj.velocity_history = deque()
    obj.zone = CertifiedBox(np.full(7, -2.), np.full(7, 2.), time.monotonic())
    obj.speed_mm_s, obj.yaw_deg_s = 10., 5.
    obj.max_speed, obj.max_age, obj.latency, obj.acceleration = 5., .1, .05, 10.
    obj._enter_mode = lambda: None
    obj.gripper = HeldGripper(5, 850, 100)
    obj.velocity_active = False
    return obj


def test_smaller_box_keeps_pipeline_workspace_and_passes_recorded_stop_pose():
    from pathlib import Path
    from cloth_agent.gamepad_arm7 import JogRuntime
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.dual_arm.sequential_home import certify_segment
    config_path = Path(__file__).resolve().parents[1] / 'config/dual_arm.local.json'
    if not config_path.exists():
        pytest.skip('requires the local measured robot configuration')
    runtime = JogRuntime(config_path)
    pipeline = object.__new__(Arm7Runtime).load_config(config_path)
    for key in ('min_mm', 'max_mm'):
        np.testing.assert_array_equal(runtime.config.arms['right'].workspace[key],
                                      pipeline.arms['right'].workspace[key])
    q = np.array([-6.13592, 16.134205, -.190451, 83.669944, -2.503654, 67.065283, 9.645687])
    static = np.array(runtime.config.static_arm_obstacles['left']['joint_angles_deg'])
    with pytest.raises(DualArmError, match='workspace'):
        certify_segment({'left': static, 'right': q-2}, {'left': static, 'right': q+2},
                        runtime.config, runtime.models)
    box = certify_adaptive_box(q, static, runtime.config, runtime.models, 5, .1, .05, certify_segment)
    assert np.max(box.upper-q) == pytest.approx(1)
    assert box.speed_limit < 5
    assert box.permits(q, box.checked_at, box.checked_at+.05, box.speed_limit, .1, .05)


def test_smaller_box_does_not_ignore_unexpired_faster_velocity():
    obj = controller()
    box = CertifiedBox(np.full(7, -.5), np.full(7, .5), time.monotonic(), 1.)
    obj.velocity_history.append((time.monotonic(), 5.))
    speed = obj._lease_speed(box)
    assert speed == 5
    assert not box.permits(np.zeros(7), time.monotonic(), time.monotonic(), speed, .1, .05)
    obj.velocity_history[0] = (time.monotonic()-1, 5.)
    assert obj._lease_speed(box) == 1


def test_adaptive_box_never_accepts_a_rejected_region():
    def reject(*args):
        raise DualArmError('outside workspace')
    with pytest.raises(DualArmError, match='outside workspace'):
        certify_adaptive_box(np.zeros(7), np.zeros(6), None, None, 5, .1, .05, reject)


def test_velocity_stream_has_no_completion_wait_and_zero_on_lb_release():
    obj = controller()
    sent = []
    def send(v, **kw):
        assert kw['duration'] == VELOCITY_DURATION > 0
        sent.append(v)
        if len(sent) == 2:
            obj.guard.update(Intent())
        if len(sent) == 4:
            obj.closed.set()
        return 0
    obj.arm = SimpleNamespace(vc_set_joint_velocity=send)
    obj._control_loop()
    assert any(sent[0]) and any(sent[1])
    assert sent[2] == sent[3] == [0.]*7


@pytest.mark.parametrize('condition', ['no_zone', 'expired_input', 'reversal'])
def test_unvalidated_stale_or_changed_input_never_moves(condition):
    obj = controller()
    if condition == 'no_zone':
        obj.zone = None
    elif condition == 'expired_input':
        obj.guard.updated = time.monotonic() - 1
    else:
        def feedback():
            obj.guard.update(Intent(forward=-1, enabled=True))
            return np.zeros(7), np.zeros(6), time.monotonic()
        obj._feedback = feedback
    sent = []
    def send(v, **kw):
        sent.append(v); obj.closed.set(); return 0
    obj.arm = SimpleNamespace(vc_set_joint_velocity=send)
    if condition == 'expired_input':
        with pytest.raises(DualArmError):
            obj._control_loop()
        assert sent == []
    else:
        obj._control_loop()
        assert sent == [[0.]*7]


def test_gripper_does_not_wait_for_robot_or_full_travel():
    obj = controller()
    obj.guard.update(Intent(gripper='open', enabled=True))
    calls = []
    def set_position(value, **kw):
        assert kw['wait'] is False and kw['wait_motion'] is False
        calls.append(value)
        if len(calls) == 1:
            obj.guard.update(Intent(enabled=True))
        else:
            obj.closed.set()
        return 0
    obj.gripper_api = SimpleNamespace(set_gripper_mode=lambda v:0,
        set_gripper_enable=lambda v:0, set_gripper_speed=lambda v:0,
        get_gripper_position=lambda:(0, 400), set_gripper_position=set_position)
    obj._gripper_loop()
    assert calls == [412, 400]


def test_slow_gripper_does_not_block_velocity_worker():
    obj = controller()
    obj.guard.update(Intent(forward=1, gripper='open', enabled=True))
    blocked = threading.Event()
    sent = []
    def set_position(value, **kw):
        blocked.set(); time.sleep(.12); return 0
    obj.gripper_api = SimpleNamespace(set_gripper_mode=lambda v:0,
        set_gripper_enable=lambda v:0, set_gripper_speed=lambda v:0,
        get_gripper_position=lambda:(0, 400), set_gripper_position=set_position)
    def send(v, **kw):
        sent.append(v)
        if len(sent) >= 4:obj.closed.set()
        return 0
    obj.arm = SimpleNamespace(vc_set_joint_velocity=send)
    grip_thread = threading.Thread(target=obj._gripper_loop)
    grip_thread.start()
    try:
        assert blocked.wait(1)
        obj._control_loop()
    finally:
        obj.closed.set(); grip_thread.join(1)
    assert len(sent) == 4 and all(any(v) for v in sent)


def test_cancel_during_mode_switch_never_reenables():
    obj = controller()
    obj.mode_entered = False
    obj.activation_lock = threading.Lock()
    obj.transitioning = threading.Event()
    writes = []
    def mode(value):
        writes.append(('mode', value))
        obj.runtime.cancel.set()
        return 0
    obj.arm = SimpleNamespace(get_state=lambda:(0, 2), get_err_warn_code=lambda:(0, [0, 0]),
        get_is_moving=lambda:False, get_cmdnum=lambda:(0, 0), set_mode=mode,
        set_state=lambda value:writes.append(('state', value)) or 0)
    # Remove the per-test stub to exercise the actual mode switch.
    del obj._enter_mode
    with pytest.raises(DualArmError, match='cancelled'):
        obj._enter_mode()
    assert writes == [('mode', 4)]
    assert not obj.transitioning.is_set()


def test_feedback_checks_normal_report_errors_and_fast_report_age():
    obj = controller()
    obj.transitioning = threading.Event()
    obj.arm = SimpleNamespace(connected=True, error_code=11, warn_code=0)
    obj.fast = SimpleNamespace(connected=True, state=2)
    obj.connection = SimpleNamespace(_report_lock=threading.Lock(),
        _latest_report=([0.]*7, [0.]*6, time.monotonic()))
    del obj._feedback
    with pytest.raises(DualArmError, match='fault'):
        obj._feedback()
    obj.arm.error_code = 0
    assert len(obj._feedback()[0]) == 7
    obj.connection._latest_report = ([0.]*7, [0.]*6, time.monotonic()-1)
    with pytest.raises(DualArmError, match='stale'):
        obj._feedback()


def test_cleanup_closes_connections_even_if_final_gripper_hold_fails():
    obj = controller()
    obj.mode_entered = False
    obj.workers = []
    obj.watchdog = None
    calls = []
    obj.runtime.close = lambda:calls.append('robot disconnected')
    obj.gripper.active = True
    obj.gripper_api = SimpleNamespace(get_gripper_position=lambda:(0, 400),
        set_gripper_position=lambda *a, **kw:1,
        disconnect=lambda:calls.append('gripper disconnected'))
    obj.close()
    assert calls == ['gripper disconnected', 'robot disconnected']


def test_gripper_never_moves_without_certified_bounds():
    obj = controller()
    obj.zone = None
    obj.guard.update(Intent(gripper='close', enabled=True))
    calls = []
    obj.gripper_api = SimpleNamespace(set_gripper_position=lambda *a, **kw:calls.append(a))
    thread = threading.Thread(target=obj._gripper_loop)
    thread.start()
    try:
        time.sleep(.04)
    finally:
        obj.closed.set();thread.join(1)
    assert calls == []


def test_arm6_box_moves_six_left_joints_and_keeps_seven_right_joints():
    visited = []
    def certify(low, high, config, models):
        visited.append((low, high))
    box = certify_adaptive_box(np.zeros(6), np.arange(7.), None, None,
                              5, .1, .05, certify, active_arm='left')
    low, high = visited[0]
    assert low['left'].shape == (6,) and high['left'].shape == (6,)
    assert low['right'] == pytest.approx(np.arange(7.))
    assert high['right'] == pytest.approx(np.arange(7.))
    assert box.lower.shape == (6,)


def test_arm6_velocity_loop_dispatches_six_axis_commands():
    from cloth_agent.gamepad_continuous import ContinuousArm6
    obj = controller()
    obj.__class__ = ContinuousArm6
    obj.runtime.config.arms['left'] = obj.runtime.config.arms.pop('right')
    obj._feedback = lambda: (np.zeros(6), np.zeros(6), time.monotonic())
    obj.zone = CertifiedBox(np.full(6, -2.), np.full(6, 2.), time.monotonic())
    seen = []
    def send(velocity, **kwargs):
        seen.append(velocity); obj.closed.set(); return 0
    obj.arm = SimpleNamespace(vc_set_joint_velocity=send)
    obj._control_loop()
    assert len(seen) == 1 and len(seen[0]) == 6
    assert seen[0][0] > 0


def test_live_inactive_arm_drift_or_stale_feedback_blocks_arm6():
    from cloth_agent.gamepad_continuous import ContinuousArm6
    obj = object.__new__(ContinuousArm6)
    obj.max_age = .1; obj.static = np.zeros(7)
    obj.runtime = SimpleNamespace(config=SimpleNamespace(limits={'stationary_tolerance_deg': .1}))
    other = SimpleNamespace(arm=SimpleNamespace(connected=True, error_code=0, warn_code=0),
        _report_arm=SimpleNamespace(connected=True, state=0), _report_lock=threading.Lock(),
        _latest_report=(np.zeros(7), None, time.monotonic()))
    obj.other_connection = other
    obj._check_other_arm()
    other._latest_report = (np.ones(7), None, time.monotonic())
    with pytest.raises(DualArmError, match='inactive arm moved'):
        obj._check_other_arm()
    other._latest_report = (np.zeros(7), None, time.monotonic()-1)
    with pytest.raises(DualArmError, match='expired'):
        obj._check_other_arm()


def test_collection_manual_control_uses_joint_limits_without_collision_proof():
    model = SimpleNamespace(lower=np.full(6, -1.), upper=np.full(6, 1.))
    def forbidden(*a): pytest.fail('collision proof must not run')
    zone = certify_adaptive_box(np.zeros(6), np.zeros(7),
        SimpleNamespace(path_validation_enabled=False), {'left': model}, 5, .1, .05,
        forbidden, active_arm='left')
    now = time.monotonic()
    assert zone.permits(np.zeros(6), now, now, 5, .1, .05)
    assert not zone.permits(np.degrees(model.upper), now, now, 5, .1, .05)


def test_standalone_arm6_connects_only_six_axis_controller(monkeypatch, tmp_path):
    import json
    from xarm import wrapper
    from cloth_agent.dual_arm import cli
    from cloth_agent.gamepad_continuous import ContinuousArm6
    opened = []; writes = []
    class Connection:
        simulated = False
        def __init__(self, config, cancel, **kwargs):
            opened.append((config.arm_id, config.axis, config.ip))
            self.arm = SimpleNamespace(version_number=(2,5,1), connected=True)
            self._report_arm = self.arm
        def prepare_position(self): writes.append('prepare')
        def disconnect(self): pass
    class Gripper:
        def __init__(self, ip, **kwargs): opened.append(('gripper', ip))
        def disconnect(self): pass
    monkeypatch.setattr('cloth_agent.gamepad_arm7.fcntl.flock', lambda *args: None)
    monkeypatch.setattr(cli, 'XArmConnection', Connection)
    monkeypatch.setattr(wrapper, 'XArmAPI', Gripper)
    backend = ContinuousArm6('config/dual_arm.local.json', tmp_path/'manual', InputGuard(),
                             start_workers=False)
    try:
        assert opened == [('left', 6, '192.168.2.232'), ('gripper', '192.168.2.232')]
        assert writes == []  # --connect-only stays read-only.
        assert backend.axis == 6 and backend.model.config.axis == 6
        assert backend.other_connection is None
        report = json.loads((tmp_path/'manual/continuous_config.json').read_text())
        assert report['arm'] == 'left' and report['axis'] == 6
        assert report['robot_ip'] == '192.168.2.232'
    finally:
        backend.close()


def test_arm6_smooth_gripper_does_not_restart_held_motion():
    from cloth_agent.gamepad_continuous import SmoothHeldGripper
    grip = SmoothHeldGripper(5, 850, 500)
    assert grip.target(800, 'close') == 5
    for position in (795, 770, 700, 676, 676, 676):
        assert grip.target(position, 'close') is None
    assert grip.target(676, None) == 676
    assert grip.target(676, None) is None
    assert grip.target(676, 'close') == 5
    assert grip.target(670, 'open') == 850
    assert grip.target(680, 'open') is None
    assert grip.target(690, None) == 690


@pytest.mark.parametrize('speed', [300, 2000, 5000])
def test_arm6_smooth_loop_uses_sdk_speed_and_holds_on_release(speed):
    from cloth_agent.gamepad_continuous import SmoothHeldGripper
    obj = controller()
    obj.gripper = SmoothHeldGripper(5, 850, speed)
    obj.guard.update(Intent(gripper='close', enabled=True))
    targets = []; speeds = []
    def send(value, **kwargs):
        assert kwargs['wait'] is False and kwargs['wait_motion'] is False
        targets.append(value)
        if len(targets) == 1:
            obj.guard.update(Intent())  # LB released.
        else:
            obj.closed.set()
        return 0
    obj.gripper_api = SimpleNamespace(set_gripper_mode=lambda _:0,
        set_gripper_enable=lambda _:0,set_gripper_speed=lambda v:speeds.append(v) or 0,
        get_gripper_position=lambda:(0,650),set_gripper_position=send)
    obj._gripper_loop()
    assert targets == [5,650]
    assert speeds == [speed]


def test_smooth_gripper_fault_requests_independent_hold():
    from cloth_agent.gamepad_continuous import SmoothHeldGripper
    obj = controller(); obj.failure = None; obj.failure_lock = threading.Lock()
    obj.mode_entered = False
    obj.gripper = SmoothHeldGripper(5,850,500)
    obj.gripper.target(600,'close')
    targets = []
    obj.fast = SimpleNamespace(get_gripper_position=lambda:(0,570),
        set_gripper_position=lambda value, **kw:targets.append(value) or 0)
    obj._fail('input heartbeat expired')
    assert targets == [570]
    assert obj.runtime.cancel.is_set()


def test_arm6_idle_switches_to_position_hold_and_checks_drift():
    from cloth_agent.gamepad_continuous import ContinuousArm6
    obj = controller(); obj.__class__ = ContinuousArm6
    obj.activation_lock = threading.Lock(); obj.transitioning = threading.Event()
    obj.fast = SimpleNamespace(mode=4, state=0)
    obj._feedback = lambda: (np.zeros(6), np.zeros(6), time.monotonic())
    calls = []
    def mode(value):
        calls.append(('mode', value)); obj.fast.mode = value; obj.arm.mode = value
        return 0
    obj.arm = SimpleNamespace(
        vc_set_joint_velocity=lambda values, **kw: calls.append(('velocity', values)) or 0,
        get_err_warn_code=lambda: (0, [0,0]), get_state=lambda: (0,0),
        set_mode=mode, set_state=lambda value: calls.append(('state', value)) or 0)
    obj._hold_position_mode()
    assert calls == [('velocity', [0.]*6), ('mode', 0), ('state', 0)]
    assert not obj.mode_entered and not obj.transitioning.is_set()
    obj._check_idle_hold(np.zeros(6), np.zeros(6))
    pose = np.zeros(6); pose[2] = -1.1
    with pytest.raises(DualArmError, match='drift'):
        obj._check_idle_hold(np.zeros(6), pose)
    with pytest.raises(DualArmError, match='drift'):
        obj._check_idle_hold(np.ones(6)*.11, np.zeros(6))


@pytest.mark.parametrize('intent', [Intent(), Intent(enabled=True, gripper='close')])
def test_arm6_no_motion_input_enters_hold_even_while_closing_gripper(intent):
    from cloth_agent.gamepad_continuous import ContinuousArm6
    obj = controller(); obj.__class__ = ContinuousArm6
    obj._feedback = lambda: (np.zeros(6), np.zeros(6), time.monotonic())
    obj.zone = CertifiedBox(np.full(6,-2.), np.full(6,2.), time.monotonic())
    obj.guard.update(intent)
    calls = []
    obj.arm = SimpleNamespace(vc_set_joint_velocity=lambda values, **kw:
        calls.append(values) or 0)
    def hold():
        calls.append('hold'); obj.closed.set()
    obj._hold_position_mode = hold
    obj._control_loop()
    assert calls == [[0.]*6, 'hold']


def test_drift_failure_stops_arm_after_leaving_velocity_mode():
    obj = controller(); obj.mode_entered = False; obj.motion_started = True
    obj.failure = None; obj.failure_lock = threading.Lock()
    obj.activation_lock = threading.Lock()
    calls = []
    obj.fast = SimpleNamespace(set_state=lambda value: calls.append(value) or 0)
    obj._fail('idle position drift')
    assert calls == [4]
    assert obj.runtime.cancel.is_set()


def test_velocity_waits_for_command_connection_mode_before_dispatch(monkeypatch):
    obj = controller(); obj.mode_entered = False
    obj.activation_lock = threading.Lock(); obj.transitioning = threading.Event()
    obj.fast = SimpleNamespace(mode=4, state=0)
    calls = []
    obj.arm = SimpleNamespace(mode=0, get_state=lambda:(0,0),
        get_err_warn_code=lambda:(0,[0,0]), get_is_moving=lambda:False,
        get_cmdnum=lambda:(0,0), set_mode=lambda _:0, set_state=lambda _:0)
    def velocity(*args, **kwargs):
        assert obj.arm.mode == obj.fast.mode == 4
        calls.append('velocity'); return 0
    obj.arm.vc_set_joint_velocity = velocity
    def report_update(_):
        assert not calls
        calls.append('report')
        obj.arm.mode = 4
    monkeypatch.setattr('cloth_agent.gamepad_continuous.time.sleep', report_update)
    del obj._enter_mode
    obj._enter_mode()
    assert calls == ['report', 'velocity']
    assert not obj.transitioning.is_set()


@pytest.mark.parametrize('case', ['mode_latency', 'changing_stick', 'temporary_zero'])
def test_arm6_active_motion_does_not_cycle_position_mode(case):
    from cloth_agent.gamepad_continuous import ContinuousArm6
    obj = controller(); obj.__class__ = ContinuousArm6
    obj.runtime.config.arms['left'] = obj.runtime.config.arms.pop('right')
    obj.zone = CertifiedBox(np.full(6,-2.), np.full(6,2.), time.monotonic())
    count = 0
    def feedback():
        nonlocal count
        count += 1
        if case == 'changing_stick':
            obj.guard.update(Intent(forward=.5, enabled=True))
        return np.zeros(6), np.zeros(6), time.monotonic()
    obj._feedback = feedback
    if case == 'mode_latency':
        obj.mode_entered = False
        def enter():
            time.sleep(.11)
            obj.guard.update(Intent(forward=1, enabled=True))
            obj.mode_entered = True
        obj._enter_mode = enter
    if case == 'temporary_zero':
        obj.zone = None
    obj._hold_position_mode = lambda: pytest.fail('active input must not switch to position mode')
    sent = []
    def send(values, **kwargs):
        sent.append(values); obj.closed.set(); return 0
    obj.arm = SimpleNamespace(vc_set_joint_velocity=send)
    obj._control_loop()
    if case == 'temporary_zero':
        assert sent == [[0.]*6]
    else:
        assert sent[0][0] > 0
    if case == 'mode_latency':
        assert count == 2


def test_arm6_downward_gain_preserves_up_and_caps_combined_speed():
    from cloth_agent.gamepad_continuous import ContinuousArm6
    gain = ContinuousArm6.downward_gain
    down = world_twist(Intent(up=-1,enabled=True),10,5,gain)
    up = world_twist(Intent(up=1,enabled=True),10,5,gain)
    assert down[2] == -15
    assert up[2] == 10
    combined = world_twist(Intent(up=-1,forward=1,enabled=True),15,5,gain)
    assert np.linalg.norm(combined[:3]) <= 15 + 1e-9
    assert world_twist(Intent(up=-1,enabled=False),10,5,gain).tolist() == [0.]*6
