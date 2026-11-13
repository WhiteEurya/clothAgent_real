import copy
import pytest

from cloth_agent.site_observation import load_site
from cloth_agent.site_robot_check import read_arm


class ReadOnlyRobot:
    """No setters or motion API: any accidental dispatch fails the test."""
    def __init__(self, config):
        self.axis = 7
        self.control_box_sn = 'fixture-box'
        self.sn = 'fixture-arm'
        self.mode = 0
        self.tcp_offset = list(config['tcp_offset_mm_deg'])
        self.world_offset = list(config['world_offset_mm_deg'])
        self.tcp_load = [.82, [0, 0, 48]]
        self.motor_enable_states = [1] * 7
        self.motor_brake_states = [1] * 7
        self.q = list(config['home']['reference_joints_deg'])
        self.pose = list(config['home']['pose_base_mm_deg'])
        self.closed = False
    def register_report_callback(self, cb): self.callback = cb
    def connect(self):
        self.connected = True
        for _ in range(2): self.callback(dict(joints=self.q, cartesian=self.pose))
    def disconnect(self): self.closed = True
    def get_robot_sn(self): return 0, self.sn
    def get_servo_angle(self, **kwargs): return 0, self.q
    def get_position(self, **kwargs): return 0, self.pose
    def get_state(self): return 0, 2
    def get_err_warn_code(self): return 0, [0, 0]
    def get_gripper_position(self): return 0, 850
    def get_gripper_err_code(self): return 0, 0


def test_read_only_snapshot_and_disconnect():
    profile = load_site()
    robot = ReadOnlyRobot(profile['arms']['right'])
    result = read_arm(profile, 'right', factory=lambda *args, **kwargs: robot)
    assert result['status'] == 'READ_ONLY_CHECKS_PASS'
    assert result['execution_allowed'] is False and result['motion_commands_sent'] == 0
    assert robot.closed


def test_tcp_mismatch_is_not_corrected():
    profile = load_site()
    robot = ReadOnlyRobot(profile['arms']['right']); robot.tcp_offset[2] += 10
    result = read_arm(profile, 'right', factory=lambda *args, **kwargs: robot)
    assert result['status'] == 'NEEDS_ATTENTION'
    assert result['checks']['tcp_offset'] is False
    assert robot.tcp_offset[2] == 182 and robot.closed


def test_fault_and_disconnect():
    profile = load_site()
    robot = ReadOnlyRobot(profile['arms']['right'])
    robot.get_state = lambda: (7, None)
    result = read_arm(profile, 'right', factory=lambda *args, **kwargs: robot)
    assert result['status'] == 'READ_FAILED' and robot.closed


@pytest.mark.parametrize('blocked', [None, 'fault', 'queued', 'moving', 'identity', 'mode', 'disabled'])
def test_explicit_startup_stop_recovery(blocked):
    profile = load_site()
    robot = ReadOnlyRobot(profile['arms']['right'])
    identity = profile['arms']['right']['identity_verification']
    robot.control_box_sn, robot.sn = identity['control_box_sn'], identity['robot_sn']
    state = [4]
    writes = []
    robot.get_state = lambda: (0, state[0])
    robot.get_cmdnum = lambda: (0, int(blocked == 'queued'))
    robot.get_is_moving = lambda: blocked == 'moving'
    robot.get_err_warn_code = lambda: (0, [int(blocked == 'fault'), 0])
    robot.motion_enable = lambda **kw: writes.append(('enable', True)) or 0
    def normal(value):
        writes.append(('state', value)); state[0] = 2
        return 0
    robot.set_state = normal
    if blocked == 'identity': robot.sn = 'wrong-device'
    if blocked == 'mode': robot.mode = 2
    if blocked == 'disabled': robot.motor_enable_states[0] = 0
    factory = lambda *a, **kw: robot
    # Ordinary diagnostics must remain read-only, even in STOP.
    assert read_arm(profile, 'right', factory=factory)['status'] == 'NEEDS_ATTENTION'
    assert writes == []
    result = read_arm(profile, 'right', factory=factory, recover_stopped=True)
    if blocked:
        assert result['status'] != 'READ_ONLY_CHECKS_PASS'
        assert writes == []
    else:
        assert result['status'] == 'READ_ONLY_CHECKS_PASS'
        assert result['stop_recovered'] and result['state'] == 2
        assert result['configuration_commands_sent'] == 2
        assert result['motion_commands_sent'] == 0
        assert writes == [('enable', True), ('state', 0)]
    assert robot.closed


@pytest.mark.parametrize('failure', [None,'mode','fault_after_mode','queue_after_mode','cancel'])
def test_stopped_servo_restores_position_without_motion(failure):
    import threading
    from cloth_agent.site_robot_check import recover_stopped_controller
    robot=ReadOnlyRobot(load_site()['arms']['right']);robot.mode=1
    state=[4];writes=[];fault=[0];queue=[0];cancel=threading.Event()
    robot.get_state=lambda:(0,state[0])
    robot.get_err_warn_code=lambda:(0,[fault[0],0])
    robot.get_cmdnum=lambda:(0,queue[0])
    robot.get_is_moving=lambda:False
    def mode(n):
        writes.append(('mode',n));robot.mode=n;state[0]=5
        if failure=='fault_after_mode':fault[0]=1
        if failure=='queue_after_mode':queue[0]=1
        if failure=='cancel':cancel.set()
        return 1 if failure=='mode' else 0
    robot.set_mode=mode
    robot.motion_enable=lambda **k:writes.append(('enable',True)) or 0
    def ready(n):writes.append(('state',n));state[0]=2;return 0
    robot.set_state=ready
    if failure:
        with pytest.raises(RuntimeError):recover_stopped_controller(robot,cancel)
        assert writes==[('mode',0)]
    else:
        assert recover_stopped_controller(robot,cancel)==3
        assert writes==[('mode',0),('enable',True),('state',0)]
        assert robot.mode==0 and state[0]==2
