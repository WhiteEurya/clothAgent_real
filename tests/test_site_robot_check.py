import copy

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
