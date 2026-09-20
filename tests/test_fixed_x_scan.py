import json
import threading
from unittest.mock import Mock

import pytest

from cloth_agent import fixed_x_scan as scan
from cloth_agent.config import RobotConfig
from cloth_agent.robot_api import ControllerTrajectoryValidation
from scripts import live_tcp_overlay as overlay


@pytest.fixture
def config():
    return RobotConfig.load(overlay.PROJECT_ROOT, overlay.PROJECT_ROOT/'config/robot.example.json')


class Arm:
    mode = 0

    def __init__(self):
        self.commands = []
        self.pose = [420., -140., 50., -178., 0., 0.]

    def get_state(self):
        return 0, 1 if self.commands else 0

    def get_err_warn_code(self):
        return 0, [0, 0]

    def get_is_moving(self):
        return bool(self.commands) and self.pose[1] != 220

    def get_servo_angle(self, **kwargs):
        return 0, [1., 2., 3., 4., 5., 6., 0.]

    def set_position(self, **kwargs):
        self.commands.append(kwargs)
        return 0

    def set_state(self, state):
        self.commands.append({'stop_state': state})
        return 0


def execute(monkeypatch, tmp_path, config, positions, cancel=False):
    arm = Arm()
    settings = scan.ScanSettings(-140, 220, 50, 10)
    planner = Mock(return_value=ControllerTrajectoryValidation({}, 0, (0, 0, 172, 0, 0, 0), 36))
    monkeypatch.setattr(scan, '_controller_trajectory_with_arm', planner)
    iterator = iter(positions)
    stop = threading.Event()

    def poll(*args):
        point = next(iterator)
        if point is None:
            return {'status': 'READ_ERROR', 'read_duration_s': 0, 'sample_monotonic': 1}
        arm.pose[:3] = point
        if cancel:
            stop.set()
        return {'status': 'OK', 'read_duration_s': .01, 'sample_monotonic': point[1]+200,
                'tcp_pose_mm_deg': list(arm.pose), 'projection': {'status': 'VISIBLE', 'raw_pixel_xy': [50, 50]}}

    report = scan.run_scan(arm, config, settings, None, None, stop, tmp_path, poll,
                           lambda *args: None, lambda *args: list(arm.pose))
    return arm, report, planner


def test_single_continuous_line_keeps_commanded_x_and_z(monkeypatch, tmp_path, config):
    arm, report, planner = execute(monkeypatch, tmp_path, config,
                                   [(420, -140, 50), (420, 0, 50), (420, 220, 50)])
    assert report['status'] == 'COMPLETE'
    assert len(arm.commands) == 1
    assert arm.commands[0]['x'] == 420
    assert arm.commands[0]['z'] == 50
    assert arm.commands[0]['y'] == 220
    assert arm.commands[0]['speed'] == 10
    assert arm.commands[0]['wait'] is False
    assert arm.commands[0]['radius'] == -1
    local = planner.call_args.args[1]
    assert local.init_joints_deg == (1, 2, 3, 4, 5, 6, 0)
    rows = [json.loads(line) for line in (tmp_path/'scan_samples.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert all(row['x_error_mm'] == 0 for row in rows)
    assert 'measured_y_speed_mm_s' in rows[-1]


@pytest.mark.parametrize('position', [(421, 0, 50), (420, 0, 51), (420, 222, 50), None])
def test_tracking_or_feedback_failure_requests_stop(monkeypatch, tmp_path, config, position):
    arm, report, _ = execute(monkeypatch, tmp_path, config, [position])
    assert report['status'] == 'FAILED'
    assert arm.commands[-1] == {'stop_state': 4}
    assert report['samples'] == 1


def test_cancel_requests_stop(monkeypatch, tmp_path, config):
    arm, report, _ = execute(monkeypatch, tmp_path, config, [(420, 0, 50)], cancel=True)
    assert report['status'] == 'FAILED'
    assert arm.commands[-1] == {'stop_state': 4}


def test_wrong_start_never_sends_motion(config):
    arm = Arm()
    arm.pose[0] = 430
    with pytest.raises(RuntimeError, match='Manually position'):
        scan.prepare(arm, config, scan.ScanSettings(-140, 220, 50, 10), lambda *args: arm.pose)
    assert arm.commands == []


def test_ik_failure_never_sends_motion(monkeypatch, config):
    arm = Arm()
    monkeypatch.setattr(scan, '_controller_trajectory_with_arm', Mock(side_effect=RuntimeError('IK reject')))
    with pytest.raises(RuntimeError, match='IK reject'):
        scan.prepare(arm, config, scan.ScanSettings(-140, 220, 50, 10), lambda *args: arm.pose)
    assert arm.commands == []


@pytest.mark.parametrize('args', [
    ['--capture-current', '--scan-y', '-140', '220', '--scan-z', '50', '--scan-speed', '10'],
    ['--real', '--confirm-real', '--scan-y', '-140', '220'],
])
def test_scan_requires_motion_flags_and_explicit_parameters(args):
    with pytest.raises(SystemExit) as error:
        overlay.main(args)
    assert error.value.code == 2
