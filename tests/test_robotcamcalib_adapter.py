import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cloth_agent.experiments.robotcamcalib_adapter import (
    adapt_proposal, check_cartesian_interface, main,
)
from cloth_agent.site_observation import load_site
from .test_prepare_partitioned_trial import CheckOnlyRobot


PROPOSALS = Path('deployment-artifacts/topdown_physical_trial_001/proposals_v2')
STATE = Path('outputs/onsite_acceptance_20261104_01/robot_readonly_host.json')


@pytest.fixture
def request_data():
    return adapt_proposal(PROPOSALS, json.loads(STATE.read_text()), load_site())


class RecordedStartRobot(CheckOnlyRobot):
    def __init__(self, request):
        super().__init__()
        self.start = copy.deepcopy(request['start'])
        self.axis = 7
        self.mode = 0
        self.tcp_offset = self.start['tcp_offset_mm_deg']
        self.world_offset = self.start['world_offset_mm_deg']

    def get_state(self): return 0, self.start['state']
    def get_err_warn_code(self): return 0, self.start['errors_warnings']
    def get_gripper_position(self): return 0, self.start['gripper_position_response'][1]
    def get_gripper_err_code(self): return 0, self.start['gripper_error_response'][1]
    def get_position(self, **kwargs): return 0, self.start['pose_base_mm_deg']
    def get_servo_angle(self, **kwargs): return 0, self.start['joints_deg']


def test_real_proposal_to_cartesian_sdk_check_interface(request_data, tmp_path):
    robot = RecordedStartRobot(request_data)
    report = check_cartesian_interface(robot, request_data, {})
    count = sum(op['kind'] == 'move_tcp' for op in request_data['operations'])
    assert robot.calls == [1] + [3] * (count - 2) + [2]
    assert report['execution_allowed'] is False
    assert report['external_collision_checked'] is False
    assert report['gripper_events_checked'] is False
    assert len(request_data['gripper_events']) == 3
    assert request_data['execution_blockers']
    original = json.loads((PROPOSALS / 'primary_plan.json').read_text())
    for phase in ('grasp', 'placed'):
        expected = next(op for op in original['operations'] if op['phase'] == phase)
        actual = next(op for op in request_data['operations'] if op['phase'] == phase)
        np.testing.assert_allclose(actual['pose_base_mm_deg'], expected['pose_base_mm_deg'])
    report['test_transport'] = 'fake SDK; no hardware connection or controller feasibility claim'
    (tmp_path / 'cartesian_interface_test.json').write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize('change', ['fault', 'state', 'axis', 'mode', 'tcp', 'closed', 'start'])
def test_original_start_checks_reject_before_path_queries(request_data, change):
    robot = RecordedStartRobot(request_data)
    if change == 'fault': robot.start['errors_warnings'] = [1, 0]
    if change == 'state': robot.start['state'] = 4
    if change == 'axis': robot.axis = 6
    if change == 'mode': robot.mode = 1
    if change == 'tcp': robot.tcp_offset = [0] * 6
    if change == 'closed': robot.start['gripper_position_response'] = [0, 0]
    if change == 'start': robot.start['joints_deg'][0] += 2
    with pytest.raises(RuntimeError):
        check_cartesian_interface(robot, request_data, {})
    assert robot.calls == []


@pytest.mark.parametrize('failure', ['tcp', 'ik', 'joint', 'detail', 'code'])
def test_controller_failures_remain_blocking(request_data, failure):
    robot = RecordedStartRobot(request_data)
    if failure == 'tcp': robot.is_tcp_limit = lambda *args, **kwargs: (0, True)
    if failure == 'ik': robot.get_inverse_kinematics = lambda *args, **kwargs: (-6, [])
    if failure == 'joint': robot.is_joint_limit = lambda *args, **kwargs: (0, True)
    if failure == 'detail': robot.only_check_result = 24
    if failure == 'code': robot.set_position = lambda *args, **kwargs: -1
    report = {}
    with pytest.raises(RuntimeError):
        check_cartesian_interface(robot, request_data, report)
    assert report['status'] == 'BLOCKED'
    assert report['execution_allowed'] is False


def test_legacy_semantic_conflicts_are_not_silently_rewritten(request_data):
    assert request_data['legacy_speed_conflicts'] == ['lifted', 'across', 'retreat']
    assert request_data['status'].endswith('EXECUTION_INCOMPATIBLE')
    assert {op['action'] for op in request_data['gripper_events']} == {'open', 'close'}


def test_reject_wrong_saved_identity():
    state = json.loads(STATE.read_text())
    state['arms']['right']['control_box_sn'] = 'different'
    with pytest.raises(ValueError, match='identity mismatch'):
        adapt_proposal(PROPOSALS, state, load_site())


def test_replay_actual_cartesian_tcp_rejection(request_data):
    history = json.loads(Path(
        'deployment-artifacts/topdown_physical_trial_001/tcp_gate_diagnosis_live.json'
    ).read_text())
    robot = RecordedStartRobot(request_data)
    def recorded_tcp_limit(pose, **kwargs):
        robot.pose = pose
        matched = [row for row in history['endpoint_queries']
                   if np.allclose(row['pose'], pose, atol=1e-6, rtol=0)]
        assert len(matched) == 1
        return tuple(matched[0]['tcp_limit'])
    robot.is_tcp_limit = recorded_tcp_limit
    report = {}
    with pytest.raises(RuntimeError, match='approach: controller TCP limit'):
        check_cartesian_interface(robot, request_data, report)
    assert robot.calls == []
    assert report['waypoint_checks'][-1]['phase'] == 'approach'
    assert report['execution_allowed'] is False


def test_original_bounds_not_replaced_by_expanded_site_workspace():
    profile = load_site()
    profile['arms']['right']['source_target_bounds_mm'][0] = [0, 100]
    with pytest.raises(ValueError, match='original backend target bounds'):
        adapt_proposal(PROPOSALS, json.loads(STATE.read_text()), profile)


def test_no_execution_option():
    with pytest.raises(SystemExit):
        main(['--execute'])
