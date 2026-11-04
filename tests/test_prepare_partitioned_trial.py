import copy

import pytest

from cloth_agent.experiments.prepare_partitioned_trial import staged_operations, controller_checks, main


def plan():
    operations = [dict(kind='gripper', phase='open', arm='right', action='open')]
    for phase, pose in [('approach', [200, -300, 110, 180, 0, 0]),
                        ('grasp', [200, -300, 10, 180, 0, 0]),
                        ('lifted', [200, -300, 90, 180, 0, 0]),
                        ('across', [350, -300, 90, 180, 0, 0]),
                        ('placed', [350, -300, 15, 180, 0, 0]),
                        ('retreat', [350, -300, 95, 180, 0, 0])]:
        operations.append(dict(kind='move_tcp', phase=phase, arm='right', pose_base_mm_deg=pose))
        if phase in ('grasp', 'placed'):
            operations.append(dict(kind='gripper', phase='close' if phase=='grasp' else 'release',
                arm='right', action='close' if phase=='grasp' else 'open'))
    return dict(selected_arm='right', operations=operations)


class CheckOnlyRobot:
    def __init__(self):
        self.calls = []
        self.only_check_result = 0
    def is_tcp_limit(self, pose, **kwargs): self.pose = pose; return 0, False
    def get_inverse_kinematics(self, pose, **kwargs): return 0, [0]*7
    def is_joint_limit(self, joints, **kwargs): return 0, False
    def get_forward_kinematics(self, joints, **kwargs): return 0, self.pose
    def set_position(self, *args, **kwargs):
        assert kwargs['only_check_type'] in (1, 2, 3)
        assert kwargs['wait'] is False
        self.calls.append(kwargs['only_check_type'])
        return 0


def test_staging_preserves_grasp_and_target_and_single_arm():
    original = plan(); before = copy.deepcopy(original)
    operations = staged_operations(original, [210, -40, 450, 178, 0, 0])
    assert original == before
    assert [o['phase'] for o in operations[:4]] == ['open', 'orient_at_height', 'transit_above_grasp', 'approach']
    for phase in ('grasp', 'placed'):
        assert next(o for o in operations if o['phase']==phase)['pose_base_mm_deg'] == next(o for o in before['operations'] if o['phase']==phase)['pose_base_mm_deg']
    assert {o['arm'] for o in operations} == {'right'}
    assert operations[2]['pose_base_mm_deg'] == [200, -300, 450, 180, 0, 0]


def test_controller_only_check_never_dispatches_motion():
    robot = CheckOnlyRobot(); report = {}
    operations = staged_operations(plan(), [210, -40, 450, 178, 0, 0])
    controller_checks(robot, operations, [0]*7, report)
    assert robot.calls == [1] + [3]*6 + [2]
    assert len(report['waypoint_checks']) == 8
    assert all(r['execution'] is False for r in report['controller_path_checks'])


def test_pitch_candidate_preserves_xyz_and_removes_stale_common_pose():
    source = plan()
    source['operations'][1]['pose_common_mm_deg'] = source['operations'][1]['pose_base_mm_deg'][:]
    before = copy.deepcopy(source)
    operations = staged_operations(source, [210, -40, 450, 178, 0, 0], pitch_offset_deg=30.)
    assert source == before
    for original in source['operations']:
        if original['kind'] == 'move_tcp':
            changed = next(o for o in operations if o['phase']==original['phase'])
            assert changed['pose_base_mm_deg'][:3] == original['pose_base_mm_deg'][:3]
            assert changed['pose_base_mm_deg'][4] == original['pose_base_mm_deg'][4] + 30
            assert 'pose_common_mm_deg' not in changed
    with pytest.raises(ValueError, match='Pitch offset'):
        staged_operations(source, [210, -40, 450, 178, 0, 0], pitch_offset_deg=90.)


def test_ik_failure_stops_before_any_path_command():
    robot = CheckOnlyRobot(); robot.get_inverse_kinematics = lambda *a, **k: (-6, [])
    with pytest.raises(RuntimeError, match='IK'):
        controller_checks(robot, staged_operations(plan(), [210, -40, 450, 178, 0, 0]), [0]*7, {})
    assert robot.calls == []


def test_path_check_failure_is_preserved():
    robot = CheckOnlyRobot(); robot.only_check_result = 7
    report = {}
    with pytest.raises(RuntimeError, match='path check failed'):
        controller_checks(robot, staged_operations(plan(), [210, -40, 450, 178, 0, 0]), [0]*7, report)
    assert report['controller_path_checks'][-1]['detail'] == 7
    assert robot.calls == [1]


def test_no_execute_option():
    with pytest.raises(SystemExit) as error:
        main(['--execute'])
    assert error.value.code == 2
