"""Adapt validated Cloth proposals to the legacy Cartesian check-only interface.

No hardware factory or execution option is provided. Full-fold incompatibilities
with execute_point_plan.py remain explicit; no legacy approval file is forged.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from cloth_agent.site_observation import DEFAULT_PROFILE, convert_pose_mm_deg, load_site
from cloth_agent.site_robot_check import value
from .prepare_partitioned_trial import controller_checks, pose_errors, staged_operations, validate_source


def vector(data, size, label):
    result = np.asarray(data, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'Invalid {label}')
    return result


def adapt_proposal(proposals, state, profile):
    plan = validate_source(Path(proposals), profile)
    active, inactive = plan['selected_arm'], plan['inactive_arm']
    for arm in ('left', 'right'):
        config, recorded = profile['arms'][arm], state['arms'][arm]
        if config['robot_axis'] != 7 or recorded['axis'] != 7:
            raise ValueError('This adapter requires the current 7+7 system')
        if config.get('installation_confirmed') is not True:
            raise ValueError(f'{arm}: installation is not confirmed')
        if (recorded['status'] != 'READ_ONLY_CHECKS_PASS'
                or recorded['ip'] != config['robot_ip']):
            raise ValueError(f'{arm}: saved state/config mismatch')
        for key in ('robot_sn', 'control_box_sn'):
            if recorded[key] != config['identity_verification'][key]:
                raise ValueError(f'{arm}: identity mismatch: {key}')
        for key in ('tcp_offset_mm_deg', 'world_offset_mm_deg'):
            if not np.allclose(vector(recorded[key], 6, key), config[key], atol=.01, rtol=0):
                raise ValueError(f'{arm}: {key} mismatch')
        vector(recorded['joints_deg'], 7, 'joints')
        vector(recorded['pose_base_mm_deg'], 6, 'pose')
    if not np.allclose(state['arms'][inactive]['joints_deg'],
                       profile['arms'][inactive]['home']['reference_joints_deg'], atol=.1, rtol=0):
        raise ValueError('Inactive arm differs from the saved obstacle pose')
    for operation in plan['operations']:
        if operation['kind'] == 'move_tcp':
            expected = convert_pose_mm_deg(profile, operation['pose_common_mm_deg'], 'common', active)
            if not np.allclose(expected, operation['pose_base_mm_deg'], atol=1e-6, rtol=0):
                raise ValueError('Common/base pose mismatch')
    bounds = np.asarray(profile['arms'][active]['source_target_bounds_mm'], dtype=float)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Invalid original backend target bounds')
    for key in ('grasp_common_mm', 'target_common_mm'):
        point = convert_pose_mm_deg(profile, [*plan[key], 0, 0, 0], 'common', active)[:3]
        if np.any(point < bounds[:, 0]) or np.any(point > bounds[:, 1]):
            raise ValueError(f'{key}: outside original backend target bounds')
    operations = staged_operations(plan, state['arms'][active]['pose_base_mm_deg'])
    moves = [op for op in operations if op['kind'] == 'move_tcp']
    legacy_speed_conflicts = []
    for index, operation in enumerate(moves):
        speed_limit, acceleration_limit = (50., 100.) if index < 4 else (10., 20.)
        if (operation['speed_mm_s'] > speed_limit
                or operation['acceleration_mm_s2'] > acceleration_limit):
            legacy_speed_conflicts.append(operation['phase'])
    return {
        'schema_version': 1,
        'status': 'CARTESIAN_CHECK_INPUT_READY_EXECUTION_INCOMPATIBLE',
        'execution_allowed': False,
        'backend_interface': 'set_position/motion_type=0/only_check_type=1,3,2',
        'source_proposals': str(Path(proposals).resolve()),
        'calibration_id': profile['calibration_id'],
        'selected_arm': active,
        'inactive_arm': inactive,
        'arm_config': copy.deepcopy(profile['arms'][active]),
        'start': copy.deepcopy(state['arms'][active]),
        'operations': operations,
        'gripper_events': [op for op in operations if op['kind'] == 'gripper'],
        'legacy_speed_conflicts': legacy_speed_conflicts,
        'execution_blockers': [
            'execute_point_plan.py requires an open gripper and never performs close/release events.',
            'The separate descend_selected_point.py supports closure only 5 mm above its selected surface; this proposal requests the original surface without that offset.',
            'The legacy final endpoint is one selected point plus 5..150 mm clearance, not a complete fold.',
            'Legacy validate_selection consumes raw depth/confidence capture files; this proposal uses hashed world XYZ and median-depth anchors.',
            'The saved observation/state is not a fresh execution-start observation.',
            'External collision validation for the controller Cartesian path is not supplied by execute_point_plan.py; joint-path certificates do not transfer.',
        ] + (['Legacy index-based speed limits reject phases: ' + ', '.join(legacy_speed_conflicts)]
             if legacy_speed_conflicts else []),
        'physical_execution': False,
        'external_collision_checked': False,
    }


def check_start(robot, request):
    config, start = request['arm_config'], request['start']
    if (robot.get_state() != (0, 2) or robot.get_err_warn_code() != (0, [0, 0])
            or robot.mode != 0):
        raise RuntimeError('Require idle position control without faults')
    if robot.axis != config['robot_axis']:
        raise RuntimeError('Controller axis mismatch')
    for actual, expected, tolerance in (
        (robot.tcp_offset, config['tcp_offset_mm_deg'], .1),
        (robot.world_offset, config['world_offset_mm_deg'], .01),
    ):
        if not np.allclose(vector(actual, 6, 'controller offset'), expected, atol=tolerance, rtol=0):
            raise RuntimeError('Controller TCP/world offset mismatch')
    opening = value(robot.get_gripper_position(), 'gripper opening')
    if not np.isfinite(opening) or opening < 500 or robot.get_gripper_err_code() != (0, 0):
        raise RuntimeError('Require open gripper without faults')
    pose = vector(value(robot.get_position(is_radian=False), 'pose'), 6, 'pose')
    distance, angle = pose_errors(pose, start['pose_base_mm_deg'])
    if distance > 1 or angle > 1:
        raise RuntimeError('Current pose differs from planned start')
    joints = vector(value(robot.get_servo_angle(is_radian=False), 'joints'), 7, 'joints')
    if not np.allclose(joints, start['joints_deg'], atol=1, rtol=0):
        raise RuntimeError('Start joint configuration changed')
    return joints


def check_cartesian_interface(robot, request, report):
    """Check motion waypoints only; gripper events stay in the compatibility report."""
    if request.get('execution_allowed') is not False:
        raise ValueError('Only non-executable adapter requests are supported')
    report.update(execution_allowed=False, external_collision_checked=False,
                  status='CHECKING', gripper_events_checked=False)
    try:
        initial = check_start(robot, request)
        controller_checks(robot, request['operations'], initial, report)
        final = check_start(robot, request)
        if not np.allclose(initial, final, atol=.1, rtol=0):
            raise RuntimeError('Robot moved during path checks')
        report['status'] = 'CARTESIAN_INTERFACE_CHECKS_PASS_NOT_EXECUTION_APPROVAL'
    except Exception as exc:
        report.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}')
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', required=True, type=Path)
    parser.add_argument('--robot-state', required=True, type=Path)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    request = adapt_proposal(args.proposals, json.loads(args.robot_state.read_text()), load_site(args.profile))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'adapter_request.json').write_text(json.dumps(request, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': request['status'], 'robot_calls': 0, 'execution_allowed': False,
                      'legacy_speed_conflicts': request['legacy_speed_conflicts']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
