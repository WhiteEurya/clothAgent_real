"""Prepare one complete top-down fold and ask controllers to CHECK paths only.

No execution option. set_position is used only with documented nonzero
only_check_type; no enable, mode/state, gripper or executing motion commands.
Controller feasibility does not validate external obstacles or cloth grasp.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cloth_agent.dual_arm.observation import Observation, sha
from cloth_agent.site_observation import DEFAULT_PROFILE, load_site
from cloth_agent.site_robot_check import read_arm, value
from cloth_agent.tabletop_view import TabletopView
from .partitioned_folding import DEFAULT_GRAPH, compile_templates, load_policy, plan_step


def staged_operations(plan, start_pose, *, pitch_offset_deg=0.):
    """Add high transit before descending; keep the proposed grasp/target intact."""
    start = np.asarray(start_pose, dtype=float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError('Invalid starting pose')
    arm = plan['selected_arm']
    if not np.isfinite(pitch_offset_deg) or abs(pitch_offset_deg) > 45:
        raise ValueError('Pitch offset must be within +/-45 degrees for preparation')
    original = [dict(op) for op in plan['operations']]
    for op in original:
        if op['kind'] == 'move_tcp':
            pose = list(op['pose_base_mm_deg'])
            pose[4] += pitch_offset_deg
            op['pose_base_mm_deg'] = pose
            # A changed base-frame orientation must not retain stale common RPY.
            op.pop('pose_common_mm_deg', None)
    if not original or {op['arm'] for op in original} != {arm}:
        raise ValueError('Expected exactly one active arm')
    approach = next(op for op in original if op['phase'] == 'approach')['pose_base_mm_deg']
    high = max(start[2], max(op['pose_base_mm_deg'][2] for op in original if op['kind'] == 'move_tcp') + 50)
    result = [dict(original[0])]
    for phase, pose in [
        ('rise_at_start', [*start[:2], high, *start[3:]]),
        ('orient_at_height', [*start[:2], high, *approach[3:]]),
        ('transit_above_grasp', [*approach[:2], high, *approach[3:]]),
    ]:
        if phase == 'rise_at_start' and abs(high - start[2]) < .01:
            continue
        result.append(dict(kind='move_tcp', phase=phase, arm=arm, pose_base_mm_deg=list(pose)))
    result.extend(dict(op) for op in original[1:])
    for op in result:
        if op['kind'] == 'move_tcp':
            slow = op['phase'] in ('grasp', 'placed', 'orient_at_height')
            op.update(speed_mm_s=5. if slow else 20., acceleration_mm_s2=5. if slow else 20.)
        elif op['kind'] == 'gripper':
            op.update(position_pulse=850 if op['action'] == 'open' else 0, speed_pulse_s=100,
                parameters_status='proposal_requires_grasp_validation')
        else:
            raise ValueError('Unsupported operation')
    return result


def pose_errors(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    return (float(np.linalg.norm(a[:3] - b[:3])),
        float(np.degrees((Rotation.from_euler('xyz', a[3:], degrees=True).inv()
            * Rotation.from_euler('xyz', b[3:], degrees=True)).magnitude())))


def controller_checks(robot, operations, start_joints, report):
    moves = [op for op in operations if op['kind'] == 'move_tcp']
    reference = list(start_joints)
    report['waypoint_checks'] = []
    report['controller_path_checks'] = []
    for op in moves:
        pose = op['pose_base_mm_deg']
        row = dict(phase=op['phase'], pose_base_mm_deg=pose)
        report['waypoint_checks'].append(row)
        row['tcp_limit'] = bool(value(robot.is_tcp_limit(pose, is_radian=False), 'TCP limit'))
        if row['tcp_limit']:
            raise RuntimeError(f"{op['phase']}: controller TCP limit")
        response = robot.get_inverse_kinematics(pose, input_is_radian=False,
            return_is_radian=False, ref_angles=reference)
        row['ik_response'] = response
        reference = value(response, f"{op['phase']} IK")
        row['joint_limit'] = bool(value(robot.is_joint_limit(reference, is_radian=False), 'Joint limit'))
        if row['joint_limit']:
            raise RuntimeError(f"{op['phase']}: joint limit")
        actual = value(robot.get_forward_kinematics(reference, input_is_radian=False, return_is_radian=False), 'FK')
        distance, angle = pose_errors(actual, pose)
        row.update(fk_translation_error_mm=distance, fk_rotation_error_deg=angle)
        if distance > 2 or angle > 1:
            raise RuntimeError(f"{op['phase']}: IK/FK mismatch")
    for index, op in enumerate(moves):
        check_type = 1 if index == 0 else (2 if index == len(moves) - 1 else 3)
        code = robot.set_position(*op['pose_base_mm_deg'], speed=op['speed_mm_s'],
            mvacc=op['acceleration_mm_s2'], is_radian=False, wait=False,
            motion_type=0, only_check_type=check_type)
        detail = robot.only_check_result
        report['controller_path_checks'].append(dict(phase=op['phase'], code=code,
            detail=detail, only_check_type=check_type, execution=False))
        if code != 0 or detail != 0:
            raise RuntimeError(f"{op['phase']}: controller path check failed, code={code}, detail={detail}")


def validate_source(proposals, profile):
    observation = Observation.load(proposals / 'observation')
    view = TabletopView.load(proposals / 'topdown', observation)
    plan = json.loads((proposals / 'primary_plan.json').read_text())
    if (plan.get('status') != 'PREVIEW_READY' or plan.get('execution_allowed') is not False
            or plan.get('decision_role') != 'primary' or plan.get('request', {}).get('decision_view') != 'topdown'):
        raise ValueError('Requires a ready top-down primary preview')
    if plan['request']['decision_image_sha256'] != sha(view.image):
        raise ValueError('Decision view changed')
    graph, _, templates = compile_templates(DEFAULT_GRAPH)
    regenerated = plan_step(graph, templates, profile, observation, plan['request'], load_policy(), view.meta['table_normal_common'])
    for field in ('operations', 'selected_arm', 'grasp_common_mm', 'target_common_mm', 'geometry_policy'):
        if regenerated[field] != plan[field]:
            raise ValueError('Plan differs from calibrated source: ' + field)
    if not regenerated['within_imported_local_target_bounds']:
        raise ValueError('Grasp or target lies outside imported local bounds')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Previously observed controller identities')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--pitch-offsets-deg', type=float, nargs='+', default=[0.],
        help='Check these tool pitch offsets in order, within +/-45; never moves hardware')
    args = parser.parse_args(argv)
    profile = load_site(args.profile)
    if (not 1 <= len(args.pitch_offsets_deg) <= 7
            or any(not np.isfinite(v) or abs(v) > 45 for v in args.pitch_offsets_deg)):
        parser.error('Provide 1..7 finite pitch offsets within +/-45 degrees')
    plan = validate_source(args.proposals, profile)
    baseline = json.loads(args.baseline.read_text())
    if args.output.exists():
        parser.error('Output already exists')
    args.output.mkdir(parents=True)
    report = dict(status='PREPARING', source_plan=str((args.proposals / 'primary_plan.json').resolve()),
        source_plan_sha256=sha(args.proposals / 'primary_plan.json'),
        source_observation_id=plan['observation_id'], profile_sha256=sha(args.profile),
        selected_arm=plan['selected_arm'], inactive_arm=plan['inactive_arm'],
        execution_allowed=False, motion_commands_sent=0, gripper_commands_sent=0,
        external_collision_checked=False, grasp_contact_validated=False,
        limitations=['Controller check-only validates its internal kinematic/path limits, not external collisions.',
            'Tool/table clearance, inactive-arm geometry and scene stability require separate verification.',
            'No grasp acquisition, real path tracking or physical fold outcome has been tested.'],
        before={}, after={})
    robot = None
    try:
        for arm in ('left', 'right'):
            snapshot = read_arm(profile, arm)
            report['before'][arm] = snapshot
            if snapshot['status'] != 'READ_ONLY_CHECKS_PASS':
                raise RuntimeError(f'{arm}: live controller preconditions failed')
            for key in ('control_box_sn', 'robot_sn', 'axis'):
                if snapshot[key] != baseline['arms'][arm][key]:
                    raise RuntimeError(f'{arm}: identity changed: {key}')
        inactive = report['before'][plan['inactive_arm']]
        if inactive['home_joint_max_error_deg'] > .5:
            raise RuntimeError('Inactive arm is not at its recorded Home')
        arm = plan['selected_arm']
        start = report['before'][arm]
        from xarm.wrapper import XArmAPI
        robot = XArmAPI(profile['arms'][arm]['robot_ip'], is_radian=False, do_not_open=True)
        robot.connect()
        q = value(robot.get_servo_angle(is_radian=False), 'precheck joints')
        if np.max(np.abs(np.asarray(q) - start['joints_deg'])) > .1:
            raise RuntimeError('Start changed before path checks')
        report['candidates'] = []
        for offset in args.pitch_offsets_deg:
            operations = staged_operations(plan, start['pose_base_mm_deg'], pitch_offset_deg=offset)
            candidate = dict(pitch_offset_deg=offset, operations=operations)
            report['candidates'].append(candidate)
            try:
                controller_checks(robot, operations, q, candidate)
                candidate['status'] = 'CONTROLLER_CHECKS_PASS'
            except RuntimeError as exc:
                candidate.update(status='BLOCKED', error=str(exc))
            check_q = value(robot.get_servo_angle(is_radian=False), 'post-candidate joints')
            if (np.max(np.abs(np.asarray(check_q) - q)) > .1
                    or value(robot.get_state(), 'post-candidate state') != 2
                    or value(robot.get_err_warn_code(), 'post-candidate errors') != [0, 0]):
                raise RuntimeError('Robot state changed during candidate checks')
            if candidate['status'] == 'CONTROLLER_CHECKS_PASS':
                report.update(operations=operations, selected_pitch_offset_deg=offset,
                    orientation_changed_from_visual_proposal=offset != 0.,
                    waypoint_checks=candidate['waypoint_checks'],
                    controller_path_checks=candidate['controller_path_checks'])
                break
        else:
            raise RuntimeError('No checked tool orientation passed the full controller path; see candidates')
        report['status'] = 'CONTROLLER_CHECKS_PASS_EXTERNAL_REVIEW_REQUIRED'
    except Exception as exc:
        report.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}')
    finally:
        if robot is not None:
            robot.disconnect()
        for arm, start in report['before'].items():
            if start.get('status') != 'READ_ONLY_CHECKS_PASS':
                continue
            try:
                after = read_arm(profile, arm)
            except Exception as exc:
                after = dict(status='READ_FAILED', error=str(exc))
            report['after'][arm] = after
            if (after['status'] != 'READ_ONLY_CHECKS_PASS'
                    or np.max(np.abs(np.asarray(after['joints_deg']) - start['joints_deg'])) > .1):
                report.update(status='BLOCKED', error=f'{arm}: state changed during preparation')
        (args.output / 'preparation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'selected_arm', 'error', 'execution_allowed',
        'motion_commands_sent', 'gripper_commands_sent')}, indent=2))
    return 0 if report['status'] == 'CONTROLLER_CHECKS_PASS_EXTERNAL_REVIEW_REQUIRED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
