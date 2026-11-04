"""Read-only comparison of controller IK, FK and TCP/joint-limit queries.

No motion, check-only motion, gripper, mode, enable or configuration setters.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI


def value(response, label):
    if response[0] != 0:
        raise RuntimeError(f'{label}: {response}')
    return response[1]


def pose_error(actual, expected):
    distance = float(np.linalg.norm(np.asarray(actual[:3]) - expected[:3]))
    angle = float(np.degrees((Rotation.from_euler('xyz', actual[3:], degrees=True).inv()
        * Rotation.from_euler('xyz', expected[3:], degrees=True)).magnitude()))
    return dict(position_mm=distance, orientation_deg=angle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    preparation = json.loads(args.preparation.read_text())
    arm_id = preparation['selected_arm']
    identity = preparation['before'][arm_id]
    operations = preparation['operations']
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), source=str(args.preparation.resolve()),
        selected_arm=arm_id, ip=identity['ip'], execution_allowed=False,
        motion_commands_sent=0, gripper_commands_sent=0, configuration_commands_sent=0, queries=[])
    robot = XArmAPI(identity['ip'], is_radian=False, do_not_open=True)
    try:
        robot.connect()
        time.sleep(.5)
        value(robot.get_robot_sn(), 'identity')
        if robot.control_box_sn != identity['control_box_sn'] or robot.axis != identity['axis']:
            raise RuntimeError('Controller identity changed')
        q0 = value(robot.get_servo_angle(is_radian=False), 'start joints')
        actual = value(robot.get_position(is_radian=False), 'start pose')
        report.update(start_joints=q0, start_pose=actual, state=robot.get_state(), mode=robot.mode,
            errors=robot.get_err_warn_code(), tcp_offset=list(robot.tcp_offset),
            world_offset=list(robot.world_offset), reduced_states=robot.get_reduced_states(is_radian=False))
        if report['state'] != (0, 2) or report['errors'] != (0, [0, 0]) or report['mode'] != 0:
            raise RuntimeError('Expected idle fault-free position mode')
        if (not np.allclose(robot.tcp_offset, identity['tcp_offset_mm_deg'], atol=.1, rtol=0)
                or not np.allclose(robot.world_offset, identity['world_offset_mm_deg'], atol=.01, rtol=0)):
            raise RuntimeError('Controller coordinate offsets changed')
        sequence_reference = q0
        targets = [dict(phase='current_measured_pose', kind='move_tcp', pose_base_mm_deg=actual)] + operations
        for op in targets:
            if op['kind'] != 'move_tcp':
                continue
            pose = op['pose_base_mm_deg']
            row = dict(phase=op['phase'], target_pose=pose, tcp_limit_response=robot.is_tcp_limit(pose, is_radian=False), solutions={})
            report['queries'].append(row)
            for name, reference in [('default_no_reference', None), ('current_joints_reference', q0),
                                    ('previous_waypoint_reference', sequence_reference)]:
                answer = robot.get_inverse_kinematics(pose, input_is_radian=False,
                    return_is_radian=False, limited=True, ref_angles=reference)
                solution = dict(ik_response=answer)
                row['solutions'][name] = solution
                if answer[0] != 0:
                    continue
                q = answer[1]
                if len(q) != 7 or not np.isfinite(q).all():
                    raise RuntimeError('Malformed successful IK response')
                solution['joint_limit_response'] = robot.is_joint_limit(q, is_radian=False)
                solution['fk_response'] = robot.get_forward_kinematics(q, input_is_radian=False, return_is_radian=False)
                if solution['fk_response'][0] == 0:
                    solution['fk_error'] = pose_error(solution['fk_response'][1], pose)
                    solution['endpoint_checks_pass'] = (
                        solution['joint_limit_response'] == (0, False)
                        and solution['fk_error']['position_mm'] <= 2
                        and solution['fk_error']['orientation_deg'] <= 1)
                if name == 'previous_waypoint_reference':
                    sequence_reference = q
        report.update(end_joints=value(robot.get_servo_angle(is_radian=False), 'end joints'),
            final_state=robot.get_state(), final_errors=robot.get_err_warn_code())
        report['maximum_joint_change_deg'] = float(np.max(np.abs(np.asarray(report['end_joints']) - q0)))
        report['status'] = 'READ_ONLY_DIAGNOSIS_COMPLETE'
    except Exception as exc:
        report.update(status='DIAGNOSIS_FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        robot.disconnect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'maximum_joint_change_deg')}, indent=2))
    for row in report['queries']:
        print(row['phase'], 'TCP:', row['tcp_limit_response'], {name:dict(code=s['ik_response'][0],
            joint_limit=s.get('joint_limit_response'), fk_error=s.get('fk_error')) for name,s in row['solutions'].items()})


if __name__ == '__main__':
    main()
