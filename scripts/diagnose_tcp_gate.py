"""Diagnose TCP screening versus controller check-only planning; NEVER execute.

All set_position/set_servo_angle calls use literal nonzero only_check_type.
No changes to limits, mode, enable, tool settings, or gripper state.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

import numpy as np
from xarm.wrapper import XArmAPI
from diagnose_partitioned_ik import value, pose_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    prep = json.loads(args.preparation.read_text())
    identity = prep['before'][prep['selected_arm']]
    moves = [op for op in prep['operations'] if op['kind'] == 'move_tcp']
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), execution_allowed=False,
                  motion_commands_sent=0, gripper_commands_sent=0, configuration_commands_sent=0,
                  check_only_calls=[], source=str(args.preparation.resolve()))
    robot = None
    q0 = None

    def stationary():
        q = value(robot.get_servo_angle(is_radian=False), 'joints')
        if (np.max(np.abs(np.asarray(q)-q0)) > .1 or robot.get_state() != (0, 2)
                or robot.get_err_warn_code() != (0, [0, 0]) or robot.mode != 0):
            raise RuntimeError('Live state changed during diagnostic')
        return q

    def joint_check(q, check_type, label):
        assert check_type in (1, 2)
        # Fixed non-executing values: there is no type=0 path in this script.
        if check_type == 1:
            code = robot.set_servo_angle(angle=q, speed=8, mvacc=10, is_radian=False,
                                        wait=False, only_check_type=1)
        else:
            code = robot.set_servo_angle(angle=q, speed=8, mvacc=10, is_radian=False,
                                        wait=False, only_check_type=2)
        row = dict(label=label, kind='joint_check_only', only_check_type=check_type,
                   code=code, detail=robot.only_check_result)
        report['check_only_calls'].append(row)
        print(json.dumps(row), flush=True)
        stationary()
        if label.startswith('reset') and (code != 0 or row['detail'] != 0):
            raise RuntimeError('Virtual-state reset check failed; subsequent path checks would be inconclusive')
        return row

    try:
        with socket.create_connection((identity['ip'], 502), timeout=3):
            pass
        robot = XArmAPI(identity['ip'], is_radian=False, do_not_open=True)
        robot.connect()
        time.sleep(.6)
        value(robot.get_robot_sn(), 'identity')
        if robot.control_box_sn != identity['control_box_sn'] or robot.axis != 7:
            raise RuntimeError('Controller identity mismatch')
        q0 = value(robot.get_servo_angle(is_radian=False), 'initial joints')
        if np.max(np.abs(np.asarray(q0)-identity['joints_deg'])) > .1:
            raise RuntimeError('Robot has changed from preparation start')
        stationary()
        if not np.allclose(robot.tcp_offset, identity['tcp_offset_mm_deg'], atol=.1, rtol=0):
            raise RuntimeError('TCP offset changed')
        if not np.allclose(robot.world_offset, identity['world_offset_mm_deg'], atol=.01, rtol=0):
            raise RuntimeError('World offset changed')
        report.update(start_joints=q0, control_box_sn=robot.control_box_sn,
                      firmware=robot.version, reduced_states=robot.get_reduced_states(is_radian=False))
        report['safety_properties'] = {key: getattr(robot, key) for key in (
            'self_collision_params', 'is_fence_mode', 'is_reduced_mode', 'is_collision_rebound',
            'collision_sensitivity', 'gravity_direction', 'tcp_offset', 'world_offset', 'tcp_load')}
        report['endpoint_queries'] = []
        for op in moves:
            pose = op['pose_base_mm_deg']
            q = value(robot.get_inverse_kinematics(pose, input_is_radian=False,
                      return_is_radian=False, ref_angles=q0), 'IK')
            row = dict(phase=op['phase'], pose=pose, q=q,
                       tcp_limit=robot.is_tcp_limit(pose, is_radian=False),
                       joint_limit=robot.is_joint_limit(q, is_radian=False),
                       fk_error=pose_error(value(robot.get_forward_kinematics(q,
                           input_is_radian=False, return_is_radian=False), 'FK'), pose))
            report['endpoint_queries'].append(row)
        # Original staged Cartesian path, controller simulation only, stop at first failure.
        for index, op in enumerate(moves):
            stationary()
            check_type = 1 if index == 0 else (2 if index == len(moves)-1 else 3)
            assert check_type in (1, 2, 3)
            code = robot.set_position(*op['pose_base_mm_deg'], speed=op['speed_mm_s'],
                    mvacc=op['acceleration_mm_s2'], is_radian=False, wait=False,
                    motion_type=0, only_check_type=check_type)
            row = dict(label=op['phase'], kind='cartesian_check_only',
                       only_check_type=check_type, code=code, detail=robot.only_check_result,
                       tcp_limit_after_virtual_step=robot.is_tcp_limit(op['pose_base_mm_deg'], is_radian=False))
            report['check_only_calls'].append(row)
            stationary()
            if code != 0 or row['detail'] != 0:
                break
        joint_check(q0, 1, 'reset_virtual_from_actual')
        joint_check(q0, 2, 'reset_virtual_to_actual')
        report['independent_cartesian_checks'] = []
        for op in moves:
            if op['phase'] not in ('transit_above_grasp', 'approach', 'grasp', 'placed'):
                continue
            stationary()
            code = robot.set_position(*op['pose_base_mm_deg'], speed=5, mvacc=5,
                    is_radian=False, wait=False, motion_type=0, only_check_type=1)
            row = dict(label='from_actual_home_to_' + op['phase'], kind='cartesian_check_only',
                       only_check_type=1, code=code, detail=robot.only_check_result)
            report['check_only_calls'].append(row)
            report['independent_cartesian_checks'].append(row)
            stationary()
            joint_check(q0, 2, 'reset_after_direct_' + op['phase'])
        # Compare valid explicit IK branches to the TCP screening API, without moving.
        for row in report['endpoint_queries']:
            if row['phase'] not in ('approach', 'grasp', 'placed'):
                continue
            if row['joint_limit'] != (0, False) or row['fk_error']['position_mm'] > 2 or row['fk_error']['orientation_deg'] > 1:
                raise RuntimeError('Explicit branch failed endpoint validation')
            row['joint_path_check'] = joint_check(row['q'], 1, row['phase'])
            row['tcp_limit_after_virtual_joint'] = robot.is_tcp_limit(row['pose'], is_radian=False)
            joint_check(q0, 2, 'reset_after_' + row['phase'])
            row['tcp_limit_after_reset'] = robot.is_tcp_limit(row['pose'], is_radian=False)
        report['end_joints'] = stationary()
        report['maximum_joint_change_deg'] = float(np.max(np.abs(np.asarray(report['end_joints'])-q0)))
        report['status'] = 'DIAGNOSIS_COMPLETE_NOT_EXECUTION_APPROVAL'
    except Exception as exc:
        report.update(status='DIAGNOSIS_FAILED', error=f'{type(exc).__name__}: {exc}')
    finally:
        if robot is not None:
            try:
                report['final_state'] = robot.get_state()
                report['final_errors'] = robot.get_err_warn_code()
                report['end_joints'] = value(robot.get_servo_angle(is_radian=False), 'final joints')
                if q0 is not None:
                    report['maximum_joint_change_deg'] = float(np.max(np.abs(np.asarray(report['end_joints'])-q0)))
            except Exception as exc:
                report['final_snapshot_error'] = str(exc)
            robot.disconnect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
