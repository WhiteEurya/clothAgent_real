"""Read-only firmware TCP-screening geometry probe; no motion or setters."""
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
        parser.error('Output exists')
    prep = json.loads(args.preparation.read_text())
    identity = prep['before'][prep['selected_arm']]
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), motion_commands_sent=0,
                  configuration_commands_sent=0, execution_allowed=False, queries=[], boundaries=[])
    robot = None
    try:
        with socket.create_connection((identity['ip'], 502), timeout=3):
            pass
        robot = XArmAPI(identity['ip'], is_radian=False, do_not_open=True)
        robot.connect()
        time.sleep(.6)
        value(robot.get_robot_sn(), 'identity')
        q0 = value(robot.get_servo_angle(is_radian=False), 'initial joints')
        if (robot.control_box_sn != identity['control_box_sn'] or robot.axis != 7
                or np.max(np.abs(np.asarray(q0)-identity['joints_deg'])) > .1
                or robot.get_state() != (0, 2) or robot.get_err_warn_code() != (0, [0, 0])
                or robot.mode != 0):
            raise RuntimeError('Baseline identity or state changed')
        if not np.allclose(robot.tcp_offset, identity['tcp_offset_mm_deg'], atol=.1, rtol=0):
            raise RuntimeError('TCP offset changed')
        report.update(start_joints=q0, dh_params=robot.get_dh_params(),
                      ik_redundancy=robot.get_xarm7_ik_redundancy(),
                      approximate_motion=robot.is_approx_motion,
                      cartesian_continuous=robot.is_cart_continuous,
                      tcp_offset=robot.tcp_offset, world_offset=robot.world_offset,
                      reduced_states=robot.get_reduced_states(is_radian=False))

        def query(pose, label):
            radians = list(pose[:3]) + np.deg2rad(pose[3:]).tolist()
            row = dict(label=label, pose=pose,
                       raw_tcp_limit=robot._arm.arm_cmd.is_tcp_limit(radians), solutions={})
            report['queries'].append(row)
            for name, ref in [('default', None), ('current', q0)]:
                answer = robot.get_inverse_kinematics(pose, input_is_radian=False,
                          return_is_radian=False, ref_angles=ref)
                sol = dict(ik=answer)
                row['solutions'][name] = sol
                if answer[0] == 0:
                    sol['joint_limit'] = robot.is_joint_limit(answer[1], is_radian=False)
                    sol['fk_error'] = pose_error(value(robot.get_forward_kinematics(answer[1],
                       input_is_radian=False, return_is_radian=False), 'FK'), pose)
            return row

        for op in prep['operations']:
            if op['kind'] != 'move_tcp' or op['phase'] not in ('approach', 'grasp', 'placed'):
                continue
            pose = op['pose_base_mm_deg']
            for rpy in ([180., 0., 0.], [-180., 0., 0.], [0., 180., 180.]):
                query(list(pose[:3])+rpy, op['phase']+'_equivalent_orientation')
        for x in (212., 300., 421., 500.):
            for z in (0., 50., 100., 150., 200., 250., 300., 400.):
                query([x, -95., z, 180., 0., 0.], 'xz_grid')
        for x in (212., 300., 421., 500.):
            lo, hi = 0., 400.
            values = []
            for _ in range(9):
                z = (lo+hi)/2
                flag = value(robot.is_tcp_limit([x,-95.,z,180.,0.,0.], is_radian=False), 'boundary')
                values.append([z, flag])
                if flag:
                    lo = z
                else:
                    hi = z
            report['boundaries'].append(dict(x=x, y=-95., restricted_below_mm=lo,
                unrestricted_above_mm=hi, queries=values))
        report.update(end_joints=value(robot.get_servo_angle(is_radian=False), 'end joints'),
                      final_state=robot.get_state(), final_errors=robot.get_err_warn_code())
        report['maximum_joint_change_deg'] = float(np.max(np.abs(np.asarray(report['end_joints'])-q0)))
        report['status'] = 'READ_ONLY_COMPLETE'
    except Exception as exc:
        report.update(status='FAILED', error=str(exc))
    finally:
        if robot is not None:
            robot.disconnect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'queries'}, indent=2), flush=True)
    for row in report['queries']:
        print(row['label'], row['pose'], row['raw_tcp_limit'], flush=True)


if __name__ == '__main__':
    main()
