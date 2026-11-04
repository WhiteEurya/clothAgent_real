"""Bounded IK/path diagnosis. All path calls are check-only; never executes."""
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import urllib.request

import numpy as np
from xarm.wrapper import XArmAPI
from diagnose_partitioned_ik import value, pose_error


def main():
    trial = Path(__file__).resolve().parents[1] / 'deployment-artifacts/topdown_physical_trial_001'
    out = trial / 'branch_path_diagnosis'
    out.mkdir(exist_ok=True)
    if (out / 'report.json').exists():
        raise RuntimeError('Refusing to overwrite evidence')
    prep = json.loads((trial / 'controller_preparation/preparation.json').read_text())
    identity = prep['before'][prep['selected_arm']]
    report = dict(started_at=datetime.now(timezone.utc).isoformat(), physical_motion_commands=0,
                  configuration_commands=0, execution_authorized=False, checks=[], samples=[])
    robot = XArmAPI(identity['ip'], is_radian=False, do_not_open=True)
    q0 = None

    def stationary():
        q = value(robot.get_servo_angle(is_radian=False), 'joints')
        if (np.max(np.abs(np.asarray(q)-q0)) > .1 or robot.get_state() != (0, 2)
                or robot.get_err_warn_code() != (0, [0, 0]) or robot.mode != 0):
            raise RuntimeError('Live state changed')
        return q

    def joint_check(q, label, initial):
        stationary()
        start = time.monotonic()
        if initial:
            code = robot.set_servo_angle(angle=q, speed=8, mvacc=10, is_radian=False,
                                        wait=False, only_check_type=1)
        else:
            code = robot.set_servo_angle(angle=q, speed=8, mvacc=10, is_radian=False,
                                        wait=False, only_check_type=2)
        row = dict(label=label, code=code, detail=robot.only_check_result,
                   elapsed_seconds=time.monotonic()-start)
        report['checks'].append(row)
        print(json.dumps(row), flush=True)
        stationary()
        if code != 0 or row['detail'] != 0:
            raise RuntimeError('Check-only did not pass; stopped without queuing further path checks')

    try:
        robot.connect()
        time.sleep(.7)
        value(robot.get_robot_sn(), 'identity')
        if robot.control_box_sn != identity['control_box_sn'] or robot.axis != 7:
            raise RuntimeError('Identity mismatch')
        q0 = value(robot.get_servo_angle(is_radian=False), 'baseline joints')
        if np.max(np.abs(np.asarray(q0)-identity['joints_deg'])) > .1:
            raise RuntimeError('Start differs from saved preparation')
        if not np.allclose(robot.tcp_offset, identity['tcp_offset_mm_deg'], atol=.1, rtol=0):
            raise RuntimeError('TCP offset changed')
        if not np.allclose(robot.world_offset, identity['world_offset_mm_deg'], atol=.01, rtol=0):
            raise RuntimeError('World offset changed')
        stationary()
        report['start_joints'] = q0
        # FK query of the candidate printed by H46, not a movement command.
        logged_q = np.rad2deg([-.099578,-1.145740,-.180942,-.242684,-.068890,1.866380,-.142069]).tolist()
        report['logged_grasp_branch'] = dict(joints_deg=logged_q,
            fk=robot.get_forward_kinematics(logged_q,input_is_radian=False,return_is_radian=False),
            joint_limit=robot.is_joint_limit(logged_q,is_radian=False))
        moves = [op for op in prep['operations'] if op['kind']=='move_tcp']
        # Maintain the preceding solution as the seed while descending / folding.
        ref = q0
        previous = value(robot.get_position(is_radian=False), 'current pose')
        for op in moves:
            target = op['pose_base_mm_deg']
            count = max(1, int(np.ceil(np.linalg.norm(np.asarray(target[:3])-previous[:3])/20)))
            for i in range(1, count+1):
                pose = (np.asarray(previous)+(np.asarray(target)-previous)*i/count).tolist()
                q = value(robot.get_inverse_kinematics(pose,input_is_radian=False,
                    return_is_radian=False,ref_angles=ref), 'IK')
                row = dict(phase=op['phase'],pose=pose,joints_deg=q,
                    joint_limit=robot.is_joint_limit(q,is_radian=False),
                    fk_error=pose_error(value(robot.get_forward_kinematics(q,
                        input_is_radian=False,return_is_radian=False),'FK'),pose),
                    max_joint_step_deg=float(np.max(np.abs(np.asarray(q)-ref))))
                report['samples'].append(row)
                ref=q
            previous=target
        stationary()
        # Baseline first. A failed no-op makes every subsequent path result inconclusive.
        joint_check(q0, 'baseline_current_to_current', True)
        joint_check(q0, 'baseline_restore_virtual', False)
        row = next(s for s in reversed(report['samples']) if s['phase']=='approach')
        if row['joint_limit'] != (0, False) or row['fk_error']['position_mm'] > 2:
            raise RuntimeError('Approach candidate failed endpoint checks')
        joint_check(row['joints_deg'], 'home_to_explicit_approach_branch', True)
        # End this diagnostic by restoring virtual planning state, without moving.
        joint_check(q0, 'restore_virtual_to_actual', False)
        report['status']='CHECKS_COMPLETE_NOT_EXECUTION_APPROVAL'
    except Exception as exc:
        report.update(status='STOPPED', reason=str(exc))
    finally:
        try:
            report['end_joints']=value(robot.get_servo_angle(is_radian=False),'final joints')
            report['final_state']=robot.get_state()
            report['final_errors']=robot.get_err_warn_code()
            if q0 is not None:
                report['maximum_joint_change_deg']=float(np.max(np.abs(np.asarray(report['end_joints'])-q0)))
        finally:
            robot.disconnect()
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with urllib.request.urlopen(f"http://{identity['ip']}:18333/handler/log/download",timeout=60) as response:
        (out/'controller_log_export.tar.gz').write_bytes(response.read())
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2),flush=True)


if __name__=='__main__':
    main()
