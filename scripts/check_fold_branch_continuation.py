"""Check the saved fold candidate in controller simulation. No motion mode exists."""
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import urllib.request
import numpy as np
from xarm.wrapper import XArmAPI
from diagnose_partitioned_ik import value


def main():
    trial=Path(__file__).resolve().parents[1]/'deployment-artifacts/topdown_physical_trial_001'
    out=trial/'branch_path_continuation'
    out.mkdir(exist_ok=True)
    if (out/'report.json').exists():
        raise RuntimeError('Output already exists')
    saved=json.loads((trial/'branch_path_diagnosis/report.json').read_text())
    prep=json.loads((trial/'controller_preparation/preparation.json').read_text())
    identity=prep['before'][prep['selected_arm']]
    q0=np.asarray(saved['start_joints'])
    report=dict(started_at=datetime.now(timezone.utc).isoformat(),checks=[],
                physical_motion_commands=0,configuration_commands=0,gripper_commands=0,
                execution_authorized=False)
    robot=XArmAPI(identity['ip'],is_radian=False,do_not_open=True)

    def stationary():
        q=value(robot.get_servo_angle(is_radian=False),'joints')
        if (np.max(np.abs(np.asarray(q)-q0))>.1 or robot.get_state()!=(0,2)
            or robot.get_err_warn_code()!=(0,[0,0]) or robot.mode!=0):
            raise RuntimeError('Live state changed')
        return q

    def record(label,code,start,**extra):
        row=dict(label=label,code=code,detail=robot.only_check_result,
                 elapsed_seconds=time.monotonic()-start,**extra)
        report['checks'].append(row)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        stationary()
        if code!=0 or row['detail']!=0:
            raise RuntimeError('Path check failed; no additional path checks queued')

    try:
        robot.connect();time.sleep(.7)
        value(robot.get_robot_sn(),'identity')
        if robot.control_box_sn!=identity['control_box_sn'] or robot.axis!=7:
            raise RuntimeError('Identity mismatch')
        if not np.allclose(robot.tcp_offset,identity['tcp_offset_mm_deg'],atol=.1,rtol=0):
            raise RuntimeError('Tool offset changed')
        if not np.allclose(robot.world_offset,identity['world_offset_mm_deg'],atol=.01,rtol=0):
            raise RuntimeError('World offset changed')
        stationary()
        start=time.monotonic()
        code=robot.set_servo_angle(angle=q0.tolist(),speed=8,mvacc=10,is_radian=False,
            wait=False,only_check_type=1)
        record('baseline',code,start)
        approach=next(s for s in reversed(saved['samples']) if s['phase']=='approach')
        start=time.monotonic()
        code=robot.set_servo_angle(angle=approach['joints_deg'],speed=8,mvacc=10,
            is_radian=False,wait=False,only_check_type=1)
        record('preposition_approach',code,start,joints_deg=approach['joints_deg'])
        samples=[s for s in saved['samples'] if s['phase'] in ('grasp','lifted','across','placed','retreat')]
        settings={o['phase']:o for o in prep['operations'] if o['kind']=='move_tcp'}
        for i,sample in enumerate(samples):
            op=settings[sample['phase']]
            stationary();start=time.monotonic()
            code=robot.set_position(*sample['pose'],speed=op['speed_mm_s'],
                mvacc=op['acceleration_mm_s2'],is_radian=False,wait=False,
                motion_type=0,only_check_type=3)
            record(f"{sample['phase']}_{i}",code,start,pose=sample['pose'])
        start=time.monotonic()
        code=robot.set_servo_angle(angle=q0.tolist(),speed=8,mvacc=10,is_radian=False,
            wait=False,only_check_type=2)
        record('restore_virtual',code,start)
        report['status']='FULL_CANDIDATE_CONTROLLER_CHECK_PASSED_NOT_EXECUTION_APPROVAL'
    except Exception as exc:
        report.update(status='STOPPED',reason=str(exc))
    finally:
        try:
            report['end_joints']=value(robot.get_servo_angle(is_radian=False),'final joints')
            report['maximum_joint_change_deg']=float(np.max(np.abs(np.asarray(report['end_joints'])-q0)))
            report['final_state']=robot.get_state();report['final_errors']=robot.get_err_warn_code()
        finally:
            robot.disconnect()
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with urllib.request.urlopen(f"http://{identity['ip']}:18333/handler/log/download",timeout=60) as response:
        (out/'controller_log_export.tar.gz').write_bytes(response.read())
    print(report['status'],flush=True)


if __name__=='__main__':
    main()
