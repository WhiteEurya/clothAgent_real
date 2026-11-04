"""Check continuous explicit IK joint waypoints only; never executes motion."""
from pathlib import Path
from datetime import datetime, timezone
import json,time,urllib.request
import numpy as np
from xarm.wrapper import XArmAPI
from diagnose_partitioned_ik import value

def main():
    trial=Path(__file__).resolve().parents[1]/'deployment-artifacts/topdown_physical_trial_001'
    out=trial/'explicit_joint_path'
    out.mkdir(exist_ok=True)
    if (out/'report.json').exists(): raise RuntimeError('Output already exists')
    saved=json.loads((trial/'branch_path_diagnosis/report.json').read_text())
    prep=json.loads((trial/'controller_preparation/preparation.json').read_text())
    identity=prep['before'][prep['selected_arm']]
    q0=np.asarray(saved['start_joints'])
    report=dict(started_at=datetime.now(timezone.utc).isoformat(),checks=[],
                physical_motion_commands=0,configuration_commands=0,execution_authorized=False)
    robot=XArmAPI(identity['ip'],is_radian=False,do_not_open=True)
    def stationary():
        q=value(robot.get_servo_angle(is_radian=False),'joints')
        if np.max(np.abs(np.asarray(q)-q0))>.1 or robot.get_state()!=(0,2) or robot.get_err_warn_code()!=(0,[0,0]) or robot.mode!=0: raise RuntimeError('Live state changed')
    def check(q,label):
        stationary(); t=time.monotonic(); code=robot.set_servo_angle(angle=list(q),speed=8,mvacc=10,is_radian=False,wait=False,only_check_type=3)
        row=dict(label=label,code=code,detail=robot.only_check_result,elapsed_seconds=time.monotonic()-t,joints_deg=list(q));report['checks'].append(row);print(json.dumps(row),flush=True);stationary()
        if code!=0 or row['detail']!=0: raise RuntimeError('Failed explicit joint path')
    try:
        robot.connect();time.sleep(.7);value(robot.get_robot_sn(),'id')
        if robot.control_box_sn!=identity['control_box_sn'] or robot.axis!=7: raise RuntimeError('identity')
        if np.max(np.abs(np.asarray(value(robot.get_servo_angle(is_radian=False),'q'))-q0))>.1: raise RuntimeError('baseline joints')
        if not np.allclose(robot.tcp_offset,identity['tcp_offset_mm_deg'],atol=.1,rtol=0): raise RuntimeError('tcp changed')
        stationary()
        # Use all continuous IK samples, but check each target from the previous virtual joint state.
        for i,s in enumerate(saved['samples']): check(s['joints_deg'],f"{s['phase']}_{i}")
        check(q0,'restore_virtual')
        report['status']='EXPLICIT_JOINT_PATH_PASSED_NOT_EXECUTION_APPROVAL'
    except Exception as e: report.update(status='STOPPED',reason=str(e))
    finally:
        try:
            report['end_joints']=value(robot.get_servo_angle(is_radian=False),'end')
            report['maximum_joint_change_deg']=float(np.max(np.abs(np.asarray(report['end_joints'][1])-q0)))
            report['final_state']=robot.get_state();report['final_errors']=robot.get_err_warn_code()
        finally:
            robot.disconnect();(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with urllib.request.urlopen(f"http://{identity['ip']}:18333/handler/log/download",timeout=60) as r:(out/'controller_log_export.tar.gz').write_bytes(r.read())
    print(report['status'],flush=True)
if __name__=='__main__':main()
