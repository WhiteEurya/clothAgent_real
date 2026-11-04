"""Controller check-only validation of an exported native ZED folding path."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np

from cloth_agent.site_robot_check import value


def validate_export(artifact, config):
    active=artifact['active_arm']
    if active not in ('left','right'):
        raise ValueError('Invalid active arm')
    current={arm:np.asarray(joints,dtype=float) for arm,joints in artifact['initial_joints_deg'].items()}
    if set(current)!={'left','right'}:
        raise ValueError('Both saved joint states required')
    for arm,joints in current.items():
        if joints.shape!=(config['arms'][arm]['axis'],) or not np.isfinite(joints).all():
            raise ValueError('Invalid initial joint state')
    moves=[]
    for operation in artifact['operations']:
        data=operation['data']
        if data['arm']!=active:
            raise ValueError('Inactive arm operation')
        if operation['kind']=='gripper':
            if data['action'] not in ('open','close'):
                raise ValueError('Invalid gripper action')
            continue
        if operation['kind']!='move':
            raise ValueError('Unsupported operation')
        for phase in ('start','end'):
            if set(data[phase])!=set(current):
                raise ValueError('Both arms required in segment')
            for arm,joints in data[phase].items():
                values=np.asarray(joints,dtype=float)
                if values.shape!=current[arm].shape or not np.isfinite(values).all():
                    raise ValueError('Invalid segment joint vector')
                if (phase=='start' or arm!=active) and not np.allclose(values,current[arm],atol=1e-7,rtol=0):
                    raise ValueError('Discontinuous path or inactive arm changed')
        speed=float(data['speed_deg_s'])
        acceleration=float(data['acceleration_deg_s2'])
        if not (0<speed<=config['limits']['joint_speed_deg_s']
                and 0<acceleration<=config['limits']['joint_accel_deg_s2']):
            raise ValueError('Segment speed/acceleration outside configured bounds')
        moves.append(data)
        current={arm:np.asarray(joints) for arm,joints in data['end'].items()}
    if not moves:
        raise ValueError('No movement to check')
    return moves


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output already exists')
    artifact=json.loads(args.operations.read_text())
    config_path=Path(artifact['config_path'])
    if hashlib.sha256(config_path.read_bytes()).hexdigest()!=artifact['config_sha256']:
        raise ValueError('Config changed since offline path proof')
    config=json.loads(config_path.read_text())
    moves=validate_export(artifact,config)
    active=artifact['active_arm']
    expected=config['arms'][active]
    initial=np.asarray(artifact['initial_joints_deg'][active])
    report=dict(started_at=datetime.now(timezone.utc).isoformat(),active_arm=active,
        physical_motion_commands=0,gripper_commands=0,configuration_commands=0,
        execution_allowed=False,source_operations=str(args.operations.resolve()),
        operations_sha256=hashlib.sha256(args.operations.read_bytes()).hexdigest(),checks=[])
    robot=None
    def stationary():
        joints=value(robot.get_servo_angle(is_radian=False),'joints')
        if np.max(np.abs(np.asarray(joints)-initial))>.1:
            raise RuntimeError('Robot moved from the saved starting state')
        if value(robot.get_state(),'state')!=2 or robot.mode!=0:
            raise RuntimeError('Robot is not stationary in position mode')
        if value(robot.get_err_warn_code(),'errors')!=[0,0]:
            raise RuntimeError('Controller fault')
        return joints
    try:
        from xarm.wrapper import XArmAPI
        with socket.create_connection((expected['ip'],502),timeout=3):pass
        robot=XArmAPI(expected['ip'],is_radian=False,do_not_open=True)
        robot.connect()
        time.sleep(.7)
        value(robot.get_robot_sn(),'identity')
        if (robot.control_box_sn!=expected['control_box_sn'] or robot.sn!=expected['robot_sn']
                or robot.axis!=expected['axis']):
            raise RuntimeError('Robot identity mismatch')
        if (not np.allclose(robot.tcp_offset,expected['tcp_offset_mm_deg'],atol=.1,rtol=0)
                or not np.allclose(robot.world_offset,0,atol=.01,rtol=0)):
            raise RuntimeError('TCP/world offset changed')
        stationary()
        for index,move in enumerate(moves):
            stationary()
            check_type=1 if index==0 else 2 if index==len(moves)-1 else 3
            began=time.monotonic()
            code=robot.set_servo_angle(angle=move['end'][active],speed=move['speed_deg_s'],
                mvacc=move['acceleration_deg_s2'],is_radian=False,wait=False,only_check_type=check_type)
            row=dict(index=index,code=code,detail=robot.only_check_result,only_check_type=check_type,
                     elapsed_s=time.monotonic()-began)
            report['checks'].append(row)
            stationary()
            if code!=0 or row['detail']!=0:
                raise RuntimeError(f'Controller rejected segment {index}: {row}')
            if index%10==0 or index==len(moves)-1:
                print(f'Controller check-only: {index+1}/{len(moves)} passed',flush=True)
        report.update(status='CONTROLLER_PATH_PASS',joint_drift_deg=float(np.max(np.abs(
            np.asarray(stationary())-initial))))
    except Exception as exc:
        report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    finally:
        if robot is not None:robot.disconnect()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:item for key,item in report.items() if key!='checks'},indent=2))
    return 0 if report['status']=='CONTROLLER_PATH_PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
