"""Isolated empty-tool timing probe; normal RobotAPI remains unchanged.

Requires both arms already Home and active gripper open. Uses existing complete
path validation, stream executor and watchdog. Never grasps cloth or retries.
"""
from dataclasses import replace
import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from cloth_agent.experiments.partitioned_unfold import ROOT, configure_runtime
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
from cloth_agent.experiments.full_grasp_adapter import build_robot_config
from cloth_agent.robot_collision import RobotCollision
from cloth_agent.robot_api import XArmBackend, _connect_xarm, _read_gripper_feedback
from cloth_agent.site_robot_check import read_arm, value


def release_progress(start_y, y):
    progress=(float(y)-float(start_y))/60.
    if not np.isfinite(progress):raise ValueError('invalid position feedback')
    return progress


def scale_timing(config, scale):
    if scale == 1:return config
    return replace(config, home_speed_deg_s=config.home_speed_deg_s*scale,
                   home_acceleration_deg_s2=config.home_acceleration_deg_s2*scale)


def fast_release_config(config):
    return replace(config, home_speed_deg_s=80., home_acceleration_deg_s2=320.)


def move_with_release(backend, observer, config, move, start_xyz, result, sample_callback=None, expected_distance_mm=60., trigger_progress=.2):
    if not .05<=trigger_progress<=.2:raise ValueError("release trigger outside trial range")
    start_xyz=np.asarray(start_xyz,float)
    delta=np.array([move['args'][k] for k in ('x','y','z')])-start_xyz
    if expected_distance_mm not in (60.,100.,150.):
        raise ValueError('unsupported release travel')
    if not np.isclose(np.linalg.norm(delta),expected_distance_mm,atol=.01):
        raise ValueError('release trajectory length does not match checked trial')
    cancel=threading.Event();finished=threading.Event();worker=None
    try:
        begin=time.monotonic()
        def monitor():
            try:
                sent=False
                while not cancel.is_set():
                    if time.monotonic()-begin>20:raise TimeoutError('release probe feedback timeout')
                    if backend._stop_requested.is_set() or backend._guard_error:raise RuntimeError('execution stopped')
                    p=value(observer.get_position(),'probe TCP feedback')
                    g=_read_gripper_feedback(observer)
                    if g['read_errors'] or g['error_code']!=0 or g['position_pulse'] is None:
                        raise RuntimeError(f'gripper feedback failed: {g}')
                    progress=float(np.dot(np.asarray(p[:3])-start_xyz,delta)/np.dot(delta,delta))
                    result['samples'].append(dict(elapsed_s=time.monotonic()-begin,pose=p,gripper=g,progress=progress))
                    if sample_callback:sample_callback(time.monotonic()-begin)
                    if not sent and progress>=trigger_progress:
                        if progress>=.8:raise RuntimeError('release trigger missed; no late release')
                        with backend._mode_stop_lock:
                            if cancel.is_set() or backend._stop_requested.is_set() or backend._guard_error:
                                raise RuntimeError('release cancelled')
                            code=observer.set_gripper_position(config.gripper_open,speed=config.gripper_speed,
                                                               wait=False,wait_motion=False)
                            backend._check('moving release',code)
                        sent=True
                        result['events'].append(dict(event='OPEN_COMMAND',elapsed_s=time.monotonic()-begin,progress=progress))
                    if sent and abs(g['position_pulse']-config.gripper_open)<=config.gripper_open_tolerance_pulse:
                        result['events'].append(dict(event='OPEN_MEASURED',elapsed_s=time.monotonic()-begin,progress=progress))
                        finished.set();return
                    time.sleep(.05)
            except BaseException as exc:
                result['monitor_error']=f'{type(exc).__name__}: {exc}'
                backend.stop();finished.set()
        worker=threading.Thread(target=monitor,name='empty-release-feedback')
        worker.start()
        result['motion']=backend.joint_sequence(config,[move])[1]
        if not finished.wait(5):raise TimeoutError('no measured opening completion')
        if result.get('monitor_error'):raise RuntimeError(result['monitor_error'])
    finally:
        cancel.set()
        if worker:worker.join(timeout=10)
        if worker and worker.is_alive():
            backend.stop()
            raise RuntimeError('release worker did not terminate')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute-empty',action='store_true')
    ap.add_argument('--timing-scale',type=int,choices=(1,2),default=1)
    ap.add_argument('--fast-release',action='store_true')
    args=ap.parse_args()
    root=ROOT/'reports'/('moving_release_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    root.mkdir(parents=True)
    r=configure_runtime(PartitionedReplayRuntime(ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json',planning_only=True))
    backend=observer=None;worker=None;cancel=threading.Event();finished=threading.Event()
    result={'status':'PREPARING','empty_tool_only':True,'samples':[],'events':[]}
    def save():
        (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        initial={k:np.array(a.home_joints) for k,a in r.config.arms.items()}
        if args.execute_empty:
            rows={k:read_arm(r.profile,k) for k in initial}
            for k,row in rows.items():
                if row['status']!='READ_ONLY_CHECKS_PASS' or row['control_box_sn']!=r.config.arms[k].serial:
                    raise ValueError(f'{k}: device not ready/identity mismatch')
                if np.max(np.abs(np.array(row['joints_deg'])-initial[k]))>r.config.limits['stationary_tolerance_deg']:
                    raise ValueError(f'{k}: must already be Home')
            initial={k:np.array(row['joints_deg']) for k,row in rows.items()}
        pose=r.models['left'].forward(initial['left'])
        config=build_robot_config(r.config.raw,{},'left',*pose[3:5],config_path=r.config_path)
        config=scale_timing(config,args.timing_scale)
        result['timing_scale']=args.timing_scale
        if not config.continuous_joint_stream:raise ValueError('probe requires existing checked continuous stream')
        yaw=config.relative_yaw_from_absolute_deg(float(pose[5]))
        move={'name':'move','args':dict(x=float(pose[0]),y=float(pose[1]+60),z=float(pose[2]),yaw=yaw)}
        actions=[{'name':'close_gripper','args':{}},move,{'name':'open_gripper','args':{}},{'name':'home','args':{}}]
        collision=RobotCollision(r.config_path,initial,'left',root=r.config.root)
        release_config=fast_release_config(config) if args.fast_release else config
        collision.require_execution(release_config,actions)
        result['release_timing']=dict(speed_deg_s=release_config.home_speed_deg_s,acceleration_deg_s2=release_config.home_acceleration_deg_s2)
        result.update(status='PATH_CHECKED',actions=actions,start_pose=pose.tolist(),physical_execution=False)
        save()
        if not args.execute_empty:return
        from xarm.wrapper import XArmAPI
        observer=_connect_xarm(XArmAPI,config.robot_ip)
        feedback=_read_gripper_feedback(observer)
        if (feedback['read_errors'] or feedback['error_code']!=0 or feedback['position_pulse'] is None
                or abs(feedback['position_pulse']-config.gripper_open)>config.gripper_open_tolerance_pulse):
            raise ValueError('active tool must already be fully open; no cloth handling in this probe')
        backend=XArmBackend(config,collision_checker=collision)
        result['physical_execution']=True
        backend.close_gripper(config)
        move_with_release(backend,observer,release_config,move,pose[:3],result,trigger_progress=.1 if args.fast_release else .2)
        backend.home(config)
        measured=next(e for e in result['events'] if e['event']=='OPEN_MEASURED')
        result['status']='COMPLETED'
        result['fully_open_before_endpoint']=measured['progress']<.95
        result['cloth_release_time_measured']=False
    except BaseException as exc:
        result.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        if backend:backend.stop()
    finally:
        cancel.set()
        if worker:worker.join(timeout=10)
        if worker and worker.is_alive():
            result.update(status='ERROR',error='release worker did not terminate')
            if backend:backend.stop()
        if observer:observer.disconnect()
        if backend:backend.close()
        r.close();save()
        print(json.dumps(dict(output=str(root),status=result['status'],error=result.get('error'),events=result['events']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
