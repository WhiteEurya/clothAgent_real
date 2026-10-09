import copy
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cloth_agent.dual_arm import homing
from cloth_agent.dual_arm.execution import XArmConnection
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.sequential_home import (
    certify_segment, command_limits, execute_sequential_home, prepare_segments,
)
from .test_dual_arm_runtime import scene, coordinator
from .test_dual_arm_home import start_above_home


def native_program(scene):
    scene.config.execution_mode = 'controller_sequential'
    initial = start_above_home(scene)
    program, attempts = homing.plan_home(initial,scene.config,scene.models)
    for k,q in initial.items():
        scene.connections[k].joints = q.copy()
    return initial,program,attempts


def test_native_home_sends_one_position_command_per_direct_arm_and_no_servo(scene,monkeypatch):
    initial,program,attempts = native_program(scene)
    assert attempts[-1]['strategy'] == 'joint_left_then_right'
    segments = prepare_segments(program,scene.config,scene.models)
    assert [s.arm for s in segments] == ['left','right']
    order = []
    for k,connection in scene.connections.items():
        original = connection.move_joint_position
        def move(*args, key=k, fn=original):
            order.append(key+'_start')
            fn(*args)
            order.append(key+'_complete')
        monkeypatch.setattr(connection,'move_joint_position',move)
        monkeypatch.setattr(connection,'servo',lambda *_:pytest.fail('must not stream servo targets'))
        monkeypatch.setattr(connection,'prepare',lambda:None)
    c = coordinator(scene)
    try:
        result = execute_sequential_home(c,program,initial)
        assert result['status']=='COMPLETED', result
        assert order==['left_start','left_complete','right_start','right_complete']
        for k,connection in scene.connections.items():
            assert [cmd[0] for cmd in connection.commands]==['position','gripper']
            assert np.allclose(connection.joints,scene.config.arms[k].home_joints)
    finally:
        c.close()


def test_native_failure_never_starts_second_arm_or_opens_grippers(scene,monkeypatch):
    initial,program,_ = native_program(scene)
    def fail(*args):
        raise DualArmError('injected native command failure')
    monkeypatch.setattr(scene.connections['left'],'move_joint_position',fail)
    c = coordinator(scene)
    try:
        result=execute_sequential_home(c,program,initial)
        assert result['status']=='FAILED'
        assert all(connection.stopped for connection in scene.connections.values())
        assert scene.connections['right'].commands==[]
        assert not any(cmd[0]=='gripper' for cmd in scene.connections['left'].commands)
    finally:
        c.close()


def test_native_requires_arrival_before_second_arm(scene,monkeypatch):
    initial,program,_=native_program(scene)
    monkeypatch.setattr(scene.connections['left'],'move_joint_position',lambda *args:None)
    c=coordinator(scene)
    try:
        result=execute_sequential_home(c,program,initial)
        assert result['status']=='FAILED'
        assert scene.connections['right'].commands==[]
    finally:
        c.close()


def test_native_rejects_simultaneous_plan_before_dispatch(scene):
    initial=start_above_home(scene)
    program,_=homing.plan_home(initial,scene.config,scene.models)
    scene.config.execution_mode='controller_sequential'
    with pytest.raises(DualArmError,match='sequentially'):
        prepare_segments(program,scene.config,scene.models)


def test_joint_box_rejects_collision_away_from_nominal_joint_line(scene):
    scene.config.execution_mode='controller_sequential'
    start={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    end=copy.deepcopy(start)
    end['left'][0]+=40
    end['left'][2]+=40
    # This is off the diagonal nominal TCP path but inside the independent-joint box.
    scene.config.obstacles.append({'name':'box_corner','min_mm':[249,-121,214],
                                   'max_mm':[251,-119,216]})
    with pytest.raises(DualArmError):
        certify_segment(start,end,scene.config,scene.models)


def test_native_sdk_uses_blocking_nonblended_position_command():
    connection=object.__new__(XArmConnection)
    connection.cancel=threading.Event()
    connection.arm=SimpleNamespace(mode=0,set_servo_angle=Mock(return_value=0))
    connection.move_joint_position([0]*6,5,10,40)
    connection.arm.set_servo_angle.assert_called_once_with(
        angle=[0]*6,speed=5,mvacc=10,is_radian=False,wait=True,timeout=40,radius=-1)


def test_path_box_rejection_precedes_expensive_stopping_proof(scene, monkeypatch):
    from cloth_agent.dual_arm import sequential_home
    from cloth_agent.dual_arm.home_path import corridor
    scene.config.execution_mode='controller_sequential'
    start={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    end={k:q.copy() for k,q in start.items()};end['left'][0]+=10
    bounds=corridor(start,scene.config,scene.models)
    bounds['left']['height']=start['left'][2]+.1
    monkeypatch.setattr(sequential_home,'validate_native_sweep',
                        lambda *a,**kw:pytest.fail('unnecessary stopping proof'))
    with pytest.raises(DualArmError,match='Home height limit'):
        certify_segment(start,end,scene.config,scene.models,path_bounds=bounds)


def test_native_home_uses_joint_speed_without_global_reach_slowdown(scene):
    arm=scene.config.arms['right']
    start=arm.home_joints.copy();end=start+10
    scene.config.limits['joint_speed_deg_s']=5
    scene.config.limits['joint_accel_deg_s2']=10
    model=SimpleNamespace(capsule_reaches={'tool':np.full(7,1500.)})
    speed,accel,timeout=command_limits(scene.config,model,arm,start,end)
    assert speed==5 and accel==10
    assert timeout==pytest.approx(40)
    scene.config.raw['safety']['arms']['right']['max_joint_speed_deg_s']=3
    assert command_limits(scene.config,model,arm,start,end)[0]==3


def test_streaming_executor_cannot_use_native_envelope_profile(scene):
    initial,program,_=native_program(scene)
    c=coordinator(scene)
    try:
        result=c.execute(program,initial,None,None,homing=True)
        assert result['status']=='FAILED'
        assert 'streaming executor' in result['error']
        assert all(not connection.commands for connection in scene.connections.values())
    finally:
        c.close()


def test_native_snapshot_allows_motion_only_during_own_position_command():
    connection=object.__new__(XArmConnection)
    connection.lock=threading.RLock()
    connection.config=SimpleNamespace(axis=6,arm_id='left')
    connection.arm=SimpleNamespace(
        connected=True, get_state=lambda:(0,1), get_err_warn_code=lambda:(0,[0,0]),
        get_servo_angle=lambda **kw:(0,[0]*6),get_position=lambda **kw:(0,[0]*6))
    connection._checked_gripper_feedback=lambda:{'usable_for_completion':True}
    connection.position_command_active=True
    assert connection.snapshot()['joints']==[0]*6
    connection.position_command_active=False
    with pytest.raises(DualArmError,match='fault/state'):
        connection.snapshot()


def test_native_snapshot_pairs_joints_and_pose_from_same_report():
    connection=object.__new__(XArmConnection)
    connection.lock=threading.RLock()
    connection._report_lock=threading.Lock()
    connection._report_callback=lambda report:None
    captured=time.monotonic()
    connection._latest_report=([10.]*7,[10.]*6,captured)
    connection.config=SimpleNamespace(axis=6,arm_id='left')
    connection.arm=SimpleNamespace(
        connected=True,get_state=lambda:(0,1),get_err_warn_code=lambda:(0,[0,0]),
        get_servo_angle=lambda **kw:pytest.fail('asynchronous joints query'),
        get_position=lambda **kw:pytest.fail('asynchronous TCP query'))
    connection._checked_gripper_feedback=lambda:{'usable_for_completion':True}
    connection.position_command_active=True
    result=connection.snapshot()
    assert result['joints']==result['pose']==[10.]*6
    assert result['sampled_monotonic_s']==captured
    assert result['sample_started_monotonic_s']<=captured


def test_old_report_timestamp_still_fails_freshness_gate(scene):
    c=coordinator(scene)
    try:
        with pytest.raises(DualArmError,match='stale'):
            c.validate_fresh({'left':{'sampled_monotonic_s':time.monotonic()-1}})
    finally:
        c.close()


@pytest.mark.parametrize('fault,queued', [(False,False),(True,False),(False,True)])
def test_home_stop_recovery_requires_clear_faults_and_empty_queue(fault,queued):
    c=object.__new__(XArmConnection)
    c.cancel=threading.Event()
    c.recover_stopped=True
    c.snapshot=lambda:{}
    state=[4]
    commands=[]
    def normal(n):
        commands.append(('state',n));state[0]=n
        return 0
    c.arm=SimpleNamespace(mode=0,get_state=lambda:(0,state[0]),
        get_is_moving=lambda:False,get_cmdnum=lambda:(0,int(queued)),
        get_err_warn_code=lambda:(0,[1 if fault else 0,0]),
        motion_enable=lambda **kw:commands.append(('enable',True)) or 0,
        set_state=normal,set_gripper_mode=lambda n:0,set_gripper_enable=lambda n:0)
    if fault or queued:
        with pytest.raises(DualArmError):
            c.prepare_position()
        assert commands==[]
    else:
        c.prepare_position()
        assert commands==[('enable',True),('state',0)]
        assert not c.recover_stopped


def test_snapshot_waits_for_new_report_without_retimestamping_old_data():
    c=object.__new__(XArmConnection)
    c.lock=threading.RLock();c._report_lock=threading.Lock()
    c._report_received=threading.Event();c._report_callback=lambda r:None
    c._latest_report=([0.]*6,[0.]*6,time.monotonic()-1)
    c.config=SimpleNamespace(axis=6,arm_id='left')
    c.position_command_active=False
    c.arm=SimpleNamespace(connected=True,get_state=lambda:(0,0),
                          get_err_warn_code=lambda:(0,[0,0]))
    c._checked_gripper_feedback=lambda:{'usable_for_completion':True}
    def report():
        time.sleep(.01)
        with c._report_lock:
            c._latest_report=([1.]*6,[1.]*6,time.monotonic())
        c._report_received.set()
    worker=threading.Thread(target=report);worker.start()
    try:
        result=c.snapshot()
        assert result['joints']==result['pose']==[1.]*6
        assert result['sampled_monotonic_s']==c._latest_report[2]
    finally:
        worker.join()


def test_failed_monitor_drains_sdk_operation_after_stop(scene,monkeypatch):
    from cloth_agent.dual_arm import sequential_home
    initial,program,_=native_program(scene)
    c=coordinator(scene)
    finished=threading.Event()
    def blocking(*args):
        c.cancel.wait(2)
        time.sleep(.02)
        finished.set()
    def reject(*args):
        raise DualArmError('injected monitoring failure')
    monkeypatch.setattr(scene.connections['left'],'move_joint_position',blocking)
    monkeypatch.setattr(sequential_home,'check_feedback',reject)
    try:
        result=execute_sequential_home(c,program,initial)
        assert result['status']=='FAILED'
        assert result['operation_settled_after_stop'] is True
        assert finished.is_set()
        assert all(connection.stopped for connection in scene.connections.values())
    finally:
        c.close()


def test_native_checks_inactive_arm_during_blocking_motion(scene,monkeypatch):
    initial,program,_=native_program(scene)
    def drifting_motion(*args):
        scene.connections['right'].joints[0]+=2
        scene.cancel.wait(1)
    monkeypatch.setattr(scene.connections['left'],'move_joint_position',drifting_motion)
    c=coordinator(scene)
    try:
        result=execute_sequential_home(c,program,initial)
        assert result['status']=='FAILED'
        assert 'inactive arm moved' in result['error']
        assert scene.connections['right'].commands==[]
    finally:
        c.close()
