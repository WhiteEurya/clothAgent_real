import copy
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cloth_agent.dual_arm import homing
from cloth_agent.dual_arm.execution import XArmConnection
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.sequential_home import (
    certify_segment, execute_sequential_home, prepare_segments,
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
