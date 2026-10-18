"""Partial release uses measured holding width and the existing completion gate."""
import pytest
from cloth_agent.dual_arm.single_arm_backend import XArmBackend, RobotExecutionError, partial_release_target
from cloth_agent.dual_arm.execution import XArmConnection
from tests.test_gripper_completion import config, backend, sample


def copied_backend(factory,samples):
    result=factory(samples)
    result.__class__=XArmBackend
    return result


def test_partial_open_waits_for_measured_target_and_stop(config,backend):
    b=copied_backend(backend,[sample(250,2),sample(250,2),sample(300,1),sample(370,1),sample(370)])
    result,_=b._command_gripper(config,target='release_partial',opening_delta_pulse=120.)
    assert b.arm.commands==[(370.,dict(speed=config.gripper_speed,wait=False))]
    trace=result['completion']
    assert trace['before_command']['position_pulse']==250
    assert trace['target_position_pulse']==370 and trace['actual_opening_delta_pulse']==120
    assert trace['reason']=='measured_target_reached'
    assert len(trace['samples'])==4
    assert all(row['completion_reason'] is None for row in trace['samples'][:-1])


def test_partial_open_is_distinct_from_full_open_near_limit(config,backend):
    b=copied_backend(backend,[sample(700,2),sample(775)])
    result,_=b._command_gripper(config,target='release_partial',opening_delta_pulse=120.)
    assert b.arm.commands[0][0]==775<config.gripper_open
    assert result['completion']['actual_opening_delta_pulse']==75
    b.arm.samples=[sample(775),sample(850)];b.arm.index=0
    full,_=b._command_gripper(config,target='open')
    assert b.arm.commands[-1][0]==850
    assert full['completion']['reason']=='measured_target_reached'


@pytest.mark.parametrize('position',[850,830,float('nan')])
def test_insufficient_partial_release_travel_rejected(config,position):
    with pytest.raises(RobotExecutionError):partial_release_target(config,position,120.)


@pytest.mark.parametrize('delta',[None,0,-1,True,float('nan'),float('inf'),20,900])
def test_invalid_release_increment_never_sends_sdk_command(config,backend,delta):
    b=copied_backend(backend,[sample(250)])
    with pytest.raises(RobotExecutionError):b._command_gripper(config,target='release_partial',opening_delta_pulse=delta)
    assert not b.arm.commands


def test_stalled_partial_open_never_completes(config,backend):
    b=copied_backend(backend,[sample(250,2)])
    reads=[0]
    def cancel(started):
        reads[0]+=1
        if reads[0]>8:raise RobotExecutionError('injected cancellation')
    b.check_gripper_wait=cancel
    with pytest.raises(RobotExecutionError) as error:
        b._command_gripper(config,target='release_partial',opening_delta_pulse=120.)
    assert error.value.gripper_completion['status']=='FAILED'
    assert all(x['completion_reason'] is None for x in error.value.gripper_completion['samples'])


def test_connection_forwards_partial_target_to_feedback_gate(config):
    from types import SimpleNamespace
    connection=object.__new__(XArmConnection);connection.config=SimpleNamespace(gripper=config)
    seen=[]
    def command(cfg,**kwargs):
        seen.append((cfg,kwargs));return {'completion':{'status':'COMPLETED'}},None
    connection._command_gripper=command
    result=connection.gripper('release_partial',opening_delta_pulse=120.)
    assert seen==[(config,dict(target='release_partial',opening_delta_pulse=120.))]
    assert result['completion']['status']=='COMPLETED'


def test_arm6_automatic_close_dispatches_gamepad_speed(backend):
    from cloth_agent.gamepad_collect_runtime import CollectionRuntime
    runtime = CollectionRuntime('config/dual_arm.local.json')
    try:
        config = runtime.config.arms['left'].gripper
        assert config.gripper_speed == 2000
        assert runtime.config.arms['right'].gripper.gripper_speed == 500
        b = copied_backend(backend, [sample(850), sample(600,1), sample(20,2), sample(20,2), sample(20,2)])
        result, _ = b._command_gripper(config, target='close')
        assert b.arm.commands == [(config.gripper_close, dict(speed=2000., wait=False))]
        assert result['completion']['status'] == 'COMPLETED'
    finally:
        runtime.close()
