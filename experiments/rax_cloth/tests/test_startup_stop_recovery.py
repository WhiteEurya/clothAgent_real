import json
import threading
from types import SimpleNamespace
import pytest
from cloth_agent.dual_arm.execution import XArmConnection
from rax_cloth.xarm_driver import NativePort


@pytest.mark.parametrize('fault,queued,moving,cancelled',[
    (False,False,False,False),(True,False,False,False),
    (False,True,False,False),(False,False,True,False),(False,False,False,True)])
def test_startup_uses_existing_recovery_guards(tmp_path,fault,queued,moving,cancelled):
    conn=object.__new__(XArmConnection)
    conn.cancel=threading.Event()
    if cancelled:conn.cancel.set()
    conn.recover_stopped=True;conn.snapshot=lambda:{}
    state=[4];commands=[]
    def normal(value):commands.append(('state',value));state[0]=value;return 0
    conn.arm=SimpleNamespace(mode=0,get_state=lambda:(0,state[0]),
        get_is_moving=lambda:moving,get_cmdnum=lambda:(0,int(queued)),
        get_err_warn_code=lambda:(0,[11 if fault else 0,0]),
        motion_enable=lambda **kw:commands.append(('enable',True)) or 0,
        set_state=normal,set_gripper_mode=lambda n:0,set_gripper_enable=lambda n:0)
    port=NativePort.__new__(NativePort);port.root=tmp_path
    blocked=fault or queued or moving or cancelled
    if blocked:
        with pytest.raises(Exception):port._recover_startup_stop(conn)
        assert commands==[]
    else:
        port._recover_startup_stop(conn)
        assert commands==[('enable',True),('state',0)]
        assert conn.recover_stopped is False
    record=json.loads((tmp_path/'startup_stop_recovery.json').read_text())
    assert record['status']==('FAILED' if blocked else 'RECOVERED')
    assert record['motion_command_sent'] is False


def test_normal_startup_does_not_prepare_or_enable(tmp_path):
    from unittest.mock import Mock
    conn=Mock();conn.checked_value.return_value=0
    port=NativePort.__new__(NativePort);port.root=tmp_path
    port._recover_startup_stop(conn)
    conn.prepare_position.assert_not_called()
    assert conn.recover_stopped is False
