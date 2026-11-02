import threading
import time
from unittest.mock import Mock
from dataclasses import replace
import pytest
from rax_cloth.baseline import configure,prepare,register
from rax_cloth.onsite_console import OnsiteConsole
from rax_cloth.xarm_backend import XArmBackend


def console(tmp_path):
    register()
    backend=Mock(latched=False,closed=False)
    c=OnsiteConsole(tmp_path,backend,threading.Event(),time.monotonic()+60,Mock())
    return c,backend


def test_startup_permission_and_manual_reset_handshake(tmp_path):
    c,b=console(tmp_path)
    c.enable()
    assert c.arming.read(tmp_path)['armed']
    c.pause()
    request=c.operator.open_request(tmp_path,'reset','Reset the scene')
    c.current=request['id']
    assert not c.arming.read(tmp_path)['armed']
    c.accept('\n')
    assert c.arming.read(tmp_path)['armed']
    result=c.operator.wait(tmp_path,request,1)
    assert result['status']=='answered' and result['done']
    b.hold.assert_not_called()


@pytest.mark.parametrize('line',['','stop\n','q\n'])
def test_eof_or_stop_cancels_session(tmp_path,line):
    c,b=console(tmp_path)
    c.accept(line)
    b.hold.assert_called_once()
    assert c.stop.is_set()


def test_fault_latch_cannot_be_rearmed_by_enter(tmp_path):
    c,b=console(tmp_path)
    c.current=c.operator.open_request(tmp_path,'help','Fault')['id']
    b.latched=True
    with pytest.raises(PermissionError):c.accept('\n')
    assert not c.arming.read(tmp_path)['armed']


def test_manual_reset_not_an_agent_tool(tmp_path):
    _,cfg=configure(tmp_path/'run','mock')
    cfg=replace(cfg,backend_options={**cfg.backend_options,'manual_reset':True})
    ws=prepare(cfg);b=XArmBackend(cfg)
    try:
        assert 'xarm_reset' not in {d['name'] for d in b.tools()}
        with pytest.raises(PermissionError):b.call('xarm_reset',{'seed':1})
        text=(ws/'.agents/skills/robot-interface/SKILL.md').read_text()
        assert 'No separate arm request is needed' in text
        assert 'xarm_reset(seed=uint32)' not in text
        assert '"name": "xarm_reset"' not in (ws/'MCP_SESSION.md').read_text()
    finally:b.close()
