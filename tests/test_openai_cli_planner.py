import base64
import json
import subprocess
from pathlib import Path

import pytest
from cloth_agent.experiments import openai_cli_planner as cli


def body():
    return {'model':'gpt-6-astra','reasoning':{'effort':'high'},'instructions':'full skills', 'tools':[],
            'input':[{'role':'user','content':[{'type':'input_text','text':'current'},
              {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(b'png').decode()},
              {'type':'input_text','text':'goal'}]}],
            'text':{'format':{'schema':{'type':'object','properties':{'decision':{'enum':['DONE']}},'required':['decision'],'additionalProperties':False}}}}


def fake_process(monkeypatch, *, tool=False, invalid=False, timeout=False, reconnect=False, failure=None):
    class Process:
        pid=123
        returncode=None
        def __init__(self, command, **kwargs):
            assert command[command.index('-p')+1]=='openai'
            assert 'model_reasoning_effort="high"' in command
            self.output=Path(command[command.index('-o')+1]);self.events=kwargs['stdout']
        def communicate(self,prompt,timeout=None):
            assert prompt.index('current') < prompt.index('[Attached image 1]') < prompt.index('goal')
            if timeout_flag:raise subprocess.TimeoutExpired('codex',1)
            if failure:
                events=[{'type':'turn.started'},{'type':'error','message':failure},
                        {'type':'turn.failed','error':{'message':failure}}]
                self.events.write('\n'.join(map(json.dumps,events)));self.returncode=1
                return
            self.output.write_text(json.dumps({'decision':'WRONG' if invalid else 'DONE'}))
            events=[{'type':'turn.completed','usage':{'input_tokens':42}}]
            if reconnect:events.insert(0,{'type':'error','message':'Reconnecting... 2/5 (stream disconnected before completion: Connection reset by peer (os error 104))'})
            if tool:events.append({'type':'item.completed','item':{'type':'command_execution'}})
            self.events.write('\n'.join(map(json.dumps,events)));self.returncode=0
        def poll(self):return self.returncode
        def wait(self,timeout=None):self.returncode=-15
    timeout_flag=timeout
    monkeypatch.setattr(cli.subprocess,'Popen',Process)


def test_success_and_original_payload(tmp_path,monkeypatch):
    fake_process(monkeypatch)
    value,metrics=cli.direct_once(tmp_path,body(),{},10)
    assert value=={'decision':'DONE'} and metrics['status']=='VALID_JSON'
    assert json.loads((tmp_path/'request_payload.json').read_text())==body()
    assert metrics['model_response_count']==1 and metrics['tool_calls']==0


@pytest.mark.parametrize('case',['tool','invalid'])
def test_invalid_output_not_dispatched(tmp_path,monkeypatch,case):
    fake_process(monkeypatch,**{case:True})
    with pytest.raises(Exception):cli.direct_once(tmp_path,body(),{},10)
    assert json.loads((tmp_path/'model_metrics.json').read_text())['status']=='FAILED'


def test_timeout_stops_cli_group(tmp_path,monkeypatch):
    fake_process(monkeypatch,timeout=True);killed=[]
    monkeypatch.setattr(cli.os,'killpg',lambda pid,sig:killed.append(pid))
    with pytest.raises(subprocess.TimeoutExpired):cli.direct_once(tmp_path,body(),{},1)
    assert killed==[123]
    assert 'TimeoutExpired' in json.loads((tmp_path/'model_metrics.json').read_text())['error']


def test_recovered_transport_error_accepts_final_schema_valid_response(tmp_path,monkeypatch):
    fake_process(monkeypatch,reconnect=True)
    value,metrics=cli.direct_once(tmp_path,body(),{},10)
    assert value=={'decision':'DONE'} and metrics['status']=='VALID_JSON'
    assert metrics['transport_reconnect_events']==1
    assert metrics['cli_attempts']==1 and metrics['retries']==0


def test_failed_cli_preserves_rate_limit_detail(tmp_path,monkeypatch):
    fake_process(monkeypatch,failure='exceeded retry limit, last status: 429 Too Many Requests')
    with pytest.raises(RuntimeError,match='429 Too Many Requests'):
        cli.direct_once(tmp_path,body(),{},10)
    metrics=json.loads((tmp_path/'model_metrics.json').read_text())
    assert metrics['service_error'].endswith('429 Too Many Requests')


@pytest.mark.parametrize('events',[
    [{'type':'error','message':'Reconnecting... 2/5 (stream disconnected before completion: reset)'}],
    [{'type':'turn.completed'},{'type':'error','message':'Reconnecting... 2/5 (stream disconnected before completion: reset)'}],
    [{'type':'error','message':'authentication failed'},{'type':'turn.completed'}],
    [{'type':'turn.failed'},{'type':'turn.completed'}],
    [{'type':'turn.completed'},{'type':'turn.completed'}],
])
def test_unresolved_errors_and_multiple_turns_still_rejected(events):
    with pytest.raises(ValueError):cli.validate_events(events)
