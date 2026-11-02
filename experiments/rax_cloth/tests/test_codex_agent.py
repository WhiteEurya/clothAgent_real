"""Offline acceptance checks; no provider connection, SDK or generated host code."""
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import tomli
from rax_cloth import codex_agent as agent
from rax_cloth import responses_relay as transport


def relay(tmp_path, **kw):
    auth=tmp_path/'host-auth.json'
    auth.write_text(json.dumps({'auth_mode':'chatgpt','tokens':{'access_token':'FAKE-TEST-TOKEN','account_id':'fake-account'}}))
    return transport.Relay(tmp_path/'relay',auth,'test-model',kw.pop('max_calls',2),
                           kw.pop('deadline',time.monotonic()+60),**kw)


def payload(**kw):
    return dict(model='test-model',stream=True,store=False,tools=[],**kw)


def test_call_reservations_survive_restart(tmp_path):
    r=relay(tmp_path,max_calls=1)
    assert r.reserve(payload())==0
    r=relay(tmp_path,max_calls=1)
    with pytest.raises(PermissionError):r.reserve(payload())
    assert r.state['cost_usd'] is None
    assert r.state['requests'][0]['status']=='reserved'


@pytest.mark.parametrize('change',[{'model':'other'},{'stream':False},{'store':True},
                                  {'tools':[{'type':'web_search'}]},{'tools':'bad'}])
def test_request_restrictions(tmp_path,change):
    r=relay(tmp_path);p=payload();p.update(change)
    with pytest.raises(ValueError):r.reserve(p)
    assert r.state['calls_reserved']==0


def test_time_and_dollar_caps(tmp_path):
    with pytest.raises(ValueError):relay(tmp_path,usd_budget=1)
    with pytest.raises(PermissionError):relay(tmp_path,deadline=time.monotonic()-1).reserve(payload())


class LocalSocket:
    """HTTP handler test without network sockets or bind permission."""
    def __init__(self,request):self.request=io.BytesIO(request);self.output=bytearray()
    def makefile(self,*args):return self.request
    def sendall(self,data):self.output.extend(data)
    def settimeout(self,_):pass


def invoke(r,body,path='/responses'):
    body=json.dumps(body).encode()
    raw=(f'POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {len(body)}\r\n\r\n').encode()+body
    connection=LocalSocket(raw)
    r.handler()(connection,('127.0.0.1',1234),None)
    return bytes(connection.output)


def test_stream_forwarding_fixed_destination_usage_no_credentials_logged(tmp_path,monkeypatch):
    r=relay(tmp_path)
    usage={'input_tokens':11,'output_tokens':7}
    response=Mock(status=200)
    response.getheader.return_value='text/event-stream'
    data=b'data: '+json.dumps({'response':{'usage':usage}}).encode()+b'\n\n'
    response.read1.side_effect=[data[:17],data[17:],b'']
    connection=Mock();connection.getresponse.return_value=response
    factory=Mock(return_value=connection)
    monkeypatch.setattr(transport.http.client,'HTTPSConnection',factory)
    result=invoke(r,payload())
    assert b'200 OK' in result and data in result
    assert factory.call_args.args==('chatgpt.com',)
    assert connection.request.call_args.args==('POST','/backend-api/codex/responses')
    assert r.state['requests'][0]['usage']==usage
    assert connection.request.call_args.kwargs['headers']['Authorization']=='Bearer FAKE-TEST-TOKEN'
    assert all(b'FAKE-TEST-TOKEN' not in p.read_bytes() for p in r.root.iterdir())


def test_unknown_route_cannot_reach_provider(tmp_path,monkeypatch):
    r=relay(tmp_path)
    factory=Mock();monkeypatch.setattr(transport.http.client,'HTTPSConnection',factory)
    result=invoke(r,payload(),'/http://robot/move')
    assert b'502' in result
    factory.assert_not_called()
    assert r.state['calls_reserved']==0


def test_operator_pause_blocks_new_billing_and_provider_calls(tmp_path,monkeypatch):
    import threading
    from rax_cloth.onsite_console import OnsiteConsole
    from rax_cloth.baseline import register
    register()
    r=relay(tmp_path)
    c=OnsiteConsole(tmp_path,Mock(latched=False,closed=False),threading.Event(),time.monotonic()+10,Mock())
    c.pause();c.current=c.operator.open_request(tmp_path,'reset','test')['id']
    entered=threading.Event()
    def gate():entered.set();c.wait_for_operator()
    r.before_request=gate
    response=Mock(status=200);response.getheader.return_value='text/event-stream'
    response.read1.return_value=b''
    connection=Mock();connection.getresponse.return_value=response
    factory=Mock(return_value=connection)
    monkeypatch.setattr(transport.http.client,'HTTPSConnection',factory)
    worker=threading.Thread(target=lambda:invoke(r,payload()),daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        assert r.state['calls_reserved']==0
        factory.assert_not_called()
        c.accept('\n');worker.join(2)
        assert not worker.is_alive() and r.state['calls_reserved']==1
        factory.assert_called_once()
    finally:c.stop.set();worker.join(2)


def test_config_and_mounts_do_not_expose_host_credentials(tmp_path):
    cfg=tomli.loads(agent.cli_config('test-model'))
    assert cfg['model_providers']['rax_offline']['requires_openai_auth'] is False
    assert cfg['web_search']=='disabled'
    cmd=agent.isolated_command(tmp_path/'vendor',tmp_path/'workspace',tmp_path/'channels',
                               tmp_path/'frames',tmp_path/'private','test-model')
    assert '--unshare-all' in cmd and '--share-net' not in cmd and '--clearenv' not in cmd
    mounts=[cmd[i+1:i+3] for i,a in enumerate(cmd) if a in ('--bind','--ro-bind')]
    assert all(src!='/' and 'auth.json' not in src for src,dst in mounts)
    assert not any(dst in ('/home/sja','/mnt/newssd') for src,dst in mounts)
    assert [str(tmp_path/'private/codex-config.toml'),'/home/agent/.codex/config.toml'] in mounts


def test_code_snapshots_never_follow_symlinks_or_execute(tmp_path):
    ws=tmp_path/'workspace';code=ws/'code';code.mkdir(parents=True)
    root=tmp_path/'private';root.mkdir()
    (code/'controller.py').write_text('raise RuntimeError("must never run on host")\n')
    secret=tmp_path/'not-code';secret.write_text('sensitive')
    (code/'bad.py').symlink_to(secret)
    rows=agent.archive_programs(ws,root)
    assert [r['file'] for r in rows]==['code/controller.py']
    assert rows[0]['physical_verified'] is False
    (code/'controller.py').write_text('def main(ctx): return ctx.value("observe")\n')
    new=agent.archive_programs(ws,root)
    assert new[0]['sha256']!=rows[0]['sha256']
    assert len(list((root/'programs').iterdir()))==2


def test_crash_resume_charges_time_and_recovers_thread(tmp_path):
    ep=tmp_path/'episode_0000';ep.mkdir()
    (ep/'events.jsonl').write_text('{"type":"thread.started","thread_id":"thread-1"}\n{truncated')
    state={'status':'running','elapsed_seconds':0,'thread_id':None,
           'attempts':[{'directory':ep.name,'started_unix_s':time.time()-30}]}
    agent.recover_interrupted(state,tmp_path,600)
    assert state['elapsed_seconds']>=30 and state['thread_id']=='thread-1'
    charged=state['elapsed_seconds'];agent.recover_interrupted(state,tmp_path,600)
    assert state['elapsed_seconds']==charged


def test_isolation_failure_stops_before_model_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(agent,'probe',lambda cmd:dict(passed=False,stderr='test namespace denial'))
    factory=Mock();monkeypatch.setattr(agent,'Relay',factory)
    args=SimpleNamespace(session=tmp_path/'run',mode='mock',replay_manifest=None,
                         model='test-model',max_model_calls=2,max_seconds=60,seed=0,
                         budget_usd=None,preflight_only=False,resume=False)
    with pytest.raises(PermissionError,match='isolation preflight failed'):agent.launch(args)
    factory.assert_not_called()
    state=json.loads((args.session/'codex-adaptation/agent-state.json').read_text())
    assert state['status']=='ISOLATION_FAILED' and not state['attempts']

@pytest.mark.parametrize('succeeded,errors,calls,timed,status',[
    (False,[{'message':'429 Too Many Requests'}],8,False,'interrupted_or_failed'),
    (False,[{'message':'429 Too Many Requests'}],1,False,'interrupted_or_failed'),
    (False,[],1,True,'session_time_budget_exhausted'),
    (True,[],8,False,'agent_completed'),
])
def test_completion_distinguishes_local_budget_and_provider_failure(succeeded,errors,calls,timed,status):
    assert agent.completion_outcome(succeeded,{'errors':errors},{'calls_reserved':calls},8,timed)[0]==status


def test_socket_access_requires_outer_network_namespace(tmp_path):
    assert tomli.loads(agent.cli_config('test-model'))['sandbox_workspace_write']['network_access'] is True
    cmd=agent.isolated_command(tmp_path/'vendor',tmp_path/'workspace',tmp_path/'channels',
                               tmp_path/'frames',tmp_path/'private','test-model')
    assert '--unshare-all' in cmd and '--share-net' not in cmd
    r=relay(tmp_path,max_calls=1)
    r.reserve(payload())
    with pytest.raises(PermissionError):r.reserve(payload())
    assert json.loads((r.root/'ledger.json').read_text())['last_denial']=='model_call_budget_exhausted'
    assert agent.completion_outcome(False,{'errors':[]},r.state,1)[0]=='model_call_budget_exhausted'


def test_status_reports_actual_relay_budget_without_fictitious_dollars(tmp_path):
    r=relay(tmp_path,max_calls=2)
    r.reserve(payload())
    value=agent.relay_budget_status(r)
    assert value['calls_used']==1 and value['calls_remaining']==1
    assert value['reported_cost_usd'] is None and value['session_cost_limit_usd'] is None
    assert 0<value['seconds_remaining']<=60
