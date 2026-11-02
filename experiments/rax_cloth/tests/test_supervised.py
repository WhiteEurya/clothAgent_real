"""Fault-injection tests, no SDK/robot connection and no physical success claims."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from cloth_agent.gamepad_teleop import InputGuard,Intent
from rax_cloth.supervised import Supervisor,motion_source,validate_request,real_tools


class Port:
    def __init__(self):self.executed=[];self.stops=0;self.proof='SAFE';self.expired=False;self.plan_hook=None
    def state(self):return {'live':True}
    def validate_observation(self,obs,camera):
        if self.expired:raise ValueError('observation expired')
    def plan(self,source,obs):
        if self.plan_hook:self.plan_hook()
        return {'status':self.proof,'operations':['certified-operation']}
    def execute(self,plan,folder):self.executed.append(plan);return {'status':'EXECUTED','physical_execution':True}
    def stop(self):self.stops+=1;return {'left':'sent','right':'sent'}
    def close(self):pass


@pytest.fixture
def service(tmp_path):
    p=Port();guard=InputGuard();s=Supervisor(p,guard,tmp_path)
    s.heartbeat('owner');s.observations['frame']=(object(),'camA')
    yield s,p,guard
    s.close()


def request():return {'arm':'arm6','joints_deg':[0]*6,'observation_id':'frame'}


def test_authorization_is_never_agent_controlled(service):
    s,p,g=service
    assert not s.status()['armed']
    assert s.call('owner','move_joints',request())['status']=='NOT_EXECUTED'
    with pytest.raises(ValueError):s.call('owner','arm',{})
    assert not p.executed


@pytest.mark.parametrize('proof',['UNSAFE','UNKNOWN'])
def test_non_safe_never_dispatches_and_stops(service,proof):
    s,p,g=service;g.update(Intent(enabled=True));p.proof=proof
    r=s.call('owner','move_joints',request())
    assert r['safety']==proof and r['status']=='REFUSED'
    assert p.stops==1 and not p.executed and s.stopped


def test_release_repress_during_planning_invalidates_request(service):
    s,p,g=service;g.update(Intent(enabled=True))
    def changed():g.update(Intent());g.update(Intent(enabled=True))
    p.plan_hook=changed
    assert s.call('owner','move_joints',request())['status']=='REFUSED'
    assert not p.executed and s.stopped


def test_stale_observation_cannot_execute(service):
    s,p,g=service;g.update(Intent(enabled=True));p.expired=True
    assert s.call('owner','move_joints',request())['status']=='REFUSED'
    assert not p.executed and s.stopped


def test_fresh_safe_authorized_dispatch_and_invalidation(service):
    s,p,g=service;g.update(Intent(enabled=True))
    result=s.call('owner','move_joints',request())
    assert result['status']=='EXECUTED' and len(p.executed)==1
    assert result['physical_actions']==1 and not s.observations
    assert result['task_success'] is None


def test_watchdog_stops_active_motion_on_deadman_loss(service):
    s,p,g=service;g.update(Intent(enabled=True));s.active=True;s.generation=g.token()
    g.update(Intent())
    deadline=time.monotonic()+1
    while not s.stopped and time.monotonic()<deadline:time.sleep(.01)
    assert s.stopped and p.stops==1
    g.update(Intent(enabled=True));assert not s.permitted() # latched; no auto resume


def test_host_disconnect_stops_even_without_active_motion(service):
    s,p,g=service;s.ping_at=time.monotonic()-2
    deadline=time.monotonic()+1
    while not s.stopped and time.monotonic()<deadline:time.sleep(.01)
    assert s.stopped and p.stops==1
    with pytest.raises(PermissionError):s.heartbeat('owner')


def test_single_owner_and_no_mode_switch(service):
    s,p,g=service
    with pytest.raises(PermissionError):s.heartbeat('other')
    with pytest.raises(ValueError):validate_request('move_joints',{**request(),'mode':'unsafe'})
    assert not p.executed


def test_units_and_zero_table_contact_are_unchanged():
    from cloth_agent.codex_motion import parse_code
    cmd=motion_source('move_tcp',{'arm':'arm6','pose_mm_deg':[100,200,0,180,0,90],'observation_id':'frame'})
    name,args=parse_code(cmd)[0]
    assert name=='move_tcp' and args['left'][2]==0 and args['left'][0]==.1
    assert args['left'][3]==pytest.approx(3.141592653589793)
    assert 'arm' not in {t['name'] for t in real_tools()}


def test_real_workspace_has_correct_contract_and_empty_new_library(tmp_path):
    from rax_cloth.official import configure,prepare
    _,config=configure(tmp_path/'run','real_supervised',control_socket=tmp_path/'operator/control.sock')
    ws=prepare(config)
    assert 'Supervised real mode' in (ws/'AGENTS.md').read_text()
    text=(ws/'.agents/skills/robot-interface/SKILL.md').read_text()
    assert 'observation_id' in text and 'Z=0' in text
    assert not list((ws/'skill-library/skills').glob('*/SKILL.md'))
    contract=json.loads((ws/'MCP_CONTRACT.json').read_text())
    assert any(t['name']=='observe' for t in contract['supervised_backend_tools'])


def test_real_requires_explicit_operator_service(tmp_path):
    from rax_cloth.official import configure
    with pytest.raises(PermissionError):configure(tmp_path/'run','real_supervised')


def test_rpc_authentication_and_whitelisted_primitives(service):
    from rax_cloth.control_service import Handler
    from test_codex_agent import LocalSocket
    s,p,g=service
    server=SimpleNamespace(secret='test-capability',supervisor=s)
    def send(packet):
        sock=LocalSocket(json.dumps(packet).encode()+b'\n')
        Handler(sock,('local',0),server)
        return json.loads(sock.output)
    assert not send({'token':'wrong','owner':'owner','op':'call','tool':'move_joints','arguments':request()})['ok']
    assert not p.executed
    assert send({'token':'test-capability','owner':'owner','op':'heartbeat'})['ok']
    assert not send({'token':'test-capability','owner':'owner','op':'call','tool':'eval','arguments':{}})['ok']
    assert not p.executed


def test_readonly_observation_camera_and_pixel_mapping(service,tmp_path):
    s,p,g=service
    def capture(camera,folder):
        folder.mkdir(parents=True);(folder/'rgb.png').write_bytes(b'fake image fixture')
        return SimpleNamespace(meta={'observation_id':camera+'-frame','captured_unix_s':time.time(),'calibration_id':camera+'-cal'},
                               sample=lambda pixel,spread:SimpleNamespace(tolist=lambda:[1,2,0]))
    p.capture=capture
    for camera in ('camA',):
        obs=s.call('owner','observe',{'camera':camera})
        point=s.call('owner','pixel_to_world',{'pixel':[10,10],'observation_id':obs['observation_id']})
        assert point['calibration_id']==camera+'-cal' and point['world_xyz_mm'][2]==0
    assert not p.executed


def test_native_port_delegates_without_height_rewriting(tmp_path):
    from rax_cloth.supervised import NativePort
    from unittest.mock import Mock
    p=NativePort.__new__(NativePort);p.runtime=Mock()
    p.runtime.plan.return_value=['original-certified-operation']
    obs=SimpleNamespace(meta={'joints_deg':{'left':[0]*6,'right':[0]*7}})
    source=motion_source('move_tcp',{'arm':'arm6','pose_mm_deg':[100,200,0,0,0,0],'observation_id':'frame'})
    proof=p.plan(source,obs)
    p.runtime.plan.assert_called_once_with(source,obs.meta['joints_deg'],preserve_command_boundaries=True)
    p.execute(proof,tmp_path)
    p.runtime._execute_operations.assert_called_once_with(proof['operations'],tmp_path)
    p.runtime.execute.assert_not_called()
    p.runtime.observe.assert_not_called()


def test_real_backend_stop_is_idempotent_and_keeps_final_counts():
    from unittest.mock import Mock
    from rax_cloth.real_backend import RealBackend
    b=RealBackend.__new__(RealBackend);b.closed=False;b.client=Mock();b.final_status=None
    b.client.close.return_value={'physical_actions':2,'stopped':True}
    b.close();b.close()
    b.client.close.assert_called_once()
    assert b.status()['physical_actions']==2


def test_busy_service_never_queues_action_for_later_authorization(service):
    s,p,g=service
    with s.lock:
        with pytest.raises(PermissionError,match='never queued'):s.call('owner','move_joints',request())
    assert not p.executed


def test_missing_operator_service_explains_startup_without_creating_capability(tmp_path):
    from rax_cloth.control_service import Client,ServiceUnavailable
    path=tmp_path/'operator/control.sock'
    with pytest.raises(ServiceUnavailable,match='终端 1'):
        Client(path,'test-session')
    assert not path.parent.exists()


def test_capability_without_live_socket_is_not_ready(tmp_path):
    from rax_cloth.control_service import Client,ServiceUnavailable
    (tmp_path/'capability').write_text('test-only')
    with pytest.raises(ServiceUnavailable,match='尚未就绪'):
        Client(tmp_path/'control.sock','test-session')
