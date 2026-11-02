import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rax_cloth.baseline import configure,prepare,register
from rax_cloth.xarm_backend import XArmBackend


def test_upstream_workflow_and_registry_are_used(tmp_path):
    _,cfg=configure(tmp_path/'run','mock');ws=prepare(cfg)
    from services.workspace import build,recipe_for
    from services.workspace.setup import Setup
    from services.backends import make_backend
    expected=tmp_path/'upstream-render';build(recipe_for(),Setup.from_config(cfg,gemini=False),expected)
    assert (ws/'START_PROMPT.md').read_bytes()==(expected/'START_PROMPT.md').read_bytes()
    for name in ('autonomous-control','manual-robot-control','experience-memory','continual-learning'):
        relative='.agents/skills/'+name+'/SKILL.md'
        assert (ws/relative).read_bytes()==(expected/relative).read_bytes()
    b=make_backend(cfg)
    assert isinstance(b,XArmBackend)
    try:
        value,images=b.call('xarm_observe',{'camera':'camA'})
        assert value['provenance']=='synthetic_interface_fixture'
        result,_=b.call('xarm_move_joints',{'arm':'arm6','joints_deg':[0]*6,'observation_id':value['observation_id']})
        assert result['status']=='NOT_EXECUTED' and not b.status()['hardware_connected']
    finally:b.close()
    b=make_backend(cfg)
    try:b.call('xarm_observe',{'camera':'camA'})
    finally:b.close()
    assert len(list((cfg.root/'observations').iterdir()))==2


def live(tmp_path):
    register()
    cfg=SimpleNamespace(root=tmp_path,backend_options={'mode':'real_supervised','robot_config':'operator-owned.json'})
    port=Mock();port.plan.return_value={'status':'SAFE','operations':['certified']}
    port.execute.return_value={'status':'EXECUTED','physical_execution':True}
    factory=Mock(return_value=port);b=XArmBackend(cfg,port_factory=factory)
    b.observations['frame']=(SimpleNamespace(meta={}), 'camA')
    return b,port,factory


def action():return {'arm':'arm6','joints_deg':[0]*6,'observation_id':'frame'}


def test_upstream_arming_refuses_before_connection(tmp_path):
    b,p,f=live(tmp_path)
    try:
        status=b.status()
        assert status['safety']=='UNKNOWN'
        assert status['safety_scope']=='per_proposed_action'
        assert status['connection_state']=='not_yet_requested'
        with pytest.raises(PermissionError,match='services.arming'):b.call('xarm_move_joints',action())
        f.assert_not_called();p.execute.assert_not_called()
    finally:b.close()


def test_authorized_safe_uses_existing_driver(tmp_path):
    b,p,f=live(tmp_path)
    try:
        b.arming.write(tmp_path,True,minutes=1)
        r,_=b.call('xarm_move_joints',action())
        assert r['status']=='EXECUTED' and b.actions==1
        p.plan.assert_called_once();p.execute.assert_called_once()
    finally:b.close()


@pytest.mark.parametrize('proof',['UNKNOWN','UNSAFE'])
def test_original_tri_state_never_relaxed(tmp_path,proof):
    b,p,f=live(tmp_path);p.plan.return_value={'status':proof,'operations':[]}
    try:
        b.arming.write(tmp_path,True,minutes=1)
        with pytest.raises(Exception):b.call('xarm_move_joints',action())
        p.execute.assert_not_called();p.stop.assert_called()
        assert not b.arming.read(tmp_path)['armed']
    finally:b.close()


def test_upstream_disarm_during_plan_prevents_dispatch(tmp_path):
    b,p,f=live(tmp_path)
    def plan(*args):
        b.arming.write(tmp_path,False)
        return {'status':'SAFE','operations':['certified']}
    p.plan.side_effect=plan
    try:
        b.arming.write(tmp_path,True,minutes=1)
        with pytest.raises(PermissionError):b.call('xarm_move_joints',action())
        p.execute.assert_not_called()
    finally:b.close()


def test_watchdog_reads_official_arm_file(tmp_path):
    b,p,f=live(tmp_path)
    try:
        b.port=p;b.arming.write(tmp_path,True,minutes=1)
        b.arm_generation=b._generation(b._arm_record());b.active=True
        b.arming.write(tmp_path,False)
        end=time.monotonic()+1
        while not p.stop.called and time.monotonic()<end:time.sleep(.01)
        assert b.latched;p.stop.assert_called()
    finally:b.close()


@pytest.mark.parametrize('fails',[False,True])
def test_execution_records_survive_backend_restart(tmp_path,fails):
    b,p,f=live(tmp_path)
    if fails:p.execute.side_effect=RuntimeError('lost execution response')
    try:
        b.arming.write(tmp_path,True,minutes=1)
        if fails:
            with pytest.raises(RuntimeError,match='lost execution'):b.call('xarm_move_joints',action())
        else:b.call('xarm_move_joints',action())
        before=b.status()
        assert before['physical_action_attempts']==1
        assert before['physical_actions']==int(not fails)
        assert before['unresolved_physical_action_attempts']==int(fails)
        assert not b.observations
    finally:b.close()
    fresh,p2,f2=live(tmp_path)
    try:
        after=fresh.status()
        for key in ('physical_actions','physical_action_attempts','robot_execution_seconds',
                    'unresolved_physical_action_attempts'):
            assert after[key]==before[key]
        assert not after['armed']
        f2.assert_not_called()
    finally:fresh.close()


def test_incomplete_dispatch_is_unknown_not_replayed(tmp_path):
    from rax_cloth.session import write_json
    folder=tmp_path/'actions'/'interrupted';folder.mkdir(parents=True)
    write_json(folder/'dispatch.json',{'meaning':'synthetic crash fixture'})
    b,p,f=live(tmp_path)
    try:
        state=b.status()
        assert state['physical_action_attempts']==1
        assert state['physical_actions']==0
        assert state['unresolved_physical_action_attempts']==1
        assert state['task_success'] is None
        f.assert_not_called()
    finally:b.close()


def test_readonly_close_disconnects_without_sending_stop(tmp_path):
    b,p,f=live(tmp_path)
    b.port=p
    b.close()
    p.stop.assert_not_called()
    p.close.assert_called_once()
    assert not b.arming.read(tmp_path)['armed']


def test_close_after_dispatch_still_stops(tmp_path):
    b,p,f=live(tmp_path)
    b.port=p;b.attempts=1
    b.close()
    p.stop.assert_called_once()
    p.close.assert_called_once()
