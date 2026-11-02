from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import pytest
from rax_cloth.baseline import configure
from rax_cloth.xarm_backend import XArmBackend
from rax_cloth.xarm_driver import validate_request
from rax_cloth.reset_adapter import run_existing_reset


@pytest.mark.parametrize('mode',['mock','replay'])
def test_reset_offline_never_connects(tmp_path,mode):
    manifest=Path(__file__).resolve().parents[1]/'configs/replay.example.json'
    _,cfg=configure(tmp_path/'run',mode,replay_manifest=manifest)
    factory=Mock(side_effect=AssertionError('no hardware'))
    b=XArmBackend(cfg,port_factory=factory)
    try:
        result,_=b.call('xarm_reset',{'seed':4})
        assert result['status']=='NOT_EXECUTED'
        assert result['physical_outcome'] is None
        factory.assert_not_called()
    finally:b.close()


def test_reset_requires_authorization_before_connect(tmp_path):
    _,cfg=configure(tmp_path/'run','real_supervised',robot_config=tmp_path/'unused')
    factory=Mock();b=XArmBackend(cfg,port_factory=factory)
    try:
        with pytest.raises(PermissionError):b.call('xarm_reset',{'seed':0})
        factory.assert_not_called()
    finally:b.close()


@pytest.mark.parametrize('seed',[True,-1,4294967296,1.5])
def test_bad_reset_seed_rejected(seed):
    with pytest.raises(ValueError):validate_request('reset',{'seed':seed})


def test_existing_recipe_delegates_only_certified_arm6_execution(tmp_path,monkeypatch):
    from cloth_agent import random_fold_reset
    from cloth_agent.codex_native_runtime import NativeRuntime
    source='motion.move_joints(left=[0,0,0,0,0,0])\n'
    operations=[('fixture','certified')]
    planner=Mock(return_value=operations);arrival=Mock()
    monkeypatch.setattr(NativeRuntime,'plan',planner)
    monkeypatch.setattr(NativeRuntime,'arrival',arrival)
    port=SimpleNamespace(runtime=SimpleNamespace(),capture=Mock())
    authorize=Mock();dispatch=Mock(return_value={'status':'EXECUTED'})
    def recipe(runtime,folder,*,seed):
        assert seed==42 and folder==tmp_path/'reset'
        planned=runtime.plan(source,{'left':[0]*6,'right':[0]*7})
        runtime._execute_operations(planned,folder/'stage')
        with pytest.raises(PermissionError,match='not certified'):
            runtime._execute_operations(planned,folder/'duplicate')
        with pytest.raises(PermissionError,match='arm6'):
            runtime.plan('motion.move_joints(right=[0,0,0,0,0,0,0])',{})
        runtime.capture_current(folder/'after')
        return {'status':'EXECUTED','outcome':'UNCERTAIN'}
    monkeypatch.setattr(random_fold_reset,'run_reset',recipe)
    result=run_existing_reset(port,tmp_path/'reset',42,authorize,dispatch)
    assert result['outcome']=='UNCERTAIN'
    dispatch.assert_called_once();assert dispatch.call_args.args[0]['status']=='SAFE'
    arrival.assert_called_once()
    port.capture.assert_called_once_with('camA',tmp_path/'reset'/'after')


def test_reset_dispatch_uses_same_safety_gate_and_clears_observations(tmp_path,monkeypatch):
    from rax_cloth import reset_adapter
    _,cfg=configure(tmp_path/'run','real_supervised',robot_config=tmp_path/'unused')
    port=Mock();b=XArmBackend(cfg,port_factory=Mock(return_value=port))
    def reset(port,folder,seed,authorize,dispatch):
        authorize()
        return dispatch({'status':'UNKNOWN','operations':['fixture']},0,folder/'stage')
    monkeypatch.setattr(reset_adapter,'run_existing_reset',reset)
    try:
        b.arming.write(cfg.root,True,minutes=1)
        b.observations['old']=('fixture','camA')
        with pytest.raises(Exception):b.call('xarm_reset',{'seed':1})
        port.execute.assert_not_called()
        assert not b.observations and b.latched
    finally:b.close()
