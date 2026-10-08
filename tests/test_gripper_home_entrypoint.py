from concurrent.futures import ThreadPoolExecutor

import pytest

from cloth_agent.dual_arm import homing
from scripts import gripper_home as script
from .test_dual_arm_runtime import scene


def test_function_defaults_to_real_execution(monkeypatch):
    calls = []
    def run(args):
        calls.append(args)
        return {'status': 'COMPLETED'}
    monkeypatch.setattr(homing, '_run_home', run)
    assert homing.gripper_home()['status'] == 'COMPLETED'
    assert calls[0].real and calls[0].confirm_real
    assert calls[0].config.name == 'dual_arm.local.json'
    homing.gripper_home(simulated=True)
    assert not calls[1].real and not calls[1].confirm_real


@pytest.mark.parametrize('args,simulated', [([],False), (['--simulate'],True),
                                          (['--real','--confirm-real'],False)])
def test_script_dispatch_without_confirmation(monkeypatch,args,simulated):
    calls = []
    def run(*args,**kwargs):
        calls.append(kwargs)
        return {'status':'COMPLETED'}
    monkeypatch.setattr(script,'gripper_home',run)
    assert script.main(args) == 0
    assert calls[0]['simulated'] is simulated


def test_script_preserves_failure_exit_status(monkeypatch):
    monkeypatch.setattr(script,'gripper_home',lambda *a,**kw: {'status':'FAILED'})
    assert script.main([]) == 1


def test_function_runs_simulated_home_from_worker_thread(scene,monkeypatch,tmp_path):
    import json
    from cloth_agent.dual_arm import cli,kinematics,preview
    config=tmp_path/'config.json'
    config.write_text(json.dumps(scene.config.raw))
    for key,connection in scene.connections.items():
        connection.joints=scene.config.arms[key].home_joints.copy()
        connection.joints[2] += 2
    monkeypatch.setattr(kinematics,'ArmModel',lambda arm:scene.models[arm.arm_id])
    def connect(config,models,cancel,real):
        assert real is False
        return scene.connections
    monkeypatch.setattr(cli,'connect',connect)
    monkeypatch.setattr(preview,'write_preview',lambda *a,**kw:None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result=pool.submit(homing.gripper_home,config,output=tmp_path/'home',
                           simulated=True,project_root=scene.config.root).result(timeout=15)
    assert result['status']=='COMPLETED'
    assert result['physical_execution'] is False
    assert result['output_directory']==str(tmp_path/'home')
