import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
from rax_cloth.backend import ClothBackend
from rax_cloth.existing import ExistingGeometry, ProtectedControllerPort
from rax_cloth.mcp import Frontend
from rax_cloth.session import Session
from rax_cloth.budget import Budget
from rax_cloth.upstream import make_backend

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]


def backend(tmp_path, **options):
    return ClothBackend(SimpleNamespace(root=tmp_path, backend_options=options))


@pytest.mark.parametrize('mode', ['sim', 'real_supervised', 'real', ''])
def test_forbidden_modes(tmp_path, mode):
    with pytest.raises(PermissionError):
        backend(tmp_path, mode=mode)


def test_upstream_registry_contract(tmp_path):
    b = make_backend(tmp_path, {'mode': 'mock'})
    assert {'tools', 'call', 'status', 'close', 'validate'} <= set(dir(b))
    assert not b.status()['hardware_connected']
    assert 'arm' not in {t['name'] for t in b.tools()}
    b.close()
    b.close()
    with pytest.raises(PermissionError):
        b.call('observe', {})


@pytest.mark.parametrize('name,args', [
    ('move_tcp', {'arm': 'arm6', 'pose_mm_deg': [1]*6, 'unsafe': True}),
    ('move_tcp', {'arm': 'arm6', 'pose_mm_deg': [float('nan')]*6}),
    ('move_tcp', {'arm': 'arm6', 'pose_mm_deg': [True]*6}),
    ('move_joints', {'arm': 'arm7', 'joints_deg': [1]*6}),
    ('move_joints', {'arm': 'arm8', 'joints_deg': [1]*7}),
    ('arm', {}), ('coordinates', {'mode': 'real_supervised'}),
    ('pixel_to_world', {'pixel': [1.5, 2]}),
])
def test_reject_invalid_calls(tmp_path, name, args):
    with pytest.raises(ValueError):
        backend(tmp_path).call(name, args)


@pytest.mark.parametrize('name,args', [
    ('move_tcp', {'arm':'arm6','pose_mm_deg':[0]*6}),
    ('move_joints', {'arm':'arm7','joints_deg':[0]*7}),
    ('gripper', {'arm':'arm6','target':'close'})])
def test_mock_has_no_physics(tmp_path, name, args):
    b = backend(tmp_path)
    first, _ = b.call('observe', {})
    result, _ = b.call(name, args)
    last, _ = b.call('observe', {})
    assert result['status'] == 'NOT_EXECUTED'
    assert result['physical_outcome'] is None
    assert result['safety'] == 'UNKNOWN'
    assert first['sha256'] == last['sha256']
    assert last['action_conditioned'] is False


def test_replay_existing_geometry(tmp_path):
    import numpy as np
    b = backend(tmp_path, mode='replay', replay_manifest=str(HERE/'configs/replay.example.json'))
    image, images = b.call('observe', {})
    assert images[0]['mimeType'] == 'image/png'
    assert image['provenance'] == 'historical_real'
    assert b.call('robot_state', {})[0]['live'] is False
    cloud = b.observation.xyz
    # Select only a valid patch, with no garment heuristic or task skill.
    good = None
    for y in range(10, cloud.shape[0]-10, 20):
        for x in range(10, cloud.shape[1]-10, 20):
            try:
                expected = b.observation.sample([x,y],10)
                good = [x,y]
                break
            except ValueError:
                pass
        if good:
            break
    assert good is not None
    result, _ = b.call('pixel_to_world', {'pixel': good})
    assert np.allclose(result['world_xyz_mm'], expected)
    assert result['frame_id'] == 'world_mm'
    assert image['sha256'] == hashlib.sha256(Path(image['image_path']).read_bytes()).hexdigest()


def test_existing_transforms_and_three_states():
    from cloth_agent.dual_arm.stop_sweep import UnsafeBoxError, ProofIncompleteError
    g = ExistingGeometry(ROOT/'config/dual_arm.local.json', ROOT)
    assert g.base_pose('arm6', [100,0,0,0,0,0])[:3] == [100,0,0]
    assert g.require_safe({'status':'SAFE'})['status'] == 'SAFE'
    with pytest.raises(UnsafeBoxError):
        g.require_safe({'status':'UNSAFE'})
    with pytest.raises(ProofIncompleteError):
        g.require_safe({'status':'UNKNOWN'})
    with pytest.raises(PermissionError):
        ProtectedControllerPort()


def test_mcp_and_audit(tmp_path):
    records=[]
    f=Frontend(backend(tmp_path),lambda event,**data:records.append((event,data)))
    assert f.handle(dict(jsonrpc='2.0',id=1,method='initialize'))['result']['capabilities']=={'tools':{}}
    r=f.handle(dict(jsonrpc='2.0',id=2,method='tools/call',params={'name':'arm'}))
    assert 'error' in r
    assert records[-1][0]=='mcp_response'


def test_budget_and_crash_resume(tmp_path):
    config={'max_llm_calls':2,'budget_usd':1,'mode':'mock'}
    s=Session(tmp_path,config)
    _,row=s.episode()
    b=Budget(s)
    ident=b.reserve(.6,phase='Explore',model='test-provider-no-call')
    with pytest.raises(PermissionError): b.reserve(.6,phase='Explore',model='test')
    s.close()
    s=Session(tmp_path,config,resume=True)
    assert s.state['episodes'][0]['status']=='interrupted'
    assert s.state['llm_cost_usd']==.6
    Budget(s).settle(ident,actual_usd=.2,input_tokens=3,output_tokens=4,elapsed_seconds=.1,response={'test':True})
    assert s.state['llm_cost_usd']==pytest.approx(.2)
    s.close()
    with pytest.raises(ValueError): Session(tmp_path,{**config,'mode':'replay'},resume=True)


def test_program_is_not_executed(tmp_path):
    s=Session(tmp_path,{'mode':'mock'})
    marker=tmp_path/'unsafe'
    result=s.version(f'open({str(marker)!r},"w").write("bad")\ndef main(ctx): pass\n',0)
    assert result['syntax_valid'] and result['execution']=='NOT_RUN'
    assert not marker.exists()
    s.close()


def test_stop_and_resume_subprocess(tmp_path):
    command=[sys.executable,'-m','rax_cloth.cli','--session',str(tmp_path/'run'),
             '--episodes','3','--interval-seconds','0.3']
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    deadline=time.monotonic()+15
    state=tmp_path/'run/state.json'
    while time.monotonic()<deadline:
        if state.exists() and len(json.loads(state.read_text())['episodes'])>=1:
            break
        time.sleep(.02)
    p.send_signal(signal.SIGTERM)
    stdout,stderr=p.communicate(timeout=10)
    assert p.returncode==0,stderr
    assert json.loads(state.read_text())['status']=='stopped'
    completed=subprocess.run(command+['--resume'],capture_output=True,text=True,timeout=15)
    assert completed.returncode==0,completed.stderr
    result=json.loads(state.read_text())
    assert result['status']=='complete' and len(result['episodes'])==3
    assert result['task_success'] is None and result['physical_actions']==0
    assert (tmp_path/'run/summary.md').is_file()


def test_replay_manifest_hash_rejects_changes(tmp_path):
    manifest=json.loads((HERE/'configs/replay.example.json').read_text())
    manifest['observations'][0]['sha256']['rgb.png']='0'*64
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='changed since manifest'):
        backend(tmp_path/'backend',mode='replay',replay_manifest=str(path)).call('observe',{})


def test_session_exclusive_lock(tmp_path):
    s=Session(tmp_path,{'mode':'mock'})
    with pytest.raises(BlockingIOError):
        Session(tmp_path,{'mode':'mock'},resume=True)
    s.close()


def test_agent_launch_fails_closed(tmp_path):
    result=subprocess.run([sys.executable,'-m','rax_cloth.cli','--driver','agent',
                           '--session',str(tmp_path/'agent')],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0
    assert 'PermissionError' in result.stderr
    assert not (tmp_path/'agent').exists()


def test_mcp_time_budget_stops_without_stdin(tmp_path):
    command=[sys.executable,'-m','rax_cloth.cli','--driver','mcp','--session',str(tmp_path/'mcp'),
             '--max-hours','0.0001']
    p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    # Keep stdin open without input. Deadline must terminate the MCP loop.
    p.wait(timeout=10)
    _,stderr=p.communicate()
    assert p.returncode==0,stderr
    assert json.loads((tmp_path/'mcp/state.json').read_text())['status']=='time_budget_exhausted'
