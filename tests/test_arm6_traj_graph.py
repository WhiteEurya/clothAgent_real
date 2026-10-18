"""Offline import integrity for operator-approved Arm6 graph evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('arm6_graph_import',
    Path(__file__).resolve().parents[1]/'scripts/compile_arm6_traj_graph.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def make_iteration(root, name, status, approved):
    p=root/name
    save(p/'traj.json',dict(status=status,operator_accepted=approved,arm='left',camera='A',samples=1))
    save(p/'observation/observation.json',dict(captured_at='2026-10-17T00:00:00Z'))
    (p/'observation/rgb.png').write_bytes(b'fixture')
    (p/'observation/world_xyz_mm.npy').write_bytes(b'fixture')
    save(p/'points.json',[])
    save(p/'states.jsonl',dict(t_s=1,phase='manual',world_tcp_mm_deg=[0]*6,gripper_pulse=20))
    save(p/'events.jsonl',dict(kind='manual_confirmed',t_s=1))
    for phase in ('approach','transfer_place','return_perception'):
        save(p/phase/'result.json',dict(status='EXECUTED'))
    return p


def test_import_preserves_rejected_iteration_gap_and_missing_after(tmp_path):
    make_iteration(tmp_path,'traj_0000','COMPLETE',True)
    make_iteration(tmp_path,'pending/iter_0001','REJECTED',False)
    make_iteration(tmp_path,'traj_0002','COMPLETE',True)
    result=module.import_session(tmp_path)
    assert [s['step_id'] for s in result['steps']]==['s000','s002']
    assert result['steps'][0]['after_observation']=='o001'
    assert result['steps'][1]['after_observation'] is None
    assert not result['steps'][0]['outcome']['continuity_verified']
    assert result['steps'][0]['commanded_action']['axis']==6
    assert result['excluded_iterations']==[dict(iteration=1,status='REJECTED')]


def test_import_rejects_sample_count_mismatch(tmp_path):
    p=make_iteration(tmp_path,'traj_0000','COMPLETE',True)
    manifest=module.read(p/'traj.json');manifest['samples']=2
    save(p/'traj.json',manifest)
    with pytest.raises(ValueError,match='sample count'):
        module.import_session(tmp_path)


def test_explicit_rejected_override_preserves_original_verdict(tmp_path):
    p=make_iteration(tmp_path,'pending/iter_0001','REJECTED',False)
    before=(p/'traj.json').read_bytes()
    result=module.import_session(tmp_path,replay_rejected_iterations={1:'User requested replay'})
    step=result['steps'][0]
    assert step['actual_execution']['operator_accepted'] is False
    auth=step['commanded_action']['replay_authorization']
    assert auth['manifest']==step['commanded_action']['manifest']
    assert (p/'traj.json').read_bytes()==before
    save(p/'traj.json',dict(status='ABORTED',operator_accepted=False))
    with pytest.raises(ValueError,match='explicitly rejected'):
        module.import_session(tmp_path,replay_rejected_iterations={1:'User requested replay'})


def test_action_binding_uses_recorded_evidence_without_model_edits(monkeypatch):
    recorded = dict(axis=6, measured_trajectory=dict(sha256='original'))
    trajectory = dict(steps=[dict(step_id='s0', commanded_action=recorded, before_observation='o0')])
    graph = dict(edges=[dict(edge_id='e0', seed_step_id='s0',
        seed_action=dict(exact_command={'axis':7},source_observation='wrong'))])
    validated=[]
    monkeypatch.setattr(module,'graph_validation',lambda g,t:validated.append(True))
    assert module.bind_recorded_actions(graph,trajectory)==['e0']
    assert graph['edges'][0]['seed_action']==dict(exact_command=recorded,source_observation='o0')
    graph['edges'][0]['seed_action']['exact_command']['axis']=0
    assert recorded['axis']==6
    assert validated==[True]
