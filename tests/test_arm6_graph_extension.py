import json
import pytest
from scripts.extend_arm6_experience_graph import import_manual, prefix_import


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def manual(root,index):
    p=root/f'traj_{index:04d}'
    save(p/'traj.json',dict(status='COMPLETE',operator_accepted=True,
        collection_mode='single_point_manual',samples=1))
    save(p/'observation/observation.json',dict(captured_at='today'))
    (p/'observation/rgb.png').write_bytes(b'RGB')
    (p/'observation/world_xyz_mm.npy').write_bytes(b'DEPTH')
    save(p/'states.jsonl',dict(t_s=1,phase='manual',world_tcp_mm_deg=[0]*6,gripper_pulse=5))
    save(p/'events.jsonl',dict(t_s=2,kind='manual_recording_finished'))
    return p


def test_manual_import_never_fabricates_after_or_target(tmp_path):
    manual(tmp_path,0);manual(tmp_path,1)
    save(tmp_path/'pending/iter_0002/traj.json',dict(status='ABORTED',samples=0))
    t=import_manual(tmp_path)
    assert t['steps'][0]['after_observation']=='o001'
    assert t['steps'][1]['after_observation'] is None
    assert t['steps'][0]['commanded_action']['type']=='ARM6_FREE_MANUAL_DEMONSTRATION'
    assert 'handoff' not in t['steps'][0]['commanded_action']
    assert len(t['excluded_iterations'])==1
    prefixed=prefix_import(t,'session_')
    assert prefixed['steps'][0]['after_observation']=='session_o001'
    assert prefixed['steps'][1]['after_observation'] is None
    assert 'o001' in t['observations']


def test_manual_import_checks_feedback_count(tmp_path):
    p=manual(tmp_path,0)
    m=json.loads((p/'traj.json').read_text());m['samples']=2;save(p/'traj.json',m)
    with pytest.raises(ValueError,match='feedback'):
        import_manual(tmp_path)
