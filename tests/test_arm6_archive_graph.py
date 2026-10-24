import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from scripts import compile_arm6_archive_graph as archive


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def observation(path):
    path.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(128,72),'purple').save(path/'rgb.png')
    save(path/'observation.json',dict(captured_at='2026-10-19T10:00:00+00:00',synthetic=False))
    np.save(path/'world_xyz_mm.npy',np.zeros((72,128,3)))


def trajectory(root,index,*,session='session_a',after=True,status='COMPLETE',mode='automatic_pick_place',episode=0):
    p=root/session/f'traj_{index:04d}'
    manifest=dict(status=status,operator_accepted=status=='COMPLETE',arm='left',camera='A',
                  samples=1,collection_mode=mode,episode_id=episode)
    save(p/'traj.json',manifest)
    observation(p/'observation')
    if after:observation(p/'after')
    save(p/'states.jsonl',dict(t_s=1,world_tcp_mm_deg=[0]*6,phase='automatic_pick_place',gripper_pulse=850))
    save(p/'events.jsonl',dict(t_s=1,kind='stage',phase='automatic_pick_place'))
    save(p/'controller/events.jsonl',dict(event='executed'))
    save(p/'points.json',[dict(name=name,pixel_xy=[10+i*20,20],world_xyz_mm=[i*30,0,0]) for i,name in enumerate(['start','target'])])
    for phase in ['automatic_pick_place','return_perception']:
        save(p/phase/'result.json',dict(status='EXECUTED'))
        save(p/phase/'native_plan.json',[])
        (p/phase/'motion.py').write_text('motion.gripper(arm="left", action="open")\n')
    return p


def test_import_prefers_dedicated_after_and_separates_sessions(tmp_path):
    a=trajectory(tmp_path,0);trajectory(tmp_path,1,after=False)
    trajectory(tmp_path,0,session='session_b')
    t=archive.import_archive(tmp_path)
    assert len(t['steps'])==3 and len(t['observations'])==5
    assert t['steps'][0]['outcome']['after_provenance']=='dedicated_after'
    assert t['steps'][0]['after_observation']!=t['steps'][1]['before_observation']
    assert t['steps'][1]['after_observation'] is None
    assert t['observations'][t['steps'][0]['after_observation']]['rgb']['path']==str(a/'after/rgb.png')


@pytest.mark.parametrize('boundary',[False,True])
def test_legacy_after_only_next_index_same_episode(tmp_path,boundary):
    trajectory(tmp_path,0,after=False)
    trajectory(tmp_path,1,episode=1 if boundary else 0)
    t=archive.import_archive(tmp_path)
    if boundary:
        assert t['steps'][0]['after_observation'] is None
        assert t['steps'][0]['outcome']['after_provenance']=='reset_episode_boundary'
    else:
        assert t['steps'][0]['after_observation']==t['steps'][1]['before_observation']
        assert not t['steps'][0]['outcome']['continuity_verified']


def test_deleted_iteration_and_reset_are_not_bridged(tmp_path):
    trajectory(tmp_path,0,after=False);trajectory(tmp_path,2,after=False);trajectory(tmp_path,3)
    (tmp_path/'session_a/resets/after_iter_0002').mkdir(parents=True)
    t=archive.import_archive(tmp_path)
    assert t['steps'][0]['after_observation'] is None
    assert t['steps'][1]['after_observation'] is None


def test_excludes_unconfirmed_reset_and_bad_feedback(tmp_path):
    trajectory(tmp_path,0)
    trajectory(tmp_path,1,status='REJECTED')
    trajectory(tmp_path,2,mode='random_reset')
    p=trajectory(tmp_path,3);save(p/'states.jsonl',dict(t_s=1,world_tcp_mm_deg=[float('nan')]*6,phase='automatic_pick_place'))
    t=archive.import_archive(tmp_path)
    assert len(t['steps'])==1 and len(t['excluded_iterations'])==3


def node(oid):
    return dict(node_id='n000',task_context='衣服',representative_observations=oid,confidence=.9,
                observation_condition=dict(positive_evidence=['可见布料'],negative_evidence=[],uncertainty_conditions=[]))


def test_partition_and_binding_preserve_each_action(tmp_path):
    trajectory(tmp_path,0);trajectory(tmp_path,1,after=False)
    t=archive.import_archive(tmp_path)
    group=dict(nodes=[node(list(t['observations']))],limitations=[])
    semantics={s['step_id']:dict(action_family='局部搬运',grasp_relation='标记布片',target_relation='目标点',
        observed_outcome='变化不确定',seed_outcome='UNKNOWN',limitations=[]) for s in t['steps']}
    g=archive.build_graph(t,group,semantics)
    assert len(g['edges'])==2 and len(g['nodes'])==2
    assert g['edges'][0]['source_node']==g['edges'][0]['target_node']
    assert g['edges'][1]['target_node']=='unknown_s001'
    assert g['edges'][0]['seed_action']['exact_command']==t['steps'][0]['commanded_action']
    for bad in [dict(nodes=[node([])],limitations=[]),dict(nodes=[node(list(t['observations'])*2)],limitations=[])]:
        with pytest.raises(ValueError):archive.validate_partition(bad,t['observations'])


def test_merge_review_preserves_members_and_rejects_wrong_evidence(tmp_path):
    trajectory(tmp_path,0)
    t=archive.import_archive(tmp_path);ids=list(t['observations'])
    left=node([ids[0]]);right=node([ids[1]]);right['node_id']='n001'
    initial=dict(nodes=[left,right],limitations=[])
    candidates=[dict(pair_id='p0',nodes=['n000','n001'],observation_members={'n000':[ids[0]],'n001':[ids[1]]})]
    decision=dict(pair_id='p0',decision='MERGE',confidence=.9,reason='same visible coverage',task_context='展开',
                  positive_evidence=['双袖露出'],negative_evidence=[],uncertainty_conditions=[])
    result,changes=archive.apply_merge_review(t,initial,candidates,dict(decisions=[decision]))
    assert len(result['nodes'])==1 and result['nodes'][0]['representative_observations']==ids
    assert len(changes)==1 and len(initial['nodes'])==2
    decision['confidence']=.7
    result,changes=archive.apply_merge_review(t,initial,candidates,dict(decisions=[decision]))
    assert len(result['nodes'])==2 and not changes
    candidates[0]['observation_members']['n001']=['unknown']
    with pytest.raises(ValueError,match='evidence mismatch'):
        archive.apply_merge_review(t,initial,candidates,dict(decisions=[decision]))
