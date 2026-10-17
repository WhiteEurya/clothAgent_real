import json
import pytest
from tests.test_graph_policy_replay import policy,Model,Runtime,matched
from cloth_agent.experiments import graph_policy_replay as pr


@pytest.mark.parametrize('review_result',['MOVED','UNCHANGED','UNKNOWN'])
def test_self_loop_replans_and_preserves_evidence(tmp_path,monkeypatch,review_result):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0'),matched('N0'),matched('N1')])
    monkeypatch.setattr(pr.missed_grasp_retry,'review',lambda *a:dict(result=review_result,confidence=.95,evidence='before/after'))
    contexts=[]
    def repair(runtime,obs,folder,graph,**kw):
        contexts.append(kw['context'])
        return dict(source='motion.grasp(right=[.5,.1,.05,3.14,0,0])\nmotion.move_tcp(right=[.5,.2,.1,3.14,0,0])\n',outcome=dict(status='EXECUTED',physical_execution=True))
    result=pr.replay(path,tmp_path/'run',runtime,model,max_steps=4,repair_self_loop=True,fallback_fn=repair)
    assert result['edge_replans']==1
    assert result['transitions'][1]['actual_node']=='N1'
    assert result['transitions'][1]['mode']=='EDGE_REPAIR'
    data=json.load(open(result['edge_experience']))['edges']['E0']
    assert data['failures'][0]['visual_review']['result']==review_result
    # No actual runtime execution artifacts in this mock: cannot promote a template.
    assert data['revisions'][0]['status']=='MINIMAL_UNSUPPORTED'
    assert not data['revisions'][0]['replayable']
    assert contexts[0]['reason']=='REPAIR_EXISTING_EDGE_AFTER_SELF_LOOP'
    assert pr.load_frozen(path)==g


def test_repair_self_loop_does_not_repeat_forever(tmp_path,monkeypatch):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0')]*4)
    monkeypatch.setattr(pr.missed_grasp_retry,'review',lambda *a:dict(result='MOVED',confidence=.9,evidence='moved'))
    calls=[]
    def repair(*a,**kw):
        calls.append(kw);return dict(source='motion.grasp(right=[.5,.1,.05,3.14,0,0])\n',outcome=dict(status='EXECUTED',physical_execution=True))
    result=pr.replay(path,tmp_path/'run',runtime,model,max_steps=4,repair_self_loop=True,fallback_fn=repair)
    assert result['status']=='EDGE_REPAIR_SELF_LOOP' and len(calls)==1
    data=json.load(open(result['edge_experience']))['edges']['E0']
    assert data['revisions'][0]['status']=='SELF_LOOP'


def test_repair_safety_reject_is_not_promoted(tmp_path,monkeypatch):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0')]*2)
    monkeypatch.setattr(pr.missed_grasp_retry,'review',lambda *a:dict(result='UNCHANGED',confidence=.9,evidence='same'))
    def repair(*a,**kw):return dict(source=None,outcome=dict(status='PLANNING_FAILED',physical_execution=False))
    result=pr.replay(path,tmp_path/'run',runtime,model,repair_self_loop=True,fallback_fn=repair)
    assert result['status']=='SAFETY_REJECT'
    data=json.load(open(result['edge_experience']))['edges']['E0']
    assert data['revisions'][0]['status']=='NOT_EXECUTED_SUCCESSFULLY'


def test_supported_revision_exports_loadable_template_bundle(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from cloth_agent.experiments.edge_self_loop_repair import EdgeRepairs
    from cloth_agent.experiments import graph_template_compiler as compiler,action_local_frame as local
    from cloth_agent.experiments.graph_minimal_grounding import MinimalGrounder
    path,g=policy(tmp_path);edge=g['edges'][0]
    image=tmp_path/'before.png';image.write_bytes(b'fixture')
    obs=SimpleNamespace(image=image)
    bundle=dict(version='minimal-grounding-host-template-v1',edges={})
    requests=SimpleNamespace(minimal_grounder=SimpleNamespace(bundle=bundle))
    repair=EdgeRepairs(tmp_path/'out',requests,graph_path=path)
    folder=tmp_path/'planner';round_folder=folder/'round_001';round_folder.mkdir(parents=True)
    (round_folder/'effective_motion.py').write_text('motion.grasp(right=[.5,.1,.05,3.14,0,0])\nmotion.move_tcp(right=[.5,.2,.1,3.14,0,0])\n')
    (round_folder/'grasp_height_policy.json').write_text('{}')
    monkeypatch.setattr(compiler,'seed_slots',lambda *a:([dict(pixel_xy=[1,2]),dict(pixel_xy=[3,4])],[],[0,0],{}))
    monkeypatch.setattr(local,'compile_local',lambda *a,**kw:dict(version='action-local-host-template-v2'))
    transition=dict(source_node='N0',actual_node='N0')
    repair.begin(edge,transition,dict(result='MOVED'),obs,folder)
    repair.executed(edge['edge_id'],dict(source='fixture',outcome=dict(status='EXECUTED',physical_execution=True)))
    repair.finish(edge['edge_id'],'N1',obs)
    revision=repair.edges['E0']['revisions'][0]
    assert revision['status']=='MINIMAL_SUPPORTED' and revision['replayable']
    grounder=MinimalGrounder(revision['host_templates'])
    assert grounder.template(edge)['repair_provenance']['result_node']=='N1'
    assert repair.edges['E0']['preferred_revision']==1
    from cloth_agent.experiments.edge_self_loop_repair import latest_repair_templates
    assert latest_repair_templates(path).resolve()==__import__('pathlib').Path(revision['host_templates']).resolve()
    inherited=EdgeRepairs(tmp_path/'next',SimpleNamespace(minimal_grounder=grounder),graph_path=path)
    assert inherited.edges['E0']['preferred_revision']==1
    assert inherited.edges['E0']['failures']==repair.edges['E0']['failures']
    # Corrupted revision cannot silently replace the seed bundle.
    __import__('pathlib').Path(revision['host_templates']).write_text('{}')
    with pytest.raises(ValueError,match='mismatch'):latest_repair_templates(path)


@pytest.mark.parametrize('limit',[1,2])
def test_repair_then_bounded_general_fallback(tmp_path,monkeypatch,limit):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0')]*10)
    monkeypatch.setattr(pr.missed_grasp_retry,'review',lambda *a:dict(result='MOVED',confidence=.9,evidence='moved'))
    contexts=[]
    def planner(*args,**kwargs):
        contexts.append(kwargs['context'])
        return dict(source='motion.grasp(right=[.5,.1,.05,3.14,0,0])\nmotion.move_tcp(right=[.5,.2,.1,3.14,0,0])\n',outcome=dict(status='EXECUTED',physical_execution=True))
    result=pr.replay(path,tmp_path/'run',runtime,model,max_steps=8,repair_self_loop=True,max_fallbacks=limit,fallback_fn=planner)
    assert result['edge_replans']==1
    assert result['fallback_steps']==limit
    assert len(contexts)==1+limit
    assert contexts[0]['reason']=='REPAIR_EXISTING_EDGE_AFTER_SELF_LOOP'
    assert result['status']=='FALLBACK_LIMIT'
    assert pr.load_frozen(path)==g
