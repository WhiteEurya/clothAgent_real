from types import SimpleNamespace
import copy
import json
from pathlib import Path
import pytest
from PIL import Image
from cloth_agent.experiments.edge_applicability import ApplicabilityRecovery
from cloth_agent.experiments.graph_minimal_grounding import GroundingUncertain
from cloth_agent.experiments import graph_policy_replay as pr
from tests.test_graph_policy_replay import policy,Runtime,Model,candidate


def setup_case(tmp_path,monkeypatch,branch='YES',confidence=.9,uncertain=False,relation='SAME_ROLE'):
    path,g=policy(tmp_path);edge=g['edges'][0]
    image=tmp_path/'rgb.png';Image.new('RGB',(30,30)).save(image)
    obs=SimpleNamespace(image=image)
    t=dict(spec={},image_size=[30,30],seed_grasp_px=[2,3],seed_target_px=[12,13])
    calls=[]
    def ask(kind,*a,**kw):
        calls.append(kind)
        if kind=='edge_applicability':return dict(applicable=branch,confidence=confidence,evidence=['visible layer'])
        if kind=='ground_edge_minimal':return dict(grasp_px=[2,3],target_px=[12,13],confidence=.9)
        if kind=='ground_edge':return candidate()
        if kind=='applicability_outcome':return dict(action_relation=relation,evidence=['changed overlay'],decision_condition='different accessible layer')
        raise AssertionError(kind)
    def expand(*args):
        if uncertain:raise GroundingUncertain('uncertain')
        return dict(status='READY',anchors_px=dict(grasp_px=[2,3],target_px=[12,13]))
    req=SimpleNamespace(ask=ask,minimal_grounder=SimpleNamespace(template=lambda e:t,expand=expand))
    monkeypatch.setattr('cloth_agent.experiments.graph_minimal_grounding.marked_seed_reference',lambda *args:image)
    live=SimpleNamespace(update=lambda *a,**k:None)
    manager=ApplicabilityRecovery(tmp_path/'run',req,live,g)
    return manager,edge,obs,calls,g,req


@pytest.mark.parametrize('branch,confidence,expected',[('YES',.9,'YES'),('NO',.9,'NO'),('UNKNOWN',.9,'UNKNOWN'),('YES',.5,'UNKNOWN')])
def test_triage_and_bounded_attempt(tmp_path,monkeypatch,branch,confidence,expected):
    m,e,o,c,g,req=setup_case(tmp_path,monkeypatch,branch,confidence)
    result=m.prepare(e,o,tmp_path/'step')
    assert m.active['branch']==expected
    assert (result is not None)==(expected=='YES')
    assert ('ground_edge_minimal' in c)==(expected=='YES')
    with pytest.raises(ValueError,match='budget exhausted'):m.prepare(e,o,tmp_path/'again')


def test_yes_full_only_after_reference_uncertainty(tmp_path,monkeypatch):
    m,e,o,c,g,req=setup_case(tmp_path,monkeypatch,uncertain=True)
    monkeypatch.setattr(pr,'grounding_payload',lambda *a:{})
    m.prepare(e,o,tmp_path/'step')
    assert c==['edge_applicability','ground_edge_minimal','ground_edge']


@pytest.mark.parametrize('relation,expected',[('SAME_ROLE','EDGE_GROUNDING_EVIDENCE'),('DIFFERENT_ROLE','PROVISIONAL_SUBSTATE'),('UNKNOWN','UNRESOLVED')])
def test_outcome_driven_evidence_not_permanent_split(tmp_path,monkeypatch,relation,expected):
    m,e,o,c,g,req=setup_case(tmp_path,monkeypatch,relation=relation)
    original=copy.deepcopy(g)
    m.prepare(e,o,tmp_path/'step')
    from tests.test_graph_policy import SOURCE
    m.executed(SOURCE,dict(status='EXECUTED',physical_execution=True))
    m.finish(e,e['target_node'],o)
    assert m.rows[0]['status']==expected
    assert g==original
    if relation=='DIFFERENT_ROLE':
        assert m.nodes[0]['parent_node']==e['source_node']
        assert not m.actions[0]['replayable']
    if relation=='SAME_ROLE':
        assert e['edge_id'] in req.grounding_reference_overrides
        assert Path(m.rows[0]['grounding_reference']).exists()
        m2=ApplicabilityRecovery(tmp_path/'run2',req,m.live,g)
        assert e['edge_id'] in req.grounding_reference_overrides


@pytest.mark.parametrize('failure',['uncertain','depth'])
def test_runtime_only_escalates_grounding_uncertainty(tmp_path,monkeypatch,failure):
    path,g=policy(tmp_path);model=Model();seen=[]
    def failed(*a,**k):
        if failure=='depth':raise ValueError('invalid depth')
        ex=GroundingUncertain('uncertain');ex.observation=a[5];raise ex
    monkeypatch.setattr(pr,'grounding_with_recapture',failed)
    monkeypatch.setattr(ApplicabilityRecovery,'prepare',lambda self,e,o,f:seen.append(o) or candidate())
    result=pr.replay(path,tmp_path/'run',Runtime(),model,applicability_recovery=True)
    assert bool(seen)==(failure=='uncertain')
    if failure=='depth':assert result['status']=='EDGE_INSTANTIATION_FAILURE' and not result['executed_edges']


@pytest.mark.parametrize('status',['EXECUTED','PLANNING_FAILED'])
def test_frontier_dispatch_and_failure_never_promoted(tmp_path,monkeypatch,status):
    from tests.test_graph_policy import SOURCE
    from cloth_agent.experiments.graph_policy import evidence_file
    path,g=policy(tmp_path);model=Model();calls=[]
    def fail(*a,**kw):
        ex=GroundingUncertain('uncertain');ex.observation=a[5];raise ex
    monkeypatch.setattr(pr,'grounding_with_recapture',fail)
    def prepare(self,edge,obs,folder):
        self.active=dict(id=0,edge_id=edge['edge_id'],source_node=edge['source_node'],branch='NO',
            assessment=dict(applicable='NO'),before=evidence_file(obs.image))
        self.rows.append(self.active)
        return None
    monkeypatch.setattr(ApplicabilityRecovery,'prepare',prepare)
    original=model.ask
    def ask(kind,*a,**kw):
        if kind=='applicability_outcome':return dict(action_relation='DIFFERENT_ROLE',evidence=['new layer'],decision_condition='different layer access')
        return original(kind,*a,**kw)
    model.ask=ask
    def frontier(*a,**kw):
        calls.append(kw['context'])
        return dict(source=SOURCE,outcome=dict(status=status,physical_execution=status=='EXECUTED'))
    result=pr.replay(path,tmp_path/'run',Runtime(),model,applicability_recovery=True,fallback_fn=frontier)
    assert len(calls)==1
    data=json.loads((tmp_path/'run/applicability_recovery/experience.json').read_text())
    if status=='EXECUTED':
        assert data['provisional_nodes'][0]['parent_node']=='N0'
        assert result['transitions'][0]['actual_node']=='N1'
    else:
        assert result['status']=='APPLICABILITY_FRONTIER_FAILED'
        assert not data['provisional_nodes']


def test_no_execution_cannot_create_substate_or_reference(tmp_path,monkeypatch):
    m,e,o,c,g,req=setup_case(tmp_path,monkeypatch,relation='DIFFERENT_ROLE')
    m.prepare(e,o,tmp_path/'step')
    m.finish(e,e['target_node'],o)
    assert m.rows[0]['status']=='NO_VERIFIED_EXECUTION'
    assert 'applicability_outcome' not in c
    assert not m.nodes and not req.grounding_reference_overrides


@pytest.mark.parametrize('fallbacks,second_status,after,expected',[
    (0,'EXECUTED','N1','SOURCE_NODE_UNCHANGED'),
    (1,'EXECUTED','N1','NO_REPLAYABLE_EDGE'),
    (1,'EXECUTED','N0','FALLBACK_LIMIT'),
    (1,'PLANNING_FAILED','N1','SAFETY_REJECT'),
])
def test_frontier_self_loop_uses_remaining_generic_recovery(tmp_path,monkeypatch,fallbacks,second_status,after,expected):
    from tests.test_graph_policy import SOURCE
    from tests.test_graph_policy_replay import matched
    from cloth_agent.experiments.graph_policy import evidence_file,load_frozen
    path,g=policy(tmp_path);model=Model([matched('N0'),matched('N0'),matched(after)])
    preparations=[];plans=[];grounds=[]
    def fail(*a,**kw):
        grounds.append(1)
        ex=GroundingUncertain('uncertain');ex.observation=a[5];raise ex
    monkeypatch.setattr(pr,'grounding_with_recapture',fail)
    def prepare(self,edge,obs,folder):
        assert not preparations,'must not repeat exhausted edge applicability'
        preparations.append(edge['edge_id'])
        self.active=dict(id=0,edge_id=edge['edge_id'],source_node=edge['source_node'],branch='UNKNOWN',
            assessment=dict(applicable='UNKNOWN'),before=evidence_file(obs.image))
        self.rows.append(self.active)
        return None
    monkeypatch.setattr(ApplicabilityRecovery,'prepare',prepare)
    original=model.ask
    def ask(kind,*a,**kw):
        if kind=='applicability_outcome':return dict(action_relation='UNKNOWN',evidence=['visible cloth'],decision_condition='uncertain layer relation')
        return original(kind,*a,**kw)
    model.ask=ask
    def frontier(rt,obs,folder,*a,**kw):
        plans.append(kw['context'])
        status='EXECUTED' if len(plans)==1 else second_status
        return dict(source=SOURCE,outcome=dict(status=status,physical_execution=status=='EXECUTED'))
    out=pr.replay(path,tmp_path/'run',Runtime(),model,max_steps=4,max_fallbacks=fallbacks,
                  repair_self_loop=True,applicability_recovery=True,fallback_fn=frontier)
    assert out['status']==expected and out['edge_replans']==0 and out['executed_edges']==[]
    assert len(grounds)==1 and len(preparations)==1 and len(plans)==(2 if fallbacks else 1)
    assert out['transitions'][0]['progress_evaluation']['progress']=='NO_STATE_CHANGE'
    assert out['fallback_steps']==(1 if fallbacks else 0)
    if fallbacks:
        assert plans[1]['reason']=='SOURCE_NODE_UNCHANGED'
        assert plans[1]['previous_transition']['mode']=='APPLICABILITY_FRONTIER'
        assert plans[1]['previous_transition']['command_source']==SOURCE
        extended=load_frozen(out['extended_graph']);pr.validate_policy(extended)
        assert extended['edges']==g['edges']
    assert load_frozen(path)==g


@pytest.mark.parametrize('node,label',[('N0','NO_STATE_CHANGE'),('N1','PROGRESS'),('other','ALTERNATIVE_OUTCOME'),(None,'UNKNOWN')])
def test_applicability_progress_is_host_match_even_when_role_unknown(tmp_path,monkeypatch,node,label):
    from tests.test_graph_policy import SOURCE
    m,e,o,c,g,req=setup_case(tmp_path,monkeypatch,relation='UNKNOWN')
    m.prepare(e,o,tmp_path/'step')
    m.executed(SOURCE,dict(status='EXECUTED',physical_execution=True))
    m.finish(e,node,o)
    assert m.rows[0]['progress_evaluation']['progress']==label
    assert 'progress' not in m.rows[0]['role_evaluation']


def test_match_conflict_withdraws_success_reference(tmp_path,monkeypatch):
    from tests.test_graph_policy import SOURCE
    m,e,o,c,g,req=setup_case(tmp_path,monkeypatch)
    m.prepare(e,o,tmp_path/'step');m.executed(SOURCE,dict(status='EXECUTED',physical_execution=True))
    m.finish(e,e['target_node'],o)
    assert e['edge_id'] in req.grounding_reference_overrides
    m.invalidate_match(e['edge_id'],dict(reason='new match disagrees'))
    assert m.rows[0]['progress_evaluation']['progress']=='UNKNOWN'
    assert e['edge_id'] not in req.grounding_reference_overrides
    ApplicabilityRecovery(tmp_path/'run2',req,m.live,g)
    assert not req.grounding_reference_overrides
