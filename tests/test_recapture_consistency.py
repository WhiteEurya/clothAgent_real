import json
import copy
import pytest
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments.recapture_consistency import review_pair,ConfirmedSourceChange,RecaptureSourceUnmatched
from tests.test_graph_policy_matching import graph_fixture,Requests,result
from tests.test_graph_policy_replay import policy,Model,Runtime,matched


def pair(node='S4',relation='SAME_DECISION_CONDITION'):
    return dict(previous_match=result(node),current_match=result(node),observation_relation=relation,comparison_evidence=['same visible overlay in both views'])


def test_pair_uses_both_queries_and_all_references(tmp_path):
    g,t,o=graph_fixture(tmp_path)
    r=Requests([pair()]);m={}
    review_pair(g,t,o,o,r,tmp_path,m,['S3','S4'],.8)
    assert [k for k,_ in r.calls[0][1]]==['current','previous_current','oS3','oS4']
    assert m['joint_match_reviews']==1


def test_same_condition_cannot_claim_different_nodes(tmp_path):
    g,t,o=graph_fixture(tmp_path);v=pair();v['previous_match']=result('S3')
    with pytest.raises(ValueError,match='Same decision condition'):
        review_pair(g,t,o,o,Requests([v]),tmp_path,{},['S3','S4'],.8)


def test_confirmed_reentry_cancels_old_edge_without_dispatch(tmp_path,monkeypatch):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0')])
    calls=[]
    def changed(rt,req,graph,trajectory,edge,before,*a,**k):
        calls.append(edge['edge_id'])
        raise ConfirmedSourceChange(matched('N1'),before,dict(observation_relation='SAME_DECISION_CONDITION'),['N0','N1'])
    monkeypatch.setattr(pr,'grounding_with_recapture',changed)
    out=pr.replay(path,tmp_path/'run',runtime,model,max_steps=3)
    assert out['source_reentries']==1 and out['status']=='NO_REPLAYABLE_EDGE'
    assert not runtime.executions and len(model.calls)==1
    record=json.loads((tmp_path/'run/step_000/source_reentry.json').read_text())
    assert record['confirmed_source']=='N1' and not record['old_edge_dispatched']


@pytest.mark.parametrize('unmatched',[False,True])
def test_reentry_alternation_is_bounded(tmp_path,monkeypatch,unmatched):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0')])
    def changed(rt,req,graph,trajectory,edge,before,*a,**k):
        if unmatched:raise RecaptureSourceUnmatched(result(status='NO_MATCH'),before,{},['N0','N1'])
        raise ConfirmedSourceChange(matched('N0'),before,{},['N0','N1'])
    monkeypatch.setattr(pr,'grounding_with_recapture',changed)
    out=pr.replay(path,tmp_path/'run',runtime,model,max_steps=10)
    assert out['source_reentries']==2 and out['status']=='SOURCE_REENTRY_LIMIT'
    assert not runtime.executions


@pytest.mark.parametrize('fallbacks',[0,3])
@pytest.mark.parametrize('global_status',['NO_MATCH','AMBIGUOUS'])
def test_joint_no_match_reenters_globally_before_any_frontier(tmp_path,monkeypatch,fallbacks,global_status):
    from tests.test_online_graph_growth import Requests as GrowthRequests
    path,g=policy(tmp_path);runtime=Runtime();calls=[]
    req=GrowthRequests([matched('N0'),result(status=global_status)],node=g['nodes'][0])
    fresh=runtime.observe(tmp_path/'fresh')
    def unmatched(rt,requests,graph,trajectory,edge,before,*a,**k):
        raise RecaptureSourceUnmatched(result(status='NO_MATCH'),fresh,{},['N0'])
    def frontier(rt,before,*a,**k):
        assert before is fresh
        assert len([c for c in req.calls if c[0]=='node_match'])==2
        calls.append('frontier')
        return dict(outcome=dict(status='PLANNING_FAILED',physical_execution=False,error='IK failed'))
    monkeypatch.setattr(pr,'grounding_with_recapture',unmatched)
    out=pr.replay(path,tmp_path/'run',runtime,req,max_steps=3,max_fallbacks=fallbacks,fallback_fn=frontier)
    recovered=fallbacks and global_status=='NO_MATCH'
    assert out['status']==('SAFETY_REJECT' if recovered else global_status)
    assert calls==(['frontier'] if recovered else []) and not runtime.executions
    record=json.loads((tmp_path/'run/step_000/source_reentry.json').read_text())
    assert record['requires_global_match'] and not record['old_edge_dispatched']
    attempt=json.loads((tmp_path/'run/step_001/matching_attempts.json').read_text())['attempts'][0]
    assert attempt['stage']=='global' and set(attempt['candidate_nodes'])=={'N0','N1'}


def test_joint_no_match_does_not_ground_old_edge_again(tmp_path,monkeypatch):
    from cloth_agent.experiments.graph_minimal_grounding import GroundingUncertain
    from types import SimpleNamespace
    from tests.test_graph_policy import observation
    before=observation(tmp_path/'before');fresh=observation(tmp_path/'fresh');calls=[]
    def uncertain(*a,**k):calls.append(1);raise GroundingUncertain('low confidence')
    monkeypatch.setattr(pr,'selected_edge_grounding',uncertain)
    monkeypatch.setattr(pr,'topology_match',lambda *a,**k:result(status='NO_MATCH'))
    monkeypatch.setattr(pr,'reliable_match',lambda *a:False)
    monkeypatch.setattr('cloth_agent.experiments.recapture_consistency.review_pair',lambda *a,**k:dict(
        observation_relation='SAME_DECISION_CONDITION',current_match=result(status='NO_MATCH')))
    req=SimpleNamespace(no_cache=False)
    with pytest.raises(RecaptureSourceUnmatched) as caught:
        pr.grounding_with_recapture(SimpleNamespace(observe=lambda *a:fresh),req,{},None,
            dict(edge_id='E',source_node='N0',target_node='N1'),before,tmp_path,{},.8)
    assert caught.value.observation is fresh and calls==[1] and not req.no_cache


def test_full_recapture_reentry_fallback_and_after_match_pipeline(tmp_path,monkeypatch):
    """Exercise real matching/recapture/recovery orchestration with injected I/O."""
    from cloth_agent.experiments import graph_policy as gp
    from cloth_agent.experiments.graph_minimal_grounding import GroundingUncertain
    from tests.test_online_graph_growth import Requests as GrowthRequests
    from tests.test_graph_policy import SOURCE
    path,g=policy(tmp_path);runtime=Runtime();ground_calls=[];fallback_images=[]
    class Timeline(GrowthRequests):
        def ask(self,kind,payload,images,validator=None):
            if kind=='node_match_pair':
                self.calls.append((kind,copy.deepcopy(payload),list(images)))
                value=dict(previous_match=result(status='NO_MATCH'),current_match=result(status='NO_MATCH'),
                           observation_relation='SAME_DECISION_CONDITION',comparison_evidence=['Same visible folds in both queries'])
                if validator:validator(value)
                return value
            return super().ask(kind,payload,images,validator)
    req=Timeline([matched('N0'),result(status='NO_MATCH'),result(status='NO_MATCH'),matched('N1')],node=g['nodes'][0])
    def uncertain(requests,graph,edge,observation,*args):
        ground_calls.append((edge['edge_id'],observation.image))
        raise GroundingUncertain('minimal anchors unresolved/low confidence')
    def frontier(rt,observation,folder,*args,**kwargs):
        fallback_images.append(observation.image)
        return dict(source=SOURCE,outcome=rt.execute(SOURCE,observation,folder))
    monkeypatch.setattr(pr,'selected_edge_grounding',uncertain)
    out=pr.replay(path,tmp_path/'run',runtime,req,max_steps=4,max_fallbacks=1,fallback_fn=frontier)
    assert out['status']=='NO_REPLAYABLE_EDGE' and out['source_reentries']==1
    assert out['executed_edges']==[] and len(ground_calls)==1 and len(runtime.executions)==1
    fresh=tmp_path/'run/step_000/grounding_retry_001/observation/rgb.png'
    assert fallback_images==[fresh]
    matching=[c for c in req.calls if c[0]=='node_match']
    assert matching[1][2][0][1]==matching[2][2][0][1]==fresh
    assert matching[-1][2][0][1]!=fresh
    assert [c[0] for c in req.calls]==['node_match','node_match','node_match_pair',
                                      'node_match','recovery_node','node_match']
    assert out['fallback_transitions'][0]['actual_node']=='N1'
    extended=gp.load_frozen(out['extended_graph']);pr.validate_policy(extended)
    assert extended['edges']==g['edges'] and gp.load_frozen(path)==g
