import copy
from types import SimpleNamespace
import pytest

from cloth_agent.experiments import graph_policy_matching as tm
from cloth_agent.experiments.graph_policy import evidence_file
from tests.test_graph_policy import observation, match


def graph_fixture(tmp_path):
    obs=observation(tmp_path/'obs')
    nodes=[dict(node_id=n,task_context='visible '+n,
        observation_condition=dict(positive_evidence=['visible contour'],negative_evidence=['no other flap'],uncertainty_conditions=[]),
        representative_observations=['o'+n],confidence=.9) for n in ['S0','S1','S2','S3','S4','R_test']]
    graph=dict(goal=dict(subgoal='full_fold',side_convention='collar up'),nodes=nodes,match_nodes=nodes,
        edges=[dict(source_node=a,target_node=b) for a,b in [('S0','S1'),('S1','S2'),('S2','S3'),('S2','S4')]])
    trajectory=dict(observations={'o'+n['node_id']:dict(rgb=evidence_file(obs.image)) for n in nodes})
    return graph,trajectory,obs


def result(node=None,status='NO_MATCH',confidence=.95):
    v=match(status)
    v.update(node_id=node or '',candidate_nodes=[node] if node else [],confidence=confidence)
    if node:v['status']='MATCH'
    return v


class Requests:
    def __init__(self,values):self.values=list(values);self.calls=[]
    def ask(self,kind,payload,images,validator=None):
        self.calls.append((payload,images));v=self.values.pop(0)
        if validator:validator(v)
        return v


def test_local_match_current_and_candidate_references(tmp_path):
    g,t,o=graph_fixture(tmp_path);m={};r=Requests([result('S2')])
    out=tm.topology_match(g,t,o,r,tmp_path/'audit',m,pending=dict(source_node='S1',target_node='S2'))
    assert out['node_id']=='S2' and len(r.calls)==1
    payload,images=r.calls[0]
    assert [n['node_id'] for n in payload['nodes']]==['S1','S2']
    assert [k for k,_ in images]==['current','oS1','oS2']
    assert all(n['representative_observations'] for n in payload['nodes'])
    assert set(payload)=={'task','nodes'}
    assert m['local_match_time']>=0 and m['global_match_time']==m['expanded_match_time']==0


def test_expanded_includes_neighbor_and_recovery_max_four(tmp_path):
    g,t,o=graph_fixture(tmp_path);m={};r=Requests([result(),result('S3')])
    tm.topology_match(g,t,o,r,tmp_path/'audit',m,pending=dict(source_node='S1',target_node='S2'),recovery_nodes=['R_test'])
    assert [n['node_id'] for n in r.calls[1][0]['nodes']]==['S1','S2','S3','R_test']
    assert len(r.calls)==2 and m['global_match_time']==0
    assert all(len(images)==1+len(p["nodes"]) for p,images in r.calls)


def test_global_only_after_local_and_expanded_fail(tmp_path):
    g,t,o=graph_fixture(tmp_path);m={};r=Requests([result('S1',confidence=.2),result(),result('S4')])
    out=tm.topology_match(g,t,o,r,tmp_path/'audit',m,pending=dict(source_node='S1',target_node='S2'),recovery_nodes=['R_test'])
    assert out['node_id']=='S4'
    assert [len(p['nodes']) for p,_ in r.calls]==[2,4,6]
    assert m['representative_image_fallback_count']==0


def test_ambiguity_already_has_references_then_expands(tmp_path):
    g,t,o=graph_fixture(tmp_path);m={};r=Requests([result(status='AMBIGUOUS'),result('S2')])
    tm.topology_match(g,t,o,r,tmp_path/'audit',m,pending=dict(source_node='S1',target_node='S2'))
    assert [k for k,_ in r.calls[0][1]]==['current','oS1','oS2']
    assert m['representative_image_match_count']==2
    assert m['representative_image_fallback_count']==0


def test_fixed_recapture_candidates_never_expand(tmp_path):
    g,t,o=graph_fixture(tmp_path);r=Requests([result(status='AMBIGUOUS')])
    out=tm.topology_match(g,t,o,r,tmp_path/'audit',{},candidate_override=['S3','S4'])
    assert out['status']=='AMBIGUOUS' and len(r.calls)==1
    assert [k for k,_ in r.calls[0][1]]==['current','oS3','oS4']


@pytest.mark.parametrize('initial,counts',[(None,[6]),('S0',[1,6])])
def test_initial_global_or_explicit_s0(tmp_path,initial,counts):
    g,t,o=graph_fixture(tmp_path);m={};r=Requests(([result()] if initial else [])+[result('S1')])
    tm.topology_match(g,t,o,r,tmp_path/'audit',m,initial_node=initial)
    assert [len(p['nodes']) for p,_ in r.calls]==counts


def test_fallback_without_target_uses_global(tmp_path):
    g,t,o=graph_fixture(tmp_path);r=Requests([result('S1')])
    tm.topology_match(g,t,o,r,tmp_path/'audit',{},pending=dict(source_node='R_test',target_node=None))
    assert len(r.calls[0][0]['nodes'])==6


def test_out_of_candidate_id_rejected(tmp_path):
    g,t,o=graph_fixture(tmp_path);r=Requests([result('S4')])
    with pytest.raises(ValueError,match='unknown'):
        tm.topology_match(g,t,o,r,tmp_path/'audit',{},pending=dict(source_node='S1',target_node='S2'))


def test_no_forced_match_after_all_candidates_fail(tmp_path):
    g,t,o=graph_fixture(tmp_path);r=Requests([result(),result(),result()])
    out=tm.topology_match(g,t,o,r,tmp_path/'audit',{},pending=dict(source_node='S1',target_node='S2'))
    assert out['status']=='NO_MATCH' and len(r.calls)==3


def test_reference_corruption_never_falls_back_to_text(tmp_path):
    g,t,o=graph_fixture(tmp_path);r=Requests([result('S1')])
    o.image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='evidence modified'):
        tm.topology_match(g,t,o,r,tmp_path/'audit',{},candidate_override=['S1'])
    assert not r.calls


def test_missing_reference_never_falls_back_to_text(tmp_path):
    g,t,o=graph_fixture(tmp_path);r=Requests([result('S1')])
    g['match_nodes'][1]['representative_observations']=[]
    with pytest.raises(ValueError,match='lacks representative'):
        tm.topology_match(g,t,o,r,tmp_path/'audit',{},candidate_override=['S1'])
    assert not r.calls


def test_schema_field_descriptions_do_not_leak():
    from cloth_agent.experiments.graph_policy_schema import MATCH,NODE
    assert 'description' not in MATCH['properties']['matched_evidence']
    assert 'description' not in MATCH['properties']['candidate_nodes']
    assert 'description' not in NODE['properties']['representative_observations']
    assert MATCH['properties']['violated_conditions'] is not MATCH['properties']['unresolved_conditions']


def test_contradictory_match_gets_bounded_contract_repair(tmp_path):
    import json
    from cloth_agent.experiments.graph_policy_replay import policy_requests
    g,t,o=graph_fixture(tmp_path)
    bad=result('S2');bad['violated_conditions']=['S1 differs from current because its sleeve is still lateral']
    class Backend:
        model='offline-fixture'
        def __init__(self):self.calls=[]
        def invoke(self,prompt,images,schema):
            self.calls.append(prompt)
            return dict(text=json.dumps(bad if len(self.calls)==1 else result('S2')))
    backend=Backend();req=policy_requests(tmp_path/'model',backend,no_cache=True)
    out=tm.topology_match(g,t,o,req,tmp_path/'audit',{},candidate_override=['S1','S2'])
    assert out['node_id']=='S2' and len(backend.calls)==2
    assert 'Contradictory MATCH' in backend.calls[-1]
    assert 'do not erase genuine uncertainty' in backend.calls[-1]


def test_genuine_contradiction_cannot_pass_by_high_confidence(tmp_path):
    g,t,o=graph_fixture(tmp_path);bad=result('S2');bad['unresolved_conditions']=['selected fold layer is occluded']
    with pytest.raises(ValueError,match='Contradictory MATCH'):
        tm.topology_match(g,t,o,Requests([bad]),tmp_path/'audit',{},candidate_override=['S1','S2'])
