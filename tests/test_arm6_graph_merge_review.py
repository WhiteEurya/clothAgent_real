import copy

import numpy as np
import pytest

from scripts.review_arm6_graph_merges import apply_round, candidates, validate_review


def source():
    return dict(graph_id='test', nodes=[dict(node_id=n, representative_observations=[o],
        task_context=n, confidence=.9, observation_condition=dict(positive_evidence=['visible'],
        negative_evidence=[], uncertainty_conditions=[])) for n,o in [('a','o1'),('b','o2'),('c','o3')]],
        edges=[dict(edge_id='e1',source_node='b',target_node='c',seed_action={'exact':'unchanged'})],
        terminal_node_ids=[])


def pair(pid,a,b):
    refs={'a':['o1'],'b':['o2'],'c':['o3']}
    return dict(pair_id=pid,nodes=[a,b],similarity=.95,sink_nodes=[a] if a=='a' else [],
                observation_members={a:refs[a],b:refs[b]})


def decision(p, confidence=.95):
    return dict(pair_id=p['pair_id'],decision='MERGE',confidence=confidence,reason='same exposed regions',
        evidence_observations=[o for r in p['observation_members'].values() for o in r],
        task_context='shared',positive_evidence=['visible'],negative_evidence=[],uncertainty_conditions=[])


def test_preserves_edges_and_evidence_and_source_without_transitive_merge():
    original=source(); snapshot=copy.deepcopy(original)
    pairs=[pair('p0','a','b'),pair('p1','b','c')]
    result,mapping,applied,deferred=apply_round(original,pairs,dict(decisions=[decision(p) for p in pairs]))
    assert original==snapshot
    assert mapping=={'a':'a','b':'a','c':'c'}
    assert result['edges']==[dict(edge_id='e1',source_node='a',target_node='c',seed_action={'exact':'unchanged'})]
    assert result['nodes'][0]['representative_observations']==['o1','o2']
    assert len(applied)==1 and deferred[0]['pair_id']=='p1'


def test_uncertain_and_low_confidence_do_not_merge():
    pairs=[pair('p0','a','b')]; d=decision(pairs[0],.84)
    assert not apply_round(source(),pairs,dict(decisions=[d]))[2]
    d.update(decision='UNCERTAIN',confidence=.99)
    assert not apply_round(source(),pairs,dict(decisions=[d]))[2]


def test_reject_stale_membership_and_missing_review():
    p=pair('p0','a','b'); d=decision(p)
    p['observation_members']['a']=['o1','extra']
    with pytest.raises(ValueError,match='membership'):
        apply_round(source(),[p],dict(decisions=[d]))
    with pytest.raises(ValueError,match='exactly once'):
        validate_review(dict(decisions=[]),[p])


def test_reject_one_sided_evidence():
    p=pair('p0','a','b'); d=decision(p);d['evidence_observations']=['o1']
    with pytest.raises(ValueError,match='both nodes'):
        validate_review(dict(decisions=[d]),[p])


def test_candidates_include_explicit_focus_below_similarity_cutoff():
    memory={'images':[{'node_id':n} for n in ['a','b','c']]}
    result=candidates(source(),memory,np.eye(3,dtype=np.float32),focus=[['a','c']])
    assert len(result)==1 and result[0]['nodes']==['a','c']
    assert result[0]['selection_reasons']==['explicit_case_review']
    with pytest.raises(ValueError,match='distinct observed'):
        candidates(source(),memory,np.eye(3),focus=[['a','missing']])
