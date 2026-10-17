"""Regression: removed empty-grasp path must stay disabled, even with unstick debug."""
from cloth_agent.experiments import graph_policy_replay as pr
from tests.test_graph_policy_replay import policy,Model,matched,Runtime


def test_unchanged_node_has_no_empty_grasp_review_or_downward_retry(tmp_path,monkeypatch):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0'),matched('N0')])
    monkeypatch.setattr('builtins.input',lambda *a:(_ for _ in ()).throw(AssertionError('unexpected prompt')))
    result=pr.replay(path,tmp_path/'run',runtime,model,max_steps=3,debug_grasp_retry=True)
    assert result['status']=='SOURCE_NODE_UNCHANGED'
    assert len(runtime.executions)==1 and len(result['transitions'])==1
    assert [c[0] for c in model.calls]==['node_match','ground_edge','node_match']
    assert not list((tmp_path/'run').rglob('auto_grasp_retry'))
    assert 'grasp_retry_check' not in pr.SCHEMAS
