import pytest
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments.missed_grasp_retry import lowered_source,review
from cloth_agent.codex_motion import parse_code
from tests.test_graph_policy_replay import policy,Runtime,Model,matched


def test_lower_only_grasp():
    source='motion.grasp(right=[0.5,0.1,0.05,-3.14,0,0])\nmotion.move_tcp(right=[0.5,0.2,0.10,-3.14,0,0])\n'
    before=parse_code(source,allow_grasp=True);after=parse_code(lowered_source(source),allow_grasp=True)
    assert after[0][1]['right'][2]==pytest.approx(.048)
    assert after[0][1]['right'][:2]==before[0][1]['right'][:2]
    assert after[0][1]['right'][3:]==before[0][1]['right'][3:]
    assert after[1:]==before[1:]


@pytest.mark.parametrize('verdict,confidence,retries', [('UNCHANGED',.95,1),('UNCHANGED',.5,0),('MOVED',.99,0),('UNKNOWN',.99,0)])
def test_bounded_retry(tmp_path,monkeypatch,verdict,confidence,retries):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0')]*4)
    monkeypatch.setattr(pr.missed_grasp_retry,'review',lambda *a:dict(result=verdict,confidence=confidence,evidence='test'))
    # Fixture uses a different primitive skeleton; isolate retry dispatch from lowering test above.
    monkeypatch.setattr(pr.missed_grasp_retry,'lowered_source',lambda s:s)
    result=pr.replay(path,tmp_path/'run',runtime,model,max_steps=4,retry_missed_grasp=True,debug_grasp_retry=True)
    assert result['status']=='SOURCE_NODE_UNCHANGED'
    assert len(runtime.executions)==1+retries
    assert result['missed_grasp_retries']==retries
    assert runtime.auto_grasp_debug is False


@pytest.mark.parametrize('recording', [{}, {'recording': {'path': '/nonexistent/execution.mp4'}}])
def test_before_after_review_does_not_require_video(tmp_path,recording):
    import json
    class O:image=tmp_path/'after.png'
    class Reviewer:
        def ask(self,kind,payload,images):
            assert kind=='missed_grasp_review'
            assert images==[('before',tmp_path/'before.png'),('after',O.image)]
            assert 'executed_command' not in payload
            return dict(result='UNCHANGED',confidence=.95,evidence='Same garment boundary and folds')
    value=review(Reviewer(),dict(before=dict(path=str(tmp_path/'before.png')),execution=recording),O(),tmp_path/'review')
    assert value['result']=='UNCHANGED'
    saved=json.loads((tmp_path/'review'/'review.json').read_text())
    assert saved['assessment_basis']=='BEFORE_AFTER_RGB_ONLY'
    assert not list((tmp_path/'review').glob('frame_*.jpg'))


def test_retry_safety_rejection_stops(tmp_path,monkeypatch):
    path,g=policy(tmp_path);runtime=Runtime();model=Model([matched('N0')]*3)
    monkeypatch.setattr(pr.missed_grasp_retry,'review',lambda *a:dict(result='UNCHANGED',confidence=.99,evidence='test'))
    monkeypatch.setattr(pr.missed_grasp_retry,'lowered_source',lambda s:s)
    original=pr.execute_policy_edge;calls=[]
    def execute(*args):
        calls.append(args)
        return original(*args) if len(calls)==1 else dict(status='PLANNING_FAILED',physical_execution=False)
    monkeypatch.setattr(pr,'execute_policy_edge',execute)
    result=pr.replay(path,tmp_path/'run',runtime,model,max_steps=4,retry_missed_grasp=True)
    assert result['status']=='SAFETY_REJECT'
    assert len(runtime.executions)==1
    assert len(calls)==2
    assert calls[0][2].image != calls[1][2].image
