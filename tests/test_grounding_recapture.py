from types import SimpleNamespace
from pathlib import Path
import pytest
from cloth_agent.experiments import graph_policy_replay as pr
from cloth_agent.experiments.graph_minimal_grounding import GroundingUncertain

@pytest.mark.parametrize('mode',['success','uncertain','state_changed','depth_failure','first_success'])
def test_recapture_bound_cache_and_source_check(tmp_path,monkeypatch,mode):
    a=tmp_path/'a.png';a.write_bytes(b'a');b=tmp_path/'b.png';b.write_bytes(b'b')
    old=SimpleNamespace(image=a);fresh=SimpleNamespace(image=b);observations=[];calls=[]
    runtime=SimpleNamespace(observe=lambda folder:observations.append(folder) or fresh)
    requests=SimpleNamespace(no_cache=False)
    def ground(req,g,e,obs,folder,hem):
        calls.append(obs)
        if len(calls)==1:
            if mode=='depth_failure':raise ValueError('depth spread rejected')
            if mode=='first_success':return {'status':'READY'}
            raise GroundingUncertain('minimal anchors unresolved/low confidence')
        assert req.no_cache
        if mode=='uncertain':raise GroundingUncertain('still uncertain')
        return {'status':'READY'}
    monkeypatch.setattr(pr,'selected_edge_grounding',ground)
    monkeypatch.setattr(pr,'topology_match',lambda *a,**k:{'node_id':'other' if mode=='state_changed' else 'source'})
    monkeypatch.setattr(pr,'reliable_match',lambda *a:True)
    monkeypatch.setattr('cloth_agent.experiments.recapture_consistency.review_pair',lambda *a,**k:dict(observation_relation='UNKNOWN',current_match={'node_id':'other'}))
    metrics={};edge={'edge_id':'E','source_node':'source','target_node':'target'}
    def run():return pr.grounding_with_recapture(runtime,requests,{},None,edge,old,tmp_path,metrics,.8)
    if mode in {'success','first_success'}:
        candidate,obs=run();assert obs is (fresh if mode=='success' else old)
    else:
        with pytest.raises(ValueError):run()
    assert not requests.no_cache
    expected=2 if mode in {'success','uncertain'} else 1
    assert len(calls)==expected
    assert len(observations)==(0 if mode in {'depth_failure','first_success'} else 1)
    if observations:assert (tmp_path/'grounding_retry_001/result.json').exists()
