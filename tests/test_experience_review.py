import json
import pytest
from cloth_agent.harness.common import write_json, read_json
from cloth_agent.harness.experience_review import review
from cloth_agent.harness.model import RuntimeClaude
from cloth_agent.planner_backend import BackendResult


def test_text_only_adapter_has_no_image_or_upload(monkeypatch,tmp_path):
    def invoke(self,**kw):
        assert self.allow_text_only
        assert kw['image_paths']==[]
        assert kw['direct_images']
        return BackendResult(json.dumps({'type':'result','structured_output':{'ok':True}}),'',0,())
    monkeypatch.setattr('cloth_agent.harness.model.RemoteClaudeBackend.invoke',invoke)
    model=RuntimeClaude(backend='remote',text_only=True)
    assert model.invoke(prompt='reflect',schema={'type':'object'},images=[],output=tmp_path/'call',stage='review')=={'ok':True}
    message=json.loads((tmp_path/'call/input.jsonl').read_text())
    assert all(c['type']=='text' for c in message['message']['content'])
    with pytest.raises(ValueError):
        RuntimeClaude().invoke(prompt='p',schema={},images=[],output=tmp_path/'vision',stage='plan')


@pytest.mark.parametrize('bad_citation',[False,True])
def test_review_one_call_no_replay_or_latency_gate(tmp_path,bad_citation):
    run=tmp_path/'run'
    write_json(run/'report.json',{'root_evidence_hash':'x','candidates':[],
        'rollouts':[{'rollout_id':'baseline_r00','status':'READY','metrics':{'elapsed_s':9999},
                     'observation_trace':[{'skill_id':'local_boundary','status':'EXECUTED'}]}]})
    class Model:
        calls=[]
        def invoke(self,**kwargs):
            assert kwargs['images']==[]
            assert 'latency optimization' in kwargs['prompt']
            self.calls.append({'response_received':True})
            return {'summary':'summary','open_questions':[], 'lessons':[{
                'name':'inspect','scope':'visual_information','information_need':'seam',
                'when_to_use':'unclear','method':['crop'],'success_check':'readable positive or negative',
                'on_insufficient':'UNKNOWN','limitations':['text review only'],
                'evidence':{'supported_by':[],'failed_in':[],
                            'unresolved':['invented' if bad_citation else 'baseline_r00']}}]}
    model=Model();model.calls=[]
    result=review(run,tmp_path/'out',model)
    assert len(model.calls)==1
    assert result['status']==('ERROR' if bad_citation else 'SUMMARIZED')
    assert result['replays']==0 and not result['automatic_activation']
    assert read_json(tmp_path/'out/context.json')['records'][0]['observation_trace']
