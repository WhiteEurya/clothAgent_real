import json
from cloth_agent.harness import reasoning_experience_trial as trial
from cloth_agent.harness.common import write_json
from cloth_agent.harness.reasoning_contract import freeze_evidence, freeze_harness, baseline_harness
from test_observation_code_trial import setup, Model


def test_snapshot_replay_and_reflection_do_not_supply_baseline_answer(tmp_path, monkeypatch):
    evidence,_=setup(tmp_path)
    baseline=tmp_path/'old';baseline.mkdir()
    frozen=freeze_evidence(evidence,baseline/'prepared')
    freeze_harness(baseline_harness(),baseline/'reasoning_version.json')
    write_json(baseline/'result.json',{'secret_old_answer':'NEVER_SEND_OLD_ACTION'})
    def replay(version, current, evidence_dir, model, output, budget, **kw):
        assert current==frozen
        assert 'NEVER_SEND_OLD_ACTION' not in json.dumps(current)
        return {'status':'READY','reason':'test','action':{'fresh':'answer'},
                'stages':[{'stage_id':'plan','judgment':{'evidence_summary':'fresh findings'}}],
                'metrics':{},'host_operations':[]}
    monkeypatch.setattr(trial,'run_rollout',replay)
    text=Model({'summary':'test','lessons':[],'open_questions':['Needs reuse test']})
    output=tmp_path/'new'
    result=trial.run(baseline,output,Model(),text)
    assert result['status']=='SUMMARIZED'
    assert (output/'baseline/result.json').read_text()==(baseline/'result.json').read_text()
    assert 'NEVER_SEND_OLD_ACTION' not in text.requests[0]['prompt']
    assert not text.requests[0]['images']
    assert result['robot_actions']==0 and not result['learned_policy_replayed']
