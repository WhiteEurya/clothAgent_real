from types import SimpleNamespace
import json
import pytest
from rax_cloth.overnight import run
from rax_cloth.session import write_json


def args(tmp_path,**changes):
    values=dict(session=tmp_path/'batch',mode='mock',model='fixture-model',profile='chatgpt',
        replay_manifest=tmp_path/'unused',max_hours=8,episodes=3,max_model_calls=5,
        episode_calls=3,episode_seconds=20,seed=0,budget_usd=None,resume=False,progress_interval=10)
    values.update(changes)
    return SimpleNamespace(**values)


def finish(options,status='agent_completed',calls=2,seconds=4):
    private=options.session/'codex-adaptation'
    (private/'model').mkdir(parents=True,exist_ok=True)
    write_json(private/'agent-state.json',dict(status=status,elapsed_seconds=seconds,
        attempts=[{'status':status}],thread_id='fixture-thread'))
    write_json(private/'model'/'ledger.json',dict(calls_reserved=calls))
    return dict(status=status,task_success=None,physical_actions=0)


def test_batch_uses_official_launcher_and_shared_budget(tmp_path):
    seen=[]
    def launch(o):
        seen.append(o)
        assert o.workflow=='upstream' and o.robot_config is None
        return finish(o,calls=min(2,o.max_model_calls))
    result=run(args(tmp_path),launch)
    assert [o.max_model_calls for o in seen]==[3,3,1]
    assert result['model_calls_reserved']==5
    assert result['task_success'] is None and result['physical_actions']==0


def test_error_stops_without_retry_and_explicit_resume_keeps_caps_thread(tmp_path):
    calls=[]
    def fail(o):
        calls.append(o)
        return finish(o,status='interrupted_or_failed')
    config=args(tmp_path,episodes=1)
    assert run(config,fail)['status']=='stopped_on_episode_error'
    assert len(calls)==1
    def resume(o):
        assert o.resume and o.max_model_calls==3 and o.max_seconds==20
        return finish(o,calls=3,seconds=8)
    config.resume=True
    result=run(config,resume)
    assert result['status']=='episodes_completed'
    assert result['model_calls_reserved']==3 and result['charged_seconds']==8


@pytest.mark.parametrize('changes',[{'mode':'real_supervised'},{'mode':'sim'}, {'budget_usd':1}])
def test_no_real_sim_or_unenforceable_dollar_cap(tmp_path,changes):
    with pytest.raises(ValueError):run(args(tmp_path,**changes),lambda o:pytest.fail('must not launch'))


def test_resume_never_changes_budgets(tmp_path):
    config=args(tmp_path,episodes=1)
    run(config,finish)
    config.resume=True;config.max_model_calls+=1
    with pytest.raises(ValueError,match='identical'):run(config,finish)


def test_missing_crash_evidence_does_not_refund_reservation(tmp_path):
    def crash(o):raise RuntimeError('fixture crash before ledger')
    config=args(tmp_path)
    result=run(config,crash)
    assert result['model_calls_reserved'] is None
    assert result['model_call_budget_charged']==3 and result['charged_seconds']==20
    assert result['status']=='stopped_on_episode_error'
