import json
from pathlib import Path
import pytest
from cloth_agent.experiments.skill_gate_online_batch import run_batch, validate_config
from cloth_agent.experiments.skill_gate_online import ROOT


def test_six_runs_with_separate_goals_and_no_expansion(tmp_path):
    from cloth_agent.experiments.discovery_data import file_hash
    goal=tmp_path/'target.png';goal.write_bytes(b'test')
    cfg={'seed':42,'goal_sha256':file_hash(goal)};calls=[]
    def execute(folder, run_id, seed):
        calls.append((run_id,seed))
        assert (folder/'goal/target.png').read_bytes()==b'test'
        return {'status':'ABSTAIN','initialization':{'status':'EXECUTED'}}
    result=run_batch(tmp_path/'batch',cfg,goal,execute,confirm_reset=lambda _: True)
    assert result['status']=='COMPLETED_SIX_RUNS'
    assert calls==[(i,42+i) for i in range(1,7)]
    with pytest.raises(FileExistsError):run_batch(tmp_path/'batch',cfg,goal,execute,confirm_reset=lambda _: True)


@pytest.mark.parametrize('status',['RESET_FAILED','EXECUTION_FAILED','ERROR'])
def test_failure_never_starts_next_run(tmp_path,status):
    from cloth_agent.experiments.discovery_data import file_hash
    goal=tmp_path/'target.png';goal.write_bytes(b'test');calls=[]
    def execute(*args):
        calls.append(1)
        return {'status':status,'initialization':{'status':'EXECUTED'}}
    result=run_batch(tmp_path/'batch',{'seed':42,'goal_sha256':file_hash(goal)},goal,execute,confirm_reset=lambda _: True)
    assert calls==[1] and result['status']=='STOPPED'


def test_current_protocol_goal_validates_without_hardware():
    cfg,goal=validate_config(ROOT/'config/skill_gate_online_pilot01.json')
    assert cfg['run_count']==6 and goal.is_file()


def test_no_confirmation_never_connects(tmp_path):
    from cloth_agent.experiments.discovery_data import file_hash
    goal=tmp_path/'target.png';goal.write_bytes(b'test')
    def execute(*args):
        pytest.fail('must not execute without manual restoration')
    result=run_batch(tmp_path/'batch', {'seed':42,'goal_sha256':file_hash(goal)},goal,execute)
    assert result['status']=='WAITING_FOR_MANUAL_RESET'
    assert result['runs']==[]


@pytest.mark.parametrize('status',['MODEL_REPORTED_DONE','ACTION_LIMIT','ABSTAIN','PLANNING_REJECTED'])
def test_every_normal_end_waits_before_next_initialization(tmp_path,status):
    from cloth_agent.experiments.discovery_data import file_hash
    goal=tmp_path/'target.png';goal.write_bytes(b'test');calls=[]
    def confirm(run_id):
        saved=json.loads((tmp_path/'batch/batch_summary.json').read_text())
        assert saved['status']=='WAITING_FOR_MANUAL_RESET'
        assert calls==list(range(1,run_id))
        return run_id==1
    def execute(folder,run_id,seed):
        calls.append(run_id)
        return {'status':status,'initialization':{'status':'EXECUTED'}}
    result=run_batch(tmp_path/'batch',{'seed':42,'goal_sha256':file_hash(goal)},goal,execute,confirm_reset=confirm)
    assert calls==[1]
    assert result['status']=='WAITING_FOR_MANUAL_RESET'
    assert result['next_run_id']==2
    assert (tmp_path/'batch/manual_reset_01.json').is_file()
    assert not (tmp_path/'batch/run_02').exists()


def test_confirmation_requires_exact_input(monkeypatch):
    from cloth_agent.experiments.skill_gate_online_batch import confirm_manual_reset
    answers=iter(['invalid', 'Y'])
    monkeypatch.setattr('builtins.input',lambda _:next(answers))
    assert confirm_manual_reset(2) is True


def test_eof_does_not_confirm(monkeypatch):
    from cloth_agent.experiments.skill_gate_online_batch import confirm_manual_reset
    def eof(_): raise EOFError
    monkeypatch.setattr('builtins.input',eof)
    assert confirm_manual_reset(2) is False


@pytest.mark.parametrize('answer',['','n','N'])
def test_default_or_no_does_not_confirm(monkeypatch,answer):
    from cloth_agent.experiments.skill_gate_online_batch import confirm_manual_reset
    monkeypatch.setattr('builtins.input',lambda _:answer)
    assert confirm_manual_reset(2) is False
