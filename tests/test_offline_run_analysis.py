import json
import pytest
from cloth_agent.experiments.offline_run_analysis import collect, write_report


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_execution_is_not_grasp_and_effect_belongs_to_previous_action(tmp_path):
    episode=tmp_path/'episode_0001'
    first=episode/'round_001';second=episode/'round_002'
    save(first/'execution.json',dict(status='EXECUTED',physical_execution=True))
    save(first/'timing.json',dict(planning_s=10,fast_planner_wall_s=10,total_round_s=20))
    save(first/'target_replan/original/plan.json',dict(decision='ACTION'))
    save(second/'execution.json',dict(status='SAFETY_REJECTED',physical_execution=False))
    save(second/'previous_action_context.json',dict(source_round='round_001'))
    save(second/'decision_evidence.json',dict(previous_effect=dict(verdict='UNCHANGED')))
    result=collect(tmp_path)
    assert result['summary']['rounds']==2
    assert result['summary']['timing_s']['planning_s']==10
    assert result['rounds'][0]['grasp']=='UNCERTAIN'
    assert result['rounds'][0]['effect_model']['verdict']=='UNCHANGED'
    assert result['rounds'][1]['effect_model'] is None
    assert result['rounds'][1]['grasp']=='NOT_EXECUTED'
    out=tmp_path/'report'
    assert write_report(result,out).exists()
    assert (out/'index.html').exists()
    with pytest.raises(FileExistsError):write_report(result,out)


def test_labels_require_evidence_and_dispatch(tmp_path):
    save(tmp_path/'round_001/execution.json',dict(status='EXECUTED',physical_execution=True))
    with pytest.raises(ValueError,match='requires evidence'):
        collect(tmp_path,{'round_001':dict(grasp='EMPTY')})
    result=collect(tmp_path,{'round_001':dict(grasp='EMPTY',evidence='人工观察提起后夹爪内无布料')})
    assert result['summary']['human_reviewed_grasps']==1
    save(tmp_path/'round_001/execution.json',dict(status='SAFETY_REJECTED',physical_execution=False))
    with pytest.raises(ValueError,match='unexecuted'):
        collect(tmp_path,{'round_001':dict(grasp='EMPTY',evidence='invalid')})


def test_corrupt_record_reported_and_no_cross_episode_effect(tmp_path):
    for ep in ('episode_0001','episode_0002'):
        save(tmp_path/ep/'round_001/execution.json',dict(status='EXECUTED',physical_execution=True))
    save(tmp_path/'episode_0002/round_002/previous_action_context.json',dict(source_round='round_001'))
    save(tmp_path/'episode_0002/round_002/decision_evidence.json',dict(previous_effect=dict(verdict='IMPROVED')))
    (tmp_path/'episode_0001/round_001/timing.json').write_text('{broken')
    result=collect(tmp_path)
    assert len(result['warnings'])==1
    assert result['rounds'][0]['effect_model'] is None
    assert result['rounds'][1]['effect_model']['verdict']=='IMPROVED'
