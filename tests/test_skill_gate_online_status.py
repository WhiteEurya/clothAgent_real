import json
import pytest
from cloth_agent.experiments.skill_gate_online_status import describe_batch, resolve_batch, print_report
from cloth_agent.experiments.skill_gate_online_batch import main


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_diagnosis_preserves_distance_and_reset_error(tmp_path,capsys):
    root=tmp_path/'skill_gate_online_test'
    write(root/'batch_summary.json',{'status':'STOPPED','reason':'IK rejected target',
        'runs':[{'run_id':1,'status':'PLANNING_REJECTED'},{'run_id':2,'status':'RESET_FAILED'}]})
    write(root/'run_01/round_000/exploration.json',{'forced_skill':'SIMPLE',
        'planning_status':'ACTION','model_requests':1,'execution':{'status':'NOT_DISPATCHED'}})
    write(root/'run_01/round_000/execution.json',{'status':'EXECUTED'})
    write(root/'run_01/round_001/exploration.json',{'forced_skill':'SIMPLE','planning_status':'ABSTAIN'})
    write(root/'run_01/round_001/translation_rejection.json',{'distance_mm':180.7,'minimum_mm':10,'maximum_mm':150})
    write(root/'run_02/episode_summary.json',{'status':'RESET_FAILED','error':'IK rejected target',
        'initialization':{'status':'PLANNING_FAILED'}})
    report=describe_batch(root)
    assert report['executed_skills']=={'SIMPLE':1,'TANGENT':0}
    assert report['runs'][0]['steps'][1]['distance_mm']==180.7
    assert report['runs'][1]['stage']=='初始化'
    print_report(report)
    output=capsys.readouterr().out
    assert '180.7 mm' in output and 'IK rejected target' in output
    assert resolve_batch('latest',tmp_path)==root


def test_partial_write_is_unknown_not_completed(tmp_path):
    (tmp_path/'batch_summary.json').write_text('{')
    assert describe_batch(tmp_path)['status']=='UNKNOWN_OR_UPDATING'


def test_status_does_not_validate_or_start_hardware(tmp_path,monkeypatch,capsys):
    from cloth_agent.experiments import skill_gate_online_batch as batch
    write(tmp_path/'batch_summary.json',{'status':'STOPPED','runs':[]})
    monkeypatch.setattr(batch,'validate_config',lambda *a:pytest.fail('status must not initialize experiment'))
    assert main(['--status',str(tmp_path),'--json'])==0
    assert json.loads(capsys.readouterr().out)['status']=='STOPPED'


def test_invalid_protocol_returns_nonzero_with_reason(tmp_path,capsys):
    assert main(['--protocol',str(tmp_path/'missing.json')])==1
    assert 'protocol mismatch' in capsys.readouterr().out
