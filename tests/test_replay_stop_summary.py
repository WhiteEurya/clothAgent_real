import json
from cloth_agent.experiments.replay_stop_summary import print_stop_summary,describe_stop


def test_changed_source_prints_after_json(tmp_path,capsys):
    step=tmp_path/'step_002';retry=step/'grounding_retry_001';retry.mkdir(parents=True)
    (step/'host_selection.json').write_text(json.dumps(dict(node_id='S3',edge_id='e009')))
    (retry/'node_match.json').write_text(json.dumps(dict(result=dict(status='MATCH',node_id='S4',confidence=.86))))
    result=dict(status='EDGE_INSTANTIATION_FAILURE',error='grounding recapture no longer reliably matches selected edge source')
    summary=print_stop_summary(tmp_path,result)
    out=capsys.readouterr().out
    assert 'S3' in out and 'S4' in out and 'e009' in out and '0.86' in out
    assert '禁止沿用旧 edge' in out
    assert summary==json.loads((tmp_path/'stop_summary.json').read_text())


def test_no_step_does_not_invent_node(tmp_path):
    s=describe_stop(tmp_path,dict(status='INTERRUPTED',error='KeyboardInterrupt'))
    assert s['node'] is None and s['edge'] is None


def test_growth_episode_uses_latest_episode(tmp_path):
    p=tmp_path/'episode_0002';p.mkdir()
    (p/'episode.json').write_text(json.dumps(dict(status='SAFETY_REJECT',error='IK failed')))
    s=describe_stop(tmp_path,dict(status='SAFETY_REJECT',episodes=2))
    assert s['error']=='IK failed' and s['details']==str(p)


def test_match_with_violations_reports_actual_rejection(tmp_path):
    p=tmp_path/'step_000';p.mkdir()
    (p/'node_match.json').write_text(json.dumps(dict(result=dict(status='MATCH',node_id='S4',confidence=.92,violated_conditions=['S3 differs'],unresolved_conditions=[]))))
    s=describe_stop(tmp_path,dict(status='AMBIGUOUS'))
    assert '违反条件' in s['reason'] and s['error']=='S3 differs'


def test_high_confidence_conflict_not_described_as_low_confidence(tmp_path,capsys):
    p=tmp_path/'step_000';p.mkdir()
    (tmp_path/'runner_config.json').write_text(json.dumps(dict(match_threshold=.8)))
    (p/'node_match.json').write_text(json.dumps(dict(result=dict(status='MATCH',node_id='S4',
        confidence=.92,candidate_nodes=['S4'],matched_evidence=['fold visible'],
        violated_conditions=['S3 differs'],unresolved_conditions=[]))))
    s=print_stop_summary(tmp_path,dict(status='AMBIGUOUS'))
    text=capsys.readouterr().out
    assert '置信度：0.92' in text and '阈值：0.8' in text
    assert '违反条件' in s['reason'] and '低于' not in s['reason']


def test_actual_low_confidence_uses_recorded_threshold(tmp_path):
    p=tmp_path/'step_000';p.mkdir()
    (tmp_path/'runner_config.json').write_text(json.dumps(dict(match_threshold=.85)))
    (p/'node_match.json').write_text(json.dumps(dict(result=dict(status='MATCH',node_id='S4',
        confidence=.82,matched_evidence=['fold visible'],candidate_nodes=['S4'],
        violated_conditions=[],unresolved_conditions=[]))))
    s=describe_stop(tmp_path,dict(status='AMBIGUOUS'))
    assert '0.82' in s['reason'] and '0.85' in s['reason'] and '低于' in s['reason']


def test_model_failure_does_not_invent_visual_uncertainty(tmp_path):
    s=describe_stop(tmp_path,dict(status='ERROR',error='ModelInvocationError: process exited 1'))
    assert '调用失败' in s['reason'] and s['match_result'] is None
