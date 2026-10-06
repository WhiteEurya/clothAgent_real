import pytest

from cloth_agent.harness.common import read_json
from cloth_agent.harness.reasoning_timing import run_once
from tests.test_reasoning_code import Model, baseline, final


def test_one_fresh_call_without_learning_or_historical_answers(tmp_path):
    source = baseline(tmp_path)
    model = Model(final()['judgment'])
    output = tmp_path/'timed'
    report = run_once(source, output, model)
    assert report['status'] == 'READY'
    assert report['call_attempts'] == len(model.requests) == 1
    assert report['learning_calls'] == report['robot_actions'] == 0
    assert model.requests[0]['stage'] == 'reasoning_rollout'
    assert 'HISTORICAL_SECRET_ANSWER' not in model.requests[0]['prompt']
    assert model.requests[0]['images']
    assert (output/'reasoning/trace/trajectory.json').is_file()
    assert (output/'timing/semantic_timeline.csv').is_file()
    assert (output/'timing/summary.json').is_file()
    with pytest.raises(FileExistsError):
        run_once(source, output, model)
    assert len(model.requests) == 1


def test_invalid_reply_stops_without_retry_or_reflection(tmp_path):
    model = Model({'status': 'invalid'})
    output = tmp_path/'failed'
    report = run_once(baseline(tmp_path), output, model)
    assert report['status'] == 'ERROR'
    assert len(model.requests) == 1
    assert read_json(output/'report.json')['learning_calls'] == 0
    assert (output/'timing/summary.md').is_file()
