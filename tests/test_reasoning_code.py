import copy
import json

import pytest

from cloth_agent.harness import reasoning_code as rc
from cloth_agent.harness.common import write_json
from cloth_agent.harness.executors.restricted import RestrictedProgram
from cloth_agent.harness.policy import PolicyError
from cloth_agent.harness.reasoning_contract import baseline_harness, freeze_evidence, freeze_harness
from tests.test_observation_code_trial import setup


def function():
    return {
        'id': 'sum_parameters', 'purpose': 'Sum an explicit measured numeric list',
        'extracted_operation': 'Repeated addition', 'when_to_use': 'When a sum is needed',
        'success_check': 'Sum returned', 'on_insufficient': 'Use UNKNOWN for missing measurements',
        'input_schema_json': json.dumps({'type': 'object', 'properties': {'values': {'type': 'array', 'maxItems': 128, 'items': {'type': 'number'}}}, 'required': ['values'], 'additionalProperties': False}),
        'output_schema_json': json.dumps({'type': 'number'}),
        'source': 'def run(arguments):\n    total = 0\n    for value in arguments["values"]:\n        total = total + value\n    return total',
        'evidence': {'supported_by': ['baseline_baseline_r00'], 'failed_in': [], 'unresolved': ['Semantic correctness unknown']},
        'tests': [{'name': str(i), 'arguments_json': json.dumps({'values': v}), 'expected_json': json.dumps(e),
                   'kind': 'synthetic', 'record_id': '', 'explanation': 'Synthetic arithmetic test'}
                  for i, (v, e) in enumerate([([1, 2], 3), ([-4, 2, 7], 5), ([], 0)])],
    }


def test_generated_code_runs_new_parameters_and_bounds_execution():
    f = function()
    assert rc.execute_function(f, {'values': [100, -20, 3]})['output'] == 83
    assert rc.execute_function(f, {'values': [1]*129})['status'] == 'UNKNOWN'
    f['source'] = 'def run(arguments):\n    for a in arguments["values"]:\n        for b in arguments["values"]:\n            for c in arguments["values"]:\n                total = a + b + c\n    return 0'
    result = rc.execute_function(f, {'values': [1]*128})
    assert result['status'] == 'UNKNOWN' and 'budget' in result['error']


@pytest.mark.parametrize('source', [
    'import os\ndef run(arguments):\n return 0',
    'def run(arguments):\n return arguments.clear()',
    'def run(arguments):\n return run(arguments)',
    'def run(arguments):\n arguments["values"] = []\n return 0',
])
def test_generated_code_has_no_side_effect_capabilities(source):
    f = function(); f['source'] = source
    assert rc.execute_function(f, {'values': [1]})['status'] == 'UNKNOWN'


def test_legacy_observation_program_does_not_gain_loops():
    with pytest.raises(PolicyError):
        RestrictedProgram('def prepare(request, source, available):\n for x in request:\n  return x')


def test_gates_do_not_accept_invented_record_output():
    f = function()
    records = [{'record_id': 'baseline_baseline_r00', 'output': 99}]
    assert rc.test_function(f, records)['passed']
    f['tests'][0].update(kind='record_replay', record_id='baseline_baseline_r00')
    assert not rc.test_function(f, records)['passed']


def test_numeric_fixture_rounding_does_not_reject_correct_float_computation():
    assert rc.equivalent(10.079954335260116, 10.079954456322348)
    assert not rc.equivalent(10.07, 10.08)
    assert not rc.equivalent(1000000000, 1000000001)


def test_citation_explanations_are_resolved_without_discarding_valid_code():
    f = function()
    f['evidence']['supported_by'] = ['baseline_baseline_r00: public arithmetic result']
    records = [{'record_id': 'baseline_baseline_r00'}]
    gate = rc.test_function(f, records)
    assert gate['passed']
    assert gate['citation_resolution'][0]['record_id'] == 'baseline_baseline_r00'
    f['evidence']['supported_by'] = ['not_a_record: explanation']
    with pytest.raises(PolicyError, match='unknown'):
        rc.test_function(f, records)


class Model:
    configuration = {'actual_measurement': False}
    def __init__(self, *answers, text_only=False):
        self.answers = list(answers); self.calls = []; self.requests = []; self.text_only = text_only
    def invoke(self, **kw):
        self.requests.append(kw)
        assert bool(kw['images']) != self.text_only
        kw['output'].mkdir(parents=True)
        self.calls.append({'response_received': True})
        return copy.deepcopy(self.answers.pop(0))


def final():
    return {'kind': 'FINAL', 'calls': [], 'judgment': {
        'observation_id': 'synthetic', 'status': 'READY', 'concepts': [],
        'action': {'selected_reference': {'camera': 'A', 'reference_id': 'R001', 'reason': 'Visible fabric'},
                   'target': {'pixel_xy': [40, 40], 'relation': 'toward', 'anchor_pixel_xy': [50, 40], 'reason': 'Current geometry'}},
        'evidence_summary': 'Current image supports proposal', 'missing_information': '', 'residual_uncertainty': 'Physical validity untested'}}


def baseline(tmp_path):
    evidence, _ = setup(tmp_path)
    root = tmp_path/'baseline'; replay = root/'replays/baseline_r00'
    freeze_evidence(evidence, replay/'prepared')
    freeze_harness(baseline_harness(), replay/'reasoning_version.json')
    write_json(root/'report.json', {'root_evidence_hash': 'root', 'rollouts': [
        {'rollout_id': 'baseline_r00', 'status': 'READY', 'action': 'HISTORICAL_SECRET_ANSWER', 'reasoning': {'stages': []}}]})
    return root


def test_two_iterations_keep_original_planner_and_reuse_library_feedback(tmp_path):
    old = baseline(tmp_path)
    extraction = {'summary': 'Extract sum', 'functions': [function()], 'left_to_model': ['All visual judgments']}
    extractor = Model(extraction, {'summary': 'Existing function sufficient', 'functions': [], 'left_to_model': []}, text_only=True)
    request = {'kind': 'REQUEST_CODE', 'judgment': None, 'calls': [
        {'function_id': 'sum_parameters', 'arguments_json': '{"values":[8,9]}', 'reason': 'Sum current measurements', 'source_image_ids': ['image_0']}]}
    planner = Model(request, final(), final())
    result = rc.run_extraction(old, None, tmp_path/'run', extractor, planner, iterations=2)
    assert result['status'] == 'COMPLETED', result
    assert [r['planning_status'] for r in result['iterations']] == ['READY', 'READY']
    assert [r['code_usage'] for r in result['iterations']] == ['USED', 'NOT_USED']
    assert result['original_workflow_preserved']
    assert 'HISTORICAL_SECRET_ANSWER' in extractor.requests[0]['prompt']
    assert all('HISTORICAL_SECRET_ANSWER' not in r['prompt'] for r in planner.requests)
    assert all('record_replay' not in r['prompt'] for r in planner.requests)
    assert '"output":17' in planner.requests[1]['prompt']
    assert '"output":17' in extractor.requests[1]['prompt']
    assert 'operation_trace' in extractor.requests[1]['prompt']
    assert (tmp_path/'run/iter_00/reasoning/trace/trajectory.json').exists()
    assert all(baseline_harness()['stages'][0]['instruction'] in r['prompt'] for r in planner.requests)
    assert (tmp_path/'run/iter_00/functions/sum_parameters/function.py').exists()


def test_invalid_code_can_be_repaired_from_next_iteration_feedback(tmp_path):
    old = baseline(tmp_path)
    bad = function(); bad['source'] = 'def run(arguments):\n return 999'
    extractor = Model({'summary': 'bad', 'functions': [bad], 'left_to_model': []},
                      {'summary': 'repair', 'functions': [function()], 'left_to_model': []}, text_only=True)
    planner = Model(final())
    report = rc.run_extraction(old, None, tmp_path/'run', extractor, planner)
    assert report['iterations'][0]['status'] == 'NO_EXECUTABLE_FUNCTION'
    assert report['iterations'][1]['planning_status'] == 'READY'
    assert len(planner.requests) == 1
    assert '"passed":false' in extractor.requests[1]['prompt']


def test_unknown_function_result_is_returned_to_planner(tmp_path):
    old = baseline(tmp_path)
    extractor = Model({'summary': 'sum', 'functions': [function()], 'left_to_model': []}, text_only=True)
    request = {'kind': 'REQUEST_CODE', 'judgment': None, 'calls': [
        {'function_id': 'missing_function', 'arguments_json': '{}', 'reason': 'try', 'source_image_ids': []}]}
    planner = Model(request, final())
    report = rc.run_extraction(old, None, tmp_path/'run', extractor, planner, iterations=1)
    assert report['iterations'][0]['planning_status'] == 'READY'
    assert 'UNKNOWN' in planner.requests[1]['prompt']


def test_replay_saved_extraction_never_regenerates_it(tmp_path):
    old = baseline(tmp_path)
    extractor = Model({'summary': 'sum', 'functions': [function()], 'left_to_model': []}, text_only=True)
    first = rc.run_extraction(old, None, tmp_path/'first', extractor, Model(final()), iterations=1)
    assert first['status'] == 'COMPLETED'
    no_generation = Model(text_only=True)
    report = rc.run_extraction(old, None, tmp_path/'recovered', no_generation, Model(final()),
                               iterations=1, reuse_extractions=tmp_path/'first')
    assert report['iterations'][0]['planning_status'] == 'READY'
    assert not no_generation.requests
    assert report['extraction_reuse']['fresh_extraction_calls'] == 0
