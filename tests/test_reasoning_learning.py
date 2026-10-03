"""Synthetic-only fixtures/mocks: never evidence of real optimization or success."""
import copy
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from cloth_agent.harness.action_consensus import select_consensus
from cloth_agent.harness.common import digest, read_json, write_json
from cloth_agent.harness.model import RuntimeClaude
from cloth_agent.harness.policy import PolicyError
from cloth_agent.harness.reasoning_contract import (
    baseline_harness, evidence_from_manifest, freeze_evidence, freeze_harness,
    load_evidence, load_harness, validate_harness, verify_evidence,
)
from cloth_agent.harness.reasoning_learning import (
    CallBudget, DebugLog, call_metrics, execution_signature, main, run_learning, run_rollout,
)
from cloth_agent.image_tools_mcp import pixel_hash
from cloth_agent.planner_backend import BackendResult


@pytest.fixture
def evidence(tmp_path):
    directory = tmp_path / 'source'
    directory.mkdir()
    images = []
    for role, color in [('clean', 'white'), ('overlay', 'blue'), ('reference', 'green')]:
        im = Image.new('RGB', (100, 80), color)
        path = directory / f'{role}.png'
        im.save(path)
        images.append({'path': str(path), 'role': role, 'status': 'AVAILABLE',
                       'size': [100, 80], 'rgb_sha256': pixel_hash(im)})
    return {'schema_version': 1, 'observation_id': 'synthetic_observation', 'fold_goal': 'Fold the visible flap inward',
            'images': images, 'candidate_registry': {'observation_id': 'synthetic_observation',
                'binding': 'RAW_RGB_HASH_VERIFIED', 'raw_size': [80, 100], 'to_raw': [0, 1, 0, -1, 0, 99],
                'candidates': [{'camera': 'A', 'candidate_id': 'R001', 'pixel_xy': [20, 30], 'raw_pixel_xy': [30, 79]},
                               {'camera': 'A', 'candidate_id': 'R002', 'pixel_xy': [80, 50], 'raw_pixel_xy': [50, 19]}]}}


def answer(**changes):
    result = {'observation_id': 'synthetic_observation', 'status': 'READY',
              'concepts': [{'name': 'synthetic_anchor', 'finding': 'Visible test marker', 'source_image_ids': ['image_0']}],
              'action': {'selected_reference': {'camera': 'A', 'reference_id': 'R001', 'reason': 'Synthetic grasp evidence'},
                         'target': {'pixel_xy': [50, 30], 'relation': 'toward', 'anchor_pixel_xy': [60, 40], 'reason': 'Synthetic target evidence'}},
              'evidence_summary': 'SYNTHETIC_REASON_NOT_A_REAL_RESULT', 'missing_information': '', 'residual_uncertainty': ''}
    result.update(changes)
    return result


def changed_harness(label='refined'):
    harness = baseline_harness()
    harness['name'] = label
    harness['stages'][0]['instruction'] = 'Use the visible boundary and sufficient evidence to decide. ' + label
    return harness


def proposal(harness, parent='h00_r00'):
    return {'status': 'PROPOSE', 'reason': 'Synthetic modification only', 'harness': harness,
            'changes': [{'field': 'stages', 'observed': 'Synthetic logged planning stage',
                         'hypothesis': 'May require less repeated explanation', 'unverified': 'Latency and correctness unverified',
                         'source_rollout_ids': [parent]}]}


class MockModel:
    configuration = {'backend': 'MOCK_TEST_ONLY', 'actual_measurement': False}

    def __init__(self, *outputs):
        self.outputs, self.requests, self.calls = list(outputs), [], []

    def invoke(self, **kw):
        self.requests.append(kw)
        self.calls.append({'backend_invoked': True, 'response_received': True, 'tool_round_trips': 0,
                           'configuration': self.configuration})
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            self.calls[-1]['response_received'] = False
            raise result
        return result(kw) if callable(result) else copy.deepcopy(result)


def run(evidence, tmp_path, model, **kwargs):
    return run_learning(evidence, tmp_path / 'experiment', model, **kwargs)


def test_serial_reflection_is_externalized_and_used_without_previous_answers(evidence, tmp_path):
    second, third = changed_harness('refined'), changed_harness('combined')
    model = MockModel(answer(), proposal(second), answer(), proposal(third, 'h01_r00'), answer())
    report = run(evidence, tmp_path, model, variants=2)
    assert report['status'] == 'SELECTED'
    assert report['actual_measurement'] is False
    assert report['totals']['model_calls'] == 5
    assert report['consensus']['selected']['votes'] == 3
    assert report['versions'][2]['parent_hash'] == report['versions'][1]['harness_hash']
    plans = [r for r in model.requests if r['stage'] == 'reasoning_rollout']
    assert len(plans) == 3
    assert second['stages'][0]['instruction'] in plans[1]['prompt']
    assert third['stages'][0]['instruction'] in plans[2]['prompt']
    assert all('SYNTHETIC_REASON_NOT_A_REAL_RESULT' not in r['prompt'] for r in plans)
    assert 'SYNTHETIC_REASON_NOT_A_REAL_RESULT' in model.requests[1]['prompt']
    assert all(r['images'] == plans[0]['images'] for r in model.requests)
    output = tmp_path / 'experiment'
    for path in ['events.jsonl', 'report.json', 'report.md', 'index.html', 'selected_harness.json', 'selected_action.json',
                 'reflections/patch_01/harness.diff', 'reflections/patch_02/proposal.json', 'rollouts/h00_r00/state.json']:
        assert (output / path).is_file()
    assert load_harness(output / 'selected_harness.json')
    assert all(r['metrics']['thinking_tokens'] is None for r in report['rollouts'])


def test_branch_reflections_share_parent_and_repeated_names_do_not_vote(evidence, tmp_path):
    model = MockModel(answer(), proposal(changed_harness('one')), answer(), proposal(changed_harness('two')), answer())
    report = run(evidence, tmp_path, model, variants=2, search='branch')
    assert report['status'] == 'SELECTED'
    assert all(v['parent_hash'] == report['versions'][0]['harness_hash'] for v in report['versions'][1:])
    for request in [r for r in model.requests if r['stage'] == 'reasoning_reflection']:
        assert 'h01_r00' not in request['prompt']


def test_multistage_bound_state_recomputed_and_early_stop(evidence, tmp_path):
    harness = baseline_harness()
    harness['stages'][0]['allow_ready'] = False
    harness['stages'].append({'id': 'select', 'context': ['plan'], 'allow_ready': True, 'instruction': 'Use the freshly computed concepts.'})
    intermediate = answer(status='CONTINUE', action=None, evidence_summary='FRESH_CONCEPT_FOR_THIS_ROLLOUT')
    model = MockModel(intermediate, answer(), intermediate, answer())
    report = run(evidence, tmp_path, model, initial=harness, variants=0, repeats=2)
    assert len(report['rollouts']) == 2 and report['totals']['model_calls'] == 4
    assert all(r['status'] == 'READY' for r in report['rollouts'])
    assert 'FRESH_CONCEPT_FOR_THIS_ROLLOUT' in model.requests[1]['prompt']
    assert 'FRESH_CONCEPT_FOR_THIS_ROLLOUT' not in model.requests[2]['prompt']
    assert report['status'] == 'NO_SELECTION'  # repeated H0 is still only one vote


def test_early_ready_skips_later_stages(evidence, tmp_path):
    harness = baseline_harness()
    harness['stages'].append({'id': 'extra', 'instruction': 'Further inspect only if needed.', 'context': ['plan'], 'allow_ready': True})
    model = MockModel(answer())
    report = run(evidence, tmp_path, model, initial=harness, variants=0)
    assert report['rollouts'][0]['status'] == 'READY'
    assert len(model.requests) == 1


@pytest.mark.parametrize('fault', ['candidate', 'observation', 'target', 'nan', 'concept', 'ready_gap'])
def test_invalid_decisions_exit_without_action(evidence, tmp_path, fault):
    result = answer()
    if fault == 'candidate': result['action']['selected_reference']['reference_id'] = 'R999'
    if fault == 'observation': result['observation_id'] = 'wrong'
    if fault == 'target': result['action']['target']['pixel_xy'] = [1000, 20]
    if fault == 'nan': result['action']['target']['pixel_xy'] = [float('nan'), 20]
    if fault == 'concept': result['concepts'][0]['source_image_ids'] = ['image_999']
    if fault == 'ready_gap': result['missing_information'] = 'Cannot see target'
    report = run(evidence, tmp_path, MockModel(result), variants=0)
    row = report['rollouts'][0]
    assert row['status'] == 'ERROR' and row['action'] is None
    assert (tmp_path / 'experiment/rollouts/h00_r00/exception.txt').exists()
    assert not (tmp_path / 'experiment/selected_action.json').exists()


@pytest.mark.parametrize('fault', ['forward_reference', 'arbitrary_code', 'no_ready'])
def test_invalid_harness_is_rejected(fault):
    h = baseline_harness()
    if fault == 'forward_reference': h['stages'][0]['context'] = ['future']
    if fault == 'arbitrary_code': h['python'] = 'pass'
    if fault == 'no_ready': h['stages'][0]['allow_ready'] = False
    with pytest.raises(PolicyError): validate_harness(h)


def test_invalid_reflection_is_not_frozen_or_executed(evidence, tmp_path):
    bad = changed_harness()
    bad['stages'][0]['context'] = ['future']
    model = MockModel(answer(), proposal(bad))
    report = run(evidence, tmp_path, model, variants=1)
    assert report['reflections'][0]['status'] == 'ERROR'
    assert len(report['versions']) == len(report['rollouts']) == 1
    assert len(model.requests) == 2
    assert not (tmp_path / 'experiment/harnesses/h_01.json').exists()


def test_renaming_is_not_an_additional_vote(evidence, tmp_path):
    renamed = baseline_harness()
    renamed['name'] = 'renamed'
    renamed['stages'][0]['id'] = 'renamed_stage'
    assert execution_signature(renamed) == execution_signature(baseline_harness())
    model = MockModel(answer(), proposal(renamed))
    report = run(evidence, tmp_path, model, variants=1)
    assert len(report['versions']) == 1
    assert report['reflections'][0]['status'] == 'ERROR'


def test_duplicate_branch_patch_cannot_inflate_support(evidence, tmp_path):
    model = MockModel(answer(), proposal(changed_harness()), answer(), proposal(changed_harness()))
    report = run(evidence, tmp_path, model, search='branch', variants=2)
    assert report['reflections'][-1]['status'] == 'DUPLICATE'
    assert len(report['versions']) == 2
    assert report['status'] == 'NO_SELECTION'


def test_budget_exhaustion_and_needs_learning_do_not_launch_slow_planner(evidence, tmp_path):
    model = MockModel(answer(status='NEEDS_LEARNING', action=None, missing_information='Target obscured'))
    report = run(evidence, tmp_path, model, variants=3, max_calls=1)
    assert len(model.requests) == 1
    assert report['rollouts'][0]['status'] == 'NEEDS_LEARNING'
    assert report['reflections'][0]['status'] == 'BUDGET_EXHAUSTED'
    assert report['status'] == 'NO_SELECTION'


def test_stop_reflection_terminates_search(evidence, tmp_path):
    model = MockModel(answer(), {'status': 'STOP', 'harness': None, 'reason': 'No supported improvement', 'changes': []})
    report = run(evidence, tmp_path, model, variants=5)
    assert len(model.requests) == 2
    assert report['reflections'][0]['status'] == 'STOP'


def test_late_model_response_rejected(evidence, tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr('cloth_agent.harness.reasoning_learning.time.monotonic', lambda: clock[0])
    def late(_):
        clock[0] = 10.
        return answer()
    report = run(evidence, tmp_path, MockModel(late), variants=0, rollout_timeout=1)
    assert report['rollouts'][0]['status'] == 'BUDGET_EXHAUSTED'
    assert report['rollouts'][0]['action'] is None


def test_pixels_and_fixed_contract_are_checked_every_call(evidence, tmp_path):
    frozen = freeze_evidence(evidence, tmp_path / 'fixed')
    verify_evidence(frozen, tmp_path / 'fixed')
    Image.new('RGB', (100, 80), 'red').save(tmp_path / 'fixed/image_0.png')
    with pytest.raises(PolicyError, match='pixels changed'):
        verify_evidence(frozen, tmp_path / 'fixed')


def test_model_side_mutation_aborts_before_accepting_result(evidence, tmp_path):
    def mutate(kw):
        Image.new('RGB', (100, 80), 'red').save(kw['images'][0])
        return answer()
    report = run(evidence, tmp_path, MockModel(mutate), variants=0)
    assert report['rollouts'][0]['status'] == 'ERROR'
    assert report['rollouts'][0]['action'] is None


def test_manifest_does_not_leak_historical_answer_or_request(evidence):
    pre = {k: v for k, v in evidence.items() if k != 'schema_version'}
    historical_image = {'path': '/SECRET_POST_IMAGE.png', 'role': 'excluded_historical_or_post_action', 'status': 'AVAILABLE'}
    pre['images'] = pre['images'] + [historical_image]
    manifest = {'decisions': [{'decision_id': 'saved', 'pre_decision': pre,
                 'post_decision': {'candidate_id': 'R999', 'reason': 'SECRET_REASON', 'evaluation': 'SECRET_FEEDBACK'},
                 'compiler_only_request': {'prompt': 'SECRET_PREVIOUS_PLANNER_ROI'}, 'tool_trace': {'crop': 'SECRET_ROI'}}]}
    safe = evidence_from_manifest(manifest, 'saved')
    assert 'SECRET' not in json.dumps(safe) and 'R999' not in json.dumps(safe)


def test_missing_images_or_registry_fail_without_model_calls(evidence, tmp_path):
    evidence['candidate_registry'] = None
    model = MockModel()
    report = run(evidence, tmp_path, model)
    assert report['status'] == 'ERROR' and report['totals']['model_calls'] == 0
    assert not model.requests


def test_prepared_crop_requires_traceable_current_parent(evidence, tmp_path):
    im = Image.new('RGB', (20, 20), 'white')
    path = tmp_path / 'crop.png'
    im.save(path)
    crop = {'role': 'clean_crop', 'path': str(path), 'size': [20, 20], 'rgb_sha256': pixel_hash(im), 'status': 'AVAILABLE'}
    evidence['images'].append(crop)
    with pytest.raises(PolicyError, match='parent hash'):
        freeze_evidence(evidence, tmp_path / 'bad_crop')
    crop['lineage'] = {'availability': 'PRE_DECISION', 'parent_rgb_sha256': evidence['images'][0]['rgb_sha256'],
                       'to_original': [1, 0, 10, 0, 1, 10]}
    frozen = freeze_evidence(evidence, tmp_path / 'good_crop')
    assert len(verify_evidence(frozen, tmp_path / 'good_crop')) == 4


def test_cli_prepare_and_existing_output_preservation(evidence, tmp_path):
    path, output = tmp_path / 'input.json', tmp_path / 'prepared'
    write_json(path, evidence)
    args = ['--evidence', str(path), '--output', str(output), '--prepare-only']
    assert main(args) == 0
    assert read_json(output / 'report.json')['totals']['model_calls'] == 0
    assert main(args) == 2
    assert read_json(output / 'report.json')['status'] == 'PREPARED'


def test_cli_missing_source_writes_honest_blocker(tmp_path):
    output = tmp_path / 'blocked'
    assert main(['--manifest', str(tmp_path / 'missing.json'), '--output', str(output)]) == 2
    assert read_json(output / 'blocked.json')['model_calls'] == 0


def test_no_robot_or_other_subprocess_is_called(evidence, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline inner loop attempted robot/subprocess access')
    monkeypatch.setattr('cloth_agent.robot_api.RobotAPI.move', forbidden)
    monkeypatch.setattr('cloth_agent.robot_api.RobotAPI.home', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    report = run(evidence, tmp_path, MockModel(answer()), variants=0)
    assert report['robot_actions'] == 0
    assert report['rollouts'][0]['action']['robot_executable'] is False


def test_real_adapter_receives_images_and_runtime_reflection(evidence, tmp_path, monkeypatch):
    outputs = [answer(), proposal(changed_harness()), answer()]
    requests = []
    def invoke(self, **kw):
        requests.append(kw)
        message = json.loads(kw['input_data'])['message']
        assert len([b for b in message['content'] if b['type'] == 'image']) == 3
        assert '--safe-mode' in kw['command']
        assert kw['command'][kw['command'].index('--tools') + 1] == ''
        result = {'type': 'result', 'structured_output': outputs.pop(0),
                  'usage': {'input_tokens': 10, 'output_tokens': 20, 'cache_read_input_tokens': 5, 'cache_creation_input_tokens': 2}}
        return BackendResult(json.dumps(result), '', 0, tuple(kw['command']))
    monkeypatch.setattr('cloth_agent.harness.model.shutil.which', lambda _: '/synthetic/claude')
    monkeypatch.setattr('cloth_agent.planner_backend.LocalClaudeBackend.invoke', invoke)
    report = run(evidence, tmp_path, RuntimeClaude(), variants=1)
    assert len(requests) == 3
    assert requests[1]['usage_stage'] == 'reasoning_reflection'
    assert report['totals']['total_tokens'] == 111
    assert report['rollouts'][0]['metrics']['thinking_tokens'] is None
    assert (tmp_path / 'experiment/reflections/patch_01/call/input.jsonl').exists()


def consensus_row(key, elapsed, *, candidate='R001', target=50., grasp=20., relation='toward', repetition=0):
    return {'rollout_id': f'{key}_{repetition}', 'harness_hash': key, 'status': 'READY',
            'action': {'selected_reference': {'reference_id': candidate}, 'grasp_pixel_xy': [grasp, 30],
                       'target': {'pixel_xy': [target, 30], 'relation': relation, 'anchor_pixel_xy': [60, 40]}},
            'metrics': {'elapsed_s': elapsed, 'model_calls': 1, 'tool_calls': 0, 'thinking_tokens': None}}


def test_fastest_disagreeing_harness_is_not_selected():
    rows = [consensus_row('baseline', 310), consensus_row('a', 190), consensus_row('b', 125),
            consensus_row('fast_outlier', 90, candidate='R002', target=90, relation='away_from'), consensus_row('c', 160)]
    report = select_consensus(rows, [r['harness_hash'] for r in rows])
    assert report['status'] == 'SELECTED'
    assert report['selected']['harness_hash'] == 'b'
    assert report['selected']['votes'] == 4 and report['selected']['support'] == .8


def test_no_single_link_chaining_or_tie_breaking_as_consensus():
    rows = [consensus_row('a', 3, target=20), consensus_row('b', 2, target=35), consensus_row('c', 1, target=50)]
    report = select_consensus(rows, ['a', 'b', 'c'], epsilon_target=20, min_harnesses=2)
    assert report['status'] == 'NO_STABLE_CONSENSUS'
    assert report['reason'] == 'TIED_DOMINANT_CLUSTERS'
    assert all(c['votes'] == 2 for c in report['clusters'])


def test_relation_and_grasp_distance_both_affect_consensus():
    rows = [consensus_row('a', 3), consensus_row('b', 2, candidate='R002', grasp=22), consensus_row('c', 1, grasp=23)]
    assert select_consensus(rows, ['a', 'b', 'c'])['status'] == 'NO_STABLE_CONSENSUS'
    assert select_consensus(rows, ['a', 'b', 'c'], grasp_mode='distance', epsilon_grasp=4)['status'] == 'SELECTED'
    rows[1]['action']['target']['relation'] = 'onto'
    assert select_consensus(rows, ['a', 'b', 'c'], grasp_mode='distance', epsilon_grasp=4)['status'] == 'NO_STABLE_CONSENSUS'


def test_unstable_repeats_are_not_votes_and_median_cost_is_used():
    rows = [consensus_row(k, elapsed, repetition=i) for k, times in [('a', [1, 101, 100]), ('b', [40, 50, 60]), ('c', [80, 90, 100])]
            for i, elapsed in enumerate(times)]
    report = select_consensus(rows, ['a', 'b', 'c'], repeats=3)
    assert report['selected']['harness_hash'] == 'b'
    rows[0]['action']['target']['pixel_xy'] = [99, 79]
    report = select_consensus(rows, ['a', 'b', 'c'], repeats=3)
    assert report['rejected_harnesses']['a'] == 'UNSTABLE_WITHIN_HARNESS'
    assert report['selected'] is None


def test_unavailable_thinking_tokens_are_not_zero_cost():
    rows = [consensus_row(k, 1) for k in ['a', 'b', 'c']]
    report = select_consensus(rows, ['a', 'b', 'c'], weights={'elapsed_s': 1, 'model_calls': 0, 'tool_calls': 0, 'thinking_tokens': 1})
    assert report['status'] == 'COST_UNAVAILABLE'
    assert report['selected'] is None


def test_explicit_thinking_usage_is_counted_without_inference(tmp_path):
    write_json(tmp_path / 'stdout.jsonl', {'type': 'result', 'usage': {'input_tokens': 10, 'output_tokens': 30,
               'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0,
               'output_tokens_details': {'reasoning_tokens': 12}}})
    metrics = call_metrics([{'backend_invoked': True, 'response_received': True, 'tool_round_trips': 0}], tmp_path)
    assert metrics['thinking_tokens'] == 12
    assert metrics['usage']['total_tokens'] == 40


def test_invalid_budget_fails_before_model_call(evidence, tmp_path):
    with pytest.raises(ValueError):
        run(evidence, tmp_path, MockModel(), max_seconds=float('nan'))
    assert not (tmp_path / 'experiment').exists()


def test_frozen_evidence_roundtrip_preserves_exact_hash_and_reference_role(evidence, tmp_path):
    evidence['images'][2]['reference_kind'] = 'target'
    first = freeze_evidence(evidence, tmp_path / 'first')
    loaded = load_evidence(tmp_path / 'first/evidence.json')
    second = freeze_evidence(loaded, tmp_path / 'second')
    assert first == second
    assert second['evidence']['images'][2]['reference_kind'] == 'target'


def test_frozen_harness_reuse_recomputes_action(evidence, tmp_path):
    path = tmp_path / 'prior_harness.json'
    freeze_harness(changed_harness('reused'), path)
    new_answer = answer()
    new_answer['action']['selected_reference']['reference_id'] = 'R002'
    model = MockModel(new_answer)
    report = run(evidence, tmp_path, model, initial=load_harness(path), variants=0)
    assert report['rollouts'][0]['action']['selected_reference']['reference_id'] == 'R002'
    assert 'SYNTHETIC_REASON_NOT_A_REAL_RESULT' not in model.requests[0]['prompt']


def test_invalid_reflection_citation_does_not_create_version(evidence, tmp_path):
    model = MockModel(answer(), proposal(changed_harness(), 'nonexistent_rollout'))
    report = run(evidence, tmp_path, model, variants=1)
    assert report['reflections'][0]['status'] == 'ERROR'
    assert len(report['versions']) == 1


def test_frozen_evidence_metadata_tampering_is_rejected(evidence, tmp_path):
    frozen = freeze_evidence(evidence, tmp_path / 'fixed')
    changed = copy.deepcopy(frozen)
    changed['evidence']['fold_goal'] = 'Different goal'
    write_json(tmp_path / 'fixed/evidence.json', changed)
    with pytest.raises(PolicyError, match='file changed'):
        verify_evidence(frozen, tmp_path / 'fixed')


def test_unknown_final_stage_falls_back_without_action(evidence, tmp_path):
    report = run(evidence, tmp_path, MockModel(answer(status='CONTINUE', action=None)), variants=0)
    assert report['rollouts'][0]['status'] == 'NEEDS_LEARNING'
    assert report['rollouts'][0]['reason'] == 'STAGES_EXHAUSTED'
    assert report['rollouts'][0]['action'] is None


@pytest.mark.parametrize('fields,expected', [
    ({'output_tokens_details': {'thinking_tokens': 12}}, 12),
    ({'thinking_tokens': 0, 'output_tokens_details': {'thinking_tokens': 12}}, 0),
    ({'thinking_tokens': None, 'output_tokens_details': {'thinking_tokens': 12}}, 12),
    ({'output_tokens_details': {'reasoning_tokens': 12}}, 12),
    ({'output_tokens_details': {'thinking_tokens': True}}, None),
    ({}, None),
])
def test_provider_thinking_formats_preserve_missing_and_zero(tmp_path, fields, expected):
    write_json(tmp_path / 'stdout.jsonl', {'type': 'result', 'usage': {
        'input_tokens': 10, 'output_tokens': 30, **fields}})
    metrics = call_metrics([{'backend_invoked': True, 'response_received': True,
                             'tool_round_trips': 0}], tmp_path)
    assert metrics['thinking_tokens'] == expected


def test_k_alias_is_new_version_attempts():
    from cloth_agent.harness.reasoning_learning import build_parser
    args = build_parser().parse_args(['--evidence', 'input.json', '--output', 'out', '--k', '3'])
    assert args.variants == 3


def metered(value, tokens=100, fail=False, unknown=False):
    def output(kw):
        kw['output'].mkdir(parents=True, exist_ok=True)
        usage = {} if unknown else dict(input_tokens=10, output_tokens=tokens-30,
                                       cache_read_input_tokens=10, cache_creation_input_tokens=10)
        write_json(kw['output']/'stdout.jsonl', {'type': 'result', 'usage': usage})
        if fail:
            raise ValueError('Synthetic rejected output')
        return value
    return output


def test_token_budget_stops_before_reflection_includes_cache_and_keeps_overshoot(evidence, tmp_path):
    model = MockModel(metered(answer(), 100))
    report = run(evidence, tmp_path, model, variants=3, max_tokens=90)
    assert len(model.requests) == 1
    budget = report['totals']['token_budget']
    assert budget['known_total_tokens'] == 100
    assert budget['remaining_tokens'] == 0
    assert budget['stop_reason'] == 'TOKEN_BUDGET_EXHAUSTED'
    assert report['reflections'][0]['status'] == 'BUDGET_EXHAUSTED'


def test_reflection_counts_against_same_token_budget(evidence, tmp_path):
    model = MockModel(metered(answer(), 100), metered(proposal(changed_harness()), 100))
    report = run(evidence, tmp_path, model, variants=3, max_tokens=200)
    assert len(model.requests) == 2
    assert report['totals']['token_budget']['known_total_tokens'] == 200
    assert report['rollouts'][-1]['status'] == 'BUDGET_EXHAUSTED'


@pytest.mark.parametrize('unknown', [False, True])
def test_failed_calls_are_counted_or_stop_on_unknown_usage(evidence, tmp_path, unknown):
    model = MockModel(metered(answer(), 100, fail=True, unknown=unknown))
    report = run(evidence, tmp_path, model, variants=3, max_tokens=100)
    assert len(model.requests) == 1
    b = report['totals']['token_budget']
    assert b['stop_reason'] == ('TOKEN_USAGE_UNKNOWN' if unknown else 'TOKEN_BUDGET_EXHAUSTED')
    assert b['unknown_usage_calls'] == int(unknown)


def test_invalid_token_budget_rejected_before_output(evidence, tmp_path):
    with pytest.raises(ValueError, match='max_tokens'):
        run(evidence, tmp_path, MockModel(), max_tokens=0)


def test_ready_allows_nonblocking_uncertainty(evidence, tmp_path):
    report = run(evidence, tmp_path, MockModel(answer(residual_uncertainty='Depth and IK require later physical checks.')), variants=0)
    assert report['rollouts'][0]['status'] == 'READY'


def test_ready_blocker_reports_field_conflict_not_visual_failure(evidence, tmp_path):
    report = run(evidence, tmp_path, MockModel(answer(missing_information='Target hidden by a fold')), variants=0)
    row = report['rollouts'][0]
    assert 'READY_BLOCKING_GAP' in row['reason']
    assert 'No conclusion about grasp or target correctness' in row['stages'][0]['validation_error']
    assert row['action'] is None


def test_nonready_action_reports_exact_conflict(evidence, tmp_path):
    report = run(evidence, tmp_path, MockModel(answer(status='CONTINUE')), variants=0)
    assert 'NON_READY_ACTION' in report['rollouts'][0]['reason']


def test_harness_allows_normal_fold_name_and_slash_prose():
    from cloth_agent.harness.reasoning_contract import validate_harness
    h = changed_harness('registry_bound_fold_planner')
    h['applicability'] = 'Inspect a before/after reference pair and collar/hem orientation.'
    assert validate_harness(h) == h


@pytest.mark.parametrize('prose', [
    'Compare 2 boundary cues; use a 0.6 confidence threshold only when applicable.',
    'Check: (1) inspect the boundary; (2) derive the target from current evidence.',
    'Derive a current crop; (120, 839) is only an example coordinate.',
    'Resolve current image IDs, e.g. image_3, and current candidate labels, e.g. R101.',
    'Do not read /tmp/result.json, ../saved/action.json or results/run/action.json.',
    'Supplied images may use base64 delivery; https://example.com is not evidence.',
])
def test_harness_prose_is_not_rejected_by_keywords(prose):
    h = changed_harness('refined_v2')
    h['stages'][0]['id'] = 'stage_1'
    h['stages'][0]['instruction'] = prose
    assert validate_harness(h) == h


def test_numbered_parameterized_reflection_runs(evidence, tmp_path):
    h = changed_harness('refined_v2')
    h['stages'][0]['id'] = 'stage_1'
    h['stages'][0]['instruction'] = (
        'Check: (1) compare 2 visible boundary cues; (2) derive grasp and target '
        'from the current registry and image geometry.')
    model = MockModel(answer(), proposal(h), answer())
    report = run(evidence, tmp_path, model, variants=1)
    assert report['reflections'][0]['status'] == 'VALID'
    assert len(report['versions']) == 2
    assert report['rollouts'][-1]['status'] == 'READY'
    assert len(model.requests) == 3
    from cloth_agent.harness.reasoning_learning import REFLECT_CONTRACT
    assert 'Parameterize the reusable harness wherever practical' in REFLECT_CONTRACT
    assert 'prose must contain no digits' not in REFLECT_CONTRACT


def test_judgment_schema_exposes_cross_field_contract():
    from cloth_agent.harness.reasoning_contract import JUDGMENT_SCHEMA
    from cloth_agent.harness.policy import validate_schema
    validate_schema(answer(residual_uncertainty='Later physical checks needed'), JUDGMENT_SCHEMA)
    for bad in [answer(missing_information='hidden target'), answer(status='CONTINUE'),
                answer(status='NEEDS_LEARNING', action=None)]:
        with pytest.raises(PolicyError):
            validate_schema(bad, JUDGMENT_SCHEMA)


@pytest.mark.parametrize('tool,args,allowed', [
    ('ToolSearch', {'query': 'select:StructuredOutput', 'max_results': 3}, True),
    ('ToolSearch', {'query': 'StructuredOutput'}, True),
    ('ToolSearch', {'query': 'select:StructuredOutput,Read'}, False),
    ('ToolSearch', {'query': 'web search'}, False),
    ('Read', {'file_path': '/tmp/extra.json'}, False),
])
def test_runtime_output_lookup_is_not_visual_exploration(evidence, tmp_path, monkeypatch, tool, args, allowed):
    block = {'type': 'tool_use', 'id': 'lookup', 'name': tool, 'input': args}
    event = {'type': 'assistant', 'message': {'content': [block]}}
    result = {'type': 'result', 'structured_output': answer(), 'usage': {
        'input_tokens': 1, 'output_tokens': 2, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0}}
    stdout = '\n'.join(json.dumps(x) for x in [event, event, {'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': 'lookup', 'is_error': True, 'content': 'No such tool available'}]}}, result])
    monkeypatch.setattr('cloth_agent.harness.model.shutil.which', lambda _: '/synthetic/claude')
    monkeypatch.setattr('cloth_agent.planner_backend.LocalClaudeBackend.invoke',
                        lambda self, **kw: BackendResult(stdout, '', 0, ()))
    model = RuntimeClaude()
    report = run(evidence, tmp_path, model, variants=0)
    assert (report['rollouts'][0]['status'] == 'READY') == allowed
    assert model.calls[0]['tool_round_trips'] == 1
    assert model.calls[0]['output_tool_lookup_calls'] == int(allowed)
    assert model.calls[0]['exploratory_tool_round_trips'] == int(not allowed)


def test_local_format_repair_preserves_raw_and_semantic_fields(evidence, tmp_path):
    legacy = answer(); del legacy['residual_uncertainty']
    report = run(evidence, tmp_path, MockModel(legacy), variants=0)
    assert report['rollouts'][0]['status'] == 'READY'
    audit = read_json(tmp_path/'experiment/rollouts/h00_r00/calls/plan/format_repair.json')
    assert 'residual_uncertainty' not in audit['before']
    assert audit['after']['action'] == legacy['action']
    assert audit['model_calls'] == 0
    assert (tmp_path/'experiment/format_preflight.json').is_file()


def test_reflection_prose_preserved_without_number_repair(tmp_path):
    from cloth_agent.harness.format_preflight import repair_format
    original = 'Check these: (1) confirm boundary; (2) confirm target. Refer to step (1).'
    h = changed_harness(); h['stages'][0]['instruction'] = original
    proposed = proposal(h)
    fixed = repair_format(proposed, 'reflection', tmp_path/'repair.json')
    assert fixed == proposed
    assert read_json(tmp_path/'repair.json')['changes'] == []
    validate_harness(fixed['harness'])


def test_contract_failure_does_not_trigger_visual_learning(evidence, tmp_path):
    model = MockModel(answer(missing_information='Target not visible'))
    report = run(evidence, tmp_path, model, variants=3)
    assert len(model.requests) == 1
    assert report['reflections'][0]['status'] == 'STOP'
    assert report['reflections'][0]['reason'].startswith('FORMAT_BLOCKED')
    assert report['rollouts'][0]['status'] == 'ERROR'


def test_single_optimization_exports_methods_and_comparison_without_consensus(evidence, tmp_path):
    candidate = proposal(changed_harness())
    candidate['operations'] = [{
        'name': 'compare_changed_region', 'information_goal': 'Locate the changed garment region',
        'method': 'Compare the reference pair jointly and map the changed region into current evidence.',
        'outputs': ['region', 'correspondence'],
        'success_check': 'The changed region has a supported current-image correspondence.',
        'on_insufficient': 'Return UNKNOWN and state the missing correspondence.',
        'replaces': 'Separate full-image descriptions followed by repeated orientation inference.',
        'expected_time_saving': 'May reduce repeated model analysis; unverified.',
    }]
    model = MockModel(answer(), answer(), candidate, answer(), answer())
    report = run(evidence, tmp_path, model, variants=1, repeats=2)
    assert len(model.requests) == 5  # four planning calls, exactly one optimization
    assert report['status'] == 'NO_SELECTION'  # default three-version consensus remains separate
    assert len(report['optimization_comparison']['versions']) == 2
    assert all(v['within_version_stability'] == 'STABLE'
               for v in report['optimization_comparison']['versions'])
    assert report['optimization_comparison']['versions'][1]['baseline_over_candidate_time_ratio'] is None
    request = next(q for q in model.requests if q['stage'] == 'reasoning_reflection')
    assert 'REDUCE total' in request['prompt']
    assert 'optimization_objective' in request['prompt']
    assert 'An added stage needs' in request['prompt']
    saved = read_json(tmp_path/'experiment/operation_candidates.json')
    assert saved['approved'] is False
    assert saved['operations'][0]['evidence'] == {
        'supported_by': [], 'failed_in': [], 'unresolved': ['h01_r00', 'h01_r01']}
    plans = [q for q in model.requests if q['stage'] == 'reasoning_rollout']
    assert all('expected_time_saving' not in q['prompt'] for q in plans)


def test_timing_comparison_does_not_promote_fast_or_incomplete_candidates():
    from cloth_agent.harness.optimization import compare_versions
    def row(key, seconds, status='READY', candidate='R001'):
        action = answer()['action']
        action['selected_reference']['reference_id'] = candidate
        return {'harness_hash': key, 'status': status, 'action': action,
                'metrics': {'elapsed_s': seconds, 'model_calls': 1,
                            'usage': {'total_tokens': 100}}}
    versions = [{'harness_hash': key} for key in ['base', 'fast', 'incomplete']]
    rows = [row('base', 100), row('base', 120), row('fast', 50), row('fast', 60, candidate='R002'),
            row('incomplete', 1), row('incomplete', 2, status='NEEDS_LEARNING')]
    report = compare_versions(versions, rows, repeats=2, options={}, actual=True)
    fast, incomplete = report['versions'][1:]
    assert fast['baseline_over_candidate_time_ratio'] == 2
    assert fast['within_version_stability'] == 'UNSTABLE'
    assert fast['promoted_to_skill'] is False and fast['quality'] == 'NOT_EVALUATED'
    assert incomplete['baseline_over_candidate_time_ratio'] is None
    assert incomplete['within_version_stability'] == 'NOT_EVALUABLE'
    single = compare_versions(versions[:1], rows[:1], repeats=1, options={}, actual=True)
    assert single['versions'][0]['within_version_stability'] == 'UNMEASURED_SINGLE_REPEAT'


def test_executable_patch_runs_host_geometry_on_fresh_measurements(evidence, tmp_path):
    h = changed_harness('host_geometry')
    h['stages'] = [
        {'id': 'measure', 'instruction': 'Measure point and hinge endpoints in current clean pixels.',
         'context': [], 'allow_ready': False},
        {'id': 'decide', 'instruction': 'Use the computed reflection and verify it visually.',
         'context': ['measure'], 'allow_ready': True, 'host_operations': [
             {'id': 'mirror', 'op': 'reflect_point', 'source_stage': 'measure',
              'bindings': {'point': 'grasp', 'line_start': 'hinge_a', 'line_end': 'hinge_b'}}]}]
    measure = answer(status='CONTINUE', action=None, measurements={
        'grasp': [20, 30], 'hinge_a': [35, 0], 'hinge_b': [35, 79]})
    def decide(request):
        payload = json.loads(request['prompt'][request['prompt'].index('{"applicability":'):])
        result = payload['host_results'][0]
        assert result['status'] == 'COMPUTED' and result['output'] == pytest.approx([50, 30])
        return answer()
    model = MockModel(answer(), proposal(h), measure, decide)
    report = run(evidence, tmp_path, model, variants=1)
    assert report['rollouts'][-1]['status'] == 'READY'
    patch = read_json(tmp_path/'experiment/reflections/patch_01/executable_patch.json')
    assert patch['host_operation_count'] == 1
    assert patch['harness_hash'] == report['rollouts'][-1]['harness_hash']
    log = read_json(tmp_path/'experiment/rollouts/h01_r00/host_execution.json')
    assert log[0]['inputs']['point'] == [20, 30]
    assert log[0]['output'] == pytest.approx([50, 30])
    assert report['rollouts'][-1]['metrics']['host_operation_count'] == 1
    changed = copy.deepcopy(h)
    changed['stages'][1]['host_operations'][0]['bindings']['point'] = 'other'
    assert execution_signature(changed) != execution_signature(h)
