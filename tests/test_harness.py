"""Synthetic fixtures only. Mock policies/results are NOT experiment conclusions."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
import shutil

import pytest
from PIL import Image, ImageColor, ImageDraw

from cloth_agent.harness.collector import collect_run, collect_tool_trace
from cloth_agent.harness.common import digest, read_json, write_json
from cloth_agent.harness.compiler import compile_policy
from cloth_agent.harness.executor import execute_policy, prepare_views, validate_observation
from cloth_agent.harness.model import RuntimeClaude
from cloth_agent.harness.knowledge import collect_knowledge
from cloth_agent.harness.policy import INSPECTION_SCHEMA, PolicyError, freeze_policy, load_policy, validate_policy
from cloth_agent.harness.__main__ import main, replay_policy
from cloth_agent.image_tools_mcp import IDENTITY, image_content_summary, pixel_hash, transform_point
from cloth_agent.planner_backend import BackendResult

RUN = 'synthetic_learning_run'


def synthetic_policy(*, cropped=False):
    inspect = {'id': 'global', 'op': 'inspect', 'when': None, 'views': ['observation'], 'context': [],
               'instruction': 'Assess applicability and the movable and preserved garment regions for the current goal. Select only if evidence is sufficient.',
               'output_schema': copy.deepcopy(INSPECTION_SCHEMA)}
    steps = [inspect]
    if cropped:
        steps += [{'id': 'detail', 'op': 'prepare_views', 'when': None, 'source': 'observation', 'roi_from': 'global'},
                  {**copy.deepcopy(inspect), 'id': 'local', 'views': ['detail'], 'context': ['global']}]
    steps += [{'id': 'done', 'op': 'return_decision', 'when': None, 'from': 'local' if cropped else 'global'},
              {'id': 'fallback', 'op': 'needs_learning', 'when': None, 'reason': 'Visual evidence is insufficient'}]
    return {'schema_version': 1, 'policy_id': 'synthetic_test_only', 'applicability': 'Visible garment with current candidate overlay',
            'required_inputs': ['observation', 'fold_goal', 'candidate_registry', 'observation_id'],
            'budget': {'max_claude_calls': 3, 'max_host_image_ops': 12, 'max_seconds': 60, 'max_images_per_call': 6},
            'steps': steps}


def judgment(observation, **kwargs):
    return {'status': 'READY', 'observation_id': observation, 'applicable': True, 'candidate_id': 'R002',
            'roi': [.1, .1, .9, .9], 'rotation_clockwise': 90, 'scale': 1.5, 'needs_reference': False,
            'movable_region': 'Synthetic test flap', 'preserved_region': 'Synthetic test center', 'evidence': 'Synthetic visible marker', **kwargs}


class MockClaude:
    configuration = {'backend': 'MOCK_TEST_ONLY', 'actual_measurement': False}

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.calls = []

    def invoke(self, **kwargs):
        self.requests.append(kwargs)
        self.calls.append({'backend_invoked': True, 'response_received': True, 'tool_round_trips': 0, 'mock': True})
        value = self.responses.pop(0)
        return value(kwargs) if callable(value) else copy.deepcopy(value)


def make_record(project, segment, iteration, when, *, reused=False, missing_image=False):
    run = project / 'runs' / RUN
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'run_metadata.json', {'run_id': RUN})
    output = run / 'results' / segment
    write_json(output / 'summary.json', {'run_dir': str(run), 'created_at': when, 'iterations': []})
    directory = output / f'iteration_{iteration:03d}'
    before = directory / 'before_raw'
    before.mkdir(parents=True)
    raw = Image.new('RGB', (40, 30), 'white')
    ImageDraw.Draw(raw).rectangle((4, 6, 24, 22), fill='blue')
    raw.save(before / 'camera_0_A.png')
    clean = raw.rotate(-90, expand=True)
    clean.save(before / 'camera_A_rgb_upright.png')
    overlay = clean.copy()
    ImageDraw.Draw(overlay).text((9, 9), 'R002', fill='red')
    overlay.save(before / 'camera_A_rxxx_overlay_upright.png')
    rows = [{'reference_id': 'R001', 'pixel_xy': [10, 12]}, {'reference_id': 'R002', 'pixel_xy': [20, 14]}]
    write_json(before / 'camera_A_coordinate_guide.json', {'samples': rows})
    write_json(before / 'camera_A_upright_mapping.json', {
        'raw_image': str(before / 'camera_0_A.png'), 'coordinate_guide': str(before / 'camera_A_coordinate_guide.json'),
        'raw_size_xy': [40, 30], 'upright_size_xy': [30, 40], 'rotation': 'clockwise90'})
    record = {'iteration': iteration, 'planned_step': 'left_sleeve', 'completed_at': when,
              'before_images': [str(before / 'camera_A_rgb_upright.png'), str(before / 'camera_A_rxxx_overlay_upright.png')],
              'after_images': [str(directory / 'after_raw/SECRET_POST_IMAGE.png')],
              'evaluation': {'status': 'SECRET_POST_FEEDBACK'},
              'planning_diagnostics': {}}
    if reused:
        record['height_retry'] = {'parent_iteration': iteration - 1, 'status': 'EXECUTED_AND_EVALUATED'}
    else:
        record['planning_diagnostics']['visual_plan_result'] = {
            'created_at': when, 'duration_s': 12.5,
            'decision': {'selected_reference': {'camera': 'A', 'reference_id': 'R001', 'reason': 'SECRET_HISTORIC_REASON'}},
        }
    if missing_image:
        (before / 'camera_A_rgb_upright.png').unlink()
    write_json(directory / 'record.json', record)
    return directory


@pytest.fixture
def scene(tmp_path):
    directory = make_record(tmp_path, 'segment_b', 1, '2026-09-24T02:00:00Z')
    manifest = collect_run(RUN, tmp_path)
    assert manifest['counts']['replayable_decisions'] == 1, manifest
    return tmp_path, directory, manifest


def test_merge_sort_dedupe_and_reuse(tmp_path):
    later = make_record(tmp_path, 'segment_b', 1, '2026-09-24T02:00:00Z')
    earlier = make_record(tmp_path, 'segment_a', 1, '2026-09-24T01:00:00Z')
    make_record(tmp_path, 'segment_a', 2, '2026-09-24T01:10:00Z', reused=True)
    review = tmp_path / 'results/process_review'
    write_json(review / '01_record.json', read_json(earlier / 'record.json'))
    manifest = collect_run(RUN, tmp_path)
    assert manifest['counts']['iterations'] == 3
    assert manifest['counts']['independent_visual_decisions'] == 2
    assert manifest['counts']['reused_iterations'] == 1
    assert [x['completed_at'] for x in manifest['iterations']] == [
        '2026-09-24T01:00:00Z', '2026-09-24T01:10:00Z', '2026-09-24T02:00:00Z']
    assert len(manifest['duplicates']) == 1
    assert manifest['counts']['replayable_decisions'] == 2


def test_legacy_nonfinite_diagnostics_imported_with_audit(scene):
    root, directory, _ = scene
    path = directory / 'record.json'
    record = read_json(path)
    record['host_compilation'] = {'table_clearance_lower_z_mm': float('nan')}
    record['evaluation']['missing_measurements'] = [float('inf'), -float('inf')]
    original = json.dumps(record)
    path.write_text(original)
    manifest = collect_run(RUN, root)
    assert manifest['counts']['replayable_decisions'] == 1
    assert path.read_text() == original
    issue = next(i for i in manifest['issues'] if i['code'] == 'legacy_nonfinite_normalized')
    assert len(issue['replacements']) == 3
    assert {r['original'] for r in issue['replacements']} == {'nan', 'inf', '-inf'}
    assert manifest['decisions'][0]['post_decision']['evaluation']['missing_measurements'] == [None, None]
    write_json(root / 'normalized_manifest.json', manifest)
    with pytest.raises(ValueError):
        digest({'still_strict': float('nan')})


def test_nonfinite_candidate_geometry_still_blocks_replay(scene):
    root, directory, _ = scene
    path = directory / 'before_raw/camera_A_coordinate_guide.json'
    guide = read_json(path)
    guide['samples'][0]['pixel_xy'][0] = float('nan')
    path.write_text(json.dumps(guide))
    manifest = collect_run(RUN, root)
    assert manifest['counts']['replayable_decisions'] == 0
    assert any(i['code'] == 'invalid_candidate_coordinates' for i in manifest['decisions'][0]['issues'])


def test_final_record_supersedes_partial_but_orphan_partial_survives(scene):
    root, directory, _ = scene
    record = read_json(directory / 'record.json')
    record.pop('evaluation')
    write_json(directory / 'partial_record.json', record)
    manifest = collect_run(RUN, root)
    assert manifest['counts']['iterations'] == 1
    assert manifest['counts']['independent_visual_decisions'] == 1
    assert len(manifest['duplicates']) == 1
    assert manifest['decisions'][0]['post_decision']['evaluation']['status'] == 'SECRET_POST_FEEDBACK'
    (directory / 'record.json').unlink()
    assert collect_run(RUN, root)['counts']['independent_visual_decisions'] == 1


def test_knowledge_snapshot_is_compiler_only(scene):
    root, _, manifest = scene
    write_json(root / 'data/fold_experience/rules.json', {
        'schema_version': 1, 'rules': {'lesson': {'context': ['SECRET_PRIOR_CONTEXT'],
        'evidence_count': {'support': 1}, 'confidence': .03}}, 'processed_trials': {}})
    write_json(root / 'data/skills/approved.json', {'skills': [
        {'name': 'custom-fold', 'purpose': 'SECRET_PRIOR_SKILL', 'guidance': 'Conditional guidance',
         'version': 1, 'source': 'reviewed'}]})
    knowledge = collect_knowledge(root)
    response = {'policy': synthetic_policy(), 'evidence': [{
        'decision_id': manifest['decisions'][0]['decision_id'], 'observed': 'Recorded trial',
        'inferred': 'Conditional selection', 'unverified': 'Generalization'}]}
    compiler = MockClaude([response])
    audit = compile_policy(manifest, 'synthetic', root / 'distilled', compiler, knowledge=knowledge)
    assert audit['status'] == 'FROZEN', audit
    assert 'SECRET_PRIOR_CONTEXT' in compiler.requests[0]['prompt']
    assert 'SECRET_PRIOR_SKILL' in compiler.requests[0]['prompt']
    assert read_json(root / 'distilled/knowledge_snapshot.json') == knowledge
    provenance = read_json(Path(audit['policy_path']).with_name('provenance.json'))
    assert provenance['knowledge_hash'] == digest(knowledge)
    frozen = load_policy(audit['policy_path'])
    before = manifest['decisions'][0]['pre_decision']
    replay = MockClaude([judgment(before['observation_id'])])
    assert execute_policy(frozen, before, replay, root / 'replay')['status'] == 'READY'
    assert 'SECRET_PRIOR' not in replay.requests[0]['prompt']
    assert 'SECRET_PRIOR' not in json.dumps(frozen)


def test_compilation_subset_is_explicit_and_audited(scene):
    root, _, _ = scene
    make_record(root, 'segment_c', 1, '2026-09-24T03:00:00Z')
    manifest = collect_run(RUN, root)
    selected, excluded = [d['decision_id'] for d in manifest['decisions']]
    response = {'policy': synthetic_policy(), 'evidence': [{
        'decision_id': selected, 'observed': 'Synthetic', 'inferred': 'Synthetic', 'unverified': 'Synthetic'}]}
    model = MockClaude([response])
    audit = compile_policy(manifest, 'synthetic', root / 'subset', model, decision_ids=[selected])
    assert audit['status'] == 'FROZEN'
    assert audit['selected_decision_ids'] == [selected]
    assert audit['excluded_decision_ids'] == [excluded]
    data = read_json(root / 'subset/compiler_iterations/001/input.json')
    assert [t['decision_id'] for t in data['traces']] == [selected]
    assert len(manifest['decisions']) == 2  # replay collection remains intact
    invalid = MockClaude([])
    audit = compile_policy(manifest, 'synthetic', root / 'invalid_subset', invalid, decision_ids=['unknown'])
    assert audit['status'] == 'BLOCKED'
    assert not invalid.calls


def test_lineage_identity_noop_and_direct_image_return():
    views = [{'image_id': 'image_0', 'parent_image_id': None, 'to_original': IDENTITY, 'original_image_index': 0},
             {'image_id': 'view_crop', 'parent_image_id': 'image_0', 'to_original': [1, 0, 3, 0, 1, 4], 'rgb_sha256': 'test'}]
    events = [{'event_id': 'one', 'tool': 'map_point', 'status': 'ok', 'arguments': {'image_id': 'image_0', 'pixel_xy': [3, 4]},
               'result': {'original_image_index': 0, 'pixel_xy': [3, 4]}},
              {'event_id': 'two', 'tool': 'crop_image', 'status': 'ok', 'arguments': {'image_id': 'image_0', 'box': [3, 4, 10, 12]},
               'result': {'image_id': 'view_crop'}},
              {'event_id': 'delivery', 'kind': 'tool_lifecycle', 'status': 'completed', 'tool_use_id': 'call_crop', 'tool': 'crop_image',
               'image_metadata': {'image_id': 'view_crop'}, 'image_content': {'identity_status': 'VERIFIED'}}]
    trace = collect_tool_trace({'views': views, 'events': events})
    assert trace['calls'][0]['classification'] == 'CONFIRMED_IDENTITY_NO_OP'
    assert trace['calls'][1]['classification'] == 'DISPLAY_TRANSFORM'
    assert trace['lineage'][1]['model_input'] == 'VERIFIED_TOOL_RETURN'
    assert trace['lineage'][0]['model_input'] == 'UNKNOWN'
    events[0]['result']['pixel_xy'] = [4, 3]
    assert collect_tool_trace({'views': views, 'events': events})['calls'][0]['classification'] == 'USAGE_UNKNOWN'


def test_compiler_calls_existing_backend_with_real_image_blocks(scene, monkeypatch):
    root, _, manifest = scene
    policy = synthetic_policy()
    observed = []
    def invoke(self, **kwargs):
        observed.append(kwargs)
        msg = json.loads(kwargs['input_data'])
        assert image_content_summary(msg['message'])['image_count'] >= 2
        assert '--tools' in kwargs['command'] and kwargs['command'][kwargs['command'].index('--tools') + 1] == ''
        result = {'policy': policy, 'evidence': [{'decision_id': manifest['decisions'][0]['decision_id'],
                  'observed': 'Synthetic recorded inspection', 'inferred': 'May select directly', 'unverified': 'Necessity unknown'}]}
        return BackendResult(json.dumps({'type': 'result', 'structured_output': result}), '', 0, tuple(kwargs['command']))
    monkeypatch.setattr('cloth_agent.harness.model.shutil.which', lambda _: '/fake/claude')
    monkeypatch.setattr('cloth_agent.planner_backend.LocalClaudeBackend.invoke', invoke)
    audit = compile_policy(manifest, 'synthetic task', root / 'compiled', RuntimeClaude())
    assert audit['status'] == 'FROZEN', audit
    assert audit['compilation_backend_invoked'] is True
    assert len(observed) == 1
    frozen = load_policy(audit['policy_path'])
    assert frozen['policy_hash'] == digest(policy)
    assert 'SECRET_HISTORIC_REASON' not in json.dumps(frozen)
    assert 'SECRET_HISTORIC_REASON' in observed[0]['prompt']  # compiler sees learning data, replay must not.


def test_invalid_policy_rejected_after_bounded_repair(scene):
    root, _, manifest = scene
    invalid = synthetic_policy()
    invalid['steps'][0]['instruction'] = 'Always return R001 at [10, 12]'
    response = {'policy': invalid, 'evidence': [{'decision_id': manifest['decisions'][0]['decision_id'],
                 'observed': 'Synthetic', 'inferred': 'Synthetic', 'unverified': 'Synthetic'}]}
    model = MockClaude([response, response])
    audit = compile_policy(manifest, 'synthetic', root / 'invalid', model)
    assert audit['status'] == 'FAILED'
    assert len(model.requests) == 2
    assert not list((root / 'invalid').rglob('policy.json'))


@pytest.mark.parametrize('mutation', [
    lambda p: p['steps'][0].update(op='python'),
    lambda p: p['steps'][0].update(views=['future']),
    lambda p: p['steps'].pop(),
    lambda p: p['budget'].update(max_seconds=float('inf')),
    lambda p: p.update(applicability='Pick the point at 10 pixels'),
    lambda p: p['steps'][0].update(when={'step': 'global', 'field': 'needs_reference', 'equals': True}),
])
def test_host_policy_rejects_invalid_contract(mutation):
    policy = synthetic_policy()
    mutation(policy)
    with pytest.raises(PolicyError):
        validate_policy(policy)


def test_freeze_is_versioned_and_tampering_rejected(tmp_path):
    policy = synthetic_policy()
    first = freeze_policy(policy, {'secret': 'HISTORICAL'}, tmp_path)
    second = freeze_policy(policy, {}, tmp_path)
    assert first != second
    frozen = read_json(first)
    frozen['policy']['budget']['max_claude_calls'] = 1
    write_json(first, frozen)
    with pytest.raises(PolicyError, match='hash'):
        load_policy(first)


def test_pair_transforms_and_inverse_coordinates(scene):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    pair = validate_observation(before)
    pair['overlay'] = copy.deepcopy(pair['clean'])  # synthetic identical pixels verifies corresponding transform.
    views, lineage, count = prepare_views(pair, judgment(before['observation_id']), root / 'views', budget_remaining=6)
    assert count == 6 and len(lineage) == 6
    assert views['clean']['size'] == views['overlay']['size']
    assert views['clean']['rgb_sha256'] == views['overlay']['rgb_sha256']
    assert views['clean']['to_original'] == views['overlay']['to_original']
    from cloth_agent.harness.executor import inverse
    raw = [15., 20.]
    transformed = transform_point(inverse(views['clean']['to_original']), raw)
    assert transform_point(views['clean']['to_original'], transformed) == pytest.approx(raw)
    with pytest.raises(ValueError, match='ROI'):
        prepare_views(pair, judgment(before['observation_id'], roi=[.9, 0, .1, 1]), root / 'bad', budget_remaining=6)


def test_replay_never_receives_post_answers_and_never_calls_robot(scene, monkeypatch):
    root, _, manifest = scene
    def forbidden(*args, **kwargs):
        pytest.fail('Offline replay reached robot execution')
    monkeypatch.setattr('cloth_agent.robot_api.RobotAPI.move', forbidden)
    monkeypatch.setattr('cloth_agent.robot_api.RobotAPI.home', forbidden)
    before = manifest['decisions'][0]['pre_decision']
    policy = synthetic_policy(cropped=True)
    frozen = load_policy(freeze_policy(policy, {'old_answer': 'R001'}, root / 'freeze'))
    model = MockClaude([judgment(before['observation_id'], status='CONTINUE', candidate_id=None), judgment(before['observation_id'])])
    rows = replay_policy(frozen, manifest, root / 'replay', model)
    assert rows[0]['status'] == 'READY', rows
    assert rows[0]['replay_candidate'] == 'R002'
    assert rows[0]['historical_candidate'] == 'R001'
    assert rows[0]['exact_candidate_agreement'] is False
    assert rows[0]['original_pixel_distance'] == pytest.approx(104 ** .5)
    assert len(model.requests) == 2
    for request in model.requests:
        prompt = request['prompt']
        assert 'SECRET_HISTORIC_REASON' not in prompt and 'SECRET_POST_FEEDBACK' not in prompt
        assert 'SECRET_POST_IMAGE' not in prompt and 'historical_candidate' not in prompt
        assert 'source_record' not in prompt and 'policy_id' not in prompt
        assert all('after_raw' not in str(p) for p in request['images'])
    assert rows[0]['replay_metrics']['host_image_ops'] == 6


@pytest.mark.parametrize('change, expected', [
    ({'candidate_id': 'R999'}, 'CANDIDATE'),
    ({'observation_id': 'wrong_observation'}, 'OBSERVATION'),
    ({'status': 'NEEDS_LEARNING', 'candidate_id': None}, 'INSUFFICIENT'),
])
def test_invalid_model_selection_exits_safely(scene, change, expected):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    policy = synthetic_policy()
    model = MockClaude([judgment(before['observation_id'], **change)])
    result = execute_policy({'policy': policy, 'policy_hash': digest(policy)}, before, model, root / 'replay')
    assert result['status'] == 'NEEDS_LEARNING'
    assert expected in result['reason']
    assert result['selected_reference'] is None


def test_budget_exhaustion_does_not_reenter_learning(scene):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    policy = synthetic_policy(cropped=True)
    policy['budget']['max_claude_calls'] = 1
    model = MockClaude([judgment(before['observation_id'], status='CONTINUE', candidate_id=None)])
    result = execute_policy({'policy': policy, 'policy_hash': digest(policy)}, before, model, root / 'replay')
    assert result['status'] == 'NEEDS_LEARNING'
    assert result['reason'] == 'CLAUDE_CALL_BUDGET_EXHAUSTED'
    assert len(model.requests) == 1


def test_missing_image_registry_and_wrong_observation(scene):
    root, directory, manifest = scene
    (directory / 'before_raw/camera_A_coordinate_guide.json').unlink()
    missing = collect_run(RUN, root)
    assert not missing['decisions'][0]['replayable']
    assert any(i['code'] == 'registry_missing_or_unbound' for i in missing['decisions'][0]['issues'])
    before = manifest['decisions'][0]['pre_decision']
    before['candidate_registry']['observation_id'] = 'wrong'
    with pytest.raises(ValueError, match='MISMATCH'):
        validate_observation(before)
    (directory / 'before_raw/camera_A_rgb_upright.png').unlink()
    missing = collect_run(RUN, root)
    assert not missing['decisions'][0]['replayable']


def test_missing_run_experiment_writes_blocker_and_no_fake_policy(tmp_path):
    output = tmp_path / 'experiment'
    rc = main(['experiment', '--scope', 'candidate-selection', '--run-id', RUN, '--project-root', str(tmp_path), '--output', str(output)])
    assert rc == 2
    report = read_json(output / 'report.json')
    assert report['compilation']['status'] == 'BLOCKED'
    assert report['compilation']['compilation_backend_invoked'] is False
    assert report['policy_hash'] is None
    assert report['counts']['READY'] == 0
    assert not list(output.rglob('policy.json'))


def test_existing_experiment_is_removed_before_restart(tmp_path):
    output = tmp_path / 'experiment'
    write_json(output / 'error.json', {'old': True})
    write_json(output / 'policies/old/policy.json', {'stale': True})
    sibling = tmp_path / 'other_experiment/keep.txt'
    sibling.parent.mkdir()
    sibling.write_text('keep')
    assert main(['collect', '--run-id', RUN, '--project-root', str(tmp_path), '--output', str(output)]) == 2
    assert not (output / 'error.json').exists()
    assert not (output / 'policies').exists()
    assert (output / 'manifest.json').is_file()
    assert sibling.read_text() == 'keep'


def test_restart_does_not_delete_project_or_input_policy(tmp_path):
    output = tmp_path / 'experiment'
    policy = output / 'policies/old/policy.json'
    write_json(policy, {'keep': True})
    write_json(output / 'error.json', {})
    assert main(['collect', '--run-id', RUN, '--project-root', str(tmp_path), '--output', str(tmp_path)]) == 2
    assert main(['replay', '--run-id', RUN, '--project-root', str(tmp_path),
                 '--output', str(output), '--policy', str(policy)]) == 2
    assert read_json(policy) == {'keep': True}


def test_remote_direct_input_contains_images_and_no_tools(tmp_path, monkeypatch):
    from cloth_agent.planner_backend import RemoteClaudeBackend
    image = tmp_path / 'source.png'
    Image.new('RGB', (3, 4), 'green').save(image)
    backend = RemoteClaudeBackend(image_tools=False)
    monkeypatch.setattr(backend, '_upload', lambda _: 'https://example.test/image.png')
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({'type': 'result', 'result': '{}'}), '')
    monkeypatch.setattr('cloth_agent.planner_backend.subprocess.run', fake_run)
    backend.invoke(prompt='test', schema={}, system_prompt='offline', image_paths=[image], direct_images=True)
    transport = commands[0][-1]
    assert '--input-format stream-json' in transport
    assert '--direct-images 1' in transport
    assert "--tools ''" in transport
    assert '--allowedTools Read' not in transport
    assert 'mcp__cloth_image' not in transport


def test_remote_wrapper_serializes_image_bytes(tmp_path):
    from cloth_agent import remote_output
    Image.new('RGB', (3, 4), 'green').save(tmp_path / 'image_0.png')
    script = tmp_path / 'fake.py'
    script.write_text("import sys,json,base64\nx=json.loads(sys.stdin.read())\nb=x['message']['content'][-1]\nassert base64.b64decode(b['source']['data']).startswith(bytes([137,80,78,71]))\nprint(json.dumps({'type':'result','result':'ok'}))")
    result = subprocess.run([sys.executable, str(Path(remote_output.__file__)), '--direct-images', '1', sys.executable, str(script)],
                            input='synthetic prompt', cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['result'] == 'ok'


def add_debug(directory, name, timestamp, candidate, *, failed=False):
    root = directory / 'claude_image_tools' / name
    images = []
    views = []
    for index, name in enumerate(['camera_A_rgb_upright.png', 'camera_A_rxxx_overlay_upright.png']):
        path = directory / 'before_raw' / name
        with Image.open(path) as im:
            (root / 'images').mkdir(parents=True, exist_ok=True)
            im.save(root / 'images' / f'image_{index}.png')
            views.append({'image_id': f'image_{index}', 'original_image_index': index,
                          'source_local_path': str(path), 'parent_image_id': None, 'to_original': IDENTITY,
                          'path': str(root / 'images' / f'image_{index}.png'), 'size': list(im.size), 'rgb_sha256': pixel_hash(im)})
        images.append(str(path))
    write_json(root / 'image_debug.json', {'views': views, 'events': [], 'audit_complete': True,
               'progress': [{'stage': 'call', 'event': 'completed', 'duration_s': 9.}]})
    write_json(root / 'request.json', {'image_paths': images, 'prompt': 'synthetic'})
    write_json(root / 'claude_result.json', {'type': 'result', 'is_error': failed,
               'structured_output': {'selected_reference': {'reference_id': candidate, 'reason': 'synthetic'}}})
    (root / 'claude_events.jsonl').write_text(json.dumps({'received_at': timestamp, 'received_elapsed_s': 1.,
                                             'event': {'type': 'system'}}) + '\n')
    return root


def test_repeated_decisions_sort_by_event_time_and_keep_own_answer(scene):
    root, directory, _ = scene
    add_debug(directory, 'visual_planning_aaa', '2026-09-24T01:02:00Z', 'R002')
    add_debug(directory, 'visual_planning_zzz', '2026-09-24T01:01:00Z', 'R001', failed=True)
    manifest = collect_run(RUN, root)
    assert manifest['counts']['independent_visual_decisions'] == 2
    assert 'zzz' in manifest['decisions'][0]['debug_directory']
    assert manifest['decisions'][0]['post_decision']['candidate_id'] is None
    assert manifest['decisions'][1]['post_decision']['candidate_id'] == 'R002'
    assert manifest['decisions'][1]['historical_metrics']['visual_decision_seconds'] == 9.
    assert any(i['code'] == 'failed_historical_output' for i in manifest['decisions'][0]['issues'])


def test_registry_follows_immutable_external_perception_batch(scene):
    root, directory, _ = scene
    before = directory / 'before_raw'
    target = root / 'runs' / RUN / 'results' / 'perception' / 'original_capture'
    target.mkdir(parents=True)
    for name in ('camera_0_A.png', 'camera_A_coordinate_guide.json'):
        (before / name).replace(target / name)
    mapping = read_json(before / 'camera_A_upright_mapping.json')
    mapping.update(raw_image=str(target / 'camera_0_A.png'), coordinate_guide=str(target / 'camera_A_coordinate_guide.json'))
    write_json(before / 'camera_A_upright_mapping.json', mapping)
    assert collect_run(RUN, root)['counts']['replayable_decisions'] == 1


def test_summary_missing_record_and_unknown_timestamp_are_reported(scene):
    root, directory, _ = scene
    summary = read_json(directory.parent / 'summary.json')
    summary['iterations'] = [{'iteration': 1}, {'iteration': 2}]
    write_json(directory.parent / 'summary.json', summary)
    record = read_json(directory / 'record.json')
    record.pop('completed_at')
    record['planning_diagnostics']['visual_plan_result'].pop('created_at')
    write_json(directory / 'record.json', record)
    manifest = collect_run(RUN, root)
    assert manifest['chronology_complete'] is False
    assert any(i['code'] == 'summary_record_missing' for i in manifest['issues'])


def test_optional_reference_is_conditional_and_missing_reference_falls_back(scene):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    policy = synthetic_policy()
    reference = copy.deepcopy(policy['steps'][0])
    reference.update(id='reference_check', views=['reference'], context=['global'],
                     when={'step': 'global', 'field': 'needs_reference', 'equals': True})
    policy['steps'].insert(1, reference)
    frozen = {'policy': policy, 'policy_hash': digest(policy)}
    model = MockClaude([judgment(before['observation_id'], needs_reference=False)])
    result = execute_policy(frozen, before, model, root / 'skipped_reference')
    assert result['status'] == 'READY'
    assert len(model.requests) == 1
    model = MockClaude([judgment(before['observation_id'], needs_reference=True)])
    result = execute_policy(frozen, before, model, root / 'missing_reference')
    assert result['status'] == 'NEEDS_LEARNING'
    assert result['reason'] == 'IMAGE_BINDING_UNAVAILABLE'
    assert len(model.requests) == 1


def test_complete_experiment_cli_with_mock_is_identified_as_synthetic(scene, monkeypatch):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    compiled = {'policy': synthetic_policy(), 'evidence': [{'decision_id': manifest['decisions'][0]['decision_id'],
                'observed': 'Synthetic only', 'inferred': 'Synthetic only', 'unverified': 'Not real evidence'}]}
    model = MockClaude([compiled, judgment(before['observation_id'])])
    monkeypatch.setattr('cloth_agent.harness.__main__.RuntimeClaude', lambda **kwargs: model)
    output = root / 'synthetic_experiment'
    assert main(['experiment', '--scope', 'candidate-selection', '--run-id', RUN, '--project-root', str(root), '--output', str(output)]) == 0
    report = read_json(output / 'report.json')
    assert report['counts']['READY'] == 1
    assert report['compilation']['configuration']['backend'] == 'MOCK_TEST_ONLY'
    assert report['replays'][0]['replay_candidate'] == 'R002'


def test_compiler_one_format_repair_can_succeed(scene):
    root, _, manifest = scene
    evidence = [{'decision_id': manifest['decisions'][0]['decision_id'], 'observed': 'Synthetic observation',
                 'inferred': 'Synthetic inference', 'unverified': 'Unverified necessity'}]
    bad = {'policy': synthetic_policy(), 'evidence': evidence}
    bad['policy']['steps'][0]['views'] = ['missing']
    good = {'policy': synthetic_policy(), 'evidence': evidence}
    model = MockClaude([bad, good])
    audit = compile_policy(manifest, 'synthetic', root / 'repair', model)
    assert audit['status'] == 'FROZEN'
    assert [a['valid'] for a in audit['attempts']] == [False, True]
    assert len(model.requests) == 2
    assert 'FORMAT/CONTRACT correction' in model.requests[1]['prompt']
    assert audit['actual_measurement'] is False


def test_compile_missing_images_does_not_call_claude(scene):
    root, directory, manifest = scene
    for item in manifest['decisions'][0]['pre_decision']['images']:
        item['status'] = 'MISSING'
        item['path'] = None
    manifest['iterations'][0]['compiler_images'] = []
    model = MockClaude([])
    audit = compile_policy(manifest, 'synthetic', root / 'no_images', model)
    assert audit['status'] == 'BLOCKED'
    assert audit['compilation_seconds'] is None
    assert not model.requests


def test_time_budget_prevents_returning_late_ready(scene, monkeypatch):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    policy = synthetic_policy()
    clock = [0.]
    monkeypatch.setattr('cloth_agent.harness.executor.time.monotonic', lambda: clock[0])
    def late(_):
        clock[0] = 61.
        return judgment(before['observation_id'])
    model = MockClaude([late])
    result = execute_policy({'policy': policy, 'policy_hash': digest(policy)}, before, model, root / 'late')
    assert result['status'] == 'NEEDS_LEARNING'
    assert result['reason'] == 'TIME_BUDGET_EXHAUSTED'
    assert len(model.requests) == 1


def test_host_image_budget_and_changed_input_fail_before_selection(scene):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    policy = synthetic_policy(cropped=True)
    policy['budget']['max_host_image_ops'] = 0
    model = MockClaude([judgment(before['observation_id'], status='CONTINUE', candidate_id=None)])
    result = execute_policy({'policy': policy, 'policy_hash': digest(policy)}, before, model, root / 'no_ops')
    assert result['reason'] == 'HOST_IMAGE_BUDGET_EXHAUSTED'
    image = before['images'][0]['path']
    Image.new('RGB', (30, 40), 'black').save(image)
    with pytest.raises(ValueError, match='CHANGED'):
        validate_observation(before)


def test_runtime_rejects_tool_use_and_preserves_raw_output(scene, monkeypatch):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    monkeypatch.setattr('cloth_agent.harness.model.shutil.which', lambda _: '/fake/claude')
    stream = '\n'.join([json.dumps({'type': 'assistant', 'message': {'content': [
        {'type': 'tool_use', 'id': 'one', 'name': 'Read', 'input': {'file_path': 'forbidden'}}]}}),
        json.dumps({'type': 'result', 'structured_output': judgment(before['observation_id'])})])
    monkeypatch.setattr('cloth_agent.planner_backend.LocalClaudeBackend.invoke',
                        lambda self, **kwargs: BackendResult(stream, '', 0, tuple(kwargs['command'])))
    model = RuntimeClaude()
    with pytest.raises(ValueError, match='Evidence/tool access outside the fixed-input contract'):
        model.invoke(prompt='synthetic', schema=INSPECTION_SCHEMA, images=[Path(before['images'][0]['path'])],
                     output=root / 'bad_model', stage='test')
    assert (root / 'bad_model/stdout.jsonl').read_text() == stream
    assert model.calls[0]['tool_round_trips'] == 1


def test_malformed_policy_extra_answer_fields_rejected():
    policy = synthetic_policy()
    policy['steps'][0]['bbox'] = [1, 2, 3, 4]
    with pytest.raises(PolicyError):
        validate_policy(policy)


def test_conflicting_copies_are_not_silently_counted_twice(scene):
    root, directory, _ = scene
    record = read_json(directory / 'record.json')
    record['evaluation'] = {'status': 'CONFLICTING_TEST_DATA'}
    write_json(root / 'results/review/01_record.json', record)
    manifest = collect_run(RUN, root)
    assert manifest['counts']['iterations'] == 1
    assert manifest['counts']['replayable_decisions'] == 0
    assert any(i['code'] == 'conflicting_record' for i in manifest['issues'])


def test_conflicting_bound_registries_are_rejected(scene):
    root, directory, _ = scene
    before = directory / 'before_raw'
    alternate = directory / 'alternate'
    guide = read_json(before / 'camera_A_coordinate_guide.json')
    guide['samples'][0]['pixel_xy'] = [11, 12]
    write_json(alternate / 'camera_A_coordinate_guide.json', guide)
    mapping = read_json(before / 'camera_A_upright_mapping.json')
    mapping['coordinate_guide'] = str(alternate / 'camera_A_coordinate_guide.json')
    write_json(alternate / 'camera_A_upright_mapping.json', mapping)
    manifest = collect_run(RUN, root)
    trace = manifest['decisions'][0]
    assert not trace['replayable']
    assert trace['pre_decision']['candidate_registry'] is None
    assert any(i['code'] == 'conflicting_candidate_registries' for i in trace['issues'])


def test_registry_requires_recorded_coordinate_convention(scene):
    root, directory, _ = scene
    path = directory / 'before_raw/camera_A_upright_mapping.json'
    mapping = read_json(path)
    mapping['rotation'] = 'unknown'
    write_json(path, mapping)
    trace = collect_run(RUN, root)['decisions'][0]
    assert not trace['replayable']
    assert any(i['code'] == 'invalid_registry_mapping' for i in trace['issues'])


def test_relocated_segment_resolves_images_and_registry_from_summary(scene):
    root, directory, _ = scene
    archive = root / 'archive'
    archive.mkdir()
    shutil.move(str(directory.parent), archive / directory.parent.name)
    manifest = collect_run(RUN, root, search_roots=[archive])
    assert manifest['counts']['iterations'] == 1
    assert manifest['counts']['replayable_decisions'] == 1
    for image in manifest['decisions'][0]['pre_decision']['images']:
        assert archive in Path(image['path']).parents


def test_incomplete_debug_directory_is_still_a_visual_call(scene):
    root, directory, _ = scene
    debug = add_debug(directory, 'visual_planning_partial', '2026-09-24T01:01:00Z', 'R001')
    (debug / 'image_debug.json').unlink()
    record = read_json(directory / 'record.json')
    record['planning_diagnostics'] = {}
    write_json(directory / 'record.json', record)
    manifest = collect_run(RUN, root)
    assert manifest['counts']['independent_visual_decisions'] == 1
    assert manifest['iterations'][0]['classification'] == 'REPLANNED'
    assert any(i['code'] == 'unreadable_json' for i in manifest['issues'])
    assert any(i['code'] == 'tool_trace_unavailable' for i in manifest['decisions'][0]['issues'])


def test_missing_claude_binary_is_not_a_measured_model_call(scene, monkeypatch):
    root, _, manifest = scene
    monkeypatch.setattr('cloth_agent.harness.model.shutil.which', lambda _: None)
    policy = synthetic_policy()
    model = RuntimeClaude(binary='missing_synthetic_binary')
    result = execute_policy({'policy': policy, 'policy_hash': digest(policy)},
                            manifest['decisions'][0]['pre_decision'], model, root / 'missing_cli')
    assert result['status'] == 'NEEDS_LEARNING'
    assert result['metrics']['inspection_attempts'] == 1
    assert result['metrics']['claude_calls'] == 0
    assert result['metrics']['actual_replay'] is False


@pytest.mark.parametrize('matrix', [[float('nan'), 1, 0, -1, 0, 29], [0, 0, 0, 0, 0, 0]])
def test_nonfinite_or_singular_registry_transform_is_rejected(scene, matrix):
    _, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    before['candidate_registry']['to_raw'] = matrix
    with pytest.raises(ValueError, match='TRANSFORM'):
        validate_observation(before)


def test_invalid_roi_is_rejected_even_on_ready(scene):
    root, _, manifest = scene
    before = manifest['decisions'][0]['pre_decision']
    policy = synthetic_policy()
    model = MockClaude([judgment(before['observation_id'], roi=[.8, .1, .2, .9])])
    result = execute_policy({'policy': policy, 'policy_hash': digest(policy)}, before, model, root / 'bad_roi')
    assert result['status'] == 'NEEDS_LEARNING'
    assert result['reason'] == 'INVALID_DYNAMIC_ROI'
    assert result['selected_reference'] is None


def incremental_response(request):
    data = json.loads(request['prompt'].split('\nLEARNING DATA:\n', 1)[1])
    policy = copy.deepcopy(data['previous_policy'] or synthetic_policy())
    policy['applicability'] += ' Updated from current evidence.'
    return {'policy': policy, 'evidence': [
        {'decision_id': key, 'observed': 'Synthetic current observation',
         'inferred': 'Conditional synthetic finding', 'unverified': 'Generalization unknown'}
        for key in data['current_evidence_ids']]}


def test_incremental_compilation_orders_iterations_and_carries_draft(tmp_path):
    # Create out of order and make pixels different across iterations so the
    # full dataset exceeds the budget although each iteration fits exactly.
    for segment, when, reused, color in [
        ('last', '2026-09-24T03:00:00Z', False, 'red'),
        ('first', '2026-09-24T01:00:00Z', False, 'blue'),
        ('middle', '2026-09-24T02:00:00Z', True, 'green'),
    ]:
        directory = make_record(tmp_path, segment, 1, when, reused=reused)
        raw_path = directory / 'before_raw/camera_0_A.png'
        with Image.open(raw_path) as raw:
            raw.putpixel((0, 0), ImageColor.getrgb(color))
            raw.save(raw_path)
            upright = raw.rotate(-90, expand=True)
            upright.save(directory / 'before_raw/camera_A_rgb_upright.png')
            upright.putpixel((0, 0), (255, 0, 255))
            upright.save(directory / 'before_raw/camera_A_rxxx_overlay_upright.png')
    manifest = collect_run(RUN, tmp_path)
    manifest['iterations'].reverse()  # compiler must sort, not trust input list order
    manifest['decisions'].reverse()
    model = MockClaude([incremental_response] * 3)
    output = tmp_path / 'incremental'
    audit = compile_policy(manifest, 'synthetic', output, model, max_images=2)
    assert audit['status'] == 'FROZEN', audit
    assert audit['completed_iterations'] == 3
    assert [r['iteration_id'] for r in audit['iterations']] == ['first:1', 'middle:1', 'last:1']
    assert all(len(r['images']) == 2 for r in model.requests)
    for index, request in enumerate(model.requests, 1):
        data = read_json(output / f'compiler_iterations/{index:03d}/input.json')
        assert len(data['accumulated_evidence']) == index - 1
        if index == 1:
            assert data['previous_policy'] is None
        else:
            previous = read_json(output / f'compiler_iterations/{index-1:03d}/draft.json')
            assert data['previous_policy'] == previous['policy']
        for image in request['images']:
            assert f"/{data['iteration_id'].split(':')[0]}/" in str(image)
    middle = read_json(output / 'compiler_iterations/002/input.json')
    assert middle['classification'] == 'REUSED'
    assert middle['current_evidence_ids'] == ['iteration:middle:1']
    provenance = read_json(Path(audit['policy_path']).with_name('provenance.json'))
    assert len(provenance['evidence']) == 3
    assert len(list(output.rglob('policy.json'))) == 1
    assert load_policy(audit['policy_path'])['policy'] == read_json(output / 'compiler_iterations/003/draft.json')['policy']


def test_incremental_failure_keeps_draft_without_freezing(scene):
    root, _, _ = scene
    make_record(root, 'later', 1, '2026-09-24T03:00:00Z')
    manifest = collect_run(RUN, root)
    def invalid(request):
        result = incremental_response(request)
        result['evidence'][0]['decision_id'] = 'future_or_unknown'
        return result
    model = MockClaude([incremental_response, invalid, invalid])
    output = root / 'partial_scan'
    audit = compile_policy(manifest, 'synthetic', output, model)
    assert audit['status'] == 'FAILED'
    assert audit['completed_iterations'] == 1
    assert audit['policy_path'] is None
    assert (output / 'compiler_iterations/001/draft.json').is_file()
    assert not list(output.rglob('policy.json'))
    assert len(model.requests) == 3


def test_incremental_unknown_chronology_stops_before_model(scene):
    root, _, manifest = scene
    manifest['iterations'][0]['completed_at'] = None
    model = MockClaude([])
    audit = compile_policy(manifest, 'synthetic', root / 'unordered', model)
    assert audit['status'] == 'BLOCKED'
    assert 'chronology' in audit['error']
    assert not model.calls


def image_summary_response(request):
    data = json.loads(request['prompt'].split('\nLEARNING DATA:\n', 1)[1])
    assert data['scope'] == 'image-processing'
    assert 'SECRET_HISTORIC_REASON' not in request['prompt']
    assert 'SECRET_POST_FEEDBACK' not in request['prompt']
    assert 'candidate_registry' not in request['prompt']
    assert 'existing_knowledge' not in data
    assert 'fixed_inspection_output_schema' not in data
    previous = data['previous_policy']
    policy = copy.deepcopy(previous) if previous else {
        'schema_version': 1, 'scope': 'image_processing', 'applicability': '可见衣物的局部图像检查',
        'rules': [], 'unresolved_questions': ['需要对照实验验证能否减少图像操作']}
    policy['rules'] = [{
        'id': 'local_detail', 'operation': 'crop_image', 'when': '全图细节不清楚',
        'input_views': '同一原图及其标记图', 'procedure': '根据当前图像识别区域并同步裁剪',
        'expected_visual_information': '更清晰的边界和缝线', 'stop_condition': '目标细节已可辨认',
        'limitations': '裁剪不增加原始像素信息', 'evidence_ids': data['current_evidence_ids']}]
    return {'updates': [{'action': 'SKIP' if previous else 'ADD', 'rule_id': 'local_detail',
        'rule': None if previous else policy['rules'][0], 'conflict_id': None,
        'reason': '相同经验复用' if previous else '首次提炼', 'evidence_ids': data['current_evidence_ids']}],
        'evidence': [{'decision_id': key, 'observed': '调用过裁剪工具',
        'inferred': '局部检查可能有用', 'unverified': '必要性未验证'} for key in data['current_evidence_ids']]}


def test_image_processing_cli_default_only_summarizes(scene, monkeypatch):
    root, _, _ = scene
    model = MockClaude([image_summary_response])
    monkeypatch.setattr('cloth_agent.harness.__main__.RuntimeClaude', lambda **kwargs: model)
    def forbidden(*args, **kwargs):
        pytest.fail('Image-only mode must not load skill libraries or run candidate replay')
    monkeypatch.setattr('cloth_agent.harness.__main__.collect_knowledge', forbidden)
    monkeypatch.setattr('cloth_agent.harness.__main__.replay_policy', forbidden)
    output = root / 'image_summary'
    assert main(['experiment', '--run-id', RUN, '--project-root', str(root), '--output', str(output)]) == 0
    assert len(model.requests) == 1
    result = read_json(output / 'image_processing_summary.json')
    assert result['scope'] == 'image_processing'
    assert result['executable'] is False
    assert 'selected_reference' not in json.dumps(result)
    assert (output / 'image_processing_summary.md').is_file()
    assert read_json(output / 'report.json')['replay_performed'] is False
    with pytest.raises(PolicyError):
        load_policy(output / 'image_processing_summary.json')


def test_image_processing_skips_reuse_and_passes_previous_summary(scene):
    root, _, _ = scene
    make_record(root, 'retry', 1, '2026-09-24T02:10:00Z', reused=True)
    make_record(root, 'later', 1, '2026-09-24T02:20:00Z')
    manifest = collect_run(RUN, root)
    model = MockClaude([image_summary_response, image_summary_response])
    out = root / 'scan_images'
    audit = compile_policy(manifest, '总结图像处理', out, model, scope='image-processing',
                           knowledge={'forbidden': 'SECRET_SKILL'})
    assert audit['status'] == 'SUMMARIZED', audit
    assert audit['completed_iterations'] == 3
    assert audit['iterations'][1]['status'] == 'NO_NEW_IMAGE_TRACE'
    assert len(model.requests) == 2
    data = read_json(out / 'compiler_iterations/003/input.json')
    assert data['previous_policy'] == read_json(out / 'compiler_iterations/001/draft.json')['policy']
    assert len(data['accumulated_evidence']) == 1
    assert not (out / 'knowledge_snapshot.json').exists()


def test_image_summary_rejects_candidate_outputs_and_unknown_sources():
    from cloth_agent.harness.image_processing import validate_image_summary
    request = {'prompt': '\nLEARNING DATA:\n' + json.dumps({
        'scope': 'image-processing', 'previous_policy': None, 'current_evidence_ids': ['current']})}
    from cloth_agent.harness.image_processing import apply_image_updates
    policy, _ = apply_image_updates(None, image_summary_response(request), {'current'}, {'current'})
    assert validate_image_summary(policy, {'current'})
    policy['rules'][0]['operation'] = 'return_decision'
    with pytest.raises(PolicyError):
        validate_image_summary(policy, {'current'})
    policy['rules'][0]['operation'] = 'crop_image'
    policy['rules'][0]['procedure'] = '选择 R005'
    with pytest.raises(PolicyError):
        validate_image_summary(policy, {'current'})
    policy['rules'][0]['procedure'] = '查看细节'
    with pytest.raises(PolicyError):
        validate_image_summary(policy, {'other'})


def test_image_updates_accumulate_and_defer_conflicts():
    from cloth_agent.harness.image_processing import apply_image_updates
    request = {'prompt': '\nLEARNING DATA:\n' + json.dumps({
        'scope': 'image-processing', 'previous_policy': None, 'current_evidence_ids': ['first']})}
    state, _ = apply_image_updates(None, image_summary_response(request), {'first'}, {'first'})
    original = copy.deepcopy(state)
    known = {'first', 'second', 'third', 'fourth'}

    def update(state, action, ref, rule=None, conflict=None):
        result = {'updates': [{'action': action, 'rule_id': 'local_detail', 'rule': rule,
                  'conflict_id': conflict, 'reason': '根据当前证据更新', 'evidence_ids': [ref]}],
                  'evidence': [{'decision_id': ref, 'observed': '局部边界可见',
                                'inferred': '条件需要区分', 'unverified': '尚未验证必要性'}]}
        return apply_image_updates(state, result, {ref}, known)[0]

    state = update(state, 'SKIP', 'second')
    assert len(state['rules']) == 1
    assert state['rules'][0]['procedure'] == original['rules'][0]['procedure']
    assert state['rules'][0]['evidence_ids'] == ['first', 'second']
    merged = copy.deepcopy(state['rules'][0])
    merged['limitations'] += '；细节已清晰时停止裁剪'
    state = update(state, 'MERGE', 'second', merged)
    alternative = copy.deepcopy(merged)
    alternative['procedure'] = '先放大原图，再判断是否裁剪'
    pending = update(state, 'CONFLICT', 'second', alternative, 'crop_order')
    assert pending['rules'] == state['rules']
    assert pending['conflicts'][0]['status'] == 'PENDING'
    unchanged = update(pending, 'SKIP', 'third')
    assert unchanged['conflicts'] == pending['conflicts']
    snapshot = copy.deepcopy(pending)
    with pytest.raises(PolicyError, match='Pending conflict'):
        update(pending, 'MERGE', 'third', alternative)
    with pytest.raises(PolicyError, match='later evidence'):
        update(pending, 'RESOLVE', 'second', alternative, 'crop_order')
    assert pending == snapshot
    resolved = update(pending, 'RESOLVE', 'fourth', alternative, 'crop_order')
    assert resolved['rules'][0]['procedure'] == alternative['procedure']
    assert resolved['conflicts'][0]['status'] == 'RESOLVED'
    assert resolved['conflicts'][0]['original_rule'] == state['rules'][0]
    assert original['rules'][0]['evidence_ids'] == ['first']
