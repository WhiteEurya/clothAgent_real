import json

from cloth_agent.public_process import collect_rollout_trace, stream_process
from cloth_agent.harness.reasoning_code import public_records
from cloth_agent.harness.reasoning_contract import JUDGMENT_SCHEMA
from cloth_agent.harness.policy import validate_schema
from test_reasoning_code import final


def test_export_preserves_tool_order_links_and_excludes_private_blocks(tmp_path):
    path = tmp_path/'events.jsonl'
    events = [
        {'type': 'assistant', 'message': {'content': [
            {'type': 'thinking', 'thinking': 'PRIVATE_SENTINEL'},
            {'type': 'text', 'text': 'Need a clearer cuff edge.'},
            {'type': 'tool_use', 'id': 'crop1', 'name': 'crop_image', 'input': {'image_id': 'image_0', 'box': [1, 2, 4, 6]}}]}},
        {'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 'crop1', 'content': [
                {'type': 'text', 'text': 'Produced view_1'},
                {'type': 'image', 'source': {'type': 'base64', 'data': 'BINARY_SENTINEL'}}]}]}},
        {'type': 'assistant', 'message': {'content': [
            {'type': 'text', 'text': '[[process]] {"id":"cuff","result_summary":"Cuff edge visible","candidate_updates":[{"candidate_id":"R002","outcome":"rejected","reason":"Outside fabric"}]}'}]}},
    ]
    path.write_text('\n'.join(json.dumps({'received_elapsed_s': n+1, 'event': e}) for n, e in enumerate(events)))
    result = stream_process(path)
    assert [e['kind'] for e in result['events']] == ['public_statement', 'tool_request', 'tool_result', 'decision_record']
    assert result['events'][2]['tool_use_id'] == 'crop1'
    text = json.dumps(result)
    assert 'PRIVATE_SENTINEL' not in text and 'BINARY_SENTINEL' not in text
    assert 'Outside fabric' in text
    assert result['events'][0]['source']['timing_scope'] == 'receipt time, not topic thinking duration'


def test_full_observation_to_selection_trace_keeps_missing_comparisons_explicit(tmp_path):
    (tmp_path/'observe').mkdir()
    (tmp_path/'observe/returned.json').write_text(json.dumps({'information': [
        {'id': 'edge', 'need': 'Resolve cuff boundary', 'status': 'UNKNOWN'}]}))
    row = {'status': 'READY', 'action': {'selected': 'R001'}, 'observation_trace': [
        {'request': {'gap_id': 'edge', 'skill_id': 'local_boundary', 'source_image_id': 'image_0',
                     'expected_information_gain': 'Resolve cuff boundary'},
         'lineage': [{'image_id': 'image_1', 'parent_image_id': 'image_0', 'operation': 'crop_image',
                      'arguments': {'box': [0, 0, 20, 20]}, 'to_original': [1, 0, 0, 0, 1, 0]}],
         'delivered_image_ids': ['image_1'], 'status': 'EXECUTED_NOT_YET_INTERPRETED'}],
        'reasoning': {'stages': [{'stage_id': 'plan', 'judgment': {'concepts': [
            {'name': 'cuff', 'finding': 'Visible hem', 'source_image_ids': ['image_1']}]}}]}}
    trace = collect_rollout_trace(row, tmp_path)
    assert trace['coverage']['counts']['image_operation'] == 1
    assert 'CANDIDATE_COMPARISON_HISTORY_NOT_EXPLICITLY_RECORDED' in trace['coverage']['gaps']
    assert any(e['relation'] == 'ADDRESSES_INFORMATION_NEED' for e in trace['edges'])
    assert any(e['relation'] == 'MODEL_CITES_IMAGE_NOT_CAUSAL_PROOF' for e in trace['edges'])
    assert 'EXECUTED_NOT_YET_INTERPRETED' in json.dumps(trace)
    # Exporting the old summary never invents a candidate rejection or new intermediate step.
    assert not any(n['kind'] == 'decision_record' for n in trace['nodes'])


def test_extraction_receives_observation_operations_not_only_final_answer(tmp_path):
    folder = tmp_path/'replays/baseline_r00/observe'; folder.mkdir(parents=True)
    (folder/'returned.json').write_text(json.dumps({'information': [
        {'id': 'gap', 'need': 'Read cuff seam', 'status': 'UNKNOWN'}]}))
    report = {'rollouts': [{'rollout_id': 'baseline_r00', 'status': 'READY', 'action': None,
        'observation_trace': [{'request': {'gap_id': 'gap', 'skill_id': 'local_boundary'},
            'lineage': [{'image_id': 'image_1', 'parent_image_id': 'image_0', 'operation': 'resize_image',
                         'arguments': {'scale': 3}}], 'status': 'EXECUTED_NOT_YET_INTERPRETED'}]}]}
    records = public_records(report, 'baseline', tmp_path)
    text = json.dumps(records)
    assert 'Read cuff seam' in text and 'resize_image' in text and '"scale": 3' in text
    assert 'operation_trace' in records[0]


def test_decision_log_is_optional_and_has_explicit_candidate_evidence():
    judgment = final()['judgment']
    validate_schema(judgment, JUDGMENT_SCHEMA)
    judgment['decision_log'] = [{'id': 'cuff', 'information_need': 'Fabric membership',
        'operation': 'Assess supplied crop', 'evidence_refs': ['image_0'], 'result_summary': 'Cuff visible',
        'candidate_updates': [{'candidate_id': 'R001', 'outcome': 'selected', 'reason': 'Visible cuff fabric'}],
        'depends_on': []}]
    judgment['decision_log'].append({'id': 'selection', 'information_need': 'Choose from supported candidates',
        'operation': 'Select a candidate', 'evidence_refs': ['image_0'], 'result_summary': 'R001 selected',
        'candidate_updates': [{'candidate_id': 'R001', 'outcome': 'selected', 'reason': 'Supported by cuff observation'}],
        'depends_on': ['cuff']})
    trace = collect_rollout_trace({'status': 'READY', 'judgment': judgment})
    assert any(e['relation'] == 'MODEL_DECLARED_DEPENDENCY' for e in trace['edges'])
    validate_schema(judgment, JUDGMENT_SCHEMA)


def test_incomplete_stream_keeps_request_and_marks_missing_result(tmp_path):
    path = tmp_path/'stdout.jsonl'
    path.write_text(json.dumps({'type': 'assistant', 'message': {'content': [
        {'type': 'tool_use', 'id': 't1', 'name': 'Read', 'input': {'file_path': 'image.png'}}]}})+'\n{')
    result = stream_process(path)
    assert len(result['events']) == 1
    assert result['warnings'] == ['MALFORMED_LINE:2', 'TOOL_RESULT_NOT_RECORDED:t1']
