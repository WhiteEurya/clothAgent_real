import json

import pytest

from cloth_agent.semantic_timing import analyze_semantic_events, CATEGORIES


def event(t, kind, **data):
    return {'type': 'stream_event', 'event': {'type': kind, **data},
            '_cloth_timing': {'elapsed_s': t}}


def marker(t, phase, action):
    return [event(t, 'content_block_start', index=0, content_block={'type': 'text', 'text': ''}),
            event(t, 'content_block_delta', index=0,
                  delta={'type': 'text_delta', 'text': f'[[phase:{phase}:{action}]]\n'}),
            event(t, 'content_block_stop', index=0)]


def analyze(rows):
    return analyze_semantic_events('\n'.join(json.dumps(r) for r in rows))


def totals(result):
    return {r['phase']: r for r in result['totals']}


def test_split_stream_markers_and_duplicate_assistant_are_not_double_counted():
    rows = [event(1, 'message_start'),
            event(2, 'content_block_start', index=0, content_block={'type': 'text'}),
            event(2, 'content_block_delta', index=0,
                  delta={'type': 'text_delta', 'text': '[[phase:orien'}),
            event(3, 'content_block_delta', index=0,
                  delta={'type': 'text_delta', 'text': 'tation:start]]\n'}),
            event(3, 'content_block_stop', index=0),
            event(4, 'content_block_start', index=1, content_block={'type': 'thinking'}),
            event(6, 'content_block_delta', index=1,
                  delta={'type': 'thinking_delta', 'thinking': '[[phase:candidate:start]]\n'}),
            event(9, 'content_block_stop', index=1),
            *marker(10, 'orientation', 'end'), event(11, 'message_stop'),
            {'type': 'assistant', 'message': {'content': [{'type': 'text',
             'text': '[[phase:orientation:start]]\n[[phase:orientation:end]]'}]},
             '_cloth_timing': {'elapsed_s': 11}}]
    result = analyze(rows)
    assert len(result['markers']) == 2
    assert result['status'] == 'OBSERVED'
    t = totals(result)
    assert t['orientation']['wall_s'] == 7
    assert t['orientation']['thinking_s'] == 5
    assert t['UNKNOWN']['wall_s'] == 4
    assert t['candidate']['wall_s'] is None


def test_revisits_and_overlapping_tools_partition_wall_time():
    rows = [event(0, 'message_start'), *marker(1, 'correspondence', 'start'),
            event(2, 'content_block_start', index=1, content_block={'type': 'tool_use', 'name': 'Read'}),
            {'type': 'assistant', 'message': {'content': [{'type': 'tool_use',
             'id': 'read', 'name': 'Read', 'input': {}}]}, '_cloth_timing': {'elapsed_s': 3}},
            event(4, 'content_block_stop', index=1),
            {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'read'}]},
             '_cloth_timing': {'elapsed_s': 6}},
            *marker(8, 'correspondence', 'end'),
            *marker(9, 'correspondence', 'start'), *marker(12, 'correspondence', 'end'),
            event(13, 'message_stop')]
    t = totals(analyze(rows))
    assert t['correspondence']['visits'] == 2
    assert t['correspondence']['wall_s'] == 10
    assert t['correspondence']['tool_arguments_s'] == 2
    assert t['correspondence']['tool_wait_s'] == 2  # 3..4 overlaps emitted arguments
    assert sum(r['wall_s'] or 0 for r in t.values()) == 13
    for r in t.values():
        if r['wall_s'] is not None:
            assert sum(r[k+'_s'] for k in CATEGORIES) == r['wall_s']


def test_missing_mismatched_and_unknown_markers_do_not_invent_attribution():
    rows = [event(0, 'message_start'), *marker(1, 'orientation', 'start'),
            *marker(3, 'candidate', 'start'), *marker(5, 'correspondence', 'end'),
            *marker(6, 'invented', 'start'), *marker(7, 'motion_target', 'start'),
            event(10, 'message_stop')]
    result = analyze(rows)
    assert result['status'] == 'PARTIAL'
    assert totals(result)['UNKNOWN']['wall_s'] == 10
    assert all(r['duration_s'] is None for r in result['occurrences'])


def test_no_markers_no_timestamps_and_bad_clock_are_explicit():
    result = analyze([event(2, 'message_start'), event(8, 'message_stop')])
    assert result['status'] == 'NO_MARKERS'
    assert totals(result)['UNKNOWN']['wall_s'] == 8
    result = analyze([{'type': 'assistant', 'received_elapsed_s': 900}])
    assert result['status'] == 'NO_TIMESTAMPS'
    assert totals(result)['UNKNOWN']['wall_s'] is None
    result = analyze([event(1, 'message_start'), *marker(2, 'orientation', 'start'),
                      *marker(1, 'orientation', 'end'), event(5, 'message_stop')])
    assert result['status'] == 'INVALID_TIMESTAMPS'
    assert totals(result)['UNKNOWN']['wall_s'] == 5


def test_coalesced_or_aggregate_only_markers_are_not_measurements():
    result = analyze([event(1, 'message_start'), *marker(2, 'orientation', 'start'),
                      *marker(2, 'orientation', 'end'), event(8, 'message_stop')])
    assert result['status'] == 'PARTIAL'
    assert result['occurrences'][0]['status'] == 'UNRESOLVED'
    assert totals(result)['orientation']['wall_s'] is None
    assert totals(result)['UNKNOWN']['wall_s'] == 8
    result = analyze([{'type': 'assistant', 'message': {'content': [{'type': 'text',
                       'text': '[[phase:orientation:start]]\n[[phase:orientation:end]]'}]},
                       '_cloth_timing': {'elapsed_s': 20}}])
    assert result['status'] == 'NO_MARKERS'
    assert totals(result)['UNKNOWN']['wall_s'] == 20


def test_backend_injection_is_opt_in_and_preserves_original_schema_and_task(tmp_path, monkeypatch):
    from cloth_agent.pipeline_timing import PipelineTiming
    from cloth_agent.planner_backend import RemoteClaudeBackend, BackendResult
    backend = RemoteClaudeBackend()
    received = []
    def invoke(**kwargs):
        received.append(kwargs)
        return BackendResult('{}', '', 0, ())
    monkeypatch.setattr(backend, '_invoke', invoke)
    schema = {'type': 'object', 'properties': {'selected_reference': {'type': 'object'}}}
    def call(stage):
        return backend.invoke(prompt='original task', image_paths=[], schema=schema,
                              system_prompt='original system', usage_stage=stage)
    call('visual_planning')
    with PipelineTiming(tmp_path/'plain'):
        call('visual_planning')
    with PipelineTiming(tmp_path/'diagnostic', semantic_phases=True):
        call('visual_planning')
        call('pixel_motion')
        call('experience_update')
    assert [x['system_prompt'] != 'original system' for x in received] == [False, False, True, True, False]
    assert all(x['schema'] == schema and x['prompt'] == 'original task' for x in received)
    assert '[[phase:NAME:start]]' in received[2]['system_prompt']


def test_cli_requires_remote_timing_and_report_survives_failure(tmp_path, monkeypatch):
    from cloth_agent import fold_exploration_pipeline as pipeline
    from cloth_agent.pipeline_timing import active_timing
    def run(args):
        assert active_timing().semantic_phases
        debug = tmp_path/'trace'
        debug.mkdir()
        rows = [event(1, 'message_start'), *marker(2, 'orientation', 'start'),
                *marker(5, 'orientation', 'end'), event(6, 'message_stop')]
        (debug/'claude_stdout.txt').write_text('\n'.join(json.dumps(r) for r in rows))
        with active_timing().span('remote.invoke', debug_dir=str(debug), usage_stage='visual_planning', iteration=1):
            raise RuntimeError('schema failed')
    monkeypatch.setattr(pipeline, '_run_main', run)
    with pytest.raises(SystemExit):
        pipeline.main(['--semantic-timing'])
    with pytest.raises(SystemExit):
        pipeline.main(['--semantic-timing', '--timing-output', str(tmp_path/'invalid'), '--planner-backend', 'local'])
    assert not (tmp_path/'invalid').exists()
    with pytest.raises(RuntimeError, match='schema failed'):
        pipeline.main(['--semantic-timing', '--timing-output', str(tmp_path/'timing')])
    report = json.loads((tmp_path/'timing/semantic_timing.json').read_text())
    assert totals(report[0]['semantic'])['orientation']['wall_s'] == 3
    assert report[0]['iteration'] == 1
    assert '衣服方向判断' in (tmp_path/'timing/semantic_timing.md').read_text()
