import json
import subprocess
import sys

import pytest
from PIL import Image

from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.claude_stream import ClaudeStreamProgress
from cloth_agent.planner_backend import RemoteClaudeBackend, parse_claude_json


def collector():
    now, notices = [0.], []
    stream = ClaudeStreamProgress(lambda *a, **k: notices.append((a, k)), clock=lambda: now[0])
    return now, notices, stream


def test_1500_system_events_are_summarized_and_silence_gets_a_heartbeat():
    now, notices, stream = collector()
    for i in range(1500):
        now[0] = i / 100
        stream.consume({'type': 'system', 'subtype': 'thinking_tokens'})
    stream.tick(force=True)
    assert len(notices) == 3
    assert all(args == ('claude_stream', 'progress') for args, _ in notices)
    assert sum(details['new_events'] for _, details in notices) == 1500
    final = notices[-1][1]
    assert final['raw_events'] == 1500
    assert final['event_counts'] == {'system.thinking_tokens': 1500}
    assert final['tool_calls'] == 0
    now[0] += 5.1
    stream.tick()
    assert notices[-1][1]['new_events'] == 0
    assert notices[-1][1]['idle_s'] == 5.1


def test_text_deltas_are_live_without_repeating_completed_assistant_text():
    now, notices, stream = collector()
    def part(event):
        stream.consume({'type': 'stream_event', 'event': event})
    part({'type': 'message_start', 'message': {'id': 'm1'}})
    part({'type': 'content_block_delta', 'index': 0,
          'delta': {'type': 'thinking_delta', 'thinking': 'not public text'}})
    for text in ['Looking ', 'at the ', 'collar.']:
        now[0] += .4
        part({'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': text}})
    assert notices[0] == (('claude_text', 'received'), {
        'text_excerpt': 'Looking at the collar.', 'characters': 22, 'excerpt_only': False})
    stream.consume({'type': 'assistant', 'message': {'id': 'm1', 'content': [
        {'type': 'text', 'text': 'Looking at the collar.'}]}})
    stream.consume({'type': 'result', 'num_turns': 1, 'stop_reason': 'end_turn', 'is_error': False})
    assert sum(a[0] == 'claude_text' for a, _ in notices) == 1
    assert notices[-1][0] == ('claude_result', 'completed')
    assert notices[-1][1]['num_turns'] == 1
    assert 'not public text' not in json.dumps(notices)
    count = len(notices)
    now[0] += 10
    stream.tick()
    assert len(notices) == count


def test_tools_errors_and_retries_are_immediate_and_counted_once():
    _, notices, stream = collector()
    call = {'type': 'assistant', 'message': {'content': [
        {'type': 'tool_use', 'id': 'view1', 'name': 'mcp__cloth_image__view_image', 'input': {}}]}}
    stream.consume(call)
    stream.consume(call)
    assert len(notices) == 1
    assert notices[0][0] == ('claude_tool', 'started')
    stream.consume({'type': 'user', 'message': {'content': [
        {'type': 'tool_result', 'tool_use_id': 'view1', 'is_error': True, 'content': 'error'}]}})
    assert notices[-1][0] == ('claude_tool', 'failed')
    stream.consume({'type': 'system', 'subtype': 'api_retry', 'message': 'waiting before retry'})
    assert notices[-1][0] == ('claude_notice', 'received')
    assert notices[-1][1]['subtype'] == 'api_retry'
    stream.consume({'type': 'result', 'is_error': True, 'subtype': 'error_max_turns'})
    assert notices[-1][0] == ('claude_result', 'failed')
    assert notices[-2][1]['tool_calls'] == 1


def test_snapshot_writes_are_bounded_and_raw_events_are_complete(tmp_path, monkeypatch):
    now = [0.]
    monkeypatch.setattr('cloth_agent.claude_image_debug.time.monotonic', lambda: now[0])
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 8)).save(image)
    debug = ImageDebugSession(tmp_path / 'debug', [image], {})
    writes = []
    original_write = debug.write
    def write(name, value):
        if name == 'image_debug.json':
            writes.append(now[0])
        original_write(name, value)
    monkeypatch.setattr(debug, 'write', write)
    monkeypatch.setattr(debug, '_match_reads', lambda: pytest.fail('system events must not recheck every image'))
    for i in range(1500):
        now[0] = i / 1000
        debug.consume_claude_line(json.dumps({'type': 'system', 'subtype': 'thinking_tokens', 'counter': i}))
        debug.flush()  # The streaming loop's timer also calls this.
    assert writes == [1.0]
    debug.flush(force=True)
    assert len(writes) == 2
    rows = [json.loads(line) for line in (debug.directory / 'claude_events.jsonl').read_text().splitlines()]
    assert [row['event']['counter'] for row in rows] == list(range(1500))
    state = json.loads((debug.directory / 'image_debug.json').read_text())
    assert state['claude_event_count'] == 1500
    assert state['claude_event_counts'] == {'system.thinking_tokens': 1500}
    assert not state['progress']
    assert 'thinking_tokens' not in (debug.directory / 'claude_transcript.md').read_text()


@pytest.mark.parametrize('timeout', [False, True])
def test_real_pipe_drain_saves_burst_and_flushes_pending_text(tmp_path, monkeypatch, timeout):
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 8)).save(image)
    backend = RemoteClaudeBackend()
    notices = []
    backend.progress_callback = lambda *a, **k: notices.append((a, k))
    text_events = [
        {'type': 'stream_event', 'event': {'type': 'message_start', 'message': {'id': 'm1'}}},
        {'type': 'stream_event', 'event': {'type': 'content_block_delta', 'index': 0,
            'delta': {'type': 'text_delta', 'text': 'Inspecting cloth'}}},
    ]
    def invoke(**kwargs):
        return backend._run_streaming([sys.executable, '-c',
            'import json,sys,time; '
            '[print(json.dumps({"type":"system","subtype":"thinking_tokens","counter":i}),flush=True) for i in range(1500)]; '
            '[print(json.dumps(e),flush=True) for e in json.loads(sys.argv[1])]; '
            'time.sleep(10) if sys.argv[2] == "timeout" else print(json.dumps({"type":"result","structured_output":{"ok":True},"num_turns":1}),flush=True)',
            json.dumps(text_events), 'timeout' if timeout else 'success'], '', .7 if timeout else 10)
    monkeypatch.setattr(backend, '_invoke', invoke)
    debug_dir = tmp_path / 'debug'
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            backend.invoke(prompt='inspect', system_prompt='inspect', schema={}, image_paths=[image], debug_dir=debug_dir)
    else:
        # invoke normally returns BackendResult; use the real drain directly here.
        backend._debug_session = ImageDebugSession(debug_dir, [image], {})
        completed = invoke()
        assert parse_claude_json(completed.stdout) == {'ok': True}
        backend._debug_session.finish('COMPLETED')
    rows = [json.loads(line) for line in (debug_dir / 'claude_events.jsonl').read_text().splitlines()]
    assert len(rows) == 1502 + (not timeout)
    assert len((debug_dir / 'stdout.log').read_text().splitlines()) == len(rows)
    assert all(args[0] != 'claude_message' for args, _ in notices)
    assert len(notices) < 15
    assert any(fields.get('text_excerpt') == 'Inspecting cloth' for _, fields in notices)
    state = json.loads((debug_dir / 'image_debug.json').read_text())
    assert state['status'] == ('FAILED' if timeout else 'COMPLETED')
    assert state['claude_event_count'] == len(rows)


@pytest.mark.parametrize('timeout', [False, True])
def test_remote_wrapper_progress_through_real_backend_logger(tmp_path, timeout):
    import shutil
    from cloth_agent import remote_output
    image=tmp_path/'image.png'; Image.new('RGB',(8,8)).save(image)
    backend=RemoteClaudeBackend()
    notices=[]; backend.progress_callback=lambda *args,**kwargs:notices.append((args,kwargs))
    backend._debug_session=ImageDebugSession(tmp_path/'debug',[image],{})
    backend._seen_timings=set(); backend._seen_events=set()
    code="import json,time; print(json.dumps({'type':'system','subtype':'api_retry','error_status':524,'attempt':1}),flush=True); "
    code += "time.sleep(10)" if timeout else "print(json.dumps({'type':'result','structured_output':{'ok':True}}),flush=True)"
    command=[sys.executable,str(remote_output.__file__),'--record-event-timing']
    if timeout:
        executable=shutil.which('timeout')
        if not executable: pytest.skip('GNU timeout unavailable')
        command += [executable,'--kill-after=1s','.3s']
    command += [sys.executable,'-c',code]
    # Wrapper writes raw files in cwd; isolate it without changing global cwd.
    launcher=[sys.executable,'-c','import os,sys; os.chdir(sys.argv[1]); os.execv(sys.argv[2],sys.argv[2:])',str(tmp_path),*command]
    result=backend._run_streaming(launcher,'',4)
    assert result.returncode == (124 if timeout else 0)
    progress=[r for r in backend._debug_session.state['progress'] if r['stage']=='remote_cli']
    assert progress[0]['event']=='cli_started'
    assert progress[0]['remote_elapsed_s']==0
    assert progress[0]['elapsed_s']>=0
    assert progress[-1]['event']=='cli_finished'
    assert progress[-1]['returncode']==result.returncode
    assert 'api_retry' in result.stdout
    assert 'remote_elapsed_s' in next(fields for args,fields in notices if args[0]=='remote_cli')
    if not timeout: assert parse_claude_json(result.stdout)=={'ok':True}


def test_progress_metadata_collision_preserves_local_clock(tmp_path):
    image=tmp_path/'image.png'; Image.new('RGB',(8,8)).save(image)
    debug=ImageDebugSession(tmp_path/'debug',[image],{})
    debug.progress('local_stage','started',None,{'elapsed_s':999,'stage':'remote_stage','event':'remote_event','duration_s':12})
    row=debug.state['progress'][-1]
    assert row['stage']=='local_stage' and row['event']=='started'
    assert row['duration_s'] is None and row['elapsed_s']<999
    assert row['source_details']['elapsed_s']==999
