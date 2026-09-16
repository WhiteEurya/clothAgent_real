import json
import subprocess
import sys

import pytest
from PIL import Image

from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.planner_backend import RemoteClaudeBackend, PlannerBackendError, parse_claude_json


def test_stream_requires_terminal_result_not_assistant_proposal():
    assistant = {'type': 'assistant', 'message': {'content': [
        {'type': 'text', 'text': '{"ok":true}'}]}}
    result = {'type': 'result', 'structured_output': {'ok': True}}
    stream = '\n'.join(map(json.dumps, [assistant, result]))
    assert parse_claude_json(stream) == {'ok': True}
    for events in ([assistant], [assistant, assistant], [assistant, result, result],
                   [assistant, {**result, 'is_error': True, 'result': 'refusal'}],
                   [assistant, {**result, 'subtype': 'error_max_turns'}]):
        with pytest.raises(PlannerBackendError):
            parse_claude_json('\n'.join(map(json.dumps, events)))


def test_public_messages_prompts_and_transfer_times_saved(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 8)).save(image)
    debug = ImageDebugSession(tmp_path / 'debug', [image], {'system_prompt': 'state assessment'})
    debug.save_prompt('Exact prompt\nimage_0.png')
    events = [
        {'type': 'assistant', 'message': {'content': [
            {'type': 'text', 'text': 'The collar is visible.'},
            {'type': 'thinking', 'thinking': 'Provider-exposed explanation.'},
            {'type': 'redacted_thinking'},
            {'type': 'tool_use', 'id': 'crop1', 'name': 'crop_image',
             'input': {'image_id': 'image_0', 'box': [0, 0, 4, 4]}}]}},
        {'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 'crop1', 'content': 'view saved'}]}},
        {'type': 'result', 'result': '{"ok":true}', 'duration_api_ms': 4210,
         'num_turns': 2, 'usage': {'input_tokens': 100}, 'modelUsage': {'model-test': {}}},
    ]
    for event in events:
        debug.consume_claude_line(json.dumps(event))
    for phase, duration in [('upload_0', 2.1), ('remote_download_0', 3.2),
                            ('remote_hash_0', .002), ('remote_claude', 4.8),
                            ('ssh_download_and_claude', 9.0)]:
        debug.progress(phase, 'finished', duration, {})
    debug.finish('COMPLETED')
    rows = [json.loads(line) for line in (debug.directory / 'claude_events.jsonl').read_text().splitlines()]
    assert [r['event'] for r in rows] == events
    assert rows[0]['since_previous_event_s'] is None
    assert rows[1]['since_previous_event_s'] >= 0
    assert (debug.directory / 'prompt.txt').read_text() == 'Exact prompt\nimage_0.png'
    transcript = (debug.directory / 'claude_transcript.md').read_text()
    assert 'Provider-exposed explanation.' in transcript
    assert 'Provider withheld reasoning' in transcript
    assert 'crop1' in transcript and 'view saved' in transcript
    timing = json.loads((debug.directory / 'timing.json').read_text())
    assert timing['images'][0]['upload_s'] == 2.1
    assert timing['images'][0]['download_s'] == 3.2
    assert timing['images'][0]['bytes'] == image.stat().st_size
    assert timing['claude_metrics']['duration_api_ms'] == 4210
    assert 'phases overlap' in timing['note']


def test_timeout_preserves_public_messages_without_final_result(tmp_path, monkeypatch):
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 8)).save(image)
    event = {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Inspecting collar'}]}}
    backend = RemoteClaudeBackend()
    def invoke(**kwargs):
        return backend._run_streaming([sys.executable, '-c',
            'import sys,time; print(sys.argv[1], flush=True); time.sleep(10)', json.dumps(event)], '', .3)
    monkeypatch.setattr(backend, '_invoke', invoke)
    debug = tmp_path / 'debug'
    with pytest.raises(subprocess.TimeoutExpired) as error:
        backend.invoke(prompt='inspect', system_prompt='state', schema={}, image_paths=[image], debug_dir=debug)
    assert 'Inspecting collar' in error.value.output
    assert 'Inspecting collar' in (debug / 'claude_transcript.md').read_text()
    assert json.loads((debug / 'image_debug.json').read_text())['status'] == 'FAILED'
    assert not (debug / 'claude_result.json').exists()
    assert 'remote_download_0' not in json.loads((debug / 'timing.json').read_text())['phases_s']
