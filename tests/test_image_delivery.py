import base64
import io
import json
import sys

import pytest
from PIL import Image

from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.image_tools_mcp import (
    ImageTools, image_content_summary, image_matches, read_hook, serve_stdio,
)


@pytest.fixture
def scene(tmp_path):
    job = tmp_path / 'remote'
    job.mkdir()
    image = Image.new('RGB', (60, 40), 'green')
    image.putpixel((10, 20), (255, 0, 0))
    image.save(job / 'image_0.png')
    tools = ImageTools(job, 1, edit_limit=1)
    return tools, tools.image_result(tools.views['image_0'])


@pytest.mark.parametrize('shape', ['mcp', 'read', 'anthropic', 'wrapped'])
def test_actual_image_shapes_are_decoded_and_hashed_without_base64_in_summary(scene, shape):
    tools, response = scene
    block = response['content'][-1]
    if shape == 'read':
        response = {'type': 'image', 'file': {'base64': block['data'], 'type': block['mimeType']}}
    elif shape == 'anthropic':
        response = [{'type': 'image', 'source': {
            'type': 'base64', 'media_type': block['mimeType'], 'data': block['data']}}]
    elif shape == 'wrapped':
        response = {'result': {'tool_response': response}}
    summary = image_content_summary(response)
    assert image_matches(summary, tools.views['image_0'])
    assert summary['images'][0]['bytes'] == (tools.job / 'image_0.png').stat().st_size
    assert block['data'] not in json.dumps(summary)


@pytest.mark.parametrize('response', [None, '', {}, [], {'type': 'text', 'text': '/tmp/image.png'},
    {'content': [{'type': 'text', 'text': '{"type":"image","path":"image.png"}'}]}])
def test_successful_tool_without_pixels_is_no_image(scene, response):
    tools, _ = scene
    summary = image_content_summary(response)
    assert summary['status'] == 'NO_IMAGE'
    assert not image_matches(summary, tools.views['image_0'])


@pytest.mark.parametrize('change', ['empty', 'bad_base64', 'text_bytes', 'truncated', 'mime', 'null_source'])
def test_corrupted_images_are_never_verified(scene, change):
    tools, response = scene
    block = response['content'][-1]
    if change == 'mime':
        block['mimeType'] = 'image/jpeg'
    elif change == 'null_source':
        block = {'type': 'image', 'source': None, 'file': None}
    else:
        block['data'] = {'empty': '', 'bad_base64': 'not-base64',
            'text_bytes': base64.b64encode(b'not an image').decode(),
            'truncated': base64.b64encode(base64.b64decode(block['data'])[:30]).decode()}[change]
    summary = image_content_summary(block)
    assert summary['status'] == 'INVALID_IMAGE'
    assert not image_matches(summary, tools.views['image_0'])


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_hook_records_content_separately_from_completion(scene, monkeypatch, direct, empty):
    tools, response = scene
    payload = {'hook_event_name': 'PostToolUse', 'tool_use_id': 'inspect',
        'tool_name': 'mcp__cloth_image__view_image' if direct else 'Read',
        'tool_input': {'image_id': 'image_0'} if direct else {'file_path': 'image_0.png'},
        'tool_response': None if empty else response}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(payload)))
    read_hook(tools.job)
    event = json.loads((tools.job / 'image_tool_calls.jsonl').read_text().splitlines()[-1])
    assert event['status'] == 'completed'
    assert event['image_content']['status'] == ('NO_IMAGE' if empty else 'VALID_IMAGE')
    assert (event['image_content']['identity_status'] == 'VERIFIED') is (not empty)
    assert tools.inspection_history()[0]['validated_image_returns'] == (0 if empty else 1)
    assert response['content'][-1]['data'] not in (tools.job / 'image_tool_calls.jsonl').read_text()


def test_stdio_returns_images_directly_and_view_does_not_spend_edit_budget(scene, monkeypatch, capsys):
    tools, _ = scene
    for index, (name, args) in enumerate([
        ('crop_image', {'image_id': 'image_0', 'box': [0, 0, 30, 20]}),
        ('view_image', {'image_id': 'image_0'}),
        ('crop_image', {'image_id': 'image_0', 'box': [0, 0, 30, 20]}),
    ]):
        monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
            'jsonrpc': '2.0', 'id': index, 'method': 'tools/call',
            'params': {'name': name, 'arguments': args}}) + '\n'))
        serve_stdio(tools)
        result = json.loads(capsys.readouterr().out)['result']
        metadata = json.loads(result['content'][0]['text'])
        assert image_matches(image_content_summary(result), metadata)
        assert metadata['edit_budget']['used'] == 1
        assert len(tools.views) == 2
        if index == 2:
            assert metadata['reused'] is True


@pytest.mark.parametrize('stream_first', [False, True])
@pytest.mark.parametrize('hook,stream,expected', [
    ('valid', 'valid', 'VERIFIED'), ('missing', 'valid', 'VERIFIED'),
    ('empty', 'valid', 'UNAVAILABLE'), ('valid', 'empty', 'UNAVAILABLE'),
    ('valid', 'wrong', 'SIZE_MISMATCH'), ('failed', 'valid', 'UNAVAILABLE'),
    ('valid', 'missing', 'UNKNOWN'), ('valid', 'error', 'UNAVAILABLE'),
])
def test_delivery_requires_correlated_cli_pixels_not_just_a_completed_hook(
        scene, tmp_path, hook, stream, expected, stream_first):
    tools, response = scene
    debug = ImageDebugSession(tmp_path / 'debug', [tools.job / 'image_0.png'], {})
    original = tools.views['image_0']
    # Replay may arrive after the stdout tool_result, on a separate SSH pipe.
    rotated = tools.call('rotate_image', {'image_id': 'image_0', 'degrees_clockwise': 90})
    edit = json.loads((tools.job / 'image_tool_calls.jsonl').read_text().splitlines()[-1])
    response = tools.image_result(rotated)
    debug.consume_claude_line(json.dumps({'type': 'assistant', 'message': {'content': [{
        'type': 'tool_use', 'id': 'inspect', 'name': 'mcp__cloth_image__rotate_image',
        'input': edit['arguments']}]}}))
    hook_event = {'kind': 'tool_lifecycle', 'tool': 'mcp__cloth_image__rotate_image',
        'tool_use_id': 'inspect', 'status': 'failed' if hook == 'failed' else 'completed',
        'arguments': edit['arguments'], 'image_metadata': rotated,
        'image_content': image_content_summary(None if hook == 'empty' else response)}
    content = [] if stream == 'empty' else response['content']
    if stream == 'wrong':
        content[-1] = tools.image_result(original)['content'][-1]
    message = {'type': 'user', 'message': {'content': [{'type': 'tool_result',
        'tool_use_id': 'inspect', 'is_error': stream == 'error', 'content': content}]}}
    if stream_first and stream != 'missing':
        debug.consume_claude_line(json.dumps(message))
    debug.consume({'kind': 'session', 'images': [original]})
    debug.consume(edit)
    if hook != 'missing':
        debug.consume(hook_event)
    if not stream_first and stream != 'missing':
        debug.consume_claude_line(json.dumps(message))
    view = debug.state['views'][-1]
    assert view['image_delivery_status'] == expected
    assert view['verification'] == 'VERIFIED'
    assert 'provider receipt' in debug.state['delivery_evidence_note']
    assert response['content'][-1]['data'] not in (debug.directory / 'claude_transcript.md').read_text()


def test_orphan_result_is_not_inspection_evidence(scene, tmp_path):
    tools, response = scene
    debug = ImageDebugSession(tmp_path / 'debug', [tools.job / 'image_0.png'], {})
    debug.consume({'kind': 'session', 'images': list(tools.views.values())})
    for identity in (None, 'unissued-tool'):
        debug.consume_claude_line(json.dumps({'type': 'user', 'message': {'content': [{
            'type': 'tool_result', 'tool_use_id': identity, 'content': response['content']}]}}))
    assert debug.state['views'][0]['image_delivery_status'] == 'UNKNOWN'
