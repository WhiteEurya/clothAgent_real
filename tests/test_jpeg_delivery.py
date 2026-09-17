import base64
import io
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.image_tools_mcp import ImageTools, image_content_summary, read_hook, verify_image_delivery


@pytest.fixture
def scene(tmp_path):
    job = tmp_path / 'remote'
    job.mkdir()
    rng = random.Random(74)
    pixels = bytes(max(0, min(255, (x*3 + y*2 + c*40) % 256 + rng.randrange(-8, 9)))
                   for y in range(240) for x in range(320) for c in range(3))
    image = Image.frombytes('RGB', (320, 240), pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 60, 190, 210), fill=(180, 175, 170))
    draw.ellipse((75, 65, 120, 100), outline='black', width=4)
    draw.text((50, 150), 'LUCKY DAY', fill='navy')
    image.save(job / 'image_0.png')
    return ImageTools(job, 1), image


def jpeg(image, quality=85, sampling=2, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, subsampling=sampling, **kwargs)
    raw = buffer.getvalue()
    return raw, {'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(raw).decode()}


@pytest.mark.parametrize('quality,sampling', [(75, 2), (80, 2), (85, 2), (90, 1), (95, 0)])
def test_jpeg_transcoding_is_verified_without_relaxing_source_hash(scene, quality, sampling):
    tools, image = scene
    raw, block = jpeg(image, quality, sampling, progressive=True)
    summary = image_content_summary(block)
    check = verify_image_delivery(summary, tools.views['image_0'], tools.job / 'image_0.png', raw)
    assert summary['images'][0]['rgb_sha256'] != tools.views['image_0']['rgb_sha256']
    assert check['status'] == 'VERIFIED_TRANSCODE'
    assert check['metrics']['tile_rms_max'] == 0
    assert check['coordinate_change'] is False


@pytest.mark.parametrize('mutation', ['wrong', 'mirror', 'shift', 'crop', 'patch', 'resize', 'orientation', 'source_changed'])
def test_jpeg_does_not_allow_wrong_geometry_or_local_content_changes(scene, mutation):
    tools, image = scene
    kwargs = {}
    if mutation == 'wrong':
        image = Image.new('RGB', image.size, 'gray')
    elif mutation == 'mirror':
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif mutation == 'shift':
        image = ImageChops.offset(image, 3, 0)
    elif mutation == 'crop':
        image = image.crop((2, 2, 318, 238)).resize(image.size)
    elif mutation == 'patch':
        # A small changed region should not disappear in a global average.
        ImageDraw.Draw(image).rectangle((90, 80, 102, 92), fill='red')
    elif mutation == 'resize':
        image = image.resize((160, 120))
    elif mutation == 'orientation':
        exif = Image.Exif()
        exif[274] = 6
        kwargs['exif'] = exif
    elif mutation == 'source_changed':
        Image.new('RGB', image.size, 'white').save(tools.job / 'image_0.png')
    raw, block = jpeg(image, **kwargs)
    check = verify_image_delivery(image_content_summary(block), tools.views['image_0'],
                                  tools.job / 'image_0.png', raw)
    assert check['status'] == ('SIZE_MISMATCH' if mutation == 'resize' else 'CONTENT_MISMATCH')


@pytest.mark.parametrize('stream_first', [False, True])
@pytest.mark.parametrize('identity', ['good', 'bad_hash', 'bad_path', 'bad_size', 'missing'])
def test_correlated_jpeg_survives_hook_stream_order_and_saves_exact_bytes(scene, tmp_path, monkeypatch,
                                                                       stream_first, identity):
    tools, image = scene
    response = tools.image_result(tools.views['image_0'])
    raw, response['content'][-1] = jpeg(image)
    metadata = json.loads(response['content'][0]['text'])
    if identity == 'bad_hash':
        metadata['rgb_sha256'] = 'wrong'
    elif identity == 'bad_path':
        metadata['path'] = '/tmp/wrong/image_0.png'
    elif identity == 'bad_size':
        metadata['size'] = [12, 12]
    response['content'][0]['text'] = json.dumps(metadata)
    if identity == 'missing':
        response['content'] = response['content'][1:]
    payload = {'hook_event_name': 'PostToolUse', 'tool_use_id': 'jpeg',
        'tool_name': 'mcp__cloth_image__view_image', 'tool_input': {'image_id': 'image_0'},
        'tool_response': response}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(payload)))
    read_hook(tools.job)
    hook = json.loads((tools.job / 'image_tool_calls.jsonl').read_text().splitlines()[-1])
    if identity == 'good':
        assert hook['image_content']['identity_status'] == 'VERIFIED_TRANSCODE'
        assert tools.inspection_history()[0]['validated_image_returns'] == 1
    debug = ImageDebugSession(tmp_path / 'debug', [tools.job / 'image_0.png'], {})
    message = json.dumps({'type': 'user', 'message': {'content': [{
        'type': 'tool_result', 'tool_use_id': 'jpeg', 'content': response['content']}]}})
    if stream_first:
        debug.consume_claude_line(message)
    debug.consume({'kind': 'session', 'images': list(tools.views.values())})
    debug.consume(hook)
    if not stream_first:
        debug.consume_claude_line(message)
    debug.finish('COMPLETED')
    view = debug.state['views'][0]
    assert view['verification'] == 'VERIFIED'
    assert view['image_delivery_status'] == ('VERIFIED_TRANSCODE' if identity == 'good' else 'IDENTITY_MISMATCH')
    if identity == 'good':
        assert Path(view['delivered_image']['saved_path']).read_bytes() == raw
    else:
        assert 'delivered_image' not in view
    assert next((debug.directory / 'returned_images').iterdir()).read_bytes() == raw
    assert block_data(response) not in (debug.directory / 'image_delivery.jsonl').read_text()


def block_data(response):
    return response['content'][-1]['data']


@pytest.mark.parametrize('failed_boundary', ['hook', 'stream'])
def test_valid_jpeg_never_overrides_tool_failure(scene, tmp_path, failed_boundary):
    tools, image = scene
    response = tools.image_result(tools.views['image_0'])
    _, response['content'][-1] = jpeg(image)
    debug = ImageDebugSession(tmp_path / 'debug', [tools.job / 'image_0.png'], {})
    debug.consume({'kind': 'session', 'images': list(tools.views.values())})
    debug.consume({'kind': 'tool_lifecycle', 'tool': 'mcp__cloth_image__view_image',
                   'status': 'failed' if failed_boundary == 'hook' else 'completed',
                   'tool_use_id': 'jpeg', 'arguments': {'image_id': 'image_0'},
                   'image_content': image_content_summary(response),
                   'image_metadata': json.loads(response['content'][0]['text'])})
    debug.consume_claude_line(json.dumps({'type': 'user', 'message': {'content': [{
        'type': 'tool_result', 'tool_use_id': 'jpeg', 'content': response['content'],
        'is_error': failed_boundary == 'stream'}]}}))
    debug.finish('COMPLETED')
    view = debug.state['views'][0]
    assert view['stream_image_status'] == 'VERIFIED_TRANSCODE'
    assert view['image_delivery_status'] == 'UNAVAILABLE'
    assert 'delivered_image' not in view
