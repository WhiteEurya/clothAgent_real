import json
from pathlib import Path

import pytest
from PIL import Image

from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.image_tools_mcp import ImageTools
from cloth_agent.motion_image_sources import resolve_motion_sources


@pytest.fixture
def scene(tmp_path):
    current = tmp_path / 'camera_A_rgb_upright.png'
    reference = tmp_path / 'fold_reference_target.png'
    Image.new('RGB', (120, 80), 'green').save(current)
    Image.new('RGB', (120, 80), 'red').save(reference)
    debug = ImageDebugSession(tmp_path / 'debug', [current, reference], {})
    tools = ImageTools(debug.image_dir, 2)
    debug.consume({'kind': 'session', 'images': list(tools.views.values())})
    def call(name, **args):
        value = tools.call(name, args)
        event = json.loads((tools.job / 'image_tool_calls.jsonl').read_text().splitlines()[-1])
        debug.consume(event)
        return value
    return [current, reference], debug, call


def motion(image_id, pixel):
    return {'actions': [{'name': 'move', 'args': {'target': 'pixel', 'image_id': image_id,
        'pixel_xy': pixel, 'height_above_grasp_mm': 30, 'yaw_deg': 0}}], 'requires_lift_checkpoint': False}


def test_host_resolves_crop_rotate_resize_chain_and_rounds(scene):
    images, debug, call = scene
    crop = call('crop_image', image_id='image_0', box=[20, 10, 100, 70])
    rotate = call('rotate_image', image_id=crop['image_id'], degrees_clockwise=90)
    zoom = call('resize_image', image_id=rotate['image_id'], scale=2)
    payload = motion(zoom['image_id'], [5, 7])
    result, trace = resolve_motion_sources(payload, images, debug.state['views'])
    assert trace[0]['mapped_pixel_xy_float'] == pytest.approx([23.25, 66.75])
    assert result['actions'][0]['args']['pixel_xy'] == [23, 67]
    assert 'image_id' not in result['actions'][0]['args']
    assert payload['actions'][0]['args']['image_id'] == zoom['image_id']
    assert trace[0]['source_image_id'] == zoom['image_id']


@pytest.mark.parametrize('source', ['image_1', 'missing', None])
def test_reference_unknown_and_unspecified_sources_rejected(scene, source):
    images, debug, _ = scene
    with pytest.raises(ValueError):
        resolve_motion_sources(motion(source, [30, 30]), images, debug.state['views'])


def test_reference_derivative_padding_and_unverified_replay_rejected(scene):
    images, debug, call = scene
    reference = call('rotate_image', image_id='image_1', degrees_clockwise=90)
    with pytest.raises(ValueError, match='CURRENT'):
        resolve_motion_sources(motion(reference['image_id'], [30, 30]), images, debug.state['views'])
    rotate = call('rotate_image', image_id='image_0', degrees_clockwise=37)
    with pytest.raises(ValueError, match='padding'):
        resolve_motion_sources(motion(rotate['image_id'], [0, 0]), images, debug.state['views'])
    debug.state['views'][-1]['verification'] = 'UNVERIFIED_REPLAY'
    with pytest.raises(ValueError, match='verified'):
        resolve_motion_sources(motion(rotate['image_id'], [40, 40]), images, debug.state['views'])


def test_direct_float_point_and_changed_current_image(scene):
    images, debug, _ = scene
    result, _ = resolve_motion_sources(motion('image_0', [10.5, 12.25]), images, debug.state['views'])
    assert result['actions'][0]['args']['pixel_xy'] == [11, 12]
    Image.new('RGB', (120, 80), 'black').save(images[0])
    with pytest.raises(ValueError, match='changed'):
        resolve_motion_sources(motion('image_0', [10.5, 12.25]), images, debug.state['views'])
