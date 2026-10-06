import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from cloth_agent.image_tools_mcp import ImageTools
from cloth_agent.motion_image_sources import resolve_motion_sources
from cloth_agent.planner_backend import BackendResult
from cloth_agent.remote_fold import RemoteFoldClient
from cloth_agent.visual_preparation import build_handoff, reasoning_sources
from tests.test_remote_fold import saved_scene, visual_payload, motion_payload


def ready(identity):
    return {'status': 'READY', 'selected_views': [identity],
            'findings': [{'information_need': 'boundary', 'source_image_ids': [identity],
                          'finding': 'Boundary visible in this synthetic fixture'}],
            'success_check': 'Sufficient to distinguish panel and background',
            'missing_information': [], 'residual_uncertainty': 'Physical success untested'}


def audited_tools(directory, images):
    directory.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        with Image.open(image) as im:
            im.save(directory/f'image_{i}.png')
    return ImageTools(directory, len(images))


def verified(tools):
    return [{**v, 'verification': 'VERIFIED', 'image_delivery_status': 'VERIFIED'}
            for v in tools.views.values()]


def test_edited_image_handoff_preserves_complete_coordinate_chain(tmp_path):
    original = tmp_path/'camera_A_rgb_upright.png'
    Image.new('RGB', (40, 30), 'red').save(original)
    tools = audited_tools(tmp_path/'audit', [original])
    crop = tools.call('crop_image', {'image_id': 'image_0', 'box': [2, 3, 22, 23]})
    zoom = tools.call('resize_image', {'image_id': crop['image_id'], 'scale': 2})
    images, bundle, lineage, aliases = build_handoff(ready(zoom['image_id']), verified(tools),
        [original], tools.job, tmp_path/'handoff')
    assert bundle['findings'][0]['source_image_ids'] == ['image_1']
    redelivered = verified(audited_tools(tmp_path/'reasoning', images))
    sources = reasoning_sources(redelivered, lineage, aliases)
    payload = {'actions': [{'name': 'move', 'args': {'target': 'pixel',
                'image_id': 'image_1', 'pixel_xy': [10, 10]}}]}
    mapped, trace = resolve_motion_sources(payload, [original], sources)
    assert trace[0]['mapped_pixel_xy_float'] == pytest.approx([6.75, 7.75])
    assert mapped['actions'][0]['args']['pixel_xy'] == [7, 8]
    redelivered[1]['rgb_sha256'] = 'changed'
    with pytest.raises(ValueError, match='changed'):
        reasoning_sources(redelivered, lineage, aliases)


class SplitBackend:
    def __init__(self, *, insufficient=False, reasoning_gap=False):
        self.calls = []
        self.insufficient = insufficient
        self.reasoning_gap = reasoning_gap

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        stage = kwargs['usage_stage']
        tools = audited_tools(Path(kwargs['debug_dir']), kwargs['image_paths'])
        if stage == 'image_preparation':
            crop = tools.call('crop_image', {'image_id': 'image_0', 'box': [2, 2, 20, 25]})
            payload = ready(crop['image_id'])
            if self.insufficient:
                payload.update(status='INSUFFICIENT', missing_information=['Occluded cuff cannot be resolved'])
        else:
            assert kwargs['image_edit_limit'] == 0
            assert kwargs['information_tools'] is False
            assert len(kwargs['image_paths']) == 3  # originals plus prepared crop
            if stage == 'visual_planning':
                task = {**visual_payload(), 'skill_invocations': []}
            elif stage == 'pixel_motion':
                task = motion_payload()
                # Keep nullable source keys required by the remote action schema.
                for action in task['actions']:
                    if action['name'] == 'move': action['args']['image_id'] = None
            else:
                raise AssertionError(stage)
            payload = {'reasoning_status': 'READY', 'missing_information': [], 'result': task}
            if self.reasoning_gap:
                payload = {'reasoning_status': 'INSUFFICIENT', 'missing_information': ['Boundary remains ambiguous'], 'result': None}
        return BackendResult(json.dumps({'result': json.dumps(payload)}), '', 0, (),
                             image_sources=tuple(verified(tools)))


def test_production_planner_prepares_then_runs_both_reasoning_calls_read_only(saved_scene):
    session, images, _ = saved_scene
    backend = SplitBackend()
    client = RemoteFoldClient(backend=backend, two_stage_vision=True)
    client.plan(images, session, 'Fold the left sleeve')
    assert [c['usage_stage'] for c in backend.calls] == ['image_preparation', 'visual_planning', 'pixel_motion']
    assert backend.calls[0]['image_edit_limit'] == 6
    assert client._prepared_vision[1]['status'] == 'READY'
    for c in backend.calls[1:]:
        assert 'prepared_visual_evidence' in ''.join(c['context_files'].values())


@pytest.mark.parametrize('preparation_gap', [True, False])
def test_insufficient_evidence_stops_before_motion(saved_scene, preparation_gap):
    session, images, _ = saved_scene
    backend = SplitBackend(insufficient=preparation_gap, reasoning_gap=not preparation_gap)
    client = RemoteFoldClient(backend=backend, two_stage_vision=True)
    with pytest.raises((ValueError, RuntimeError), match='INSUFFICIENT'):
        client.plan(images, session, 'Fold the left sleeve')
    assert 'pixel_motion' not in [c['usage_stage'] for c in backend.calls]
    assert client.last_plan_result is None


def test_zero_edit_budget_removes_edit_tools_and_enforces_calls(tmp_path):
    Image.new('RGB', (10, 10), 'red').save(tmp_path/'image_0.png')
    tools = ImageTools(tmp_path, 1, edit_limit=0)
    names = {t['name'] for t in tools.available_tools()}
    assert 'view_image' in names
    assert not names & {'crop_image', 'resize_image', 'rotate_image', 'observe_information'}
    with pytest.raises(ValueError):
        tools.call('crop_image', {'image_id': 'image_0', 'box': [0, 0, 5, 5]})
