from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.claude_image_debug import ImageDebugSession
from cloth_agent.claude_molmo_view import prepare_molmo_view, map_molmo_pixel, MolmoOrientationError
from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline
from cloth_agent.image_tools_mcp import ImageTools, pixel_hash
from cloth_agent.molmo_keypoint_pipeline import (
    CONFIDENCE_DEFINITION, MolmoKeypointPipelineError, run_molmo_keypoint_pipeline,
)
from cloth_agent.planner_backend import BackendResult
from cloth_agent.remote_fold import RemoteFoldClient, rgb_evidence


class OrientationBackend:
    """Actual image tools/replay, only Claude's choices are simulated."""
    def __init__(self, *, angle=90, zoom=1, failure=None, extra_edits=0):
        self.angle, self.zoom, self.failure = angle, zoom, failure
        self.extra_edits = extra_edits
        self.called = False

    def invoke(self, **kwargs):
        assert kwargs['image_edit_limit'] == 6
        debug = ImageDebugSession(kwargs['debug_dir'], kwargs['image_paths'], kwargs['prompt'])
        tools = ImageTools(debug.image_dir, 1, edit_limit=kwargs['image_edit_limit'])
        debug.consume({'kind': 'session', 'images': list(tools.views.values())})
        selected = tools.views['image_0']
        for name, args in ([('rotate_image', {'degrees_clockwise': self.angle})] if self.angle else []) + (
                [('resize_image', {'scale': self.zoom})] if self.zoom != 1 else []) + [
                    ('resize_image', {'scale': 1})]*self.extra_edits:
            selected = tools.call(name, {'image_id': selected['image_id'], **args})
            debug.consume(json.loads((tools.job / 'image_tool_calls.jsonl').read_text().splitlines()[-1]))
        if self.failure != 'unread':
            debug.consume({'kind': 'read', 'status': 'completed', 'tool_use_id': 'read_final',
                           'arguments': {'file_path': selected['path']}})
        w, h = selected['size']
        response = {'status': 'READY', 'image_id': selected['image_id'],
                    'collar_pixel_xy': [w/2, h*.2], 'hem_pixel_xy': [w/2, h*.8],
                    'reason': 'Collar above hem in the selected RGB.'}
        if self.failure == 'unknown':
            response['image_id'] = 'stale_view'
        elif self.failure == 'sideways':
            response.update(collar_pixel_xy=[w*.2, h/2], hem_pixel_xy=[w*.8, h/2])
        elif self.failure == 'uncertain':
            response.update(status='UNCERTAIN', image_id=None, collar_pixel_xy=None, hem_pixel_xy=None)
        elif self.failure == 'padding':
            response['collar_pixel_xy'] = [0, 0]
        elif self.failure == 'unverified':
            debug.state['views'][-1]['verification'] = 'UNVERIFIED_REPLAY'
        debug.finish('COMPLETED')
        if self.failure == 'tampered':
            Image.new('RGB', (w, h), 'black').save(debug.state['views'][-1]['path'])
        self.called = True
        self.debug, self.selected = debug, selected
        return BackendResult(json.dumps({'result': json.dumps(response)}), '', 0, ('fake-claude',),
                             image_sources=tuple(debug.state['views']))


def canonical_image(tmp_path):
    path = tmp_path / 'camera_A_rgb_upright.png'
    image = Image.new('RGB', (120, 80), 'white')
    image.putpixel((30, 59), (255, 0, 0))
    image.save(path)
    return path


@pytest.mark.parametrize('angle,zoom', [(0, 1), (90, 1), (90, 2), (37, 1)])
def test_selected_pixels_are_exact_and_mapped_through_all_transforms(tmp_path, angle, zoom):
    canonical = canonical_image(tmp_path)
    backend = OrientationBackend(angle=angle, zoom=zoom)
    selection = prepare_molmo_view(backend, canonical, tmp_path / 'orientation', timeout_s=30)
    with Image.open(selection['selected_image']) as image, Image.open(backend.selected['path']) as actual:
        assert image.tobytes() == actual.convert('RGB').tobytes()
        assert pixel_hash(image) == selection['selected_rgb_sha256']
        point = [image.width/2, image.height/2]
    trace = map_molmo_pixel(selection, point)
    assert trace['source_image_id'] == backend.selected['image_id']
    assert len(trace['mapped_pixel_xy_float']) == 2
    assert selection['side_convention'] == 'COLLAR_UP_IMAGE_LEFT_RIGHT'


@pytest.mark.parametrize('failure', ['unread', 'unknown', 'sideways', 'uncertain', 'padding', 'unverified', 'tampered'])
def test_invalid_orientation_stops_handoff_and_keeps_debug(tmp_path, failure):
    output = tmp_path / 'orientation'
    with pytest.raises(ValueError):
        prepare_molmo_view(OrientationBackend(angle=37, failure=failure),
                          canonical_image(tmp_path), output, timeout_s=30)
    report = json.loads((output / 'selection.json').read_text())
    assert report['status'] == 'FAILED_NO_MOLMO'
    assert Path(report['image_debug_directory'], 'image_debug.json').is_file()
    assert not (output / 'molmo_input').exists()


def test_timeout_keeps_selection_failure(tmp_path):
    class FailedBackend:
        def invoke(self, **kwargs):
            raise TimeoutError('Claude timed out')
    with pytest.raises(MolmoOrientationError, match='no automatic budget reset'):
        prepare_molmo_view(FailedBackend(), canonical_image(tmp_path), tmp_path / 'orientation', timeout_s=30)
    assert json.loads((tmp_path / 'orientation' / 'selection.json').read_text())['status'] == 'FAILED_NO_MOLMO'


def test_select_existing_final_view_after_all_six_edits(tmp_path):
    backend = OrientationBackend(extra_edits=5)
    report = prepare_molmo_view(backend, canonical_image(tmp_path), tmp_path / 'orientation', timeout_s=30)
    assert report['status'] == 'READY'
    assert backend.selected['edit_budget']['remaining'] == 0
    assert report['edit_limit'] == 6
    assert report['automatic_retry_allowed'] is False


def test_orientation_failure_cannot_restart_unattended_with_fresh_budget():
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.unattended = True
    pipeline._last_operational_stage = 'molmo'
    calls = []
    def failed_attempt():
        calls.append(1)
        raise MolmoOrientationError('no acceptable image after the edit budget')
    pipeline._run_once = failed_attempt
    with pytest.raises(MolmoOrientationError):
        pipeline.run()
    assert len(calls) == 1
    for stage in ('planning', 'molmo', 'perception', 'supervisor', None):
        assert not pipeline._unattended_error_is_retriable(MolmoOrientationError('invalid selection'), stage)


@pytest.mark.parametrize('step', ['left_sleeve', 'right_sleeve'])
@pytest.mark.parametrize('found', [True, False])
def test_fold_handoff_reads_selected_rgb_without_depth_and_maps_back(tmp_path, monkeypatch, step, found):
    views = tmp_path / 'workspace' / 'perception_views'
    views.mkdir(parents=True)
    Image.new('RGB', (80, 120), 'white').save(views / 'camera_0_A.png')
    iteration = tmp_path / 'iteration_001'
    iteration.mkdir()
    backend = OrientationBackend()
    client = RemoteFoldClient.__new__(RemoteFoldClient)
    client.backend, client.timeout_s = backend, 30
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.client = client
    pipeline.session = SimpleNamespace(run_dir=tmp_path, workspace=tmp_path / 'workspace')
    pipeline.project_root = Path(__file__).resolve().parents[1]
    pipeline.molmo_sleeve_grounding = True
    pipeline.molmo_confidence_threshold = .5
    pipeline.molmo_timeout_s = 30
    pipeline.molmo_python = Path(sys.executable)
    pipeline.molmo_gpu_max_memory_gib = 17
    pipeline.molmo_load_in_8bit = True
    pipeline._debug = lambda *args, **kwargs: None
    def forbidden(*args, **kwargs):
        pytest.fail('no metric lookup, old Molmo axis or static garment reference in RGB handoff')
    pipeline._measure_molmo_frame = forbidden
    monkeypatch.setattr('cloth_agent.molmo_keypoint_pipeline._local_geometry', forbidden)
    monkeypatch.setattr('cloth_agent.molmo_keypoint_pipeline._load_flat_reference', forbidden)
    captured = {}
    def worker(command, **kwargs):
        assert backend.called  # Claude must finish before Molmo starts.
        image_path = Path(command[command.index('--image')+1])
        with Image.open(image_path) as image, Image.open(backend.selected['path']) as expected:
            assert image.size == (80, 120)
            assert image.tobytes() == expected.tobytes()
        assert not list(image_path.parent.glob('*.npy'))
        spec = json.loads(Path(command[command.index('--specs')+1]).read_text())[0]
        assert ('LEFT' if step == 'left_sleeve' else 'RIGHT') in spec['description']
        assert 'collar is above the hem' in spec['description']
        captured['input'] = image_path
        record = {'name': spec['name'], 'status': 'point_returned', 'pixel_xy': [20.25, 30.75],
                  'confidence': .8, 'confidence_definition': CONFIDENCE_DEFINITION,
                  'point_token_probabilities': [.8, .8, .8]}
        if not found:
            record.update(status='not_found', pixel_xy=None, confidence=0., point_token_probabilities=[])
        Path(command[command.index('--output')+1]).write_text(json.dumps({
            'views': [{'label': 'A', 'image_size': [80, 120], 'axis_reference': None, 'records': [record]}]}))
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr('cloth_agent.molmo_keypoint_pipeline._run_worker_streaming', worker)
    (views / 'garment_frame.json').write_text('{}')
    pipeline._prepare_garment_frame(iteration, views / 'camera_0_A.png')
    assert not (views / 'garment_frame.json').exists()
    hint = pipeline._locate_sleeve_with_molmo(step=step, iteration=1, iteration_dir=iteration)
    if found:
        assert hint['status'] == 'MOLMO_POINT_AVAILABLE'
        assert hint['upright_pixel_xy'] == [31, 59]
        assert hint['raw_pixel_xy'] == [59, 88]
    else:
        assert hint['status'] == 'NO_VALID_MOLMO_SLEEVE_POINT'
        assert 'raw_pixel_xy' not in hint
    assert Path(hint['input_image']) == captured['input']
    assert hint['semantic_authority'] == 'Claude'
    assert rgb_evidence([Path(hint['image']), Path(hint['processed_image'])], tmp_path)
    assert (iteration / 'molmo_sleeve_locator' / 'pixel_mapping.json').is_file()
    assert not (views / 'molmo_keypoint_grasp_references.json').exists()
    from cloth_agent.fold_exploration_viser import _markdown_for_iteration, _iter_images
    summary = _markdown_for_iteration(iteration)
    assert 'Claude → Molmo → Claude' in summary
    assert str(captured['input']) in summary
    assert captured['input'] in _iter_images(iteration)


def test_failed_orientation_never_invokes_molmo(tmp_path, monkeypatch):
    views = tmp_path / 'workspace' / 'perception_views'
    views.mkdir(parents=True)
    Image.new('RGB', (80, 120), 'white').save(views / 'camera_0_A.png')
    client = RemoteFoldClient.__new__(RemoteFoldClient)
    client.backend, client.timeout_s = OrientationBackend(failure='uncertain'), 30
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.client, pipeline.session = client, SimpleNamespace(run_dir=tmp_path)
    pipeline.molmo_sleeve_grounding = True
    pipeline._debug = lambda *args, **kwargs: None
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline.run_molmo_keypoint_pipeline',
                        lambda **kwargs: pytest.fail('Molmo must not run on an unverified orientation'))
    with pytest.raises(ValueError, match='orientation'):
        pipeline._locate_sleeve_with_molmo(step='left_sleeve', iteration=1,
                                          iteration_dir=tmp_path / 'iteration_001')


def test_rgb_hints_cannot_be_installed_as_grasp_references(tmp_path):
    with pytest.raises(MolmoKeypointPipelineError, match='install=False'):
        run_molmo_keypoint_pipeline(project_root=tmp_path, perception_dir=tmp_path,
                                   artifact_dir=tmp_path / 'out', direct_keypoints=True,
                                   rgb_only=True, install=True)
    assert not (tmp_path / 'out').exists()
