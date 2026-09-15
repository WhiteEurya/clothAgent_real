import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_agent.fold_frame import build_frame, load_frame, project_pixels, FRAME_RULE
from cloth_agent.auto_exploration import _fold_sleeve_reference_geometry
from cloth_agent.fold_exploration_pipeline import _molmo_sleeve_spec, FoldSupervisor, FoldExplorationPipeline


@pytest.mark.parametrize('angle', [0, 37, 90, 180, 270])
def test_rotated_sleeve_gate_preserves_garment_sides(angle):
    # Rasterize in the garment frame so oblique angles exercise real projections.
    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = np.array([150., 150.])
    transform = lambda p: (np.asarray(p) - center) @ rotation.T + center
    collar, hem, left, right, torso = transform([[150, 70], [150, 230], [70, 120], [230, 120], [150, 140]])
    rgb = Image.new('RGB', (300, 300))
    frame = build_frame(rgb, collar, hem)
    yy, xx = np.indices((300, 300))
    local = (np.stack((xx, yy), axis=-1) - center) @ rotation + center
    x, y = local[..., 0], local[..., 1]
    mask = ((x >= 120) & (x <= 180) & (y >= 70) & (y <= 230)) | ((x >= 60) & (x <= 240) & (y >= 105) & (y <= 150))
    raw_mask = np.rot90(mask, 1)
    def check(point, step):
        u, v = np.rint(point).astype(int)
        return _fold_sleeve_reference_geometry(raw_mask, [v, 299-u], step=step,
                                               garment_frame=frame, require_free_edge=False)
    assert check(left, 'left_sleeve')['requested_side_ok']
    assert check(right, 'right_sleeve')['requested_side_ok']
    with pytest.raises(ValueError, match='side band'):
        check(right, 'left_sleeve')
    with pytest.raises(ValueError, match='side band'):
        check(torso, 'left_sleeve')
    assert project_pixels(left, frame)[0] < 0


def test_frame_rejects_missing_degenerate_and_stale(tmp_path):
    image = Image.new('RGB', (200, 200))
    with pytest.raises(ValueError):
        build_frame(image, [50, 50], [50, 50])
    with pytest.raises(ValueError):
        build_frame(image, [float('nan'), 1], [20, 30])
    with pytest.raises(ValueError):
        build_frame(image, [-1, 1], [20, 30])
    frame = build_frame(image, [50, 50], [150, 50])
    (tmp_path / 'garment_frame.json').write_text(json.dumps(frame))
    assert load_frame(tmp_path, image) == frame
    image.putpixel((0, 0), (1, 2, 3))
    with pytest.raises(ValueError, match='stale'):
        load_frame(tmp_path, image)


def test_supervisor_and_molmo_receive_identical_frame(tmp_path):
    rgb = Image.new('RGB', (200, 200))
    frame = build_frame(rgb, [50, 100], [150, 100])
    path = tmp_path / 'camera_A_rgb_upright.png'
    rgb.save(path)
    (tmp_path / 'garment_frame.json').write_text(json.dumps(frame))
    bundle = FoldSupervisor._write_context_bundle(tmp_path, images=[path], video_evidence=[], history=[], screen={})
    instructions = (tmp_path / bundle['read_order'][0]).read_text()
    assert FRAME_RULE in instructions
    assert json.dumps(frame) in instructions
    spec = _molmo_sleeve_spec('left_sleeve', frame)
    assert json.dumps(frame) in spec.description
    assert 'negative projection' in spec.description
    assert 'compatibility' not in spec.description
    assert 'positive projection' in _molmo_sleeve_spec('right_sleeve', frame).description


@pytest.mark.parametrize('confidence', [.9, .1])
def test_axis_acquisition_uses_current_rgb_and_stops_on_low_confidence(tmp_path, monkeypatch, confidence):
    views = tmp_path / 'workspace' / 'perception_views'
    views.mkdir(parents=True)
    raw = Image.new('RGB', (80, 60), 'white')
    raw.save(views / 'camera_0_A.png')
    np.save(views / 'camera_A_base_xyz_mm.npy', np.zeros((60, 80, 3)))
    np.save(views / 'camera_A_height_above_table_mm.npy', np.zeros((60, 80)))
    output = tmp_path / 'capture'
    output.mkdir()
    rgb_path = output / 'camera_A_rgb_upright.png'
    raw.rotate(-90, expand=True).save(rgb_path)
    def worker(**kwargs):
        assert kwargs['direct_keypoints'] and not kwargs['install']
        assert kwargs['cameras'] == ('A',)
        artifacts = kwargs['artifact_dir']
        artifacts.mkdir()
        records = [dict(name=name, pixel_xy=p, status='point_returned', confidence=confidence)
                   for name, p in [('fold_collar_center', [15, 30]), ('fold_hem_center', [45, 30])]]
        (artifacts / 'molmo_keypoints_raw.json').write_text(json.dumps({'views': [{'label': 'A', 'records': records}]}))
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline.run_molmo_keypoint_pipeline', worker)
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.session = SimpleNamespace(workspace=tmp_path / 'workspace')
    pipeline.project_root = tmp_path
    pipeline._debug = lambda *a, **kw: None
    pipeline.molmo_python = None
    pipeline.molmo_confidence_threshold = .5
    pipeline.molmo_gpu_max_memory_gib = 17
    pipeline.molmo_load_in_8bit = True
    pipeline.molmo_timeout_s = 900
    if confidence < .5:
        with pytest.raises(ValueError, match='axis unavailable'):
            pipeline._prepare_garment_frame(output, rgb_path)
        assert not (views / 'garment_frame.json').exists()
    else:
        pipeline._prepare_garment_frame(output, rgb_path)
        with Image.open(rgb_path) as image:
            frame = load_frame(views, image)
        assert frame['right_unit'] == [0., -1.]
        assert (output / 'camera_A_garment_frame.png').exists()
