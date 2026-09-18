import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.check_camera_calibration import (
    audit_map, checkerboard, deproject, fixed_point_report, load_capture, main, measure_point,
)


def scene(root, translation=(0, 0, 0), offset=0):
    root.mkdir()
    Image.new('RGB', (80, 60), 'gray').save(root / 'rgb.png')
    depth = np.full((60, 80), .5)
    np.save(root / 'depth.npy', depth)
    k = np.array([[100., 0, 40], [0, 100, 30], [0, 0, 1]])
    t = np.eye(4)
    t[:3, 3] = translation
    yy, xx = np.indices(depth.shape)
    xyz = deproject(k, np.stack([xx, yy], axis=-1), depth) + np.array(translation)*1000
    xyz[:, :, 2] += offset
    np.save(root / 'camera_A_base_xyz_mm.npy', xyz.astype(np.float32))
    (root / 'camera_A_coordinate_guide.json').write_text(json.dumps({'samples': [
        {'reference_id': 'R081', 'pixel_xy': [40, 30], 'base_xyz_mm': xyz[30, 40].tolist()}]}))
    path = root / 'result.json'
    path.write_text(json.dumps({'views': [{'label': 'A', 'image': 'rgb.png', 'depth_m': 'depth.npy',
        'intrinsics': k.tolist(), 'X_base_camera': t.tolist(), 'base_z_offset_mm': offset}]}))
    return path


def test_exact_depth_reconstruction_includes_pipeline_z_offset(tmp_path):
    c = load_capture(scene(tmp_path / 'capture', translation=(.3, .2, .1), offset=7))
    measured = measure_point(c, [40, 30])
    assert measured['base_xyz_mm'] == pytest.approx([300, 200, 600])
    assert measured['pipeline_xyz_mm'] == pytest.approx([300, 200, 607])
    assert measured['map_error_mm'] < 1e-5
    assert audit_map(c, .05)['status'] == 'CONSISTENT'
    c['xyz'][30, 40, 0] += 10
    assert audit_map(c, .05)['status'] == 'MISMATCH'


def test_missing_center_depth_is_not_replaced_by_neighbor(tmp_path):
    c = load_capture(scene(tmp_path / 'capture'))
    c['depth'][30, 40] = np.nan
    result = measure_point(c, [40, 30])
    assert result['status'] == 'INVALID_DEPTH'
    assert result['valid_patch_fraction'] > .9
    with pytest.raises(ValueError, match='outside'):
        measure_point(c, [100, 30])


def test_bad_rotation_is_rejected(tmp_path):
    path = scene(tmp_path / 'capture')
    data = json.loads(path.read_text())
    data['views'][0]['X_base_camera'][0][0] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='determinant'):
        load_capture(path)


def observation(c, pixel, known=None):
    row = {'point_id': 'fixed_corner_1', 'result_json': str(c['path']),
           'X_base_camera': c['transform'].tolist(), 'measurement': measure_point(c, pixel)}
    if known is not None:
        row['known_base_xyz_mm'] = known
    return row


def test_pose_diversity_and_independent_absolute_truth(tmp_path):
    a = load_capture(scene(tmp_path / 'a'))
    b = load_capture(scene(tmp_path / 'b', translation=(.05, 0, 0)))
    rows = [observation(a, [40, 30], [0, 0, 500]), observation(b, [30, 30], [0, 0, 500])]
    report = fixed_point_report(rows, 5)
    assert report['absolute_accuracy']['status'] == 'WITHIN_TOLERANCE_AT_TESTED_POINTS'
    assert report['fixed_point_drift'][0]['max_pairwise_drift_mm'] < 1e-6
    # Wrong extrinsic can remain internally consistent but fail independent truth.
    wrong = load_capture(scene(tmp_path / 'wrong', translation=(.07, 0, 0)))
    assert audit_map(wrong, .05)['status'] == 'CONSISTENT'
    report = fixed_point_report([rows[0], observation(wrong, [30, 30], [0, 0, 500])], 5)
    assert report['fixed_point_drift'][0]['status'] == 'DRIFT_DETECTED'
    assert report['absolute_accuracy']['status'] == 'OUT_OF_TOLERANCE'
    repeated = fixed_point_report([observation(a, [40, 30])] * 3, 5)
    assert repeated['absolute_accuracy']['status'] == 'NOT_VERIFIED'
    assert repeated['fixed_point_drift'][0]['status'] == 'INSUFFICIENT_POSE_VARIATION'


def test_cli_iteration_selects_exact_before_capture(tmp_path):
    path = scene(tmp_path / 'capture')
    before = tmp_path / 'iteration_002' / 'before_raw'
    before.mkdir(parents=True)
    (before / 'camera_A_upright_mapping.json').write_text(json.dumps({
        'coordinate_guide': str(path.parent / 'camera_A_coordinate_guide.json')}))
    output = tmp_path / 'reports'
    assert main(['--iteration', str(before.parent), '--reference', 'R081', '--output-dir', str(output)]) == 0
    report = json.loads(next(output.glob('*/report.json')).read_text())
    assert report['absolute_accuracy']['status'] == 'NOT_VERIFIED'
    assert report['captures'][0]['points'][0]['reference_error_mm'] < .001
    assert list(output.glob('*/*raw_rgb_points.png'))


def test_cli_known_point_and_invalid_input_receipts(tmp_path):
    scene(tmp_path / 'capture', translation=(.02, 0, 0))
    observations = tmp_path / 'observations.json'
    observations.write_text(json.dumps([{'point_id': 'P1', 'result_json': 'capture/result.json',
        'pixel_xy': [40, 30], 'known_base_xyz_mm': [0, 0, 500]}]))
    assert main(['--observations', str(observations), '--output-dir', str(tmp_path / 'out')]) == 2
    assert main(['--perception', str(tmp_path / 'missing'), '--output-dir', str(tmp_path / 'bad')]) == 1
    report = json.loads(next((tmp_path / 'bad').glob('*/report.json')).read_text())
    assert report['status'] == 'INPUT_ERROR'


def test_checkerboard_held_out_and_metric_scale(tmp_path):
    pytest.importorskip('cv2')
    path = scene(tmp_path / 'board')
    c = load_capture(path)
    cols, rows, cell, margin = 7, 5, 40, 40
    image = np.full(((rows+1)*cell+2*margin, (cols+1)*cell+2*margin), 180, np.uint8)
    for y in range(rows+1):
        for x in range(cols+1):
            image[margin+y*cell:margin+(y+1)*cell, margin+x*cell:margin+(x+1)*cell] = 255 if (x+y)%2 else 0
    c['rgb'] = Image.fromarray(image).convert('RGB')
    c['depth'] = np.full(image.shape, .5)
    c['xyz'] = None
    c['k'] = np.array([[1000., 0, image.shape[1]/2], [0, 1000, image.shape[0]/2], [0, 0, 1]])
    result = checkerboard(c, cols, rows, 20, 1)
    assert result['status'] == 'WITHIN_TOLERANCE'
    assert result['depth_edge_length_error_mm']['max'] < .2
    wrong_scale = checkerboard(c, cols, rows, 30, 1)
    assert wrong_scale['depth_edge_length_error_mm']['median'] > 9
    assert wrong_scale['depth_scale_status'] == 'OUT_OF_TOLERANCE'
