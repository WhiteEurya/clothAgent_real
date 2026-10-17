import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.depth_level import level_support_cloud
from cloth_agent.dual_arm.observation import save_observation

ROOT = Path(__file__).resolve().parents[1]


def patch():
    policy = json.loads((ROOT/'config/arm7_support_height_patch.json').read_text())
    # Existing tests independently exercise the height-only policy.
    policy.pop('xy_patch', None)
    return policy


def xy_trial_patch():
    # Exercise the optional historical XY policy independently of live settings.
    c = patch()
    c['xy_patch'] = dict(enabled=True, mode='world_xy_linear_y',
                        x_bias_mm=12., left_y_mm=-73.10145568847656,
                        right_y_mm=188.21218872070312, left_y_bias_mm=0.,
                        right_y_bias_mm=8., outside_interval='linear_extrapolation')
    return c


def test_xy_trial_endpoints_midpoint_and_combined_affine():
    c = xy_trial_patch()
    xy = c['xy_patch']
    ys = np.linspace(xy['left_y_mm'], xy['right_y_mm'], 3)
    points = np.c_[np.full(3, 620.), ys, np.full(3, 65.)]
    result, matrix = level_support_cloud(points, c, c['reference_joints_deg'], c['reference_world_from_camera_mm'])
    np.testing.assert_allclose(result[:, 0], points[:, 0]-12)
    np.testing.assert_allclose(result[:, 1], ys-[0, 4, 8])
    height_only, _ = level_support_cloud(points, patch(), c['reference_joints_deg'], c['reference_world_from_camera_mm'])
    np.testing.assert_allclose(result[:, 2], height_only[:, 2])
    np.testing.assert_allclose(result, points @ matrix[:3, :3].T + matrix[:3, 3])


def test_xy_trial_preserves_invalid_pixels_and_has_explicit_extrapolation():
    c = xy_trial_patch()
    p = c['xy_patch']
    points = np.array([[620., p['right_y_mm']+(p['right_y_mm']-p['left_y_mm']), 65.], [np.nan]*3])
    result, _ = level_support_cloud(points, c, c['reference_joints_deg'], c['reference_world_from_camera_mm'])
    assert result[0, 1] == pytest.approx(points[0, 1]-16)
    assert np.isnan(result[1]).all()


def test_measured_plane_levels_preserving_anchor_xy_and_cloth_relief():
    c = patch()
    a, b, intercept = c['plane_z_ax_by_c']
    anchor = np.array(c['anchor_xy_mm'])
    xy = np.array([anchor, anchor+[0, 390], anchor+[-200, 100]])
    surface = np.c_[xy, xy @ [a, b]+intercept]
    cloud = np.stack([surface, surface+[0, 0, 12]])
    corrected, _ = level_support_cloud(cloud, c, c['reference_joints_deg'], c['reference_world_from_camera_mm'])
    np.testing.assert_array_equal(corrected[..., :2], cloud[..., :2])
    height = c.get('reference_height_mm', surface[0, 2])
    np.testing.assert_allclose(corrected[0, :, 2], height)
    np.testing.assert_allclose(corrected[1, :, 2]-corrected[0, :, 2], 12)
    np.testing.assert_allclose(corrected[:, 0, :2], cloud[:, 0, :2])
    assert corrected[0, 1, 2] < cloud[0, 1, 2]


@pytest.mark.parametrize('change', ['joints', 'translation', 'rotation'])
def test_patch_rejects_changed_pose_or_calibration(change):
    c = patch()
    q = np.array(c['reference_joints_deg'])
    T = np.array(c['reference_world_from_camera_mm'])
    if change == 'joints':
        q[0] += 1
    elif change == 'translation':
        T[0, 3] += 5
    else:
        theta = np.radians(1)
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        T[:3, :3] = R @ T[:3, :3]
    with pytest.raises(ValueError, match='requires the measured'):
        level_support_cloud(np.zeros((2, 2, 3)), c, q, T)


def test_saved_observation_is_corrected_once_and_keeps_rigid_extrinsic(tmp_path):
    c = patch()
    T = np.array(c['reference_world_from_camera_mm'])
    config = SimpleNamespace(raw={'calibration_id': 'patched', 'camera': {'mount_arm': 'right', 'support_height_patch': c}})
    xyz = np.array([[[600., 230., 64.], [595., -160., 54.]], [[np.nan]*3, [500., 0., 60.]]])
    obs = save_observation(tmp_path/'obs', np.zeros((2, 2, 3), dtype=np.uint8), xyz, config,
                           {'right': c['reference_joints_deg']}, synthetic=True,
                           extra={'world_from_camera_mm': T.tolist()})
    expected, affine = level_support_cloud(xyz, c, c['reference_joints_deg'], T)
    np.testing.assert_allclose(obs.xyz, expected, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(np.load(obs.directory/'uncorrected_world_xyz_mm.npy'), xyz, equal_nan=True)
    np.testing.assert_array_equal(obs.meta['world_from_camera_mm'], T)
    np.testing.assert_allclose(obs.meta['corrected_world_from_camera_affine_mm'], affine @ T)
    assert 'uncorrected_world_xyz_mm.npy' in obs.meta['sha256']


def test_all_camera_b_configs_get_patch_and_reparse_is_idempotent():
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.codex_camera_b import with_arm7_camera
    c = DualConfig.load(ROOT/'config/dual_arm.local.json', root=ROOT, homing_only=True)
    corrected = with_arm7_camera(c)
    assert 'support_height_patch' not in c.raw['camera']
    assert corrected.raw['camera']['support_height_patch']['enabled']
    raw = json.loads(json.dumps(corrected.raw))
    raw['camera'].pop('support_height_patch')
    reparsed = DualConfig.parse(raw, ROOT, homing_only=True)
    assert reparsed.raw['camera']['support_height_patch'] == corrected.raw['camera']['support_height_patch']
    assert reparsed.raw['calibration_id'] == corrected.raw['calibration_id']
    again = DualConfig.parse(reparsed.raw, ROOT, homing_only=True)
    assert again.raw == reparsed.raw


def test_shared_disable_removes_stale_patch_and_changes_identity(tmp_path):
    from cloth_agent.depth_level import bind_support_height_patch
    (tmp_path/'config').mkdir()
    policy = patch()
    file = tmp_path/'config/arm7_support_height_patch.json'
    file.write_text(json.dumps(policy))
    raw = {'calibration_id': 'test', 'camera': {'serial': '233622079809', 'mount_arm': 'right'}}
    enabled = bind_support_height_patch(raw, tmp_path)
    policy['enabled'] = False
    file.write_text(json.dumps(policy))
    disabled = bind_support_height_patch(enabled, tmp_path)
    assert 'support_height_patch' not in disabled['camera']
    assert disabled['calibration_id'] != enabled['calibration_id']
    assert 'support_height_patch' in enabled['camera']


def test_missing_shared_policy_cannot_silently_bypass_patch(tmp_path):
    from cloth_agent.depth_level import bind_support_height_patch
    raw = {'calibration_id': 'test', 'camera': {'serial': '233622079809', 'mount_arm': 'right'}}
    with pytest.raises(FileNotFoundError):
        bind_support_height_patch(raw, tmp_path)


def test_graph_geometry_retains_height_patch_provenance():
    from cloth_agent.experiments.graph_policy import observation_geometry
    metadata = {'support_height_patch': patch(),
                'corrected_world_from_camera_affine_mm': np.eye(4).tolist()}
    assert observation_geometry(SimpleNamespace(meta=metadata)) == metadata


def test_raw_plane_relevel_once_across_workspace(tmp_path):
    c = json.loads((ROOT/'config/arm7_support_height_patch.json').read_text())
    x,y = np.meshgrid(np.linspace(380,780,9), np.linspace(-280,300,9))
    a,b,intercept = c['plane_z_ax_by_c']
    raw = np.stack([x,y,a*x+b*y+intercept],-1)
    raw[3:5,3:5,2] += 12  # raised garment relief must survive
    config = SimpleNamespace(raw={'calibration_id':'raw-relevel', 'camera':{'mount_arm':'right','support_height_patch':c}})
    obs = save_observation(tmp_path/'obs',np.zeros((9,9,3),np.uint8),raw,config,
        {'right':c['reference_joints_deg']},synthetic=True,
        extra={'world_from_camera_mm':c['reference_world_from_camera_mm']})
    expected = np.full((9,9),c['reference_height_mm']);expected[3:5,3:5]+=12
    np.testing.assert_allclose(obs.xyz[:,:,2],expected,atol=1e-5)
    np.testing.assert_allclose(np.load(obs.directory/'uncorrected_world_xyz_mm.npy'),raw,atol=1e-4)
    np.testing.assert_allclose(obs.xyz[:,:,:2],raw[:,:,:2],atol=1e-4)
    # Any accidental second leveling would leave a large spatial gradient.
    twice,_=level_support_cloud(obs.xyz,c,c['reference_joints_deg'],c['reference_world_from_camera_mm'])
    assert np.max(np.abs(twice[:,:,2]-expected))>10
