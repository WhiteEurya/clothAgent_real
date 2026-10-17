import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cloth_agent.experiments import action_local_frame as al
from cloth_agent.experiments.replay_calibration_migration import migrate_bundle, validate_plane_migration
from tests.test_seed_contact_tcp import setup


def case(tmp_path):
    edge, _, obs, template = setup(tmp_path)
    old = copy.deepcopy(obs.meta['support_height_patch'])
    old.update(reference_world_from_camera_mm=np.eye(4).tolist(), anchor_xy_mm=[80., 80.],
               reference_joints_deg=[0.] * 7, max_joint_delta_deg=.2)
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler('xyz', [.5, -1.3, 1.1], degrees=True).as_matrix()
    delta[:3, 3] = [2., -1., -3.]
    new = copy.deepcopy(old)
    new['reference_world_from_camera_mm'] = delta.tolist()
    a, b, c = old['plane_z_ax_by_c']
    plane = np.linalg.solve(delta.T, [-a, -b, 1., -c])
    new['plane_z_ax_by_c'] = (-plane[[0, 1, 3]] / plane[2]).tolist()
    new['anchor_xy_mm'] = (delta @ [80., 80., 50., 1.])[:2].tolist()
    new['calibration_migration'] = dict(new_world_from_old_world_mm=delta.tolist(), new_surface_measurement=False)
    template['repair_provenance'] = {'revision': 2, 'result_node': 'validated-target'}
    bundle = dict(version='minimal-grounding-host-template-v1', edges={edge['edge_id']: template})
    return edge, obs, bundle, old, new


def test_migration_preserves_repaired_motion_and_leaves_original_frozen(tmp_path):
    edge, obs, bundle, old, new = case(tmp_path)
    untouched = copy.deepcopy(bundle)
    migrated = migrate_bundle(bundle, old, new)
    assert bundle == untouched
    eid = edge['edge_id']
    expected = copy.deepcopy(bundle['edges'][eid])
    expected['contact_reference_sha256'] = al.contact_reference(
        SimpleNamespace(meta={'support_height_patch': new}))[1]
    assert migrated['edges'][eid] == expected
    t = bundle['edges'][eid]
    value = dict(grasp_px=t['seed_grasp_px'], target_px=t['seed_target_px'], confidence=1.)
    obs.meta['support_height_patch'] = old
    original = al.expand(t, value, edge, obs, tmp_path/'before')
    obs.meta['support_height_patch'] = new
    with pytest.raises(ValueError, match='contact-height calibration changed'):
        al.expand(t, value, edge, obs, tmp_path/'rejected')
    current = al.expand(migrated['edges'][eid], value, edge, obs, tmp_path/'after')
    assert original['poses'] == current['poses']
    # Further unverified geometry changes must still be rejected by runtime.
    obs.meta['support_height_patch'] = copy.deepcopy(new)
    obs.meta['support_height_patch']['plane_z_ax_by_c'][2] += 1
    with pytest.raises(ValueError, match='contact-height calibration changed'):
        al.expand(migrated['edges'][eid], value, edge, obs, tmp_path/'changed')


@pytest.mark.parametrize('damage', ['height', 'contact', 'plane', 'anchor', 'pose', 'delta',
                                   'limits', 'new_measurement', 'unconfirmed', 'disabled'])
def test_unproven_or_physical_changes_rejected(tmp_path, damage):
    _, _, bundle, old, new = case(tmp_path)
    if damage == 'height': new['reference_height_mm'] += 1
    elif damage == 'contact': new['contact_reference']['source'] = 'different-measurement'
    elif damage == 'plane': new['plane_z_ax_by_c'][2] += 1
    elif damage == 'anchor': new['anchor_xy_mm'][0] += 1
    elif damage == 'pose': new['reference_world_from_camera_mm'][0][3] += 1
    elif damage == 'delta': new['calibration_migration']['new_world_from_old_world_mm'][0][3] += 1
    elif damage == 'limits': new['max_joint_delta_deg'] += 1
    elif damage == 'new_measurement': new['calibration_migration']['new_surface_measurement'] = True
    elif damage == 'unconfirmed': new['contact_reference']['user_confirmed'] = False
    elif damage == 'disabled': new['enabled'] = False
    with pytest.raises((ValueError, AssertionError)):
        migrate_bundle(bundle, old, new)


@pytest.mark.parametrize('damage', ['wrong_contract', 'legacy_height', 'wrong_height'])
def test_source_template_must_match_recorded_contact(tmp_path, damage):
    edge, _, bundle, old, new = case(tmp_path)
    t = bundle['edges'][edge['edge_id']]
    if damage == 'wrong_contract': t['contact_reference_sha256'] = 'unknown'
    elif damage == 'legacy_height': t['z_policy'] = 'legacy_visual_pa'
    else: t['contact_reference_height_mm'] += 1
    with pytest.raises(ValueError): migrate_bundle(bundle, old, new)


def test_active_xy_correction_is_not_silently_migrated(tmp_path):
    _, _, _, old, new = case(tmp_path)
    for p in (old, new): p['xy_patch'] = dict(enabled=True)
    with pytest.raises(ValueError, match='XY patch'): validate_plane_migration(old, new)
