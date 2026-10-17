"""Offline migration of contact-TCP templates after a verified hand-eye update.

Only a rigid re-expression of the same measured support plane is supported.
Physical contact evidence and all template motion parameters must stay unchanged.
Existing template/graph artifacts and runtime contract checks remain intact.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .action_local_frame import contact_reference, expand
from .graph_minimal_grounding import MinimalGrounder, write
from .graph_policy import file_hash, frozen, load_frozen


def validate_plane_migration(old, new):
    """Prove geometry-only migration, not a changed physical contact baseline."""
    old_height, old_hash = contact_reference(SimpleNamespace(meta={'support_height_patch': old}))
    new_height, new_hash = contact_reference(SimpleNamespace(meta={'support_height_patch': new}))
    changing = {'plane_z_ax_by_c', 'anchor_xy_mm', 'reference_world_from_camera_mm',
                'calibration_migration'}
    if {k: v for k, v in old.items() if k not in changing} != {
            k: v for k, v in new.items() if k not in changing}:
        raise ValueError('Only optical geometry may change; contact evidence and limits must match')
    if old.get('xy_patch', {}).get('enabled'):
        raise ValueError('An active empirical XY patch requires separate revalidation')
    from cloth_agent.dual_arm.geometry import transform
    before = transform(old['reference_world_from_camera_mm'], 'old camera pose')
    after = transform(new['reference_world_from_camera_mm'], 'new camera pose')
    delta = after @ np.linalg.inv(before)
    migration = new['calibration_migration']
    if migration['new_surface_measurement'] is not False:
        raise ValueError('A new surface measurement requires revalidation')
    np.testing.assert_allclose(delta, migration['new_world_from_old_world_mm'], atol=1e-9, rtol=0)
    a, b, c = np.asarray(old['plane_z_ax_by_c'], float)
    plane = np.linalg.solve(delta.T, [-a, -b, 1., -c])
    if abs(plane[2]) < .9:
        raise ValueError('Migrated support plane is too steep')
    expected = -plane[[0, 1, 3]] / plane[2]
    np.testing.assert_allclose(new['plane_z_ax_by_c'], expected, atol=1e-9, rtol=0)
    anchor = np.asarray(old['anchor_xy_mm'], float)
    moved_anchor = delta @ np.r_[anchor, anchor @ [a, b] + c, 1.]
    np.testing.assert_allclose(new['anchor_xy_mm'], moved_anchor[:2], atol=1e-9, rtol=0)
    if not np.isfinite(expected).all() or old_height != new_height:
        raise ValueError('Invalid plane or changed contact height')
    return dict(old_contract_sha256=old_hash, new_contract_sha256=new_hash,
                contact_reference_height_mm=new_height, new_world_from_old_world_mm=delta.tolist())


def migrate_bundle(bundle, old, new):
    proof = validate_plane_migration(old, new)
    result = copy.deepcopy(bundle)
    if bundle['version'] != 'minimal-grounding-host-template-v1' or not bundle['edges']:
        raise ValueError('Expected a nonempty minimal Host bundle')
    for eid, template in result['edges'].items():
        if (template.get('version') != 'action-local-host-template-v2'
                or template.get('z_policy') != 'seed_contact_tcp'):
            raise ValueError(f'{eid}: only contact-TCP templates support this migration')
        if (template['contact_reference_sha256'] != proof['old_contract_sha256']
                or template['contact_reference_height_mm'] != proof['contact_reference_height_mm']):
            raise ValueError(f'{eid}: template does not match the source contact contract')
        template['contact_reference_sha256'] = proof['new_contract_sha256']
    result['calibration_migration'] = proof
    return result


def migrate(graph_path, templates_path, activation_path, output, *, root, config_path):
    """Validate active FK/plane evidence, write a new frozen bundle, and roundtrip it."""
    from cloth_agent.codex_camera_b import with_arm7_camera
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.dual_arm.kinematics import ArmModel
    from .graph_policy_replay import validate_policy

    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh migration directory; do not overwrite frozen evidence')
    graph_path, templates_path, activation_path = map(
        lambda p: Path(p).resolve(), (graph_path, templates_path, activation_path))
    graph = load_frozen(graph_path)
    validate_policy(graph)
    grounder = MinimalGrounder(templates_path)
    edges = {e['edge_id']: e for e in graph['edges']}
    for eid in grounder.bundle['edges']:
        grounder.template(edges[eid])
    activation = json.loads(activation_path.read_text())
    for name, expected in activation['active_files_sha256'].items():
        if file_hash(root / name) != expected:
            raise ValueError('Active calibration differs from activation record: ' + name)
    config = with_arm7_camera(DualConfig.load(config_path, root=root, homing_only=True))
    if config.raw['calibration_id'] != activation['new_calibration_id']:
        raise ValueError('Loaded runtime calibration differs from activation record')
    if file_hash(config.arms['right'].urdf) != activation['runtime_urdf_sha256']:
        raise ValueError('Runtime kinematics changed')
    new = config.raw['camera']['support_height_patch']
    previous_path = root / new['calibration_migration']['previous_policy']
    if str(previous_path.relative_to(root)) not in activation['backups']:
        raise ValueError('Previous policy is not the activation backup')
    old = json.loads(previous_path.read_text())
    # Verify both optical frames against the same runtime FK and recorded hand-eyes.
    model = ArmModel(config.arms['right'])
    mount = config.arms['right'].world_from_base @ model.frame_at(new['reference_joints_deg'], 'link_eef')
    for patch, key in ((old, 'old_handeye_mm'), (new, 'new_handeye_mm')):
        np.testing.assert_allclose(patch['reference_world_from_camera_mm'],
                                   mount @ np.asarray(activation[key]), atol=1e-9, rtol=0)
    np.testing.assert_allclose(config.raw['camera']['mount_from_camera_mm'],
                               activation['new_handeye_mm'], atol=1e-12, rtol=0)
    migrated = migrate_bundle(grounder.bundle, old, new)
    for template in migrated['edges'].values():
        if template['camera_serial'] != config.raw['camera']['serial']:
            raise ValueError('Template camera differs from active camera')
        if template['image_size'] != [config.raw['camera']['width'], config.raw['camera']['height']]:
            raise ValueError('Template image resolution differs from active camera')
    sources = {str(p): file_hash(p) for p in (graph_path, templates_path, activation_path, previous_path)}
    migrated['calibration_migration'].update(source_files_sha256=sources,
        calibration_id=config.raw['calibration_id'], runtime_urdf_sha256=activation['runtime_urdf_sha256'],
        active_files_sha256=activation['active_files_sha256'],
        semantics='Only the verified optical-plane contract changes; executed profiles and repair provenance are preserved.')
    output.mkdir(parents=True)
    checks = []
    for eid, template in grounder.bundle['edges'].items():
        value = dict(grasp_px=template['seed_grasp_px'], target_px=template['seed_target_px'], confidence=1.)
        anchors = template['seed_anchor_world_mm']
        def sample(pixel, spread):
            if pixel == value['grasp_px']: return np.array(anchors[0])
            if pixel == value['target_px']: return np.array(anchors[1])
            raise ValueError('Unexpected anchor')
        w, h = template['image_size']
        # Controlled equal-anchor comparison isolates migration from new camera XY.
        obs = SimpleNamespace(sample=sample, xyz=np.empty((h, w, 0)),
            meta={'camera_serial': template['camera_serial'], 'support_height_patch': old})
        before = expand(template, value, edges[eid], obs, output/eid/'before')
        obs.meta['support_height_patch'] = new
        after = expand(migrated['edges'][eid], value, edges[eid], obs, output/eid/'after')
        if before['poses'] != after['poses']:
            raise ValueError(eid + ': migration changed motion at identical anchors')
        checks.append(dict(edge_id=eid, identical_anchor_motion_unchanged=True,
                           repair_preserved=template.get('repair_provenance') == migrated['edges'][eid].get('repair_provenance')))
    for name, expected in sources.items():
        if file_hash(name) != expected: raise ValueError('Migration input changed: ' + name)
    bundle_path = output/'host_templates.json'
    frozen(bundle_path, migrated)
    verified = MinimalGrounder(bundle_path)
    for eid in migrated['edges']: verified.template(edges[eid])
    report = dict(status='MIGRATED_OFFLINE_VALIDATED', graph=str(graph_path),
        templates=str(bundle_path), edges=checks, proof=migrated['calibration_migration'],
        robot_calls=0, model_calls=0,
        limitation='Fresh depth, physical alignment and native trajectory checks remain required; this is not a physical trial.')
    write(output/'migration.json', report)
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('graph', 'templates', 'activation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--config', type=Path, default=root/'config/dual_arm.local.json')
    args = parser.parse_args()
    result = migrate(args.graph, args.templates, args.activation, args.output, root=root, config_path=args.config)
    print(json.dumps({k: result[k] for k in ('status', 'templates', 'edges', 'robot_calls')}, indent=2))


if __name__ == '__main__':
    main()
