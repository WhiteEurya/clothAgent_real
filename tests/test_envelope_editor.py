import copy
import json

import pytest

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.envelope_editor import envelope_breakdown, save_draft
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.safety import padded_capsules
from .test_dual_arm_runtime import scene


@pytest.mark.parametrize('mode', ['servo_stream', 'controller_sequential'])
def test_breakdown_matches_runtime_padding(scene, mode):
    raw = copy.deepcopy(scene.config.raw)
    raw['safety']['status'] = 'measured'
    for k, row in raw['safety']['arms'].items():
        row.update(base_error_mm=10, geometry_error_mm=10,
                   max_joint_speed_deg_s=raw['limits']['joint_speed_deg_s'],
                   stop_excursion_deg=[1.25] * raw['arms'][k]['axis'])
    config = DualConfig.parse(raw, scene.config.root)
    config.execution_mode = mode
    models = {k: ArmModel(a) for k,a in config.arms.items()}
    # Use real URDF joints rather than the fixture's Cartesian fake joint values.
    joints = {k: [0] * a.axis for k,a in config.arms.items()}
    breakdown = envelope_breakdown(raw, models, execution_mode=mode)
    padded = padded_capsules(config, models, joints)
    for k, caps in padded.items():
        for cap in caps:
            row = breakdown[k][cap.name]
            assert row['total_mm'] == pytest.approx(cap.radius)
            assert row['total_mm'] == pytest.approx(row['nominal_mm'] + sum(row['increments_mm'].values()))
            if mode == 'controller_sequential':
                assert all(row['increments_mm'][name] == 0 for name in
                           ('feedback','watchdog','stop_latency','tick','tick_lateness','dispatch'))
                assert row['increments_mm']['base'] == 10


def test_save_is_unvalidated_atomic_draft_with_backup(scene, tmp_path):
    raw = copy.deepcopy(scene.config.raw)
    original = copy.deepcopy(raw)
    path = tmp_path / 'dual_arm.envelope_draft.json'
    save_draft(raw, {}, path)
    first = path.read_text()
    saved = json.loads(first)
    assert saved['safety']['status'] == 'estimated'
    assert saved['safety']['validation_id'] is None
    assert saved['collision_geometry_verified'] is False
    assert saved['servo_commissioned'] is False
    assert raw == original
    save_draft(raw, {}, path)
    backups = list(tmp_path.glob('*.bak'))
    assert len(backups) == 1
    assert backups[0].read_text() == first
    with pytest.raises(ValueError):
        save_draft(raw, {}, tmp_path / 'dual_arm.local.json')
    assert not (tmp_path / 'dual_arm.local.json').exists()
