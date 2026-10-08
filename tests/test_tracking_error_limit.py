import copy

import pytest

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene


def test_zero_cartesian_tracking_error_is_accepted(scene):
    raw = copy.deepcopy(scene.config.raw)
    raw['limits']['tracking_error_mm'] = 0
    config = DualConfig.parse(raw, scene.config.root)
    assert config.limits['tracking_error_mm'] == 0


@pytest.mark.parametrize('value', [-0.01, 15.01, float('nan'), float('inf'), False])
def test_invalid_cartesian_tracking_error_still_rejected(scene, value):
    raw = copy.deepcopy(scene.config.raw)
    raw['limits']['tracking_error_mm'] = value
    with pytest.raises(DualArmError, match='tracking_error_mm'):
        DualConfig.parse(raw, scene.config.root)


def test_zero_joint_tracking_error_is_accepted(scene):
    raw = copy.deepcopy(scene.config.raw)
    raw['limits']['tracking_error_deg'] = 0
    config = DualConfig.parse(raw, scene.config.root)
    assert config.limits['tracking_error_deg'] == 0


@pytest.mark.parametrize('value', [-0.01, 5.01, float('nan'), float('inf'), False])
def test_invalid_joint_tracking_error_still_rejected(scene, value):
    raw = copy.deepcopy(scene.config.raw)
    raw['limits']['tracking_error_deg'] = value
    with pytest.raises(DualArmError, match='tracking_error_deg'):
        DualConfig.parse(raw, scene.config.root)


def test_zero_additional_clearance_is_accepted(scene):
    raw = copy.deepcopy(scene.config.raw)
    raw['limits']['clearance_mm'] = 0
    assert DualConfig.parse(raw, scene.config.root).limits['clearance_mm'] == 0


def test_zero_geometry_error_does_not_disable_positive_stopping_bound(scene):
    from cloth_agent.dual_arm.safety import validate_safety
    raw = copy.deepcopy(scene.config.raw['safety'])
    raw.update(status='measured', validation_id='test',
               open_tools_and_attachments_verified=True, controller_watchdog_verified=True,
               joint_segment_tracking_verified=True)
    for k, row in raw['arms'].items():
        row.update(base_error_mm=10, geometry_error_mm=0,
                   stop_excursion_deg=[1.25]*scene.config.arms[k].axis)
    validate_safety(raw,scene.config.arms,scene.config.limits,real=True)
    raw['arms']['left']['stop_excursion_deg']=[0]*scene.config.arms['left'].axis
    with pytest.raises(DualArmError,match='stopping excursions'):
        validate_safety(raw,scene.config.arms,scene.config.limits,real=True)
