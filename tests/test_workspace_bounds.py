import numpy as np
import pytest

from cloth_agent.dual_arm.config import ArmConfig, workspace_bound
from cloth_agent.dual_arm.geometry import DualArmError


def test_shared_y_z_bounds_allow_unrestricted_x_and_upper_z():
    bounds = {
        'min_mm': workspace_bound([None, -250.578537, -0.542602], 'min', lower=True),
        'max_mm': workspace_bound([None, 246.447205, None], 'max', lower=False),
    }
    from types import SimpleNamespace
    arm = SimpleNamespace(workspace=bounds, arm_id='test')
    for x in (-1e6, 1e6):
        ArmConfig.validate_point(arm, [x, 0, 1e6])
    for point in ([0, -251, 0], [0, 247, 0], [0, 0, -1], [np.nan, 0, 0]):
        with pytest.raises(DualArmError):
            ArmConfig.validate_point(arm, point)


@pytest.mark.parametrize('value', [None, [0, 1], [0, np.inf, 1], [False, 0, 1], ['0', 0, 1]])
def test_missing_or_invalid_bounds_are_still_rejected(value):
    with pytest.raises(DualArmError):
        workspace_bound(value, 'min', lower=True)
