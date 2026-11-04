import json
from dataclasses import asdict

import pytest

from cloth_agent.config import ConfigError, SafetyError, WorkspaceBounds
from cloth_agent.experiments.full_grasp_adapter import DEFAULT_CONFIG


def test_partial_bounds_require_explicit_policy():
    values = {'x_max': 577.506409, 'y_max': 127.638359, 'z_min': -20}
    with pytest.raises(ConfigError):
        WorkspaceBounds.from_mapping(values).validate(300, 0, 50, require_complete=True)
    bounds = WorkspaceBounds.from_mapping({**values, 'boundary_policy': 'explicit_one_sided'})
    bounds.validate(-10000, -10000, 10000, require_complete=True)
    for point in ((578, 0, 50), (300, 128, 50), (300, 0, -21)):
        with pytest.raises(SafetyError):
            bounds.validate(*point, y_extension_mm=1000)
    with pytest.raises(SafetyError):
        bounds.validate(577, 0, 50, margin_mm=1)
    assert WorkspaceBounds.from_mapping(asdict(bounds)) == bounds


def test_exact_left_mirror_is_enforced():
    workspace = json.loads(DEFAULT_CONFIG.read_text())['arms']['left']['workspace']
    bounds = WorkspaceBounds.from_mapping({
        'z_min': workspace['min_mm'][2], 'boundary_policy': workspace['boundary_policy'],
        'halfspaces_mm': workspace['halfspaces_mm'],
    })
    point = [571.4520790965614, -18.5743778817968, 338.51456773900867]
    normal = workspace['halfspaces_mm'][0][:3]
    bounds.validate(*point, require_complete=True)
    bounds.validate(*(value - direction for value, direction in zip(point, normal)))
    with pytest.raises(SafetyError, match='halfspace'):
        bounds.validate(*(value + direction for value, direction in zip(point, normal)), y_extension_mm=1000)
    with pytest.raises(SafetyError):
        bounds.validate(*point, margin_mm=1)


@pytest.mark.parametrize('extra', [
    {'boundary_policy': 'typo'},
    {'halfspaces_mm': [[0, 0, 0, 5]]},
    {'halfspaces_mm': [[1, 0, 0, float('nan')]]},
    {'halfspaces_mm': [[1, 0, 0]]},
])
def test_malformed_policy_rejected(extra):
    with pytest.raises(ConfigError):
        WorkspaceBounds.from_mapping({'z_min': 0, **extra})


def test_opt_in_does_not_accept_empty_workspace():
    for values in ({'z_min': 0}, {'x_max': 500}):
        bounds = WorkspaceBounds.from_mapping({'boundary_policy': 'explicit_one_sided', **values})
        assert not bounds.complete
