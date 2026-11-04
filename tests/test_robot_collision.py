from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.config import SafetyError
from cloth_agent.dual_arm.geometry import Capsule, DualArmError
from cloth_agent.robot_collision import RobotCollision


def scene():
    checker = object.__new__(RobotCollision)
    class Model:
        def __init__(self, active):
            self.active = active
            self.config = SimpleNamespace(axis=1)
            self.lower, self.upper = np.radians([-10]), np.radians([10])
        def capsules(self, q):
            x = float(q[0]) if self.active else 0.
            return [Capsule('tool', np.array([x, 0., 0.]), np.array([x, 0., .1]), .1)]
        def capsule_motion_bounds(self, half):
            return {'tool': float(half[0]) if self.active else 0.}
    checker.models = {'left': Model(True), 'right': Model(False)}
    checker.config = SimpleNamespace(obstacles=[], limits={'clearance_mm': 0.},
                                    arms={'left': SimpleNamespace(ip='left')})
    checker.initial = {'left': np.array([-2.]), 'right': np.array([0.])}
    checker.margins = {'left': 0., 'right': 0.}
    checker.active_arm = 'left'
    return checker


def test_collision_between_clear_endpoints_is_rejected():
    c = scene()
    end = {'left': np.array([2.]), 'right': np.array([0.])}
    c._box(c.initial, c.initial)
    c._box(end, end)
    with pytest.raises(DualArmError, match='inter-arm collision'):
        c.validate_joint_path([c.initial, end])


def test_whole_joint_interval_and_static_other_arm():
    c = scene()
    end = {'left': [-1.], 'right': [0.]}
    result = c.validate_joint_path([c.initial, end])
    assert result['status'] == 'JOINT_RANGES_CLEAR'
    assert not result['controller_interpolation_covered']
    assert not result['execution_allowed']
    with pytest.raises(SafetyError, match='stationary'):
        c.validate_joint_path([c.initial, {'left': [-1.], 'right': [1.]}])


@pytest.mark.parametrize('q', [[20.], [float('nan')], []])
def test_limits_and_invalid_feedback(q):
    with pytest.raises((ValueError, SafetyError)):
        scene().validate_joint_path([{'left': q, 'right': [0.]}])


def test_table_tool_sweep_and_budget():
    c = scene()
    c.config.obstacles = [dict(name='table', min_mm=[-1.5, -1, -1], max_mm=[-1, 1, 1])]
    with pytest.raises(DualArmError, match='collision'):
        c.validate_joint_path([c.initial, {'left': [-.5], 'right': [0.]}])
    with pytest.raises(SafetyError, match='budget'):
        scene().validate_joint_path([scene().initial, {'left': [-1.], 'right': [0.]}], max_cells=0)


def test_cartesian_ik_samples_cannot_authorize_controller_motion():
    c = scene()
    with pytest.raises(SafetyError, match='CONTROLLER_PATH_UNCOVERED'):
        c.require_execution(SimpleNamespace(robot_ip='left'), [{'name': 'move'}])
    with pytest.raises(SafetyError, match='CONTROLLER_PATH_UNCOVERED'):
        c.require_execution(SimpleNamespace(robot_ip='left'), [{'name': 'home'}])
