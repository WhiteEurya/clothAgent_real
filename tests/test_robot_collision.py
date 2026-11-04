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
    for name in ('move', 'home'):
        report = c.inspect_actions(SimpleNamespace(robot_ip='left'), [{'name': name}])
        assert not report['execution_allowed']


def test_cartesian_refinement_avoids_joint_chord_dipping_into_table():
    from dataclasses import replace
    from cloth_agent.config import WorkspaceBounds
    from tests.test_loop import robot_config
    c = object.__new__(RobotCollision)
    class Model:
        def __init__(self, active):
            self.active = active
            self.config = SimpleNamespace(axis=2)
            self.lower, self.upper = np.radians([-3., -3.]), np.radians([3., 3.])
        def forward(self, q):
            xyz = [q[0], 0, q[1] + q[0]**2] if self.active else [10, 0, 10]
            return np.array([*xyz, 180, 0, 0.])
        def inverse(self, pose, seed):
            return np.array([pose[0], pose[2] - pose[0]**2])
        def tcp_motion_bound(self, half):
            return float(7 * abs(half[0]) + abs(half[1])) if self.active else 0.
        def capsule_motion_bounds(self, half):
            return {'tool': self.tcp_motion_bound(half)}
        def capsules(self, q):
            p = self.forward(q)[:3]
            return [Capsule('tool', p, p, .05)]
    c.models = {'left': Model(True), 'right': Model(False)}
    c.active_arm = 'left'
    c.initial = {'left': np.array([-1., -.4]), 'right': np.zeros(2)}
    c.margins = {'left': 0., 'right': 0.}
    c.config = SimpleNamespace(arms={'left': SimpleNamespace(ip='left')},
        obstacles=[dict(name='table', min_mm=[-100, -100, -100], max_mm=[100, 100, 0])],
        limits={'clearance_mm': 0., 'stationary_tolerance_deg': .01})
    config = replace(robot_config(), robot_ip='left', collision_active_arm='left',
        init_pose_mm_deg=(-1, 0, .6, 180, 0, 0),
        boundaries=WorkspaceBounds(x_min=-20, x_max=20, y_min=-20, y_max=20, z_min=0, z_max=20))
    end = {'left': np.array([1., -.4]), 'right': np.zeros(2)}
    with pytest.raises(DualArmError, match='table'):
        c.validate_joint_path([c.initial, end])
    plan = c.plan_action(config, 'move', dict(x=1., y=0., z=.6, yaw=0.))
    assert len(plan['path']) > 2
    assert plan['status'] == 'JOINT_RANGES_CLEAR'
    np.testing.assert_allclose(plan['path'][-1]['left'], end['left'])
    for row in plan['path']:
        assert c.models['left'].forward(row['left'])[2] == pytest.approx(.6)
