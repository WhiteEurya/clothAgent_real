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
        assert c.models['left'].forward(row['left'])[2] > 0.05
    assert c.models['left'].forward(plan['path'][-1]['left'])[2] == pytest.approx(.6)


def test_compact_clear_path_to_one_motion():
    c=scene()
    states=[{'left':np.array([x]),'right':np.array([0.])} for x in [-2.,-1.8,-1.5,-1.]]
    compact=c.compact_joint_path(states,None,0.)
    assert len(compact)==2
    c.validate_joint_path(compact)
    np.testing.assert_array_equal(compact[-1]['left'],states[-1]['left'])


def test_compaction_never_uses_rejected_shortcut(monkeypatch):
    c=scene()
    states=[{'left':np.array([x]),'right':np.array([0.])} for x in [-2.,-1.8,-1.5,-1.]]
    def validate(self,low,high,config=None):
        if abs(high['left'][0]-low['left'][0])>.51:
            raise SafetyError('shortcut crosses forbidden joint box')
    monkeypatch.setattr(RobotCollision,'_box',validate)
    compact=c.compact_joint_path(states,None,0.)
    assert len(compact)>2
    for a,b in zip(compact,compact[1:]):assert abs(b['left'][0]-a['left'][0])<=.51


def test_refinement_emits_real_waypoints_instead_of_joint_product():
    c = scene()
    for m in c.models.values():
        m.config.axis = 2
        m.lower, m.upper = np.radians([-10., -10.]), np.radians([10., 10.])
    start = {'left': np.array([-2., -2.]), 'right': np.zeros(2)}
    end = {'left': np.array([2., 2.]), 'right': np.zeros(2)}
    c.initial = start
    checked = []
    def box(low, high, config=None):
        # The synchronous diagonal is clear; off-diagonal joint combinations
        # can collide. Small command envelopes must be certified individually.
        if max(high['left'][0]-low['left'][1], high['left'][1]-low['left'][0]) > .6:
            raise SafetyError('off-diagonal collision')
        checked.append((low, high))
    c._box = box
    with pytest.raises(SafetyError):
        c.validate_joint_path([start, end], max_cells=64)
    path = c.refine_commanded_path([start, end], None, .01)
    assert len(path) > 2
    np.testing.assert_allclose(path[-1]['left'], end['left'])
    for a, b in zip(path, path[1:]):
        box({k: np.minimum(a[k],b[k])-.01 for k in a},
            {k: np.maximum(a[k],b[k])+.01 for k in a})
        np.testing.assert_array_equal(b['right'], start['right'])


def test_refinement_cannot_hide_a_real_midpath_collision():
    c = scene()
    with pytest.raises(DualArmError, match='inter-arm collision'):
        c.refine_commanded_path([c.initial, {'left': np.array([2.]), 'right': np.array([0.])}], None, 0.)
