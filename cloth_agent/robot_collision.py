"""Geometry/path checks for RobotAPI; no controller, executor or admission flags.

An IK preview is NOT the controller's Cartesian trajectory.  The existing
set_position backend has no exported joint path, so dual-arm Cartesian execution
remains blocked even if every supplied joint interval is collision free.
"""
from dataclasses import replace

import numpy as np

from .config import SafetyError
from .dual_arm.config import DualConfig
from .dual_arm.geometry import DualArmError, check_collision, finite
from .dual_arm.kinematics import ArmModel


class RobotCollision:
    def __init__(self, config_path, initial, active_arm, *, root):
        self.config = DualConfig.load_geometry(config_path, root=root)
        self.models = {k: ArmModel(a) for k, a in self.config.arms.items()}
        if active_arm not in self.models:
            raise SafetyError('unknown active arm')
        self.active_arm = active_arm
        self.initial = self.joints(initial)
        margins = self.config.raw['geometry_margin_mm']
        self.margins = {k: float(margins[k]) for k in self.models}
        if any(not np.isfinite(v) or v < 0 for v in self.margins.values()):
            raise SafetyError('invalid geometry margin')

    def joints(self, values):
        if set(values) != set(self.models):
            raise SafetyError('both actual arm joint states are required')
        result = {}
        for key, model in self.models.items():
            q = finite(values[key], (model.config.axis,), 'joint state')
            if np.any(np.radians(q) < model.lower) or np.any(np.radians(q) > model.upper):
                raise SafetyError(f'{key}: joint limits exceeded')
            result[key] = q.copy()
        return result

    def _box(self, lower, upper):
        capsules = {}
        for key, model in self.models.items():
            center, half = (lower[key] + upper[key]) / 2, (upper[key] - lower[key]) / 2
            reach = model.capsule_motion_bounds(half)
            capsules[key] = [replace(c, radius=c.radius + self.margins[key] + reach[c.name])
                             for c in model.capsules(center)]
        check_collision(capsules, self.config.obstacles, self.config.limits['clearance_mm'])

    def validate_joint_path(self, path, *, max_cells=8192):
        """Cover entire independent joint ranges, not just sampled configurations.

        This proves the supplied ranges geometrically. It does not claim an
        unobserved set_position trajectory follows them or create dispatch rights.
        """
        states = [self.joints(row) for row in path]
        if not states:
            raise SafetyError('empty joint path')
        if any(not np.allclose(states[0][k], self.initial[k], atol=1e-9, rtol=0) for k in self.models):
            raise SafetyError('path does not start at the supplied robot state')
        other = 'right' if self.active_arm == 'left' else 'left'
        if any(not np.allclose(row[other], self.initial[other], atol=1e-9, rtol=0) for row in states):
            raise SafetyError('the other arm must remain stationary for the entire path')
        visited = 0
        self._box(states[0], states[0])
        for start, end in zip(states, states[1:]):
            pending = [({k: np.minimum(start[k], end[k]) for k in start},
                        {k: np.maximum(start[k], end[k]) for k in start})]
            while pending:
                lo, hi = pending.pop()
                visited += 1
                if visited > max_cells:
                    raise SafetyError('complete collision path check exceeded cell budget')
                try:
                    self._box(lo, hi)
                    continue
                except DualArmError:
                    center = {k: (lo[k] + hi[k]) / 2 for k in lo}
                    self._box(center, center)  # A colliding point cannot be subdivided away.
                key = self.active_arm
                axis = int(np.argmax(hi[key] - lo[key]))
                if hi[key][axis] - lo[key][axis] < 1e-4:
                    raise SafetyError('complete collision path check unresolved at minimum interval')
                upper_half = {k: v.copy() for k, v in hi.items()}
                lower_half = {k: v.copy() for k, v in lo.items()}
                upper_half[key][axis] = lower_half[key][axis] = center[key][axis]
                pending.extend(((lo, upper_half), (lower_half, hi)))
        return dict(status='JOINT_RANGES_CLEAR', checked_cells=visited,
                    controller_interpolation_covered=False, execution_allowed=False)

    def inspect_actions(self, config, actions):
        if config.robot_ip != self.config.arms[self.active_arm].ip:
            raise SafetyError('RobotAPI arm differs from the collision scene')
        self.validate_joint_path([self.initial])
        kinds = sorted({row['name'] for row in actions if row['name'] in {'move', 'home', 'shake', 'shake_open'}})
        return dict(status='PATH_UNAVAILABLE' if kinds else 'STATIC_GEOMETRY_CLEAR',
                    initial_collision_check='PASS', motion_kinds=kinds,
                    controller_interpolation_covered=False, execution_allowed=False,
                    reason='set_position/shake do not expose the executed joint branch; home needs a full joint sweep from live state, not a reset to Home in the IK preview')

    def require_execution(self, config, actions):
        report = self.inspect_actions(config, actions)
        if report['motion_kinds']:
            raise SafetyError('CONTROLLER_PATH_UNCOVERED: ' + report['reason'])
        raise SafetyError('LIVE_OTHER_ARM_STATE_REQUIRED: saved geometry is not live feedback')
