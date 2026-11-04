"""Static-other-arm geometry and joint paths for the existing RobotAPI backend."""
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

    def _box(self, lower, upper, robot_config=None):
        capsules = {}
        nominal = {}
        for key, model in self.models.items():
            center, half = (lower[key] + upper[key]) / 2, (upper[key] - lower[key]) / 2
            reach = model.capsule_motion_bounds(half)
            if hasattr(model, 'local_capsule_motion_bounds'):
                local = model.local_capsule_motion_bounds(center, half)
                reach = {name: min(value, local[name]) for name, value in reach.items()}
            nominal[key] = {c.name: c for c in model.capsules(center)}
            capsules[key] = [replace(c, radius=c.radius + self.margins[key] + reach[c.name])
                             for c in nominal[key].values()]
        def box_separated(key, shape, box, margin):
            model = self.models[key]
            if not hasattr(model, 'projection_motion_interval'):
                return False
            center, half = (lower[key] + upper[key]) / 2, (upper[key] - lower[key]) / 2
            cap = nominal[key][shape.name]
            radius = cap.radius + self.margins[key] + margin
            for axis, direction in enumerate(np.eye(3)):
                low, high = model.projection_motion_interval(center, half, cap.name,
                    np.array([cap.start, cap.end]), direction)
                if low - radius > box['max_mm'][axis] or high + radius < box['min_mm'][axis]:
                    return True
            return False
        check_collision(capsules, self.config.obstacles, self.config.limits['clearance_mm'],
                        box_separated=box_separated)
        if robot_config is not None:
            key = self.active_arm
            center = (lower[key] + upper[key]) / 2
            half = (upper[key] - lower[key]) / 2
            pose = self.models[key].forward(center)
            radius = self.models[key].tcp_motion_bound(half)
            # Bound the entire TCP sweep against the existing selected planes.
            # No yaw expansion is applied to the current one-sided workspace.
            robot_config.boundaries.validate(*pose[:3],
                margin_mm=robot_config.workspace_margin_mm + radius,
                z_lower_margin_mm=robot_config.lower_z_margin_mm + radius,
                require_complete=True)

    def validate_joint_path(self, path, *, max_cells=8192, robot_config=None, tolerance_deg=0.):
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
        if not np.isfinite(tolerance_deg) or tolerance_deg < 0:
            raise SafetyError('invalid joint tolerance')
        def check(lo, hi):
            low = {k: q - tolerance_deg for k, q in lo.items()}
            high = {k: q + tolerance_deg for k, q in hi.items()}
            self.joints(low); self.joints(high)
            self._box(low, high, robot_config)
        check(states[0], states[0])
        for start, end in zip(states, states[1:]):
            pending = [({k: np.minimum(start[k], end[k]) for k in start},
                        {k: np.maximum(start[k], end[k]) for k in start})]
            while pending:
                lo, hi = pending.pop()
                visited += 1
                if visited > max_cells:
                    raise SafetyError('complete collision path check exceeded cell budget')
                try:
                    check(lo, hi)
                    continue
                except (DualArmError, SafetyError):
                    center = {k: (lo[k] + hi[k]) / 2 for k in lo}
                    check(center, center)  # A colliding point cannot be subdivided away.
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
        return dict(status='JOINT_PLAN_NOT_RUN' if kinds else 'STATIC_GEOMETRY_CLEAR',
                    initial_collision_check='PASS', motion_kinds=kinds,
                    controller_interpolation_covered=False, execution_allowed=False,
                    reason='Static inspection only; plan_action must solve and check complete joint ranges before dispatch')

    def plan_action(self, config, name, args, *, initial=None):
        """PyRoki-seeded Cartesian samples, dispatched later as joint targets.

        Each joint segment is checked as a complete independent-joint box.
        This covers different joint timing, not just the line between samples.
        """
        from copy import copy
        if (config.robot_ip != self.config.arms[self.active_arm].ip
                or config.collision_active_arm != self.active_arm):
            raise SafetyError('RobotAPI arm differs from the collision scene')
        checker = copy(self)
        checker.initial = self.joints(self.initial if initial is None else initial)
        key = self.active_arm
        model = self.models[key]
        tolerance = self.config.limits['stationary_tolerance_deg']
        states = [{k: v.copy() for k, v in checker.initial.items()}]
        if name == 'move':
            target = np.array([args['x'], args['y'], args['z'], config.orientation_roll_deg,
                               config.orientation_pitch_deg, config.command_yaw_deg(args['yaw'])])
            start = model.forward(states[0][key])
            delta = target - start
            delta[3:] = (delta[3:] + 180) % 360 - 180
            count = max(1, int(np.ceil(np.linalg.norm(delta[:3]) / 10)),
                        int(np.ceil(np.max(np.abs(delta[3:])) / 5)))
            def append_pose(pose, depth=0):
                config.validate_workspace_pose(*pose[:3],
                    relative_yaw_deg=config.relative_yaw_from_absolute_deg(pose[5]))
                q = model.inverse(pose, states[-1][key])
                row = {k: v.copy() for k, v in states[-1].items()}
                row[key] = q
                row = checker.joints(row)
                segment = copy(checker); segment.initial = states[-1]
                try:
                    segment.validate_joint_path([states[-1], row], robot_config=config, tolerance_deg=tolerance)
                except (DualArmError, SafetyError):
                    # A joint chord can dip below a straight Cartesian path.
                    # Re-solve actual Cartesian midpoints, not just interpolate
                    # more points on the same colliding joint chord.
                    endpoint = copy(checker); endpoint.initial = row
                    endpoint.validate_joint_path([row], robot_config=config, tolerance_deg=tolerance)
                    if depth >= 8:
                        raise
                    previous = model.forward(states[-1][key])
                    change = pose - previous
                    change[3:] = (change[3:] + 180) % 360 - 180
                    if np.max(np.abs(change)) < .01:
                        raise
                    append_pose(previous + change / 2, depth + 1)
                    append_pose(pose, depth + 1)
                    return
                states.append(row)
            for i in range(1, count + 1):
                append_pose(start + delta * i / count)
        elif name in {'home', 'perception_position'}:
            target = config.init_joints_deg if name == 'home' else config.perception_joints_deg
            row = {k: v.copy() for k, v in states[-1].items()}
            row[key] = target
            goal = checker.joints(row)
            start = states[0][key]
            delta = goal[key] - start
            count = max(1, int(np.ceil(np.max(np.abs(delta)) / 5.)))
            # These are actual stop-at-waypoint commands, not just samples
            # used to approve one large controller move. The existing backend
            # dispatches every segment with radius=-1, wait=True and checks
            # arrival. Validate the full joint box of each segment below.
            for i in range(1, count + 1):
                row = {k: v.copy() for k, v in goal.items()}
                row[key] = goal[key].copy() if i == count else start + delta * i / count
                states.append(row)
        elif name not in {'open_gripper', 'close_gripper'}:
            raise SafetyError(f'{name}: no checked joint path implementation')
        report = checker.validate_joint_path(states, robot_config=config, tolerance_deg=tolerance)
        report.update(command='set_servo_angle', radius=-1, path=[{k: q.tolist() for k, q in s.items()} for s in states],
                      joint_tolerance_deg=tolerance)
        # These are offline plans. Live identity/state and command monitoring
        # still belong to XArmBackend; this report is not execution permission.
        return report

    def require_execution(self, config, actions):
        current = self.initial
        plans = []
        for index, row in enumerate(actions):
            try:
                plan = self.plan_action(config, row['name'], row.get('args', {}), initial=current)
            except (ValueError, RuntimeError) as exc:
                raise SafetyError(f'action {index + 1} ({row["name"]}): {exc}') from exc
            current = plan['path'][-1]
            plans.append(plan)
        return plans
