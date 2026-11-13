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
        # IK must reserve the same tracking envelope checked on every segment.
        # Keep physical FK/collision limits unchanged; tighten only IK search.
        for model in self.models.values():
            model.ik_joint_margin_deg = float(self.config.limits['stationary_tolerance_deg']) + 1e-4
        if active_arm not in self.models:
            raise SafetyError('unknown active arm')
        self.active_arm = active_arm
        self.initial = self.joints(initial)
        margins = self.config.raw['geometry_margin_mm']
        self.margins = {k: float(margins[k]) for k in self.models}
        if any(not np.isfinite(v) or v < 0 for v in self.margins.values()):
            raise SafetyError('invalid geometry margin')
        from .dual_arm.self_collision import SelfCollisionChecker
        self.self_collision = {k: SelfCollisionChecker(m, self.margins[k], self.config.limits['clearance_mm'])
                               for k, m in self.models.items()}
        for k, model in self.models.items():
            model.ik_collision_check = lambda q, key=k: self.self_collision[key].check_box(
                q, np.full_like(q, self.config.limits['stationary_tolerance_deg']))

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
        for key, checker in getattr(self, 'self_collision', {}).items():
            checker.check_box((lower[key]+upper[key])/2, (upper[key]-lower[key])/2)
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

    def refine_commanded_path(self, states, config, tolerance, *, max_depth=12):
        """Certify small commanded segments, not a Cartesian product of a whole leg.

        Every subdivision is retained as an actual controller waypoint. Each
        segment still covers independent joint timing plus tracking tolerance.
        This is not permission to execute the original unsplit command.
        """
        states = [self.joints(row) for row in states]
        if not states:
            raise SafetyError('empty commanded path')
        other = 'right' if self.active_arm == 'left' else 'left'
        if any(not np.allclose(row[other], states[0][other], atol=1e-9, rtol=0) for row in states):
            raise SafetyError('the other arm must remain stationary')
        def check(a, b):
            low = {k: np.minimum(a[k], b[k]) - tolerance for k in a}
            high = {k: np.maximum(a[k], b[k]) + tolerance for k in a}
            self.joints(low); self.joints(high)
            self._box(low, high, config)
        check(states[0], states[0])
        result = [states[0]]
        def append(a, b, depth):
            try:
                check(a, b)
            except (DualArmError, SafetyError) as exc:
                check(b, b)
                middle = {k: (a[k] + b[k]) / 2 for k in a}
                check(middle, middle)
                if depth >= max_depth or np.max(np.abs(b[self.active_arm]-a[self.active_arm])) < 1e-3:
                    raise SafetyError('commanded segment cannot be cleared within path refinement limit') from exc
                append(a, middle, depth+1)
                append(middle, b, depth+1)
            else:
                result.append(b)
        for a, b in zip(states, states[1:]):
            append(a, b, 0)
        return result

    def compact_joint_path(self, states, config, tolerance):
        """Remove intermediate stops only after checking the larger swept joint box.

        Failed shortcuts keep the original route. No blending or unchecked
        controller Cartesian interpolation is introduced.
        """
        from copy import copy
        attempts = 0
        def span(first, last):
            nonlocal attempts
            if last <= first + 1:
                return [states[first], states[last]]
            if attempts < 64:
                attempts += 1
                checker = copy(self)
                checker.initial = states[first]
                try:
                    low = {k: np.minimum(states[first][k], states[last][k])-tolerance for k in states[first]}
                    high = {k: np.maximum(states[first][k], states[last][k])+tolerance for k in states[first]}
                    checker.joints(low); checker.joints(high)
                    checker._box(low, high, config)
                    return [states[first], states[last]]
                except (DualArmError, SafetyError):
                    pass
            middle = (first + last) // 2
            return span(first, middle)[:-1] + span(middle, last)
        if len(states) < 3:
            return states
        return span(0, len(states)-1)

    def plan_action(self, config, name, args, *, initial=None):
        """PyRoki-seeded Cartesian samples, dispatched later as joint targets.

        Refined segments become controller waypoints; each small swept box
        covers independent joint timing and the existing tracking tolerance.
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
        route = 'pyroki_cartesian_waypoints' if name == 'move' else 'stationary'
        route_failures = []
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
                    refined = segment.refine_commanded_path([states[-1], row], config, tolerance)
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
                states.extend(refined[1:])
            for i in range(1, count + 1):
                append_pose(start + delta * i / count)
        elif name in {'home', 'perception_position'}:
            target = config.init_joints_deg if name == 'home' else config.perception_joints_deg
            goal = {k: v.copy() for k, v in states[0].items()}
            goal[key] = np.asarray(target, dtype=float)
            goal = checker.joints(goal)
            start_pose = model.forward(states[0][key])
            goal_pose = model.forward(goal[key])
            failures = []
            # Use the same PyRoki Cartesian path solver as ordinary moves.
            # Lifted alternatives are only for return motion, never a grasp leg.
            for lift in (0., 25., 50.):
                candidate = [states[0]]
                poses = [goal_pose]
                if lift:
                    a, b = start_pose.copy(), goal_pose.copy()
                    a[2] = b[2] = max(a[2], b[2]) + lift
                    poses = [a, b, goal_pose]
                try:
                    for pose in poses:
                        part = copy(checker)
                        part.initial = candidate[-1]
                        moved = part.plan_action(config, 'move', dict(
                            x=pose[0], y=pose[1], z=pose[2],
                            yaw=config.relative_yaw_from_absolute_deg(pose[5])), initial=candidate[-1])
                        candidate.extend(part.joints(row) for row in moved['path'][1:])
                    # IK may find another redundancy branch. Reach the exact
                    # configured Home joints only with a separately checked leg.
                    final = checker.refine_commanded_path([candidate[-1], goal], config, tolerance)
                    candidate.extend(final[1:])
                    states = candidate
                    route = 'pyroki_cartesian_return' if not lift else f'pyroki_lifted_return_{lift:g}mm'
                    break
                except (DualArmError, SafetyError) as exc:
                    failures.append(str(exc))
            else:
                # A joint-space return can be feasible when Cartesian IK is not.
                # Its subdivisions are real queue targets, fully checked as well.
                try:
                    states = checker.refine_commanded_path([states[0], goal], config, tolerance)
                    route = 'checked_joint_return'
                except (DualArmError, SafetyError) as exc:
                    raise SafetyError('no checked return path: '+str(exc)+'; Cartesian attempts: '+repr(failures)) from exc
            route_failures = failures
        elif name not in {'open_gripper', 'close_gripper'}:
            raise SafetyError(f'{name}: no checked joint path implementation')
        original_waypoints = len(states)
        states = checker.compact_joint_path(states, config, tolerance)
        states = checker.refine_commanded_path(states, config, tolerance)
        report = dict(status='JOINT_RANGES_CLEAR', checked_segments=max(0, len(states)-1),
                      controller_interpolation_covered=False, execution_allowed=False,
                      validation='per_command_swept_joint_bounds', actual_subdivision_waypoints=True)
        report.update(original_waypoints=original_waypoints, removed_stops=max(0, original_waypoints-len(states)),
                      route=route, rejected_return_routes=route_failures)
        report.update(command='set_servo_angle', intermediate_radius=0, final_radius=-1,
                      queue_mode='continuous_zero_radius_with_exact_final_stop', path=[{k: q.tolist() for k, q in s.items()} for s in states],
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
                error = SafetyError(f'action {index + 1} ({row["name"]}): {exc}')
                if hasattr(exc, 'ik_diagnostics'):
                    error.planning_failure = dict(kind='IK_REJECTED', action_index=index+1,
                        action=row, active_arm=self.active_arm, diagnostics=exc.ik_diagnostics,
                        physical_execution=False, stage='complete_pre_execution_path_planning')
                raise error from exc
            current = plan['path'][-1]
            plans.append(plan)
        return plans
