"""Bridge the new path planner to the installed native Home executor.

The installed collision envelopes and controller stopping bounds remain the
authority. The generic PathPlanner accepts this runtime scene adapter instead
of substituting the demonstration FCL scene's attachment geometry. Smoothed
timing is a reference; native commands receive independent full-box proofs.
"""
from __future__ import annotations

import time
import math
from types import SimpleNamespace

import numpy as np

from ..geometry import DualArmError, finite
from ..home_path import check_program_path, corridor, deviation, world_pose
from ..planning import Motion, Phase, validate_sample
from ..sequential_home import certify_segment, prepare_segments
from ..stop_sweep import path_proof_budget, search_proof_budget
from .path import PathPlanner
from .trajectory import parameterize, polynomial_peak


class HomeEndpointError(DualArmError):
    def __init__(self, label, report):
        self.diagnostics = {'endpoint': label, **report}
        detail = report['reason']
        if report.get('nominal_geometry_clear') and 'inter-arm collision' in detail:
            detail = (
                f"nominal geometry is clear, but configured uncertainty envelopes overlap; "
                f"nominal gap={report['nominal_gap_mm']:.3f} mm, "
                f"combined position margin={report['combined_position_margin_mm']:.3f} mm. "
                "No separation motion has been certified from this state. " + detail)
        super().__init__(f'Home {label} rejected: {detail}')


def endpoint_diagnostics(config, models, joints, report):
    """Compare nominal geometry for reporting only; never authorizes a move."""
    from copy import deepcopy
    from ..separation import minimum_gap
    from ..stop_sweep import position_error_mm
    result = dict(report)
    nominal = deepcopy(config)
    nominal.limits['tracking_error_mm'] = 0.
    nominal.limits['tracking_error_deg'] = 0.
    nominal.limits['clearance_mm'] = 0.
    for row in nominal.raw['safety']['arms'].values():
        row['base_error_mm'] = row['geometry_error_mm'] = 0.
    try:
        validate_sample(nominal, models, joints, holding=False)
        result['nominal_geometry_clear'] = True
    except DualArmError as exc:
        result.update(nominal_geometry_clear=False, nominal_reason=str(exc))
    result['nominal_gap_mm'] = minimum_gap(models, joints)
    result['combined_position_margin_mm'] = (
        sum(position_error_mm(config, k) for k in config.arms) + config.limits['clearance_mm'])
    return result


def unpack(q):
    q = np.degrees(finite(q, (13,), 'Home joint radians'))
    return {'left': q[:6], 'right': q[6:]}


def pack(joints):
    return np.radians(np.r_[joints['left'], joints['right']])


class HomeChecker:
    def __init__(self, config, models, bounds):
        self.config, self.models, self.bounds = config, models, bounds

    def check(self, left, right):
        try:
            joints = unpack(np.r_[left, right])
            validate_sample(self.config, self.models, joints, holding=False)
            for key, arm in self.config.arms.items():
                point = world_pose(self.models[key], arm, joints[key])[:3]
                bound = self.bounds[key]
                if point[2] > bound['height'] or deviation(point, bound) > bound['radius']:
                    raise DualArmError('Home state exceeds path corridor')
            return {'safe': True}
        except DualArmError as exc:
            return {'safe': False, 'reason': str(exc)}


class HomeValidator:
    def __init__(self, config, models, bounds):
        self.config, self.models, self.bounds = config, models, bounds

    def certify_box(self, start, end, *, deadline=None, max_nodes=None):
        a, b = unpack(start), unpack(end)
        remaining = 5. if deadline is None else deadline-time.monotonic()
        if remaining <= 0:
            raise DualArmError('Home path planning budget exhausted')
        with search_proof_budget(min(5., remaining)):
            certify_segment(a, b, self.config, self.models,
                            max_nodes=max_nodes,
                            path_bounds=self.bounds)
        return {'nodes': 1}

    def certify(self, start, end, *, deadline=None):
        """Certify a joint line as multiple possible native commands.

        Splitting a line does not certify its entire endpoint box. Polynomial
        enclosures must call certify_box separately, as must actual dispatch.
        """
        deadline = min(time.monotonic()+5., deadline or float('inf'))
        pending = [(np.asarray(start), np.asarray(end), 0)]
        nodes = 0
        while pending:
            a,b,depth = pending.pop()
            if time.monotonic() >= deadline:
                raise DualArmError('Home path planning budget exhausted')
            nodes += 1
            width = float(np.max(np.abs(np.degrees(b-a))))
            try:
                self.certify_box(a,b,deadline=deadline,
                                 max_nodes=64 if width>.25 else None)
            except DualArmError:
                if depth >= 12 or width <= .25 or time.monotonic() >= deadline:
                    raise
                midpoint = (a+b)/2
                pending.extend([(midpoint,b,depth+1),(a,midpoint,depth+1)])
        return {'nodes':nodes}

    def certify_polynomial(self, coefficients):
        """Cover a curved polynomial with exact-extremum boxes, never chords."""
        deadline = time.monotonic()+5.
        pending = [(0.,1.,0)]
        while pending:
            lo,hi,depth = pending.pop()
            local = np.zeros_like(coefficients)
            for power in range(6):
                for j in range(power+1):
                    local[j] += coefficients[power]*math.comb(power,j)*lo**(power-j)*(hi-lo)**j
            center = np.polynomial.polynomial.polyval(.5,local)
            residual = local.copy(); residual[0] -= center
            half = polynomial_peak(residual)
            try:
                self.certify_box(center-half,center+half,deadline=deadline,
                                 max_nodes=64 if np.max(np.degrees(2*half))>.25 else None)
            except DualArmError:
                if depth>=12 or time.monotonic()>=deadline:
                    raise
                mid=(lo+hi)/2
                pending.extend([(mid,hi,depth+1),(lo,mid,depth+1)])


class HomePlanner:
    def __init__(self, initial, config, models):
        if config.execution_mode != 'controller_sequential':
            raise DualArmError('new Home bridge requires native position execution')
        if tuple(config.arms) != ('left', 'right') or [a.axis for a in config.arms.values()] != [6, 7]:
            raise DualArmError('Home bridge requires left xArm6 and right xArm7')
        self.initial, self.config, self.models = initial, config, models
        self.bounds = corridor(initial, config, models)
        checker = HomeChecker(config, models, self.bounds)
        validator = HomeValidator(config, models, self.bounds)
        adapted = {}
        for key, model in models.items():
            if not hasattr(model, 'lower') and config.raw['safety']['status'] != 'synthetic':
                raise DualArmError('real Home planner requires URDF joint limits')
            axis = config.arms[key].axis
            adapted[key] = SimpleNamespace(
                lower=getattr(model, 'lower', np.full(axis, -20.)),
                upper=getattr(model, 'upper', np.full(axis, 20.)))
        self.paths = PathPlanner(SimpleNamespace(models=adapted), checker, validator)
        self.paths.edge_timeout_s = .5

    def _motion(self, path, name):
        edges = [dict(phase=name, start_q_rad=a.tolist(), end_q_rad=b.tolist())
                 for a, b in zip(path, path[1:])]
        timed = parameterize(edges,
            np.full(13, np.deg2rad(self.config.limits['joint_speed_deg_s'])),
            np.full(13, np.deg2rad(self.config.limits['joint_accel_deg_s2'])),
            period_s=1/self.config.limits['rate_hz'])
        q = np.column_stack([timed['trajectory_a']['positions_rad'],
                             timed['trajectory_b']['positions_rad']])
        # Certify the exact polynomial coordinate extrema, not only its samples.
        for segment in timed['segments']:
            coefficients = np.asarray(segment['coefficients_rad'])
            self.paths.validator.certify_polynomial(coefficients)
        joints = {'left': np.degrees(q[:, :6]), 'right': np.degrees(q[:, 6:])}
        poses = {k: np.asarray([world_pose(self.models[k], arm, row) for row in joints[k]])
                 for k, arm in self.config.arms.items()}
        # Reference timing also obeys Cartesian speed/acceleration from local config.
        times = np.asarray(timed['trajectory_a']['times_s'])
        scale = 1.
        for values in poses.values():
            dt = np.diff(times)
            velocity = np.diff(values[:, :3], axis=0)/dt[:, None]
            acceleration = np.diff(velocity, axis=0)/((dt[1:]+dt[:-1])/2)[:, None]
            scale = max(scale, np.linalg.norm(velocity, axis=1).max()/self.config.limits['cartesian_speed_mm_s'])
            if len(acceleration):
                scale = max(scale, np.sqrt(np.linalg.norm(acceleration, axis=1).max()/self.config.limits['cartesian_accel_mm_s2']))
        times *= scale
        return Motion(Phase(name, 'move', {k:p[-1] for k,p in poses.items()}), times, joints, poses)

    def finish(self, current, prefix, attempts):
        target = {k:a.home_joints for k,a in self.config.arms.items()}
        for order in ('left_first', 'right_first'):
            label = f'home_ompl_{len(prefix)}separation_{order}'
            try:
                path = self.paths.joint_path(pack(current), pack(target), order=order, timeout_s=20.)
                # Keep each arm's contiguous path in one Motion. Do not turn
                # every 20 ms sample into a blocking controller instruction.
                groups = []
                for a,b in zip(path, path[1:]):
                    moving = tuple(k for k,sl in (('left',slice(0,6)),('right',slice(6,13)))
                                   if np.max(np.abs(b[sl]-a[sl])) > 1e-12)
                    if len(moving) > 1:
                        raise DualArmError('Home path attempted simultaneous arm motion')
                    if not groups or groups[-1][0] != moving:
                        groups.append((moving, [a]))
                    groups[-1][1].append(b)
                program = list(prefix)+[self._motion(rows, f'{label}_{i}')
                                       for i,(_,rows) in enumerate(groups)]
                program.append(Phase('home_open_grippers', 'open'))
                if sum(len(m.times) for m in program[:-1]) > self.config.limits['max_plan_samples']:
                    raise DualArmError('whole Home sample budget exceeded')
                check_program_path(program, self.config, self.models)
                prepare_segments(program, self.config, self.models)
                attempts.append({'strategy':label, 'status':'accepted'})
                return program
            except DualArmError as exc:
                attempts.append({'strategy':label, 'status':'rejected', 'reason':str(exc)})
        return None

    def plan(self):
        from ..separation import minimum_gap, separation_steps
        from ..stop_sweep import position_error_mm
        attempts = []
        target = {k:a.home_joints for k,a in self.config.arms.items()}
        for label, state in (('initial state', self.initial), ('Home target', target)):
            report = self.paths.checker.check(*[pack(state)[sl] for sl in (slice(0,6),slice(6,13))])
            if not report['safe']:
                raise HomeEndpointError(label, endpoint_diagnostics(self.config, self.models, state, report))
        close = 30.+sum(position_error_mm(self.config, k) for k in self.config.arms)
        at_home = all(np.max(np.abs(self.initial[k]-target[k])) <= 1e-9 for k in target)
        separate_first = not at_home and minimum_gap(self.models, self.initial) < close
        if separate_first:
            for current, prefix in separation_steps(self.initial, self.config, self.models, attempts):
                if minimum_gap(self.models, current) < close:
                    continue
                program = self.finish(current, prefix, attempts)
                if program is not None:
                    return program, attempts
        program = self.finish(self.initial, [], attempts)
        # The close-distance threshold is only a search ordering heuristic.
        # A pose just outside it can still need separation to make multi-axis
        # native stopping boxes feasible, so never skip this fallback.
        if program is None and not at_home and not separate_first:
            for current, prefix in separation_steps(self.initial, self.config, self.models, attempts):
                program = self.finish(current, prefix, attempts)
                if program is not None:
                    return program, attempts
        if program is None:
            raise DualArmError(f'No native-certified OMPL Home route: {attempts}')
        return program, attempts
