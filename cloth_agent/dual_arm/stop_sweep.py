"""Adaptive coverage of independent joint stopping ranges for native commands.

Every accepted cell contains its entire joint box. Subdivision splits ONE joint
at a time, so asynchronous braking and off-diagonal configurations are retained.
No finite sample grid is treated as a collision certificate.
"""
from __future__ import annotations

import numpy as np
import time
from contextlib import contextmanager
from contextvars import ContextVar

from .geometry import Capsule, DualArmError, check_collision, finite
from .mesh_narrowphase import check_cell_collision


_search_deadline = ContextVar('collision_search_deadline', default=None)


@contextmanager
def search_proof_budget(seconds=10.):
    """Bound exploratory candidate cost; expiration rejects, never accepts."""
    deadline=time.monotonic()+seconds
    outer=_search_deadline.get()
    token=_search_deadline.set(min(deadline,outer) if outer is not None else deadline)
    try:
        yield
    finally:
        _search_deadline.reset(token)


def position_error_mm(config, key):
    if config.raw['safety']['status'] != 'measured':
        return 0.0
    row = config.raw['safety']['arms'][key]
    return row['base_error_mm'] + row['geometry_error_mm'] + config.limits['tracking_error_mm']


def cell_capsules(config, models, center, half):
    """Local cell enclosure; angular padding shrinks with each subdivision."""
    result = {}
    for key, model in models.items():
        residual = (model.local_capsule_motion_bounds(center[key],half[key])
                    if hasattr(model,'local_capsule_motion_bounds') else model.capsule_motion_bounds(half[key]))
        error = position_error_mm(config, key)
        result[key] = [Capsule(c.name, c.start, c.end, c.radius+error+residual[c.name])
                       for c in model.capsules(center[key])]
    return result


def certify_joint_box(config, models, lower, upper, *, tool_contact=False, max_nodes=None):
    """Certify all configurations in a Cartesian product of joint intervals."""
    low, high = {}, {}
    for key, arm in config.arms.items():
        low[key] = finite(lower[key], (arm.axis,), 'joint box lower')
        high[key] = finite(upper[key], (arm.axis,), 'joint box upper')
        if np.any(low[key] > high[key]):
            raise DualArmError('joint box lower exceeds upper')
    pending = [(low,high)]
    nodes, accepted = 0, 0
    budget = config.raw['safety']['max_sweep_nodes']
    if max_nodes is not None:
        budget = min(budget, max_nodes)
    while pending:
        deadline=_search_deadline.get()
        if deadline is not None and time.monotonic()>=deadline:
            raise DualArmError('candidate collision proof time budget exhausted')
        lo, hi = pending.pop()
        nodes += 1
        if nodes > budget:
            raise DualArmError('stopping sweep proof exhausted its node budget')
        center = {k:(lo[k]+hi[k])/2 for k in lo}
        half = {k:(hi[k]-lo[k])/2 for k in lo}
        enclosed = cell_capsules(config,models,center,half)
        try:
            check_cell_collision(config,models,center,enclosed,half_ranges=half,tool_contact=tool_contact)
        except DualArmError as coarse_error:
            # This exact configuration is IN the box: a real witness rejects it.
            zero = {k:np.zeros_like(v) for k,v in half.items()}
            try:
                check_cell_collision(config,models,center,cell_capsules(config,models,center,zero),
                                     tool_contact=tool_contact)
            except DualArmError as collision:
                raise DualArmError(f'stopping sweep contains collision: {collision}') from collision
            # Split a single joint, covering both children in full. Prefer the
            # joint contributing the largest conservative Cartesian uncertainty.
            candidates = []
            conflict = str(coarse_error)
            implicated = {k: [c.name for c in enclosed[k] if f'{k}/{c.name},' in conflict + ',']
                          for k in models}
            for k,model in models.items():
                for j,width in enumerate(half[k]):
                    if width <= 1e-6:
                        continue
                    delta=np.zeros_like(half[k]);delta[j]=width
                    bounds = model.capsule_motion_bounds(delta)
                    relevant = implicated[k]
                    score=max((bounds[name] for name in relevant),default=0.) if any(implicated.values()) else max(bounds.values(),default=0.)
                    # Mesh vertices can rotate about an otherwise stationary
                    # capsule axis; include their radial reach in the priority.
                    if relevant:
                        score += max(c.radius for c in enclosed[k] if c.name in relevant)*np.radians(width)
                    candidates.append((score,k,j))
            if not candidates:
                raise DualArmError(f'stopping sweep clearance cannot be certified: {coarse_error}') from coarse_error
            _,key,joint = max(candidates)
            first_hi={k:v.copy() for k,v in hi.items()}
            second_lo={k:v.copy() for k,v in lo.items()}
            first_hi[key][joint]=center[key][joint]
            second_lo[key][joint]=center[key][joint]
            pending.extend([(second_lo,hi),(lo,first_hi)])
        else:
            accepted += 1
    return {'nodes':nodes,'accepted_cells':accepted}


def native_joint_ranges(config, start, end):
    """Bound a native move and independent absolute braking excursions.

    Stored stop limits are unsigned: cover BOTH sides of each moving joint's
    command range rather than assuming the brake cannot recoil. Uncommanded
    joints retain tracking tolerance, not fictitious commanded motion.
    """
    lower,upper={},{}
    measured = config.raw['safety']['status']=='measured'
    tracking = config.limits['tracking_error_deg'] if measured else 0.0
    for k,arm in config.arms.items():
        a=finite(start[k],(arm.axis,),'segment start')
        b=finite(end[k],(arm.axis,),'segment end')
        delta=b-a
        stop=(np.asarray(config.raw['safety']['arms'][k]['stop_excursion_deg'])
              if measured else np.zeros(arm.axis))
        # Every uncommanded axis may drift within the arrival tolerance,
        # including stationary axes belonging to the currently moving arm.
        idle = config.limits['stationary_tolerance_deg'] if measured else 0.0
        excursion=np.where(np.abs(delta)>1e-9,stop,idle)
        lower[k]=np.minimum(a,b)-tracking-excursion
        upper[k]=np.maximum(a,b)+tracking+excursion
    return lower,upper


def validate_native_sweep(config, models, start, end, *, tool_contact=False, max_nodes=None):
    lower,upper=native_joint_ranges(config,start,end)
    return certify_joint_box(config,models,lower,upper,tool_contact=tool_contact,max_nodes=max_nodes)


def path_proof_budget(config, start, end):
    """Spend the full proof budget on small boxes near the stopping floor.

    Further splitting tiny path intervals does not shrink their stopping
    excursions. A small fixed budget there repeats the same expensive proof
    for hundreds of almost identical intervals.
    """
    width=max(float(np.max(np.abs(np.asarray(end[k])-start[k]))) for k in start)
    stop=max(float(max(row['stop_excursion_deg']))
             for row in config.raw['safety']['arms'].values())
    return None if width<=max(1.,4*stop) else 64


def validate_native_path(config, models, joints, *, tool_contact=False):
    """Cover a sampled joint path in large boxes, splitting only as needed.

    Include every intermediate joint extremum, including curved IK paths.
    A failed large-box proof is subdivided; failure at one interval propagates.
    """
    count = len(next(iter(joints.values())))
    pending = [(0, count-1)] if count > 1 else []
    while pending:
        lo,hi = pending.pop()
        minimum = {k:q[lo:hi+1].min(axis=0) for k,q in joints.items()}
        maximum = {k:q[lo:hi+1].max(axis=0) for k,q in joints.items()}
        lower,upper = native_joint_ranges(config, minimum, maximum)
        try:
            certify_joint_box(config,models,lower,upper,tool_contact=tool_contact,
                              max_nodes=path_proof_budget(config,minimum,maximum) if hi-lo>1 else None)
        except DualArmError:
            if hi-lo <= 1:
                raise
            mid = (lo+hi)//2
            pending.extend([(mid,hi),(lo,mid)])


def validate_native_state(config, models, joints, *, half_ranges=None, tool_contact=False):
    """Instantaneous geometry/tracking check; stopping belongs to command proof."""
    tracking=(config.limits['tracking_error_deg']
              if config.raw['safety']['status']=='measured' else 0.)
    half={k:np.full(a.axis,tracking)+(0 if half_ranges is None else half_ranges[k])
          for k,a in config.arms.items()}
    return certify_joint_box(config,models,
        {k:np.asarray(q)-half[k] for k,q in joints.items()},
        {k:np.asarray(q)+half[k] for k,q in joints.items()},tool_contact=tool_contact)
