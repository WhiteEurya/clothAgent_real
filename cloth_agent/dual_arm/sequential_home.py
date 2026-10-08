"""Prevalidated, non-blended controller position commands, one arm at a time."""
from dataclasses import dataclass
import json
import math
import time

import numpy as np

from .geometry import DualArmError, pose_error
from .planning import Motion, validate_sample
from .stop_sweep import validate_native_sweep, native_joint_ranges, path_proof_budget


@dataclass
class PositionSegment:
    arm: str
    start: dict
    end: dict
    speed: float
    acceleration: float
    timeout: float

    def to_dict(self):
        return dict(arm=self.arm, start={k: q.tolist() for k,q in self.start.items()},
                    end={k: q.tolist() for k,q in self.end.items()},
                    speed_deg_s=self.speed, acceleration_deg_s2=self.acceleration,
                    timeout_s=self.timeout)


def tcp_weights(model, arm):
    """Configuration-independent TCP lever bounds for actual URDF models."""
    if not hasattr(model, 'capsule_reaches'):
        return None  # Deterministic test model; never a real ArmModel.
    return np.max(list(model.capsule_reaches.values()), axis=0) + np.linalg.norm(arm.tcp_offset[:3])


def certify_segment(start, end, config, models, *, max_nodes=None, path_bounds=None):
    """Cover the entire joint box, not only the nominal synchronized joint line."""
    changed = [k for k in start if np.max(np.abs(end[k]-start[k])) > 1e-9]
    if len(changed) > 1:
        raise DualArmError('controller segment must move at most one arm')
    midpoint = {k: (start[k]+end[k])/2 for k in start}
    half = {k: np.abs(end[k]-start[k])/2 for k in start}
    validate_native_sweep(config, models, start, end, max_nodes=max_nodes)
    for joints in (start, midpoint, end):
        validate_sample(config, models, joints, holding=False)
    for k, arm in config.arms.items():
        weights = tcp_weights(models[k], arm)
        if weights is None:
            if config.raw['safety']['status'] != 'synthetic':
                raise DualArmError('controller proof requires real URDF reach bounds')
            radius = float(np.linalg.norm(half[k][:3]))
        else:
            radius = float(weights @ np.radians(half[k]))
        center = models[k].forward(midpoint[k])[:3]
        arm.validate_point(center - radius)
        arm.validate_point(center + radius)
        if path_bounds is not None:
            from .home_path import deviation
            world=arm.world_from_base[:3,:3]@center+arm.world_from_base[:3,3]
            bound=path_bounds[k]
            if world[2]+radius>bound['height']+1e-6:
                raise DualArmError(f'{k}: controller segment exceeds Home height limit')
            if float(deviation(world,bound))+radius>bound['radius']+1e-6:
                raise DualArmError(f'{k}: controller segment exceeds Home corridor')


def command_limits(config, model, arm, start, end):
    speed = config.limits['joint_speed_deg_s']
    acceleration = config.limits['joint_accel_deg_s2']
    weights = tcp_weights(model, arm)
    if weights is not None:
        active = np.abs(end-start) > 1e-9
        reach = float(weights[active].sum())
        if reach > 0:
            axes = max(1, int(active.sum()))
            # Bound sum(J*qdot), plus acceleration and rotational cross terms.
            speed = min(speed, math.degrees(config.limits['cartesian_speed_mm_s']/reach),
                        math.degrees(math.sqrt(config.limits['cartesian_accel_mm_s2']/(2*reach*axes))))
            acceleration = min(acceleration, math.degrees(config.limits['cartesian_accel_mm_s2']/(2*reach)))
    timeout = 30 + 4*(float(np.max(np.abs(end-start)))/speed + speed/acceleration)
    return speed, acceleration, timeout


def prepare_segments(program, config, models):
    """Finish proof of ALL commands before any dispatch; split uncertain boxes."""
    from .homing import validate_home_program
    if config.execution_mode != 'controller_sequential':
        raise DualArmError('native Home requires controller-sequential configuration')
    validate_home_program(program, config)
    from .home_path import check_program_path
    path_bounds,_=check_program_path(program,config,models)
    segments = []
    previous = None
    for motion in program[:-1]:
        moving = [k for k,q in motion.joints.items() if np.max(np.abs(q-q[0])) > 1e-9]
        if len(moving) > 1:
            raise DualArmError('native Home must move arms sequentially')
        if previous is not None and any(not np.allclose(q[0], previous[k], atol=1e-9, rtol=0)
                                        for k,q in motion.joints.items()):
            raise DualArmError('discontinuous native Home program')
        previous = {k:q[-1].copy() for k,q in motion.joints.items()}
        if not moving:
            continue
        k = moving[0]
        pending = [(0, len(motion.times)-1)]
        while pending:
            lo, hi = pending.pop()
            start = {a:q[lo].copy() for a,q in motion.joints.items()}
            end = {a:q[hi].copy() for a,q in motion.joints.items()}
            try:
                certify_segment(start, end, config, models,
                                max_nodes=path_proof_budget(config,start,end) if hi-lo>1 else None,
                                path_bounds=path_bounds)
            except DualArmError:
                if hi-lo <= 1:
                    raise
                mid = (lo+hi)//2
                pending.extend([(mid,hi),(lo,mid)])
                continue
            speed, acceleration, timeout = command_limits(config, models[k], config.arms[k], start[k], end[k])
            segments.append(PositionSegment(k,start,end,speed,acceleration,timeout))
            if len(segments) > 256:
                raise DualArmError('native Home exceeds 256 preplanned position commands')
    return segments


def check_feedback(coordinator, segment):
    config, models = coordinator.config, coordinator.models
    state = coordinator.snapshots()
    joints = {k:np.asarray(v['joints']) for k,v in state.items()}
    lower,upper=native_joint_ranges(config,segment.start,segment.end)
    for k,q in joints.items():
        if k != segment.arm:
            if np.max(np.abs(q-segment.start[k])) > config.limits['stationary_tolerance_deg']:
                raise DualArmError(f'{k}: inactive arm moved during native Home')
        else:
            if np.any(q < lower[k]) or np.any(q > upper[k]):
                raise DualArmError(f'{k}: controller left certified joint segment')
        distance, angle = pose_error(state[k]['pose'], models[k].forward(q))
        if distance > 2 or angle > 1:
            raise DualArmError(f'{k}: live controller/model FK mismatch')
    validate_sample(config, models, joints, holding=False)


def wait_operation(coordinator, operation, timeout, monitor):
    future = coordinator.pool.submit(operation)
    began = time.monotonic()
    while not future.done():
        if coordinator.cancel.is_set():
            raise DualArmError('native Home cancelled')
        if time.monotonic()-began > timeout:
            raise DualArmError('native position command completion timeout')
        monitor()
        coordinator.cancel.wait(.05)
    future.result()
    if coordinator.cancel.is_set():
        raise DualArmError('native Home cancelled')


def execute_sequential_home(coordinator, program, initial, segments=None, *, confirmed=False):
    """Position-mode commands only. Feedback supervises, never streams targets."""
    result = {'status':'RUNNING', 'execution_mode':'controller_sequential',
              'physical_execution':not coordinator.simulated, 'completed_segments':0}
    config, models = coordinator.config, coordinator.models
    try:
        if not coordinator.simulated:
            if not confirmed:
                raise DualArmError('real native Home requires confirmation')
            config.require_real(homing=True)
        # Rebuild from the program so callers cannot inject uncertified segments.
        segments = prepare_segments(program, config, models)
        expected = {k:np.asarray(q).copy() for k,q in initial.items()}
        if any(not np.allclose(program[0].joints[k][0],q,atol=1e-9,rtol=0) for k,q in expected.items()):
            raise DualArmError('native program does not start at supplied initial joints')

        def arrival(target):
            # A native joint command arrives by joint position, not by the old
            # streamer's Cartesian tracking tolerance (which may validly be 0).
            # Keep the independent controller/model consistency gate.
            state = coordinator.snapshots()
            actual={k:np.asarray(v['joints']) for k,v in state.items()}
            for k,v in state.items():
                if np.max(np.abs(actual[k]-target[k])) > config.limits['stationary_tolerance_deg']:
                    raise DualArmError(f'{k}: native command did not arrive at target')
                distance,angle=pose_error(v['pose'],models[k].forward(actual[k]))
                if distance>2 or angle>1:
                    raise DualArmError(f'{k}: live controller/model FK mismatch ({distance:.3f} mm, {angle:.3f} deg)')
            validate_sample(config,models,actual,holding=False)

        arrival(expected)
        for c in coordinator.connections.values():
            c.prepare_position()
        for index, segment in enumerate(segments):
            arrival(segment.start)
            coordinator.event('position_command_start', index=index, **segment.to_dict())
            connection = coordinator.connections[segment.arm]
            wait_operation(coordinator,
                lambda s=segment,c=connection: c.move_joint_position(s.end[s.arm],s.speed,s.acceleration,s.timeout),
                segment.timeout, lambda s=segment: check_feedback(coordinator,s))
            arrival(segment.end)
            expected = segment.end
            result['completed_segments'] += 1
            coordinator.event('position_command_complete',index=index,arm=segment.arm)
        arrival({k:a.home_joints for k,a in config.arms.items()})
        for k,connection in coordinator.connections.items():
            wait_operation(coordinator,lambda c=connection:c.gripper('open'),
                           config.arms[k].gripper.gripper_completion_timeout_s+1,
                           lambda:arrival(expected))
            coordinator.event('home_gripper_opened',arm=k)
        arrival(expected)
        result['status'] = 'COMPLETED'
    except BaseException as exc:
        result['status'] = 'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['stop_results'] = coordinator.stop_all()
        coordinator.event('native_home_failed',error=result['error'])
    finally:
        (coordinator.directory/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
