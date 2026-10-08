"""Exact joint-home planning through the shared dual-arm collision executor."""
from __future__ import annotations

import math
import numpy as np

from .geometry import DualArmError, finite
from .planning import Motion, Phase, compile_motion, validate_sample
from .safety import validate_sweep


def joint_motion(start, end, config, models, name):
    """Trapezoidal joint speed; validate every sample and swept interval."""
    limits = config.limits
    start = {k: finite(start[k], (a.axis,), 'start joints') for k, a in config.arms.items()}
    end = {k: finite(end[k], (a.axis,), 'home joints') for k, a in config.arms.items()}
    delta = max(float(np.max(np.abs(end[k] - start[k]))) for k in start)
    profile_speed = limits['joint_speed_deg_s']
    profile_acceleration = limits['joint_accel_deg_s2']
    ramp = min(profile_speed / profile_acceleration, math.sqrt(delta / profile_acceleration))
    cruise = max(0., delta / profile_speed - ramp)
    nominal_duration = 2 * ramp + cruise
    duration = max(.5, nominal_duration)
    for _ in range(12):
        count = math.ceil(duration * limits['rate_hz'])
        if count + 1 > limits['max_plan_samples']:
            raise DualArmError('home trajectory sample budget exceeded')
        times = np.linspace(0, duration, count + 1)
        if delta == 0:
            progress = np.zeros_like(times)
        else:
            t = times * nominal_duration / duration
            distance = np.where(
                t < ramp, .5 * profile_acceleration * t**2,
                np.where(t <= ramp + cruise,
                         .5 * profile_acceleration * ramp**2 + profile_acceleration * ramp * (t-ramp),
                         delta - .5 * profile_acceleration * (nominal_duration-t)**2))
            progress = distance / delta
        joints = {k: start[k] + progress[:, None]*(end[k]-start[k]) for k in start}
        for k in joints:
            joints[k][-1] = end[k]  # Preserve the taught redundant-joint solution.
        poses = {k: [] for k in start}
        for i in range(len(times)):
            world = validate_sample(config, models, {k: q[i] for k, q in joints.items()}, holding=False)
            for k in poses:
                poses[k].append(world[k])
        poses = {k: np.asarray(v) for k, v in poses.items()}
        factor = 1.0
        dt = times[1] - times[0]
        for values, speed, acceleration, cartesian in [
            (joints, limits['joint_speed_deg_s'], limits['joint_accel_deg_s2'], False),
            (poses, limits['cartesian_speed_mm_s'], limits['cartesian_accel_mm_s2'], True),
        ]:
            for q in values.values():
                if cartesian:
                    q = q[:, :3]
                velocity = np.diff(q, axis=0)/dt
                accel = np.diff(np.vstack([np.zeros_like(q[0]), velocity, np.zeros_like(q[0])]), axis=0)/dt
                peak = lambda v: float(np.max(np.linalg.norm(v, axis=1) if cartesian else np.abs(v)))
                factor = max(factor, peak(velocity)/speed, math.sqrt(peak(accel)/acceleration))
        max_step = max(float(np.max(np.abs(np.diff(q, axis=0)))) for q in joints.values())
        factor = max(factor, max_step/limits['max_ik_step_deg'])
        if factor <= 1.001:
            if config.execution_mode == 'controller_sequential':
                from .stop_sweep import validate_native_path
                validate_native_path(config, models, joints)
            else:
                for i in range(1, len(times)):
                    validate_sweep(config, models, {k: q[i-1] for k, q in joints.items()},
                                   {k: q[i] for k, q in joints.items()})
            return Motion(Phase(name, 'move', {k: p[-1] for k, p in poses.items()}), times, joints, poses)
        duration *= factor * 1.05
    raise DualArmError('unable to time-scale home trajectory')


def retreat_targets(initial, config, models, distance_mm):
    """Retract horizontally toward each arm's own base, preserving TCP attitude."""
    from .geometry import matrix_pose, pose_matrix

    targets = {}
    for k, arm in config.arms.items():
        pose = np.array(models[k].forward(initial[k]), dtype=float, copy=True)
        radius = float(np.linalg.norm(pose[:2]))
        if radius <= distance_mm:
            raise DualArmError(f'{k}: retreat would reach or cross its base axis')
        pose[:2] *= (radius - distance_mm) / radius
        arm.validate_point(pose[:3])
        targets[k] = matrix_pose(arm.world_from_base @ pose_matrix(pose))
    return targets


def plan_home(initial, config, models):
    """Try direct routes, then bounded retreats; never return a partial plan."""
    target = {k: a.home_joints.copy() for k, a in config.arms.items()}
    keys = tuple(config.arms)
    attempts = []
    for label, joints in [('initial state', initial), ('Home target', target)]:
        try:
            validate_sample(config, models, joints, holding=False)
        except DualArmError as exc:
            raise DualArmError(f'No collision-checked home route: {label}: {exc}') from exc
    orders = [('simultaneous', [keys]),
              ('left_then_right', [(keys[0],), (keys[1],)]),
              ('right_then_left', [(keys[1],), (keys[0],)])]
    if config.execution_mode == 'controller_sequential':
        orders = orders[1:]

    def finish(current, prefix, route_label):
        native=config.execution_mode == 'controller_sequential'
        methods=('cartesian','joint','interleaved','clearance_cartesian','clearance_joint') if native else ('joint',)
        for method in methods:
            if method=='interleaved':
                from .home_path import interleaved_return
                from .sequential_home import prepare_segments
                for steps,clearance in ((6,False),(6,True),(12,False),(12,True),(24,True)):
                    strategy=f'{route_label}interleaved_{steps}'+('_clearance' if clearance else '')
                    try:
                        program=list(prefix)+interleaved_return(current,target,config,models,
                                                              f'home_{strategy}',steps=steps,clearance=clearance)
                        if sum(len(m.times) for m in program)>config.limits['max_plan_samples']:
                            raise DualArmError('whole home program sample budget exceeded')
                        program.append(Phase('home_open_grippers','open'))
                        prepare_segments(program,config,models)
                        attempts.append({'strategy':strategy,'status':'accepted'})
                        return program
                    except DualArmError as exc:
                        attempts.append({'strategy':strategy,'status':'rejected','reason':str(exc)})
                continue
            for label,groups in orders:
                strategy=f'{route_label}{method+"_" if native else ""}{label}'
                q={k:v.copy() for k,v in current.items()}
                program=list(prefix)
                try:
                    for index,group in enumerate(groups):
                        name=f'home_{strategy}_{index}'
                        if native:
                            from .home_path import coordinated_return
                            motions=coordinated_return(q,target,group,config,models,name,
                                                       clearance=method.startswith('clearance_'),
                                                       joint=method.endswith('joint'))
                        else:
                            end={k:target[k] if k in group else v for k,v in q.items()}
                            motions=[joint_motion(q,end,config,models,name)]
                        program.extend(motions)
                        q={k:v[-1].copy() for k,v in motions[-1].joints.items()}
                    if sum(len(m.times) for m in program)>config.limits['max_plan_samples']:
                        raise DualArmError('whole home program sample budget exceeded')
                    program.append(Phase('home_open_grippers','open'))
                    if native:
                        from .sequential_home import prepare_segments
                        prepare_segments(program,config,models)
                    attempts.append({'strategy':strategy,'status':'accepted'})
                    return program
                except DualArmError as exc:
                    attempts.append({'strategy':strategy,'status':'rejected','reason':str(exc)})
        return None

    # Close collaborative grasps need to separate before rotating toward Home.
    # Plan the WHOLE separating prefix and return route before any dispatch.
    if config.execution_mode == 'controller_sequential':
        from .separation import minimum_gap, separation_steps
        from .stop_sweep import position_error_mm
        at_home=all(np.max(np.abs(initial[k]-target[k])) <= config.limits['stationary_tolerance_deg']
                    for k in keys)
        close_threshold=30.+sum(position_error_mm(config,k) for k in keys)
        if not at_home and minimum_gap(models,initial) < close_threshold:
            for current,prefix in separation_steps(initial,config,models,attempts):
                program=finish(current,prefix,f'separate_{len(prefix)}steps_return_')
                if program is not None:
                    return program,attempts

    program = finish(initial, [], '')
    if program is not None:
        return program, attempts
    inverse = {k: model.inverse for k, model in models.items()}
    for distance in (50., 100., 150.):
        try:
            targets = retreat_targets(initial, config, models, distance)
        except DualArmError as exc:
            attempts.append({'strategy': f'retreat_{distance:g}mm', 'status': 'rejected', 'reason': str(exc)})
            continue
        for label, groups in orders:
            strategy = f'retreat_{distance:g}mm_{label}'
            current = {k: np.asarray(q).copy() for k, q in initial.items()}
            prefix = []
            try:
                for index, group in enumerate(groups):
                    poses = validate_sample(config, models, current, holding=False)
                    phase = Phase(f'home_{strategy}_{index}', 'move',
                                  {k: targets[k] if k in group else poses[k] for k in keys},
                                  fixed_arms=tuple(k for k in keys if k not in group))
                    motion = compile_motion(phase, current, config, models, inverse)
                    prefix.append(motion)
                    current = {k: q[-1].copy() for k, q in motion.joints.items()}
                program = finish(current, prefix, f'{strategy}_return_')
                if program is not None:
                    return program, attempts
            except DualArmError as exc:
                attempts.append({'strategy': strategy, 'status': 'rejected', 'reason': str(exc)})
    raise DualArmError(f'No collision-checked home route: {attempts}')


def validate_home_program(program, config):
    """The observation-free execution path is limited to home and final opening."""
    if len(program) < 2 or not isinstance(program[-1], Phase):
        raise DualArmError('invalid home program')
    final = program[-1]
    if final.kind != 'open' or final.name != 'home_open_grippers' or set(final.gripper_arms) != set(config.arms):
        raise DualArmError('home must finish by opening both grippers')
    for item in program:
        phase = item.phase if isinstance(item, Motion) else item
        if phase.holding or phase.tool_contact or phase.pin_arm or phase.checkpoint:
            raise DualArmError('home cannot contain cloth/contact operations')
    for item in program[:-1]:
        if not isinstance(item, Motion) or item.phase.kind != 'move' or not item.phase.name.startswith('home_'):
            raise DualArmError('home requires only checked joint motions')
    for k, arm in config.arms.items():
        if not np.allclose(program[-2].joints[k][-1], arm.home_joints, atol=1e-6, rtol=0):
            raise DualArmError('home must reach both saved joint configurations')


def run_home(args):
    """Compatibility adapter for the dual-arm command-line interface."""
    result = _run_home(args)
    return 0 if result['status'] in {'COMPLETED', 'PREFLIGHT_ONLY'} else 1


def gripper_home(config_path=None, *, output=None, simulated=False,
                 preflight_only=False, project_root=None):
    """Block until both real arms return Home and both grippers open.

    Uses the saved local configuration by default. Returns an execution result
    dictionary including status and output_directory; preflight errors raise
    DualArmError. Importing this module never connects to or moves hardware.
    """
    from datetime import datetime, timezone
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
    directory = (Path(output) if output is not None else
                 root/'results/dual_home'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = SimpleNamespace(
        config=Path(config_path) if config_path is not None else root/'config/dual_arm.local.json',
        output=directory, project_root=root, real=not simulated,
        confirm_real=not simulated, preflight_only=preflight_only)
    return _run_home(args)


def _run_home(args):
    """Read both robots, finish all planning, then optionally execute together."""
    import signal
    import threading
    from .cli import connect, write_json
    from .config import DualConfig
    from .execution import DualArmCoordinator
    from .kinematics import ArmModel
    from .preview import write_preview

    if args.real != args.confirm_real:
        raise DualArmError('real home requires both --real and --confirm-real')
    if args.real:
        import json
        from .readiness import home_missing_fields
        missing = home_missing_fields(json.loads(args.config.read_text()),
                                      execution_mode='controller_sequential')
        if missing:
            raise DualArmError('Home prerequisites missing (no robot connection):\n - '
                               + '\n - '.join(missing))
    config = DualConfig.load(args.config, root=args.project_root, homing_only=True)
    config.execution_mode = 'controller_sequential'
    if args.real:
        config.require_real(homing=True)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    models = {k: ArmModel(a) for k, a in config.arms.items()}
    cancel = threading.Event()
    connections = connect(config, models, cancel, args.real)
    coordinator = DualArmCoordinator(config, models, connections, cancel, directory, realtime=args.real)
    install_signal_handler = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if install_signal_handler else None
    def terminate(signum, frame):
        cancel.set()
        raise KeyboardInterrupt('SIGTERM')
    if install_signal_handler:
        signal.signal(signal.SIGTERM, terminate)
    try:
        state = coordinator.snapshots()
        initial = {k: np.asarray(v['joints']) for k, v in state.items()}
        program, attempts = plan_home(initial, config, models)
        validate_home_program(program, config)
        from .sequential_home import prepare_segments, execute_sequential_home
        segments = prepare_segments(program, config, models)
        from .home_path import check_program_path
        _,path_metrics=check_program_path(program,config,models)
        write_json(directory/'config.json', config.raw)
        write_json(directory/'plan.json', {
            'kind': 'dual_joint_home', 'attempts': attempts,
            'execution_mode': config.execution_mode,
            'controller_segments': [s.to_dict() for s in segments],
            'path_metrics': path_metrics,
            'initial_joints_deg': {k: q.tolist() for k, q in initial.items()},
            'home_joints_deg': {k: a.home_joints.tolist() for k, a in config.arms.items()},
            'continuous_collision_checked': True,
            'duration_s': sum(float(m.times[-1]) for m in program if isinstance(m, Motion)),
        })
        for i, motion in enumerate(program[:-1]):
            np.savez_compressed(directory/f'home_{i}.npz', times=motion.times,
                                **{f'{k}_joints_deg': q for k, q in motion.joints.items()})
        write_preview(program, config, models, directory/'preview.html')
        if args.preflight_only:
            result = {'status': 'PREFLIGHT_ONLY', 'physical_execution': False,
                      'output_directory': str(directory)}
            write_json(directory/'execution.json', result)
            print(f'PREFLIGHT_ONLY: {directory}')
            return result
        result = execute_sequential_home(coordinator, program, initial, segments,
                                         confirmed=args.confirm_real)
        print(f"{result['status']}: {directory}")
        return {**result, 'output_directory': str(directory)}
    except BaseException as exc:
        coordinator.stop_all()
        write_json(directory/'failure.json', {'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        if install_signal_handler:
            signal.signal(signal.SIGTERM, previous)
        coordinator.close()
