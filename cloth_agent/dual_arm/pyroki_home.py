"""PyRoki Home candidates, accepted only by the installed dual-arm validators."""
import numpy as np

from .geometry import DualArmError, finite
from .homing import joint_motion
from .planning import Phase, validate_sample
from .sequential_home import prepare_segments, execute_sequential_home


def optimize_joint_path(model, start, goal, count=17):
    import jax.numpy as jnp
    import jaxls
    import pyroki as pk
    from .pyroki_ik import PyrokiIK

    adapter = PyrokiIK(model)
    robot = adapter.robot
    seed = np.radians(np.linspace(start, goal, count))[:, adapter.to_pyroki]
    variables = [robot.joint_var_cls(index) for index in range(count)]
    costs = [pk.costs.limit_constraint(robot, variable) for variable in variables]
    costs.extend(pk.costs.smoothness_cost(current, previous, weight=1.)
                 for previous, current in zip(variables, variables[1:]))
    costs.extend(pk.costs.rest_cost(variables[index], rest_pose=jnp.asarray(seed[index]), weight=1e5)
                 for index in (0, count - 1))
    solution = jaxls.LeastSquaresProblem(costs, variables).analyze().solve(
        initial_vals=jaxls.VarValues.make(tuple(variable.with_value(jnp.asarray(row))
                                               for variable, row in zip(variables, seed))),
        verbose=False, linear_solver='dense_cholesky',
        termination=jaxls.TerminationConfig(max_iterations=50),
    )
    result = np.degrees(np.asarray([solution[variable] for variable in variables]))[:, adapter.from_pyroki]
    if not np.isfinite(result).all() or not np.allclose(result[[0, -1]], [start, goal], atol=1e-4, rtol=0):
        raise DualArmError('PyRoki Home did not converge to requested endpoints')
    result[0], result[-1] = start, goal
    if np.any(np.radians(result) < model.lower) or np.any(np.radians(result) > model.upper):
        raise DualArmError('PyRoki Home exceeds URDF joint limits')
    return result


def plan_home(initial, config, models):
    if config.execution_mode != 'controller_sequential':
        raise DualArmError('PyRoki Home requires the existing sequential position executor')
    if not getattr(config, 'path_validation_enabled', True) or getattr(config, 'collision_envelopes_disabled_arms', ()):
        raise DualArmError('Home requires both arms and installed collision geometry checks enabled')
    initial = {key: finite(initial[key], (arm.axis,), 'current joints').copy()
               for key, arm in config.arms.items()}
    target = {key: arm.home_joints.copy() for key, arm in config.arms.items()}
    validate_sample(config, models, initial, holding=False)
    validate_sample(config, models, target, holding=False)
    attempts = []
    for order in (('left', 'right'), ('right', 'left')):
        current = {key: value.copy() for key, value in initial.items()}
        program = []
        try:
            for active in order:
                rows = (np.asarray([current[active], target[active]])
                        if np.array_equal(current[active], target[active])
                        else optimize_joint_path(models[active], current[active], target[active]))
                if rows.ndim != 2 or rows.shape[1] != config.arms[active].axis or len(rows) < 2:
                    raise DualArmError('Invalid PyRoki path dimensions')
                if not np.array_equal(rows[0], current[active]) or not np.array_equal(rows[-1], target[active]):
                    raise DualArmError('PyRoki path endpoints changed')
                for index, row in enumerate(rows[1:]):
                    end = {key: value.copy() for key, value in current.items()}
                    end[active] = row.copy()
                    program.append(joint_motion(current, end, config, models, f'home_pyroki_{active}_{index}'))
                    current = end
                    if sum(len(motion.times) for motion in program) > config.limits['max_plan_samples']:
                        raise DualArmError('PyRoki Home sample budget exceeded')
            program.append(Phase('home_open_grippers', 'open'))
            segments = prepare_segments(program, config, models)
            attempts.append({'order': list(order), 'status': 'accepted'})
            return program, segments, attempts
        except DualArmError as exc:
            attempts.append({'order': list(order), 'status': 'rejected', 'reason': str(exc)})
    raise DualArmError(f'No accepted PyRoki Home candidate: {attempts}')


def execute_home(coordinator, *, confirmed=False):
    try:
        snapshots = coordinator.snapshots()
        initial = {key: finite(row['joints'], (coordinator.config.arms[key].axis,), 'live joints')
                   for key, row in snapshots.items()}
        program, segments, attempts = plan_home(initial, coordinator.config, coordinator.models)
        return execute_sequential_home(coordinator, program, initial, segments, confirmed=confirmed)
    except BaseException:
        coordinator.stop_all()
        raise
