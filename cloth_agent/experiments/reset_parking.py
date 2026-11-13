"""Bounded reset drop and inactive-arm parking search."""
import ast
import json
import numpy as np
from dataclasses import replace
from pathlib import Path
from .discovery_data import save
from .full_grasp_adapter import build_robot_config
from cloth_agent.codex_motion import parse_code
from cloth_agent.robot_collision import RobotCollision


def _program_actions(program):
    actions = []
    for node in ast.parse(program).body[0].body:
        call = node.value
        values = [ast.literal_eval(a) for a in call.args]
        actions.append(dict(name=call.func.id, args=dict(zip(('x','y','z','yaw'), values))))
    return actions


def shift_midline_drop(source, tangent_xy, distance_mm):
    """Move the release leg along the arm bisector without changing grasp/lift."""
    tangent = np.asarray(tangent_xy, float)
    if tangent.shape != (2,) or not np.isfinite(tangent).all() or not np.isclose(np.linalg.norm(tangent), 1., atol=1e-6):
        raise ValueError('invalid reset midline tangent')
    if not np.isfinite(distance_mm) or abs(distance_mm) > 50:
        raise ValueError('reset midline shift must be within 50 mm')
    calls = parse_code(source, allow_grasp=True)
    if len(calls) < 3 or calls[0][0] != 'grasp':
        raise ValueError('reset source requires grasp, lift and drop')
    lines = []
    for index, (name, kwargs) in enumerate(calls):
        moved = {arm:list(pose) for arm,pose in kwargs.items()}
        if index >= 2:
            for pose in moved.values():
                pose[0] += float(tangent[0] * distance_mm / 1000.)
                pose[1] += float(tangent[1] * distance_mm / 1000.)
        lines.append(f'motion.{name}(' + ', '.join(f'{arm}={pose!r}' for arm,pose in moved.items()) + ')')
    return '\n'.join(lines) + '\n'


def _save_selected_reset(folder, source, program, assignment, shift_mm, tangent):
    folder = Path(folder)
    (folder/'motion.py').write_text(source, encoding='utf-8')
    (folder/'robotapi_preview.py').write_text(program, encoding='utf-8')
    geometry_path = folder/'geometry.json'
    if geometry_path.exists():
        geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
        calls = parse_code(source, allow_grasp=True)
        release = np.asarray(next(iter(calls[-1][1].values()))[:3], float) * 1000.
        transfer = np.asarray(next(iter(calls[2][1].values()))[:3], float) * 1000.
        midline = transfer.copy()
        midline[2] = release[2]
        geometry.update(assignment=assignment, midline_shift_mm=float(shift_mm),
                        transfer_world_mm=transfer.tolist(),
                        midline_shift_tangent_xy=np.asarray(tangent,float).tolist(),
                        midline_before_translation_world_mm=midline.tolist(),
                        drop_world_mm=release.tolist())
        save(geometry_path, geometry)


def find_parking(runtime, before, source, folder):
    initial = {k:np.asarray(v) for k,v in before.meta['joints_deg'].items()}
    _, _, original = runtime.compile(source, before, post_release_descent_mm=50.)
    active, inactive = original['active_arm'], original['inactive_arm']
    axis = np.asarray(original['right_axis_xy'], float)
    tangent = np.array([-axis[1], axis[0]], float)
    tangent /= np.linalg.norm(tangent)
    home = runtime.models[inactive].forward(initial[inactive])
    rejected = []
    shifts = (0., 10., -10., 20., -20., 30., -30., 40., -40., 50., -50.)
    for shift in shifts:
        candidate_source = shift_midline_drop(source, tangent, shift)
        robot, program, assignment = runtime.compile(
            candidate_source, before, post_release_descent_mm=50.)
        if assignment['active_arm'] != active:
            raise ValueError('reset arm assignment changed during midline search')
        actions = _program_actions(program)
        try:
            check = RobotCollision(runtime.config_path, initial, active, root=runtime.config.root)
            check.require_execution(robot, actions)
            actual = runtime.models[inactive].forward(initial[inactive])
            result = dict(joints_deg=initial[inactive].tolist(), pose_base_mm_deg=actual.tolist(),
                          retreat_base_x_mm=0., parking_required=False, midline_shift_mm=shift,
                          midline_shift_tangent_xy=tangent.tolist(), rejected=rejected,
                          arm=inactive, active_arm=active, reset_source=candidate_source,
                          status='RESET_COMPLETE_PATH_CHECKED')
            _save_selected_reset(folder,candidate_source,program,assignment,shift,tangent)
            save(folder/'parking_plan.json', result)
            return result
        except (ValueError, RuntimeError) as exc:
            rejected.append(dict(midline_shift_mm=shift, retreat_base_x_mm=0., error=str(exc)))
            print(f'active {active} midline shift {shift:g} mm rejected: {exc}',flush=True)
            # Moving the other arm cannot repair an active-arm IK failure.
            if 'PyRoki IK rejected target' in str(exc):
                continue
        for distance in (30., 50., 75., 100., 125., 150.):
            try:
                target = home.copy()
                target[0] -= distance
                q = runtime.models[inactive].inverse(target, initial[inactive])
                actual = runtime.models[inactive].forward(q)
                config = build_robot_config(runtime.config.raw, {}, inactive, *actual[3:5], config_path=runtime.config_path)
                config = replace(config, perception_joints_deg=tuple(q), perception_pose_mm_deg=tuple(actual))
                retreat = RobotCollision(runtime.config_path, initial, inactive, root=runtime.config.root)
                retreat.require_execution(config, [dict(name='perception_position',args={})])
                parked = dict(initial);parked[inactive]=q
                check = RobotCollision(runtime.config_path, parked, active, root=runtime.config.root)
                check.require_execution(robot, actions)
                result = dict(joints_deg=q.tolist(), pose_base_mm_deg=actual.tolist(),
                              retreat_base_x_mm=distance, parking_required=True,
                              midline_shift_mm=shift, midline_shift_tangent_xy=tangent.tolist(),
                              rejected=rejected,arm=inactive,active_arm=active,
                              reset_source=candidate_source,status='BOTH_COMPLETE_PATHS_CHECKED')
                _save_selected_reset(folder,candidate_source,program,assignment,shift,tangent)
                save(folder/'parking_plan.json', result)
                return result
            except (ValueError, RuntimeError) as exc:
                rejected.append(dict(midline_shift_mm=shift,retreat_base_x_mm=distance,error=str(exc)))
                print(f'{inactive} retreat {distance:g} mm at midline shift {shift:g} rejected: {exc}',flush=True)
                if 'PyRoki IK rejected target' in str(exc):
                    break
    save(folder/'parking_plan.json', dict(status='NO_CLEAR_PATH',rejected=rejected))
    raise ValueError('No valid midline drop within +/-50 mm or inactive-arm retreat within 150 mm: '
                     +rejected[-1]['error'])


def execute_parking(runtime, before, plan, folder):
    from cloth_agent.site_robot_check import read_arm
    from copy import deepcopy
    def snapshots():
        rows = {}
        for key in runtime.models:
            row = read_arm(runtime.profile,key)
            if (row['status'] != 'READ_ONLY_CHECKS_PASS'
                    or row['control_box_sn'] != runtime.config.arms[key].serial):
                raise ValueError(f'{key}: controller not ready or identity mismatch')
            rows[key] = dict(joints=row['joints_deg'])
        return rows
    current = snapshots()
    for key in current:
        if np.max(np.abs(np.array(current[key]['joints'])-before.meta['joints_deg'][key])) > runtime.config.limits['stationary_tolerance_deg']:
            raise ValueError('Arms changed after capture; parking canceled')
    arm=plan.get('arm','left')
    after = runtime._perception_before_capture(folder/f'{arm}_retreat',current,snapshots,arm=arm,saved=plan)
    observation = deepcopy(before)
    observation.meta['joints_deg'] = {k:v['joints'] for k,v in after.items()}
    save(folder/'post_parking_joints.json',observation.meta['joints_deg'])
    return observation
