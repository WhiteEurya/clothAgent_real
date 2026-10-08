"""Homing uses exact taught joints and never bypasses the collision planner."""
import numpy as np
import pytest

from cloth_agent.dual_arm import homing
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.planning import Motion
from cloth_agent.dual_arm.cli import main
from .test_dual_arm_runtime import scene, coordinator


def test_home_readiness_lists_all_missing_and_ignores_visual_fields():
    from cloth_agent.dual_arm.readiness import home_missing_fields
    missing = home_missing_fields({})
    assert 'arms.left.workspace.min_mm' in missing
    assert 'arms.right.workspace.max_mm' in missing
    assert 'safety.arms.right.stop_excursion_deg' in missing
    assert 'calibration_status' in missing
    assert not any('camera' in key or 'grasp_rpy' in key for key in missing)


def test_home_config_does_not_require_unused_grasp_orientation(scene):
    import copy
    from cloth_agent.dual_arm.config import DualConfig
    raw = copy.deepcopy(scene.config.raw)
    for arm in raw['arms'].values():
        arm['grasp_rpy_world_deg'] = None
    home = DualConfig.parse(raw, scene.config.root, homing_only=True)
    assert all(a.grasp_rpy is None for a in home.arms.values())
    with pytest.raises(DualArmError):
        DualConfig.parse(raw, scene.config.root)
    with pytest.raises(DualArmError, match='grasp orientation'):
        home.require_real()


def start_above_home(scene):
    start = {k: a.home_joints.copy() for k, a in scene.config.arms.items()}
    for q in start.values():
        q[2] += 5
    start['right'][6] = 12  # Same TCP can have a different seventh joint.
    return start


def test_home_reaches_both_exact_joint_vectors_and_opens(scene):
    initial = start_above_home(scene)
    program, attempts = homing.plan_home(initial, scene.config, scene.models)
    homing.validate_home_program(program, scene.config)
    assert attempts[-1]['status'] == 'accepted'
    for k, arm in scene.config.arms.items():
        assert np.array_equal(program[-2].joints[k][-1], arm.home_joints)
        scene.connections[k].joints = initial[k].copy()
    c = coordinator(scene)
    try:
        result = c.execute(program, initial, None, None, homing=True)
        assert result['status'] == 'COMPLETED', result
    finally:
        c.close()


def test_home_has_constant_speed_cruise_and_bounded_acceleration(scene):
    initial = start_above_home(scene)
    target = {k: a.home_joints.copy() for k, a in scene.config.arms.items()}
    motion = homing.joint_motion(initial, target, scene.config, scene.models, 'home_test')
    dt = np.diff(motion.times)
    velocity = np.diff(motion.joints['right'][:, 6]) / dt
    peak = np.max(np.abs(velocity))
    assert peak <= scene.config.limits['joint_speed_deg_s'] * 1.001
    assert np.count_nonzero(np.isclose(np.abs(velocity), peak)) > len(velocity) // 4
    assert abs(velocity[0]) < peak
    assert abs(velocity[-1]) < peak
    acceleration = np.diff(np.r_[0., velocity, 0.]) / dt[0]
    assert np.max(np.abs(acceleration)) <= scene.config.limits['joint_accel_deg_s2'] * 1.001


def test_time_scaling_never_changes_joint_path_or_overshoots(scene):
    scene.config.limits['cartesian_speed_mm_s']=1
    initial=start_above_home(scene)
    target={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    motion=homing.joint_motion(initial,target,scene.config,scene.models,'home_slow')
    for k,q in motion.joints.items():
        assert np.all(q>=np.minimum(initial[k],target[k])-1e-9)
        assert np.all(q<=np.maximum(initial[k],target[k])+1e-9)
        delta=target[k]-initial[k]
        assert np.all(np.diff(q,axis=0)*delta>=-1e-9)


def test_home_rejects_collision_on_entire_route(scene):
    initial = start_above_home(scene)
    # Put an obstacle at the left TCP's midpoint, keeping endpoints clear.
    scene.config.obstacles.append({'name': 'blocking', 'min_mm': [244,-126,181], 'max_mm':[256,-114,184]})
    with pytest.raises(DualArmError, match='No collision-checked home route'):
        homing.plan_home(initial, scene.config, scene.models)


def test_home_falls_back_to_sequential_and_holds_other_joints(scene, monkeypatch):
    original = homing.joint_motion
    def reject_simultaneous(start, end, config, models, name):
        if 'simultaneous' in name:
            raise DualArmError('simultaneous route blocked')
        return original(start, end, config, models, name)
    monkeypatch.setattr(homing, 'joint_motion', reject_simultaneous)
    initial = start_above_home(scene)
    program, attempts = homing.plan_home(initial, scene.config, scene.models)
    assert attempts[-1]['strategy'] == 'left_then_right'
    assert np.all(program[0].joints['right'] == initial['right'])
    assert np.all(program[1].joints['left'] == scene.config.arms['left'].home_joints)


def test_no_partial_program_when_later_leg_fails(scene, monkeypatch):
    original = homing.joint_motion
    def reject_late(start, end, config, models, name):
        if 'simultaneous' in name or name.endswith('_1'):
            raise DualArmError('blocked')
        return original(start, end, config, models, name)
    monkeypatch.setattr(homing, 'joint_motion', reject_late)
    with pytest.raises(DualArmError, match='No collision-checked home route'):
        homing.plan_home(start_above_home(scene), scene.config, scene.models)


def test_home_program_cannot_skip_exact_targets(scene):
    program, _ = homing.plan_home(start_above_home(scene), scene.config, scene.models)
    program[-2].joints['right'][-1,6] += 2
    with pytest.raises(DualArmError, match='both saved'):
        homing.validate_home_program(program, scene.config)


def test_incomplete_home_config_never_connects(tmp_path, monkeypatch):
    import json
    from cloth_agent.dual_arm.setup import initial_config
    from .test_dual_arm_runtime import ROOT
    path=tmp_path/'config.json'
    path.write_text(json.dumps(initial_config(ROOT,ROOT/'data/robot/dual_arm_home.json')))
    def fail_connect(*a,**kw):
        pytest.fail('must reject before hardware connection')
    monkeypatch.setattr('cloth_agent.dual_arm.cli.connect',fail_connect)
    assert main(['home','--config',str(path),'--real','--confirm-real','--output',str(tmp_path/'out')]) == 1


def test_home_fault_stops_both_without_opening(scene, monkeypatch):
    initial = start_above_home(scene)
    program, _ = homing.plan_home(initial, scene.config, scene.models)
    for k in initial:
        scene.connections[k].joints = initial[k].copy()
    def fail_servo(q):
        raise DualArmError('injected servo failure')
    monkeypatch.setattr(scene.connections['right'], 'servo', fail_servo)
    c = coordinator(scene)
    try:
        result = c.execute(program, initial, None, None, homing=True)
        assert result['status'] == 'FAILED'
        assert all(a.stopped for a in scene.connections.values())
        assert all(not any(cmd[0] == 'gripper' for cmd in a.commands) for a in scene.connections.values())
    finally:
        c.close()


def test_homing_does_not_bypass_real_safety_requirements(scene):
    with pytest.raises(DualArmError):
        scene.config.require_real(homing=True)


def test_retreat_routes_around_obstacle_on_direct_home_path(scene):
    initial = {k: a.home_joints.copy() for k, a in scene.config.arms.items()}
    for q in initial.values():
        q[2] += 60
    scene.config.obstacles.append({
        'name': 'direct_path_block', 'min_mm': [244, -126, 208],
        'max_mm': [256, -114, 212]})
    program, attempts = homing.plan_home(initial, scene.config, scene.models)
    assert all(a['status'] == 'rejected' for a in attempts[:3])
    assert attempts[-1]['strategy'].startswith('retreat_100mm_')
    retreat = program[0]
    for k in initial:
        before = scene.models[k].forward(initial[k])
        after = scene.models[k].forward(retreat.joints[k][-1])
        assert np.linalg.norm(after[:2]) == pytest.approx(np.linalg.norm(before[:2]) - 100)
        assert np.allclose(after[2:], before[2:])
    homing.validate_home_program(program, scene.config)


def test_retreat_sequential_keeps_other_arm_joint_solution(scene, monkeypatch):
    original_joint = homing.joint_motion
    original_cartesian = homing.compile_motion

    def only_retreat_return(start, end, config, models, name):
        if 'retreat_' not in name:
            raise DualArmError('direct route blocked')
        return original_joint(start, end, config, models, name)

    def only_sequential(phase, *args):
        if 'simultaneous' in phase.name:
            raise DualArmError('paired retreat blocked')
        return original_cartesian(phase, *args)

    monkeypatch.setattr(homing, 'joint_motion', only_retreat_return)
    monkeypatch.setattr(homing, 'compile_motion', only_sequential)
    # Preserve the redundant joint in this synthetic IK, as the real seeded IK does.
    monkeypatch.setattr(scene.models['right'], 'inverse', lambda p, seed: np.r_[p, seed[6:]])
    initial = start_above_home(scene)
    program, attempts = homing.plan_home(initial, scene.config, scene.models)
    assert 'retreat_50mm_left_then_right_return_' in attempts[-1]['strategy']
    assert np.all(program[0].joints['right'] == initial['right'])
    assert np.all(program[1].joints['left'] == program[0].joints['left'][-1])
    homing.validate_home_program(program, scene.config)


def test_retreat_never_escapes_colliding_initial_state(scene, monkeypatch):
    initial = start_above_home(scene)
    scene.config.obstacles.append({
        'name': 'initial_block', 'min_mm': [249, -121, 184],
        'max_mm': [251, -119, 186]})
    def forbidden(*args, **kwargs):
        pytest.fail('must reject colliding initial state before planning retreat')
    monkeypatch.setattr(homing, 'compile_motion', forbidden)
    with pytest.raises(DualArmError, match='initial state'):
        homing.plan_home(initial, scene.config, scene.models)


def test_retreat_never_returns_partial_plan(scene, monkeypatch):
    def blocked(*args, **kwargs):
        raise DualArmError('return leg blocked')
    monkeypatch.setattr(homing, 'joint_motion', blocked)
    with pytest.raises(DualArmError, match='No collision-checked home route'):
        homing.plan_home(start_above_home(scene), scene.config, scene.models)


def test_retreat_rejects_crossing_own_base_axis(scene):
    with pytest.raises(DualArmError, match='base axis'):
        homing.retreat_targets(start_above_home(scene), scene.config, scene.models, 300)
