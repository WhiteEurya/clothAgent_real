import numpy as np
import pytest

from cloth_agent.dual_arm import pyroki_home
from cloth_agent.dual_arm.geometry import DualArmError
from .test_dual_arm_runtime import scene, coordinator


def test_sequential_checked_program_executes_from_actual_state(scene, monkeypatch):
    scene.config.execution_mode = 'controller_sequential'
    monkeypatch.setattr(pyroki_home, 'optimize_joint_path',
                        lambda model, start, goal: np.linspace(start, goal, 3))
    initial = {key: arm.home_joints.copy() for key, arm in scene.config.arms.items()}
    for key in initial:
        initial[key][2] += 5
        scene.connections[key].joints = initial[key].copy()
    program, segments, attempts = pyroki_home.plan_home(initial, scene.config, scene.models)
    assert attempts[-1]['status'] == 'accepted'
    assert segments
    previous = initial
    for segment in segments:
        for key in initial:
            np.testing.assert_allclose(segment.start[key], previous[key])
            if key != segment.arm:
                np.testing.assert_array_equal(segment.start[key], segment.end[key])
        previous = segment.end
    runtime = coordinator(scene)
    try:
        result = pyroki_home.execute_home(runtime)
        assert result['status'] == 'COMPLETED', result
        for key, arm in scene.config.arms.items():
            np.testing.assert_allclose(scene.connections[key].joints, arm.home_joints)
    finally:
        runtime.close()


def test_collision_refusal_never_reaches_execution(scene, monkeypatch):
    scene.config.execution_mode = 'controller_sequential'
    def reject(*args, **kwargs):
        raise DualArmError('collision with inactive arm')
    monkeypatch.setattr(pyroki_home, 'validate_sample', reject)
    monkeypatch.setattr(pyroki_home, 'execute_sequential_home',
                        lambda *args, **kwargs: pytest.fail('executed rejected path'))
    runtime = coordinator(scene)
    try:
        with pytest.raises(DualArmError, match='inactive arm'):
            pyroki_home.execute_home(runtime)
    finally:
        runtime.close()


def test_disabled_geometry_is_rejected(scene):
    scene.config.execution_mode = 'controller_sequential'
    scene.config.path_validation_enabled = False
    with pytest.raises(DualArmError, match='geometry checks enabled'):
        pyroki_home.plan_home({}, scene.config, scene.models)


def test_tilted_workspace_plane_covers_segment_volume(scene):
    from cloth_agent.dual_arm.sequential_home import certify_segment
    scene.config.execution_mode = 'controller_sequential'
    start = {key: arm.home_joints.copy() for key, arm in scene.config.arms.items()}
    end = {key: value.copy() for key, value in start.items()}
    end['left'][0] += 2
    center = (start['left'][:3] + end['left'][:3]) / 2
    normal = np.array([1., -1., 0.]) / np.sqrt(2)
    limit = float(normal @ center + .5)
    scene.config.arms['left'].raw['workspace']['halfspaces_mm'] = [[*normal, limit]]
    with pytest.raises(DualArmError, match='halfspace'):
        certify_segment(start, end, scene.config, scene.models)


def test_real_pyroki_optimizer_uses_current_seven_axis_model():
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.dual_arm.kinematics import ArmModel
    from cloth_agent.experiments.full_grasp_adapter import DEFAULT_CONFIG, ROOT
    config = DualConfig.load(DEFAULT_CONFIG, root=ROOT, homing_only=True)
    model = ArmModel(config.arms['right'])
    goal = config.arms['right'].home_joints
    start = goal.copy()
    start[0] += 2
    rows = pyroki_home.optimize_joint_path(model, start, goal, count=5)
    assert rows.shape == (5, 7)
    np.testing.assert_array_equal(rows[0], start)
    np.testing.assert_array_equal(rows[-1], goal)
    assert np.isfinite(rows).all()
