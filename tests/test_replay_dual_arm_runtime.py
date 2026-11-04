import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime, route_grasp


def source(x, y=0.):
    return (f'motion.grasp(right={[x, y, .02, 0., 0., 0.]})\n'
            'motion.move_tcp(right=[0.9, 0.2, 0.03, 0., 0., 1.])\n')


@pytest.mark.parametrize('x,arm', [(-.2, 'left'), (.2, 'right'), (0., 'right')])
def test_route_and_crossing_target(x, arm):
    routed, audit = route_grasp(source(x), {'left': [-300, 0, 150], 'right': [300, 0, 100]})
    assert audit['active_arm'] == arm
    original, actual = parse_code(source(x), allow_grasp=True), parse_code(routed, allow_grasp=True)
    assert all(set(args) == {arm} for _, args in actual)
    assert [args[arm] for _, args in actual] == [args['right'] for _, args in original]


def test_rotated_and_translated_tcp_boundary():
    # Physical left/right are along Y here, not image X or world X.
    _, audit = route_grasp(source(.8, .1), {'left': [800, 0, 100], 'right': [800, 400, 100]})
    assert audit['active_arm'] == 'left'
    _, moved = route_grasp(source(.8, .1), {'left': [800, -400, 100], 'right': [800, 0, 100]})
    assert moved['active_arm'] == 'right'


def test_reject_ambiguous_tcp_and_dual_grasp():
    with pytest.raises(DualArmError, match='indistinguishable'):
        route_grasp(source(0), {'left': [0, 0, 100], 'right': [0, 0, 200]})
    with pytest.raises(DualArmError, match='logical right'):
        route_grasp(source(0).replace('right=', 'left='), {'left': [-300, 0, 100], 'right': [300, 0, 100]})


def test_current_config_loads_both_live_models_without_old_arm6_pose(tmp_path):
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json', planning_only=True)
    try:
        assert runtime.active_arms == ('left', 'right')
        assert not hasattr(runtime.config, "static_arm_obstacles")
        assert runtime.config.execution_mode == "geometry_only"
        assert all(a.axis == 7 for a in runtime.config.arms.values())
        assert runtime.config.raw['camera']['backend'] == 'zed'
        with pytest.raises(DualArmError, match='cannot execute'):
            runtime.execute('', None, tmp_path)
    finally:
        runtime.close()


def test_factory_selects_partitioned_runtime_for_current_site(monkeypatch):
    from cloth_agent.experiments.graph_policy import make_runtime
    calls = []
    runtime = SimpleNamespace(config=object())
    def create(config):
        calls.append(config)
        return runtime
    monkeypatch.setattr('cloth_agent.replay_dual_arm_runtime.PartitionedReplayRuntime', create)
    monkeypatch.setattr('cloth_agent.trajectory_preview.TrajectoryPreview', lambda *a, **k: 'preview')
    path = 'config/dual_arm.zed_dual_xarm7_20261103.json'
    assert make_runtime(path, 8088) is runtime
    assert calls == [path]


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_graph_compiles_to_original_experiment_api_for_either_arm(tmp_path, arm):
    from cloth_agent.dual_arm.geometry import matrix_pose, pose_matrix
    from cloth_agent.experiment import ExperimentRunner
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json', planning_only=True)
    joints = {k: a.home_joints.copy() for k, a in runtime.config.arms.items()}
    pose = matrix_pose(runtime.config.arms[arm].world_from_base @ pose_matrix(runtime.models[arm].forward(joints[arm])))
    grasp = np.r_[pose[:3] / 1000, np.radians(pose[3:])].tolist()
    target = grasp.copy(); target[2] += .02
    source = f'motion.grasp(right={grasp!r})\nmotion.move_tcp(right={target!r})\n'
    config, program, assignment = runtime.compile(source, SimpleNamespace(meta={'joints_deg': joints}))
    assert assignment['active_arm'] == arm
    assert config.robot_ip == runtime.config.arms[arm].ip
    assert config.collision_scene_path
    runner = ExperimentRunner(tmp_path / arm, config)
    path = runner.workspace / 'action.py'; path.write_text(program)
    preflight = runner.preflight(path)
    assert preflight.error is None
    assert [a['name'] for a in preflight.actions] == ['open_gripper', 'move', 'move', 'close_gripper',
                                                    'move', 'open_gripper', 'move', 'home']
    np.testing.assert_allclose(preflight.actions[2]['actual_ee_pose'][:3], runtime.models[arm].forward(joints[arm])[:3])
    runtime.close()
