import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.codex_motion import parse_code
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime, route_grasp
from cloth_agent.single_arm_runtime import SingleArmRuntime


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


def test_execute_routes_both_turns_from_live_fk_and_keeps_graph_source(tmp_path, monkeypatch):
    runtime = object.__new__(PartitionedReplayRuntime)
    runtime._routing_lock = threading.Lock()
    runtime.config = SimpleNamespace(arms={k: SimpleNamespace(world_from_base=np.eye(4)) for k in ('left', 'right')})
    runtime.models = {k: SimpleNamespace(forward=lambda q: [*q[:3], 0, 0, 0]) for k in ('left', 'right')}
    observation = SimpleNamespace(meta={'joints_deg': {'left': [-300, 0, 100], 'right': [300, 0, 100]}})
    calls = []
    def execute(self, code, obs, folder):
        calls.append((self.arm_key, code))
        with pytest.raises(DualArmError, match='already running'):
            self.execute(source(0), obs, folder)
        return {'status': 'EXECUTED'}
    monkeypatch.setattr(SingleArmRuntime, 'execute', execute)
    for i, x in enumerate([-.2, .2]):
        result = runtime.execute(source(x), observation, tmp_path / str(i))
        assert result['arm_assignment']['active_arm'] == ('left' if x < 0 else 'right')
    assert [arm for arm, _ in calls] == ['left', 'right']
    assert json.loads((tmp_path / '0/arm_assignment.json').read_text())['midpoint_world_mm'] == [0, 0, 100]


def test_dispatch_rejects_other_arm_or_moving_other_joints(tmp_path):
    runtime = object.__new__(PartitionedReplayRuntime)
    runtime.arm_key, runtime.other_key = 'left', 'right'
    runtime.robot_name, runtime.other_robot_name = 'left xArm7', 'right xArm7'
    with pytest.raises(DualArmError, match='dispatcher rejects'):
        runtime._execute_operations([('gripper', {'arm': 'right'})], tmp_path)
    segment = SimpleNamespace(arm='left', start={'right': np.zeros(7)}, end={'right': np.ones(7)})
    with pytest.raises(DualArmError, match='stationary'):
        runtime._execute_operations([('move', segment)], tmp_path)


def test_current_config_loads_both_live_models_without_old_arm6_pose(tmp_path):
    runtime = PartitionedReplayRuntime('config/dual_arm.zed_dual_xarm7_20261103.json', planning_only=True)
    try:
        assert runtime.active_arms == ('left', 'right')
        assert runtime.config.static_arm_obstacles == {}
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
    assert calls == [path] and runtime.preview == 'preview'
