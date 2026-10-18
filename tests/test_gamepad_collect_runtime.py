import json
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.gamepad_collect_runtime import CollectionRuntime
from cloth_agent.single_arm_runtime import SingleArmRuntime
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.static_arm import StaticArmObstacle
from tests.test_dual_arm_runtime import scene


def configure(scene, monkeypatch, tmp_path):
    scene.config.root = tmp_path
    scene.config.raw['camera'].update(mount_arm='left', serial='317222073552')
    path = tmp_path/'data/robot/xarm7_perception_pose.json'
    path.parent.mkdir(parents=True)
    arm = scene.config.arms['right']
    record = dict(robot_ip=arm.ip, axis=arm.axis, control_box_sn=arm.serial,
                  joint_angles_deg=arm.home_joints.tolist())
    path.write_text(json.dumps(record))
    monkeypatch.setattr(SingleArmRuntime, 'load_config', lambda *a: scene.config)
    runtime = object.__new__(CollectionRuntime)
    return runtime, path


def test_collection_never_connects_arm7_even_when_offline(scene, monkeypatch, tmp_path):
    from cloth_agent.dual_arm import cli
    runtime, _ = configure(scene, monkeypatch, tmp_path)
    config = runtime.load_config(None)
    opened = []
    class Connection:
        def __init__(self, arm, *args, **kwargs):
            assert arm.arm_id == 'left', 'Arm7 must never be contacted'
            opened.append(arm.arm_id)
        def disconnect(self): pass
    monkeypatch.setattr(cli, 'XArmConnection', Connection)
    connections = cli.connect(config, scene.models, scene.cancel, True)
    assert opened == ['left']
    assert isinstance(connections['right'], StaticArmObstacle)
    assert connections['right'].snapshot()['live_feedback'] is False
    assert connections['right'].stop()['status'] == 'NOT_CONNECTED'
    with pytest.raises(DualArmError, match='cannot receive'):
        connections['right'].move_joint_position()


def test_saved_geometry_identity_is_validated(scene, monkeypatch, tmp_path):
    runtime, path = configure(scene, monkeypatch, tmp_path)
    record = json.loads(path.read_text()); record['robot_ip'] = 'wrong'
    path.write_text(json.dumps(record))
    with pytest.raises(DualArmError, match='identity mismatch'):
        runtime.load_config(None)


def direct_runtime():
    import threading
    runtime = object.__new__(CollectionRuntime)
    runtime.cancel = threading.Event()
    def forbidden(*args, **kwargs):
        pytest.fail('collection must never call RRT joint_path')
    runtime.paths = SimpleNamespace(joint_path=forbidden)
    runtime._join_moves = lambda operations, boundaries: operations
    return runtime


def test_collection_joint_moves_validate_direct_path_without_rrt():
    runtime = direct_runtime()
    initial = dict(left=np.zeros(6), right=np.zeros(7))
    checked = []
    def segments(start, end):
        checked.append((start, end)); return ['direct_segment']
    runtime._segments = segments
    result = runtime.plan('motion.move_joints(left=[0.1,0,0,0,0,0])', initial,
                          preserve_command_boundaries=True)
    assert result == [('move', 'direct_segment')]
    assert checked[0][1]['left'][0] == pytest.approx(np.degrees(.1))
    assert checked[0][1]['right'] == pytest.approx(initial['right'])


def test_direct_path_failure_does_not_fall_back_to_rrt():
    runtime = direct_runtime()
    def blocked(*args): raise DualArmError('direct path blocked')
    runtime._segments = blocked
    with pytest.raises(DualArmError, match='direct path blocked'):
        runtime.plan('motion.move_joints(left=[0.1,0,0,0,0,0])',
                     dict(left=np.zeros(6), right=np.zeros(7)),
                     preserve_command_boundaries=True)


def test_collection_tcp_forces_no_detours(monkeypatch):
    runtime = direct_runtime()
    seen = []
    def tcp(self, k, pose, q, *, allow_detour):
        seen.append(allow_detour); raise DualArmError('IK failed')
    monkeypatch.setattr(SingleArmRuntime, '_tcp_segments', tcp)
    with pytest.raises(DualArmError, match='IK failed'):
        runtime._tcp_segments('left', [], {}, allow_detour=True)
    assert seen == [False]


def test_arm6_collection_uses_arm7_shared_runtime_and_collision_policy():
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.codex_arm6_runtime import Arm6Runtime
    from cloth_agent.dual_arm.safety import collision_envelopes_enabled
    arm6 = CollectionRuntime('config/dual_arm.local.json')
    arm7 = Arm7Runtime('config/dual_arm.local.json')
    try:
        assert not isinstance(arm6, Arm6Runtime)
        for name in ('plan', '_segments', '_inverse_target', '_execute_operations',
                     'adjust_motion'):
            assert getattr(type(arm6), name) is getattr(type(arm7), name)
        assert arm6.grasp_offset_mm == arm7.grasp_offset_mm == 0
        assert arm6.staged_release == arm7.staged_release is True
        assert arm6.parallel_jaw_grasp_symmetry == arm7.parallel_jaw_grasp_symmetry is True
        assert collision_envelopes_enabled(arm6.config, 'left') == collision_envelopes_enabled(arm7.config, 'right')
        assert arm6.models['left'].config.axis == 6
        assert arm6.config.raw['camera']['mount_arm'] == 'left'
        assert arm6.direct_joint_moves
    finally:
        arm6.close(); arm7.close()


def test_arm6_shared_wrist_policy_preserves_real_six_axis_tcp(monkeypatch):
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm.geometry import pose_error
    runtime = CollectionRuntime('config/dual_arm.local.json')
    runtime.planning_context = {}
    model = runtime.models['left']
    seed = model.config.home_joints.copy(); seed[-1] = 0
    target_joints = seed.copy(); target_joints[-1] = -45
    target = model.forward(target_joints)
    monkeypatch.setattr(NativeRuntime, '_inverse_target', lambda *a: target_joints.copy())
    try:
        solution = runtime._inverse_target('left', target, seed, 'host_approach_ik')
        assert len(solution) == 6
        distance, angle = pose_error(model.forward(solution), target)
        assert distance < 1e-6 and angle < 1e-6
        assert runtime.planning_context['wrist_winding']['selected_deg'] == pytest.approx(-45)
    finally:
        runtime.close()


def test_shared_arm6_dispatch_rejects_hidden_arm7_motion(tmp_path):
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    runtime = direct_runtime()
    a = dict(left=np.zeros(6), right=np.zeros(7))
    b = dict(left=np.ones(6), right=np.ones(7))
    segment = PositionSegment('left', a, b, 5, 10, 30)
    with pytest.raises(DualArmError, match='keep arm7 stationary'):
        runtime._execute_operations([('move', segment)], tmp_path)


def test_home_perception_skips_path_validation_but_keeps_limits(monkeypatch):
    runtime = CollectionRuntime('config/dual_arm.local.json')
    def forbidden(*a, **kw): pytest.fail('Home/perception must not validate or search paths')
    monkeypatch.setattr(runtime, 'plan', forbidden)
    monkeypatch.setattr(runtime, '_segments', forbidden)
    monkeypatch.setattr(runtime.paths, 'joint_path', forbidden)
    monkeypatch.setattr(runtime.paths.validator, 'certify_box', forbidden)
    initial = {k: a.home_joints.copy() for k, a in runtime.config.arms.items()}
    initial['left'][0] += 1
    target = runtime.config.arms['left'].home_joints.copy(); target[0] += 2
    try:
        _, source = runtime._preparation_program(initial, 'left', target)
        operations = runtime._plan_preparation(source, initial)
        assert [kind for kind, _ in operations] == ['gripper', 'move', 'move']
        assert operations[-1][1].end['left'] == pytest.approx(target)
        assert operations[-1][1].end['right'] == pytest.approx(initial['right'])
        assert runtime.plan_stats['path_validation'] == 'disabled_by_user'
        with pytest.raises(DualArmError, match='joint limits'):
            runtime._plan_preparation('motion.move_joints(left=[100,0,0,0,0,0])', initial)
        with pytest.raises(DualArmError, match='only accepts'):
            runtime._plan_preparation('motion.gripper(arm="right",action="open")', initial)
    finally:
        runtime.close()


def test_all_collection_stages_skip_geometric_proofs(monkeypatch):
    from cloth_agent.dual_arm.planning import validate_sample
    from cloth_agent.dual_arm.geometry import matrix_pose, pose_matrix
    from cloth_agent.gamepad_collect import tcp_command
    runtime = CollectionRuntime('config/dual_arm.local.json')
    def forbidden(*a, **kw): pytest.fail('geometric proof must not run in collection')
    monkeypatch.setattr(runtime.paths.validator, 'certify_box', forbidden)
    monkeypatch.setattr(runtime.paths, 'joint_path', forbidden)
    monkeypatch.setattr('cloth_agent.codex_native_runtime.certify_segment', forbidden)
    monkeypatch.setattr('cloth_agent.dual_arm.safety.check_state', forbidden)
    initial = {k: a.home_joints.copy() for k, a in runtime.config.arms.items()}
    goal = initial['left'].copy(); goal[0] += .1
    world = matrix_pose(runtime.config.arms['left'].world_from_base @
                        pose_matrix(runtime.models['left'].forward(goal)))
    monkeypatch.setattr(runtime, '_inverse_target', lambda *a: goal.copy())
    source = ('motion.gripper(arm="left",action="open")\n'
              + f'motion.move_joints(left={np.radians(goal).tolist()})\n'
              + tcp_command(world))
    try:
        operations = runtime.plan(source, initial, preserve_command_boundaries=True)
        assert operations and runtime.plan_stats['scope'] == 'all_collection'
        validate_sample(runtime.config, runtime.models, initial, holding=False)
        invalid = {k: v.copy() for k, v in initial.items()}; invalid['left'][0] = 10000
        with pytest.raises(DualArmError, match='joint'):
            runtime._segments(initial, invalid)
        with pytest.raises(DualArmError, match='joint'):
            validate_sample(runtime.config, runtime.models, invalid, holding=False)
    finally:
        runtime.close()


def test_startup_recovers_before_already_at_perception_branch(monkeypatch):
    runtime = direct_runtime(); runtime.prepared = False
    order = []
    runtime.coordinator = SimpleNamespace(connections={'left': SimpleNamespace(
        prepare_position=lambda: order.append('recover'))})
    monkeypatch.setattr(SingleArmRuntime, 'move_to_perception', lambda *a: order.append('perception'))
    runtime.move_to_perception('unused')
    assert order == ['recover', 'perception']
    assert runtime.prepared


def test_cartesian_ik_waypoints_become_one_controller_line(monkeypatch):
    from types import SimpleNamespace
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm.sequential_home import PositionSegment
    from cloth_agent.queued_motion import CartesianPositionSegment
    runtime = object.__new__(CollectionRuntime)
    runtime.config = SimpleNamespace(limits=dict(cartesian_speed_mm_s=10,
        cartesian_accel_mm_s2=30), arms={'left':SimpleNamespace(world_from_base=np.eye(4))})
    runtime.models = {'left': SimpleNamespace(forward=lambda q: np.asarray(q))}
    states = [dict(left=np.array([x,0,0,0,0,0.]), right=np.zeros(7)) for x in (0,5,10)]
    segments = [PositionSegment('left', a,b,5,10,30) for a,b in zip(states,states[1:])]
    def continuation(*args, **kw):
        assert kw['allow_detour'] is False
        return segments, states[-1]
    monkeypatch.setattr(NativeRuntime, '_tcp_segments', continuation)
    result, final = runtime._tcp_segments('left',[.01,0,0,0,0,0],states[0])
    assert len(result) == 1 and isinstance(result[0], CartesianPositionSegment)
    assert result[0].segments == tuple(segments)
    assert result[0].target_pose == [10,0,0,0,0,0]
    assert result[0].cartesian_speed == 5
    assert result[0].cartesian_acceleration == 30
    assert final is states[-1]


def test_linear_controller_command_has_only_one_endpoint_and_waits():
    import threading
    from types import SimpleNamespace
    from cloth_agent.dual_arm.execution import XArmConnection
    connection = object.__new__(XArmConnection)
    connection.cancel = threading.Event()
    calls = []
    connection.arm = SimpleNamespace(mode=0,
        set_position=lambda *args, **kw: calls.append((args,kw)) or 0)
    connection._check = lambda label, code: None
    connection.move_tcp_position([100,200,300,180,0,0], 10,30,60)
    assert calls == [((100,200,300,180,0,0),dict(speed=10,mvacc=30,
        is_radian=False,wait=True,timeout=60,radius=-1))]
    assert connection.position_command_active is False


def test_return_opens_only_after_home_arrival():
    runtime = CollectionRuntime('config/dual_arm.local.json')
    try:
        runtime.open_at_home_on_return = True
        initial = {k:a.home_joints.copy() for k,a in runtime.config.arms.items()}
        _, source = runtime._preparation_program(initial, 'left', initial['left'])
        from cloth_agent.codex_motion import parse_code
        calls = parse_code(source)
        assert [name for name, _ in calls] == ['move_joints', 'gripper', 'move_joints']
        assert calls[0][1]['left'] == pytest.approx(np.radians(runtime.config.arms['left'].home_joints))
        assert calls[1][1] == dict(arm='left', action='open')
    finally:
        runtime.close()
