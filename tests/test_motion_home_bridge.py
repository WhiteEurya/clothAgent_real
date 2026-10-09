"""New path-planner/native-executor integration, without hardware access."""
import json
import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.motion.home import HomePlanner, HomeEndpointError, pack
from cloth_agent.dual_arm.sequential_home import execute_sequential_home, prepare_segments
from .test_dual_arm_runtime import scene, coordinator


def configured(scene):
    scene.config.execution_mode = 'controller_sequential'
    initial = {k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    for q in initial.values():
        q[2] += 5
    for k, connection in scene.connections.items():
        connection.joints = initial[k].copy()
    return initial, HomePlanner(initial, scene.config, scene.models)


def test_new_planner_reaches_native_executor_and_opens_both_grippers(scene):
    initial, planner = configured(scene)
    program, attempts = planner.plan()
    assert attempts[-1]['status'] == 'accepted'
    assert 'ompl' in attempts[-1]['strategy']
    segments = prepare_segments(program, scene.config, scene.models)
    assert [s.arm for s in segments] == ['left', 'right']
    c = coordinator(scene)
    try:
        result = execute_sequential_home(c, program, initial, segments)
        assert result['status'] == 'COMPLETED', result
        for key, connection in scene.connections.items():
            assert np.allclose(connection.joints, scene.config.arms[key].home_joints)
            assert [row[0] for row in connection.commands] == ['position', 'gripper']
    finally:
        c.close()


def test_bridge_rejects_two_arm_controller_edge(scene):
    initial, planner = configured(scene)
    goal = {k:a.home_joints for k,a in scene.config.arms.items()}
    with pytest.raises(DualArmError, match='at most one arm'):
        planner.paths.validator.certify(pack(initial), pack(goal))


def test_bridge_uses_installed_workspace_in_search(scene):
    initial, planner = configured(scene)
    outside = {k:q.copy() for k,q in initial.items()}
    outside['left'][2] = 700
    q = pack(outside)
    assert not planner.paths.checker.check(q[:6], q[6:])['safe']


def test_entrypoint_no_longer_calls_legacy_home_planner(scene, monkeypatch, tmp_path):
    from cloth_agent.dual_arm import homing, cli, kinematics, preview
    configured(scene)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(scene.config.raw))
    monkeypatch.setattr(homing, 'plan_home', lambda *args:pytest.fail('legacy planner selected'))
    monkeypatch.setattr(kinematics, 'ArmModel', lambda arm:scene.models[arm.arm_id])
    monkeypatch.setattr(cli, 'connect', lambda *args:scene.connections)
    monkeypatch.setattr(preview, 'write_preview', lambda *args:None)
    result = homing.gripper_home(path, simulated=True, output=tmp_path/'run', project_root=scene.config.root)
    assert result['status'] == 'COMPLETED'
    recorded = json.loads((tmp_path/'run'/'plan.json').read_text())
    assert recorded['planner'] == 'motion.PathPlanner/OMPL'
    assert recorded['controller_interpolation'] == 'native_position_nonblended'


def test_endpoint_reports_uncertainty_overlap_without_relaxing_execution(scene):
    initial, planner = configured(scene)
    initial['right'][1] = -100  # Tool surfaces are separated by 10 mm.
    scene.config.raw['safety']['status'] = 'measured'
    scene.config.limits['tracking_error_mm'] = 0
    scene.config.limits['tracking_error_deg'] = 0
    scene.config.limits['clearance_mm'] = 0
    for row in scene.config.raw['safety']['arms'].values():
        row['base_error_mm'] = 10
        row['geometry_error_mm'] = 0
    with pytest.raises(HomeEndpointError, match='nominal geometry is clear') as failure:
        planner.plan()
    diagnostics = failure.value.diagnostics
    assert diagnostics['endpoint'] == 'initial state'
    assert diagnostics['nominal_gap_mm'] == pytest.approx(10)
    assert diagnostics['combined_position_margin_mm'] == pytest.approx(20)
    assert all(row['base_error_mm'] == 10 for row in scene.config.raw['safety']['arms'].values())
    assert all(not c.commands for c in scene.connections.values())


def test_planning_failure_records_state_without_sending_stop(scene, monkeypatch, tmp_path):
    from cloth_agent.dual_arm import homing, cli, kinematics
    configured(scene)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(scene.config.raw))
    monkeypatch.setattr(kinematics, 'ArmModel', lambda arm:scene.models[arm.arm_id])
    monkeypatch.setattr(cli, 'connect', lambda *args:scene.connections)
    def reject(self):
        raise DualArmError('injected planning rejection')
    monkeypatch.setattr(HomePlanner, 'plan', reject)
    with pytest.raises(DualArmError, match='injected planning rejection'):
        homing.gripper_home(path, simulated=True, output=tmp_path/'run', project_root=scene.config.root)
    assert all(not c.stopped and not c.commands for c in scene.connections.values())
    assert (tmp_path/'run'/'initial_state.json').is_file()
    assert (tmp_path/'run'/'config.json').is_file()
    failure = json.loads((tmp_path/'run'/'failure.json').read_text())
    assert failure['execution_started'] is False


def test_failed_direct_route_still_tries_separation_outside_close_threshold(scene, monkeypatch):
    from cloth_agent.dual_arm import separation
    from cloth_agent.dual_arm.homing import joint_motion
    initial, planner = configured(scene)
    assert separation.minimum_gap(scene.models, initial) > 50
    separated = {k:q.copy() for k,q in initial.items()}
    separated['right'][1] += 2
    prefix = joint_motion(initial, separated, scene.config, scene.models, 'home_test_separate')
    monkeypatch.setattr(separation, 'separation_steps', lambda *args:iter([(separated,[prefix])]))
    original = planner.finish
    calls = []
    def finish(current, prefix, attempts):
        calls.append(len(prefix))
        return original(current, prefix, attempts) if prefix else None
    monkeypatch.setattr(planner, 'finish', finish)
    program, _ = planner.plan()
    assert calls == [0,1]
    assert program[0].phase.name == 'home_test_separate'


def test_search_edge_subdivision_covers_entire_joint_line(scene, monkeypatch):
    _, planner = configured(scene)
    accepted = []
    def box(a,b,**kwargs):
        if np.max(np.abs(b-a)) > .1:
            raise DualArmError('coarse proof exhausted its node budget')
        accepted.append((a.copy(),b.copy()))
    monkeypatch.setattr(planner.paths.validator,'certify_box',box)
    a=np.zeros(13); b=a.copy(); b[0]=.25
    planner.paths.validator.certify(a,b)
    assert len(accepted)==4
    assert np.array_equal(accepted[0][0],a)
    assert np.array_equal(accepted[-1][1],b)
    for previous,next_ in zip(accepted,accepted[1:]):
        assert np.array_equal(previous[1],next_[0])


def test_curved_reference_uses_full_boxes_not_diagonal_path_certificates(scene, monkeypatch):
    _, planner = configured(scene)
    monkeypatch.setattr(planner.paths.validator,'certify',lambda *a,**kw:pytest.fail('line certificate used for curve'))
    boxes=[]
    monkeypatch.setattr(planner.paths.validator,'certify_box',lambda a,b,**kw:boxes.append((a,b)))
    c=np.zeros((6,13));c[1,0]=.1;c[1,1]=.1;c[2,1]=-.1
    planner.paths.validator.certify_polynomial(c)
    for u in np.linspace(0,1,101):
        q=np.polynomial.polynomial.polyval(u,c)
        assert any(np.all(q>=lo-1e-10) and np.all(q<=hi+1e-10) for lo,hi in boxes)
