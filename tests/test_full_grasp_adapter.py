import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cloth_agent import experiment, robot_api
from cloth_agent.experiments.full_grasp_adapter import (
    DEFAULT_CONFIG, compile_proposal, main, validate_offline,
)
from cloth_agent.site_observation import load_site


PROPOSALS = Path('deployment-artifacts/topdown_physical_trial_001/proposals_v2')
STATE = Path('outputs/onsite_acceptance_20261104_01/robot_readonly_host.json')


@pytest.fixture
def inputs():
    return (json.loads((PROPOSALS / 'primary_plan.json').read_text()), load_site(),
            json.loads(DEFAULT_CONFIG.read_text()), json.loads(STATE.read_text()))


def test_complete_real_proposal_uses_original_executor_without_hardware(tmp_path, monkeypatch, inputs):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline integration attempted hardware access')
    monkeypatch.setattr(experiment, 'XArmBackend', forbidden)
    monkeypatch.setattr(experiment, 'validate_controller_trajectory', forbidden)
    monkeypatch.setattr(robot_api, 'XArmBackend', forbidden)
    monkeypatch.setattr(robot_api, 'validate_controller_trajectory', forbidden)
    config_before = DEFAULT_CONFIG.read_bytes()
    report = validate_offline(PROPOSALS, STATE, tmp_path / 'run')
    result = json.loads((tmp_path / 'run/results/fold_sleeve.json').read_text())
    assert report['status'] == 'SIMULATED_EXECUTION_COMPLETE'
    assert result['preflight_completed'] and result['execution_completed']
    assert result['physical_execution'] is False
    assert report['robot_connections'] == 0
    assert report['action_count'] == 9
    assert report['original_execution_config_check']['status'] == 'PASS'
    assert report['controller_path_check']['status'] == 'NOT_RUN'
    plan = inputs[0]
    for operation, actual in zip(plan['operations'], result['actual_robot_actions'], strict=True):
        assert actual['success'] is True
        if operation['kind'] == 'move_tcp':
            assert actual['name'] == 'move'
            np.testing.assert_allclose(actual['actual_ee_pose'], operation['pose_base_mm_deg'], atol=1e-9)
        else:
            assert actual['name'] == operation['action'] + '_gripper'
    assert DEFAULT_CONFIG.read_bytes() == config_before


@pytest.mark.parametrize('change', ['speed', 'acceleration', 'pose', 'gripper', 'arm', 'identity'])
def test_incompatible_inputs_are_not_silently_rewritten(inputs, change):
    plan, profile, config, state = copy.deepcopy(inputs)
    move = next(op for op in plan['operations'] if op['kind'] == 'move_tcp')
    if change == 'speed': move['speed_mm_s'] = 5
    if change == 'acceleration': move['acceleration_mm_s2'] = 5
    if change == 'pose':
        move = next(op for op in plan['operations'] if op['phase'] == 'grasp')
        move['pose_base_mm_deg'][4] = 20
        move['pose_common_mm_deg'][4] = 20
    if change == 'gripper': plan['operations'][0]['position_pulse'] = 700
    if change == 'arm': plan['operations'][0]['arm'] = 'left'
    if change == 'identity': state['arms']['left']['robot_sn'] = 'wrong'
    with pytest.raises(ValueError):
        compile_proposal(plan, profile, config, state)


def test_original_preflight_workspace_rejection_is_preserved(tmp_path, inputs):
    plan, profile, execution, state = copy.deepcopy(inputs)
    execution['arms']['right']['workspace']['min_mm'][2] = 20
    config, source, mapping = compile_proposal(plan, profile, execution, state)
    runner = experiment.ExperimentRunner(tmp_path / 'run', config)
    path = runner.workspace / 'fold.py'
    path.write_text(source)
    result = runner.preflight(path)
    assert result.error is not None
    assert 'below the safe lower bound' in result.error
    assert not any(action['name'] == 'close_gripper' for action in result.actions)


def test_native_gates_are_not_promoted_or_required(inputs):
    plan, profile, execution, state = inputs
    before = copy.deepcopy(execution)
    config, source, mapping = compile_proposal(plan, profile, execution, state)
    assert execution == before
    assert 'safety' not in execution
    assert 'collision_geometry_verified' not in execution
    assert execution['geometry_provenance']['collision_geometry_verified'] is False
    assert config.speed_mm_s == execution['limits']['cartesian_speed_mm_s']
    assert config.acceleration_mm_s2 == execution['limits']['cartesian_accel_mm_s2']
    assert config.gripper_close == 5
    assert len(mapping) == 9


def test_no_real_execution_option():
    with pytest.raises(SystemExit):
        main(['--real'])


@pytest.mark.parametrize('arm',['left','right'])
def test_formal_dual_wrist_config_enables_tested_continuous_stream(arm):
    from cloth_agent.experiments.full_grasp_adapter import build_robot_config
    path=Path('config/dual_arm.dual_wrist_xarm7_20261110.json')
    raw=json.loads(path.read_text())
    config=build_robot_config(raw,{},arm,180,0,config_path=path)
    assert config.continuous_joint_stream is True
    assert config.home_speed_deg_s==20
    assert config.home_acceleration_deg_s2==40
    assert config.joint_queue_prefill_s==0
    config.validate_for_real()
    raw['limits']['continuous_joint_stream']=False
    assert not build_robot_config(raw,{},arm,180,0,config_path=path).continuous_joint_stream
    raw['limits']['continuous_joint_stream']='false'
    with pytest.raises(ValueError,match='boolean'):
        build_robot_config(raw,{},arm,180,0,config_path=path)
