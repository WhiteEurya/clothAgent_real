import json
from pathlib import Path

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.robot_capabilities import build_capabilities


ROOT = Path(__file__).resolve().parents[1]


def test_current_zed_site_reports_both_real_xarm7_models():
    config = DualConfig.load(
        ROOT / 'config/dual_arm.zed_dual_xarm7_20261103.json',
        root=ROOT,
        homing_only=True,
    )
    models = {key: ArmModel(arm) for key, arm in config.arms.items()}
    report = build_capabilities(config, models)
    assert {key: row['robot'] for key, row in report['arms'].items()} == {
        'left': 'xArm7', 'right': 'xArm7'
    }
    assert all(row['axis'] == 7 for row in report['arms'].values())


def test_current_zed_workspace_only_binds_measured_minimum_z():
    config = DualConfig.load(
        ROOT / 'config/dual_arm.zed_dual_xarm7_20261103.json',
        root=ROOT,
        homing_only=True,
    )
    assert all(a.workspace['min_mm'][0] == float('-inf') for a in config.arms.values())
    assert all(a.workspace['min_mm'][1] == float('-inf') for a in config.arms.values())
    assert all(a.workspace['max_mm'][0] == float('inf') for a in config.arms.values())
    assert all(a.workspace['max_mm'][1] == float('inf') for a in config.arms.values())
    assert config.arms['left'].workspace['min_mm'][2] == 8
    assert config.arms['right'].workspace['min_mm'][2] == -20
    raw = config.raw
    assert raw['execution_policy']['mode'] == 'sequential_single_gripper_static_other'
    assert set(raw['static_arm_models']) == {'left', 'right'}
