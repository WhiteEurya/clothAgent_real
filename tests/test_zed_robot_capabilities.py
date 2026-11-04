import json
from pathlib import Path

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.robot_capabilities import build_capabilities


ROOT = Path(__file__).resolve().parents[1]


def test_current_zed_site_reports_both_real_xarm7_models():
    config = DualConfig.load_geometry(
        ROOT / 'config/dual_arm.zed_dual_xarm7_20261103.json',
        root=ROOT,
    )
    models = {key: ArmModel(arm) for key, arm in config.arms.items()}
    report = build_capabilities(config, models)
    assert {key: row['robot'] for key, row in report['arms'].items()} == {
        'left': 'xArm7', 'right': 'xArm7'
    }
    assert all(row['axis'] == 7 for row in report['arms'].values())


def test_current_zed_workspace_preserves_operator_selected_planes():
    config = DualConfig.load_geometry(
        ROOT / 'config/dual_arm.zed_dual_xarm7_20261103.json',
        root=ROOT,
    )
    assert all(a.workspace['min_mm'][0] == float('-inf') for a in config.arms.values())
    assert all(a.workspace['min_mm'][1] == float('-inf') for a in config.arms.values())
    assert config.arms['left'].workspace['max_mm'][0] == float('inf')
    assert config.arms['left'].raw['workspace']['halfspaces_mm']
    assert config.arms['right'].workspace['max_mm'][0] == 577.506409
    assert config.arms['right'].workspace['max_mm'][1] == 127.638359
    assert config.arms['left'].workspace['min_mm'][2] == 8
    assert config.arms['right'].workspace['min_mm'][2] == -20
    raw = config.raw
    assert raw['execution_chain'] == 'ExperimentRunner / RobotAPI / XArmBackend'
    assert 'static_arm_models' not in raw
    assert 'safety' not in raw
