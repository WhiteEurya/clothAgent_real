"""Offline regression for the installed 1305 SDK / factory URDF mismatch."""
from pathlib import Path
import numpy as np
import pytest
from xarm.core.config.x_config import XCONF
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.robot_collision import RobotCollision
from cloth_agent.config import SafetyError

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('side',['left','right'])
def test_current_arm_sdk_limits_apply_to_model_and_path(side):
    config=DualConfig.load_geometry(ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json',root=ROOT)
    model=ArmModel(config.arms[side]);limits=np.asarray(XCONF.Robot.JOINT_LIMITS[7][71305])
    assert np.all(model.lower>=limits[:,0]) and np.all(model.upper<=limits[:,1])
    assert model.lower[3]==pytest.approx(-.10472)
    for i,name in enumerate(model.names):
        assert model.urdf.joint_map[name].limit.lower==model.lower[i]
        assert model.urdf.joint_map[name].limit.upper==model.upper[i]
    q=model.config.home_joints.copy();q[3]=-7.138075
    with pytest.raises(DualArmError,match='joint limits'):model.forward(q)
    checker=object.__new__(RobotCollision);checker.models={side:model}
    with pytest.raises(SafetyError,match='joint limits'):checker.joints({side:q})
    q[3]=-6.
    assert np.isfinite(model.forward(q)).all()
