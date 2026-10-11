"""Offline integration checks; never connect to an arm or camera."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.geometry import DualArmError, pose_matrix, pose_error
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.pyroki_ik import PyrokiIK

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module',params=['left','right'])
def model(request):
    config=DualConfig.load(ROOT/'config/dual_arm.local.json',root=ROOT,homing_only=True)
    arm=deepcopy(config.arms[request.param])
    # Exercise all six TCP offset components, not just the current Z offset.
    arm.tcp_offset=np.array([12.,-9.,172.,4.,-7.,11.])
    result=ArmModel(arm)
    result._pyroki_ik=PyrokiIK(result)
    return result


def test_pyroki_fk_matches_original_base_tcp_and_joint_order(model):
    rng=np.random.default_rng(731)
    for _ in range(12):
        q=np.degrees(rng.uniform(model.lower,model.upper))
        assert np.allclose(model._pyroki_ik.forward(q),pose_matrix(model.forward(q)),atol=1e-8)
    assert model._pyroki_ik.robot.joints.num_actuated_joints==model.config.axis


def test_reachable_targets_and_seed_continuity(model):
    q=model.config.home_joints.copy()
    rng=np.random.default_rng(12)
    for _ in range(5):
        intended=np.clip(q+rng.uniform(-.5,.5,len(q)),np.degrees(model.lower),np.degrees(model.upper))
        target=model.forward(intended)
        result=model.inverse(target,q)
        distance,angle=pose_error(model.forward(result),target)
        assert distance<=.5 and angle<=.2
        assert np.max(np.abs(result-q))<3
        q=result


@pytest.mark.parametrize('kind',['nan','wrong_shape','outside_limits','wrong_pose'])
def test_original_model_rejects_bad_solver_output(model,monkeypatch,kind):
    q=model.config.home_joints.copy();target=model.forward(q)
    if kind=='nan':q[:]=np.nan
    elif kind=='wrong_shape':q=q[:-1]
    elif kind=='outside_limits':q[0]=np.degrees(model.upper[0])+1
    else:q[0]+=10
    monkeypatch.setattr(model,'_pyroki_ik',SimpleNamespace(solve=lambda *args:q))
    with pytest.raises(DualArmError,match='PyRoki IK rejected'):
        model.inverse(target,model.config.home_joints)


def test_impossible_target_still_rejected(model):
    with pytest.raises(DualArmError,match='PyRoki IK rejected'):
        model.inverse([10000,10000,10000,0,0,0],model.config.home_joints)


def test_fast_budget_retains_pose_acceptance_and_reports_failure(model):
    original=model._pyroki_ik
    try:
        model.ik_max_iterations=40;model.ik_seed_limit=2
        model._pyroki_ik=PyrokiIK(model)
        q=model.config.home_joints.copy()
        target=model.forward(q+np.full_like(q,.4))
        result=model.inverse(target,q)
        distance,angle=pose_error(model.forward(result),target)
        assert distance<=.5 and angle<=.2
        with pytest.raises(DualArmError) as caught:
            model.inverse([10000,10000,10000,0,0,0],q)
        d=caught.value.ik_diagnostics
        assert d['seed_count']<=2 and d['max_iterations_per_seed']==40
        assert d['attempts'] and d['solve_elapsed_s']>=0
        assert 'does not prove' in d['explanation']
    finally:
        model._pyroki_ik=original
        del model.ik_max_iterations,model.ik_seed_limit
