"""Sequential-leg validity and budget checks before entering OMPL."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.motion.path import PathPlanner


def planner(check, certify):
    models={k:SimpleNamespace(lower=np.full(n,-2.),upper=np.full(n,2.))
            for k,n in [('left',6),('right',7)]}
    return PathPlanner(SimpleNamespace(models=models),SimpleNamespace(check=check),
                       SimpleNamespace(certify=certify))


def test_invalid_sequential_goal_rejected_before_edge_proof_or_ompl():
    def check(a,b):
        return dict(safe=not(a[0]>.5 and b[0]<.5),reason='camera pair overlap')
    p=planner(check,lambda *a,**kw:pytest.fail('proof attempted for invalid leg target'))
    with pytest.raises(DualArmError,match='path leg goal is unsafe: camera pair overlap'):
        p.joint_path(np.zeros(13),np.ones(13),order='left_first')


def test_other_arm_order_remains_available():
    calls=[]
    def check(a,b):
        return dict(safe=not(a[0]>.5 and b[0]<.5),reason='camera pair overlap')
    p=planner(check,lambda a,b,**kw:calls.append((a.copy(),b.copy())))
    path=p.joint_path(np.zeros(13),np.ones(13),order='right_first')
    assert len(path)==3 and len(calls)==2
    assert np.array_equal(path[1][:6],np.zeros(6))
    assert np.array_equal(path[1][6:],np.ones(7))


def test_expired_budget_is_not_reported_as_invalid_goal():
    p=planner(lambda *a:pytest.fail('state checked after known expiry'),
              lambda *a,**kw:pytest.fail('proof after expiry'))
    with pytest.raises(DualArmError,match='time budget exhausted'):
        p._rrt(np.zeros(13),np.ones(13),slice(0,13),time.monotonic()-1)


def test_direct_proof_reserves_budget_for_search():
    deadlines=[]
    p=planner(lambda *a:dict(safe=True),lambda *a,**kw:deadlines.append(kw['deadline']))
    began=time.monotonic()
    p._rrt(np.zeros(13),np.ones(13),slice(0,13),began+20)
    assert 4.9 < deadlines[0]-began < 5.1
