from types import SimpleNamespace
import numpy as np
import pytest
from cloth_agent.codex_motion import parse_code, compile_code

@pytest.mark.parametrize('code', ['import os', 'motion.move_joints(left=get_joints())', 'exec("x")',
 'motion.gripper(arm="both", action="open")', 'motion.move_joints(left=[0])'])
def test_invalid_code_never_executes(code):
    with pytest.raises((ValueError,SyntaxError)):
        parse_code(code)


def test_blocked_is_not_a_trajectory():
    assert not compile_code('motion.blocked(reason="no image")',None,{})['success']


def test_code_calls_existing_path_and_timing_planner(monkeypatch):
    from cloth_agent.dual_arm.motion import trajectory
    checked=[]
    def path(a,b,**kw):
        checked.append((a,b,kw));return [a,b]
    models={k:SimpleNamespace(names=list(range(n)),urdf=SimpleNamespace(joint_map={
        i:SimpleNamespace(limit=SimpleNamespace(velocity=1)) for i in range(n)})) for k,n in [('left',6),('right',7)]}
    planner=SimpleNamespace(checker=SimpleNamespace(check=lambda *a:{'safe':True}),
        paths=SimpleNamespace(joint_path=path),scene=SimpleNamespace(models=models),validator=object())
    monkeypatch.setattr(trajectory,'validate_timed',lambda timed,validator:{'checked':True})
    result=compile_code('motion.move_joints(left=[0.1,0,0,0,0,0])\nmotion.gripper(arm="left",action="close")',
                        planner,dict(units='rad',left=[0]*6,right=[0]*7))
    assert result['success'] and not result['execution_authorized']
    assert len(checked)==1
    assert result['events'][0]['waypoint']==1
    assert result['verification']=={'checked':True}
    assert np.array_equal(checked[0][1][6:],[0]*7)
