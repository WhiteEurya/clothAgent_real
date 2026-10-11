import numpy as np
import pytest

from cloth_agent.continuous_motion import compile_continuous
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.motion.trajectory import sample_segment,polynomial_peak


def row(a,b):
    return dict(kind='move',data=dict(arm='left',start={'left':a,'right':[0]*7},
        end={'left':b,'right':[0]*7},speed_deg_s=5,acceleration_deg_s2=10,timeout_s=30))


def test_continuous_velocities_boxes_and_dynamics():
    points=[[x,2*x,0,0,0,0] for x in [0,1,2,3]]
    result=compile_continuous([row(a,b) for a,b in zip(points,points[1:])])
    assert result['execution_authorized'] is False
    group=result['groups'][0]
    assert group['moving_internal_knots']==2
    prior=None
    for segment in group['segments']:
        first=sample_segment(segment,0);last=sample_segment(segment,segment['duration_s'])
        if prior is not None:
            for x,y in zip(prior,first):assert np.allclose(x,y,atol=1e-9)
        else:assert np.allclose(first[1:],0)
        c=np.array(segment['coefficients_rad']);duration=segment['duration_s']
        assert np.all(polynomial_peak(np.polynomial.polynomial.polyder(c))/duration<=np.array(group['velocity_limits_rad_s'])+1e-9)
        assert np.all(polynomial_peak(np.polynomial.polynomial.polyder(c,m=2))/duration**2<=np.array(group['acceleration_limits_rad_s2'])+1e-9)
        for t in np.linspace(0,duration,101):
            q=sample_segment(segment,t)[0]
            assert np.all(q>=np.minimum(first[0],last[0])-1e-10)
            assert np.all(q<=np.maximum(first[0],last[0])+1e-10)
            assert np.allclose(q[6:],0)
        prior=last
    assert np.allclose(prior[1:],0)


def test_gripper_events_force_stops_and_inactive_arm_rejected():
    event=dict(kind='gripper',data={'arm':'left','action':'close'})
    result=compile_continuous([row([0]*6,[1]*6),event,row([1]*6,[2]*6)])
    assert len(result['groups'])==3 and result['groups'][1]==event
    invalid=row([0]*6,[1]*6);invalid['data']['end']['right'][0]=1
    with pytest.raises(DualArmError):compile_continuous([invalid])


def test_joint_reversal_does_not_overshoot():
    result=compile_continuous([row([0]*6,[1]*6),row([1]*6,[0]*6)])
    group=result['groups'][0]
    assert group['moving_internal_knots']==0
    assert group['contained_in_original_joint_boxes']
