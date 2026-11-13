import numpy as np
import pytest
from cloth_agent.checked_joint_trajectory import compile_trajectory
from cloth_agent.config import SafetyError


def test_continuous_curve_stays_in_checked_boxes_and_respects_dynamics():
    points=np.array([[0,0,0,0,0,0,0],[2,1,-1,0,3,0,0],
                     [4,2,-2,0,2,0,0],[5,3,-3,0,0,0,0]],float)
    c=compile_trajectory(points,20,40)
    assert c['moving_internal_knots']==2
    previous=None
    for i,(coeff,t) in enumerate(zip(c['coefficients'],c['segment_durations_s'])):
        coeff=np.array(coeff);u=np.linspace(0,1,10001)
        q=np.polynomial.polynomial.polyval(u,coeff).T
        v=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coeff)).T/t
        a=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coeff,2)).T/t**2
        assert np.all(q>=np.minimum(points[i],points[i+1])-1e-8)
        assert np.all(q<=np.maximum(points[i],points[i+1])+1e-8)
        assert np.max(np.abs(v))<=20+1e-8
        assert np.max(np.abs(a))<=40+1e-8
        if previous is not None:np.testing.assert_allclose(v[0],previous,atol=1e-8)
        np.testing.assert_allclose(a[[0,-1]],0,atol=1e-8)
        if i==0:np.testing.assert_allclose(v[0],0,atol=1e-8)
        previous=v[-1]
    np.testing.assert_allclose(previous,0,atol=1e-8)
    np.testing.assert_allclose(c['positions_deg'][0],points[0])
    np.testing.assert_allclose(c['positions_deg'][-1],points[-1],atol=1e-8)


@pytest.mark.parametrize('points', [[[0]*7,[float('nan')]*7], [[0]*7,[0]*7]])
def test_invalid_path_is_rejected(points):
    with pytest.raises(SafetyError):compile_trajectory(points,20,40)


def test_local_timing_handles_uneven_segments_and_reversals():
    rng=np.random.default_rng(741)
    for _ in range(12):
        points=np.cumsum(rng.normal(size=(9,7))*rng.uniform(.01,10,size=(9,1)),axis=0)
        c=compile_trajectory(points,20,40)
        previous=None
        for i,(coeff,t) in enumerate(zip(c['coefficients'],c['segment_durations_s'])):
            coeff=np.asarray(coeff);u=np.linspace(0,1,2001)
            q=np.polynomial.polynomial.polyval(u,coeff).T
            v=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coeff)).T/t
            a=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coeff,2)).T/t**2
            assert np.max(np.abs(v))<=20+1e-7
            assert np.max(np.abs(a))<=40+1e-7
            assert np.all(q>=np.minimum(points[i],points[i+1])-1e-7)
            assert np.all(q<=np.maximum(points[i],points[i+1])+1e-7)
            if previous is not None:np.testing.assert_allclose(v[0],previous,atol=1e-7)
            previous=v[-1]


def test_local_timing_avoids_stop_at_each_straight_waypoint():
    # A straight 20-degree traverse should not pay a fresh acceleration ramp
    # for each of its ten subdivisions.
    points=np.zeros((11,7));points[:,0]=np.arange(11)*2
    c=compile_trajectory(points,20,40)
    independent_stop_lower_bound=10*2*np.sqrt(2/40)
    assert c['duration_s']<.75*independent_stop_lower_bound
    assert c['moving_internal_knots']==9


def test_tcp_limits_stretch_time_without_changing_checked_curve():
    points=np.array([[0,0],[10,5],[15,-2]],float);weights=np.array([900.,500.])
    original=compile_trajectory(points,20,40)
    slow=compile_trajectory(points,20,40,tcp_reaches=weights,tcp_speed=15.,tcp_acceleration=30.)
    assert slow['duration_s']>original['duration_s']
    np.testing.assert_allclose(slow['coefficients'],original['coefficients'])
    for c,t in zip(slow['coefficients'],slow['segment_durations_s']):
        u=np.linspace(0,1,10001);c=np.array(c)
        omega=np.abs(np.radians(np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(c)).T/t))
        alpha=np.abs(np.radians(np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(c,2)).T/t**2))
        assert np.max(omega@weights)<=15+1e-6
        assert np.max(alpha@weights+max(weights)*omega.sum(1)**2)<=30+1e-6
    np.testing.assert_allclose(slow['positions_deg'][-1],points[-1],atol=1e-8)
