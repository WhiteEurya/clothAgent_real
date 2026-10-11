import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.stop_sweep import (
    native_joint_ranges, position_error_mm, validate_native_state,
    validate_native_sweep,
)
from .test_stop_sweep import arc_scene


@pytest.mark.parametrize('active,parked',[('left','right'),('right','left')])
def test_policy_preserves_command_range_other_arm_and_feedback_bounds(active,parked):
    config, _ = arc_scene()
    config.limits.update(tracking_error_deg=.2, stationary_tolerance_deg=.1,
                         tracking_error_mm=3.)
    start={'left':np.array([0.]), 'right':np.array([0.])}
    end={active:np.array([.1]), parked:np.array([0.])}
    original=native_joint_ranges(config,start,end)
    config.collision_envelopes_disabled_arms=frozenset({active})
    low,high=native_joint_ranges(config,start,end,for_collision=True)
    assert np.array_equal(low[active],start[active])
    assert np.array_equal(high[active],end[active])
    assert np.array_equal(low[parked],original[0][parked])
    assert np.array_equal(high[parked],original[1][parked])
    feedback=native_joint_ranges(config,start,end)
    for actual,expected in zip(feedback,original):
        for key in start:
            assert np.array_equal(actual[key],expected[key])
    assert position_error_mm(config,active)==0.
    assert position_error_mm(config,parked)==3.


def test_disabled_stop_envelope_skips_collision_outside_command():
    config,models=arc_scene()
    config.obstacles=[{'name':'stop_only','min_mm':[99.8,1.5,-1],
                       'max_mm':[100.2,1.7,1]}]
    start={'left':[0.],'right':[0.]}
    end={'left':[.1],'right':[0.]}
    with pytest.raises(DualArmError):
        validate_native_sweep(config,models,start,end)
    config.collision_envelopes_disabled_arms=frozenset({'left'})
    assert validate_native_sweep(config,models,start,end)['status']=='SAFE'


def test_nominal_collision_still_rejected_with_envelope_disabled():
    config,models=arc_scene()
    config.collision_envelopes_disabled_arms=frozenset({'left'})
    config.obstacles=[{'name':'actual_collision','min_mm':[99.8,-.1,-1],
                       'max_mm':[100.2,.1,1]}]
    with pytest.raises(DualArmError):
        validate_native_state(config,models,{'left':[0.],'right':[0.]})
    with pytest.raises(DualArmError):
        validate_native_sweep(config,models,{'left':[-2.],'right':[0.]},
                              {'left':[2.],'right':[0.]})


def test_state_omits_only_arm6_tracking_and_position_padding():
    config,models=arc_scene()
    config.limits.update(tracking_error_deg=.2, tracking_error_mm=1.)
    joints={'left':[0.],'right':[0.]}
    with pytest.raises(DualArmError):
        validate_native_state(config,models,joints)
    config.collision_envelopes_disabled_arms=frozenset({'left'})
    assert validate_native_state(config,models,joints)['status']=='SAFE'
