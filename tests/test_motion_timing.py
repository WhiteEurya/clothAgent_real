"""Timing regressions independent of optional FCL/OMPL installation."""
import copy

import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.motion.trajectory import parameterize, sample_segment, validate_timed


def timed(points, phases=None):
    edges = [dict(start_q_rad=a.tolist(), end_q_rad=b.tolist(),
                  phase=phases[i] if phases else 'transit')
             for i, (a, b) in enumerate(zip(points, points[1:]))]
    return parameterize(edges, np.full(13, .25), np.full(13, .5))


class Recorder:
    def __init__(self):
        self.padding = []

    def certify(self, a, b, **kwargs):
        self.padding.append(kwargs['padding'])
        return dict(minimum_clearance_bound_m=.1, nodes=1)


def test_internal_waypoints_keep_velocity_and_phase_events_stop():
    points = np.zeros((11, 13)); points[:, 0] = np.linspace(0, .1, 11)
    plan = timed(points)
    validate_timed(plan, Recorder())
    for left, right in zip(plan['segments'], plan['segments'][1:]):
        before = sample_segment(left, left['duration_s'])
        after = sample_segment(right, 0)
        assert np.allclose(before, after, atol=1e-9)
        assert before[1][0] > 0
        assert np.allclose(before[0][6:], 0)
    assert plan['duration_s'] < 2.0  # Previously 3.398 s with ten full stops.
    events = timed(points, ['transit']*5 + ['descend']*5)
    assert np.allclose(sample_segment(events['segments'][4], events['segments'][4]['duration_s'])[1:], 0)
    validate_timed(events, Recorder())


def test_curved_polynomial_is_enclosed_not_only_its_old_chord():
    points = np.zeros((4, 13))
    points[:, :2] = [[0, 0], [.03, .01], [.06, .025], [.1, .03]]
    plan = timed(points); proof = Recorder()
    validate_timed(plan, proof)
    assert max(np.max(p) for p in proof.padding) > 1e-5
    for segment, padding in zip(plan['segments'], proof.padding):
        start = np.asarray(segment['start_q_rad'])
        delta = np.asarray(segment['end_q_rad'])-start
        for u in np.linspace(0, 1, 101):
            q, v, a = sample_segment(segment, u*segment['duration_s'])
            progress = (q-start)@delta/(delta@delta)
            assert 0-1e-9 <= progress <= 1+1e-9
            assert np.all(np.abs(q-start-progress*delta) <= padding+1e-9)
            assert np.max(np.abs(v)) <= .25+1e-9
            assert np.max(np.abs(a)) <= .5+1e-9
    class RejectCurve(Recorder):
        def certify(self, a, b, **kwargs):
            if np.max(kwargs['padding']) > 1e-5:
                raise DualArmError('curve clearance cannot be certified')
            return super().certify(a, b, **kwargs)
    with pytest.raises(DualArmError, match='curve clearance'):
        validate_timed(plan, RejectCurve())


def test_corrupted_smooth_coefficients_and_retiming_are_rejected():
    points = np.zeros((3, 13)); points[:, 0] = [0, .05, .1]
    original = timed(points)
    changed = copy.deepcopy(original)
    changed['segments'][0]['coefficients_rad'][2][0] += .01
    with pytest.raises(DualArmError, match='endpoints'):
        validate_timed(changed, Recorder())
    changed = copy.deepcopy(original)
    changed['segments'][0]['duration_s'] /= 100
    with pytest.raises(DualArmError, match='velocity/acceleration'):
        validate_timed(changed, Recorder())


def test_arm_switch_preserves_sequential_execution():
    points = np.zeros((3, 13)); points[1:, 0] = .1; points[2, 6] = .1
    plan = timed(points)
    validate_timed(plan, Recorder())
    assert np.allclose(sample_segment(plan['segments'][0], plan['segments'][0]['duration_s'])[1:], 0)
    assert np.allclose(sample_segment(plan['segments'][1], .1)[0][:6], points[1, :6])
