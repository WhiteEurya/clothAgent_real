from copy import deepcopy
from types import MethodType, SimpleNamespace
import threading

import numpy as np
import pytest

from cloth_agent import one_shot_views as v
from cloth_agent.gamepad_collect_runtime import CollectionRuntime
from cloth_agent.codex_motion import parse_code


def runtime_and_record():
    record = v.read(v.RIGHT_VIEW)
    arm = SimpleNamespace(ip=record['robot_ip'], axis=6, serial=record['control_box_sn'],
                          tcp_offset=np.array(record['tcp_offset_mm_deg']), arm_id='left')
    config = SimpleNamespace(arms={'left': arm}, limits=dict(
        joint_speed_deg_s=30., joint_accel_deg_s2=60., stationary_tolerance_deg=.5),
        raw={'camera': dict(mount_arm='left', serial=record['camera_serial']),
             'safety': {'arms': {'left': {'max_joint_speed_deg_s': 25.}}}})
    runtime = SimpleNamespace(config=config, arm_key='left', cancel=threading.Event(),
        models={'left': SimpleNamespace(lower=np.full(6, -np.pi), upper=np.full(6, np.pi))},
        perception_target=lambda: ('left', np.zeros(6)), arrival=lambda q: None)
    runtime._plan_preparation = MethodType(CollectionRuntime._plan_preparation, runtime)
    return runtime, record


@pytest.mark.parametrize('field,value', [
    ('mount_arm', 'right'), ('camera_serial', 'wrong'), ('robot_ip', 'wrong'),
    ('control_box_sn', 'wrong'), ('axis', 7), ('tcp_offset_mm_deg', [0]*6),
    ('joint_angles_deg', [float('nan')]*7), ('joint_angles_deg', [0]*5),
])
def test_view_identity_and_calibration_fail_closed(field, value):
    runtime, record = runtime_and_record()
    record[field] = value
    with pytest.raises(v.ViewError):
        v.view_target(runtime, record)


def test_both_routes_planned_at_reduced_speed_without_changing_manipulation_limits(tmp_path):
    runtime, record = runtime_and_record()
    initial = {'left': [0]*6, 'right': [5]*7}
    before = SimpleNamespace(meta={'joints_deg': initial}, validate_for=lambda *a, **k: None)
    old = dict(runtime.config.limits)
    tour = v.ViewTour(runtime, before, tmp_path, record)
    assert runtime.config.limits == old
    for route in (tour.outbound, tour.inbound):
        folder, start, end, ops = route
        calls = parse_code((folder/'motion.py').read_text())
        assert len(calls) == 1 and calls[0][0] == 'move_joints'
        assert set(calls[0][1]) == {'left'}
        assert len(ops) == 1 and ops[0][0] == 'move'
        seg = ops[0][1]
        assert seg.speed == 10 and seg.acceleration == 20 and seg.timeout > 30
        np.testing.assert_equal(end['right'], initial['right'])
    np.testing.assert_allclose(tour.inbound[2]['left'], initial['left'])
    np.testing.assert_allclose(tour.outbound[2]['left'], record['joint_angles_deg'][:6])


def test_bad_perception_or_route_rejected_before_any_motion(tmp_path):
    runtime, record = runtime_and_record()
    before = SimpleNamespace(meta={'joints_deg': {'left': [1]*6}}, validate_for=lambda *a, **k: None)
    with pytest.raises(v.ViewError, match='not at perception'):
        v.ViewTour(runtime, before, tmp_path/'wrong_pose', record)
    before.meta['joints_deg']['left'] = [0]*6
    record['joint_angles_deg'][0] = 400
    limits = deepcopy(runtime.config.limits)
    (tmp_path/'wrong_target').mkdir()
    with pytest.raises(v.DualArmError, match='joint limits'):
        v.ViewTour(runtime, before, tmp_path/'wrong_target', record)
    assert runtime.config.limits == limits


@pytest.mark.parametrize('mode', ['success', 'early_model', 'model_error', 'model_timeout',
                                 'return_error', 'interrupt', 'cancel', 'capture_error'])
def test_model_and_return_are_parallel_but_only_joined_result_is_usable(tmp_path, monkeypatch, mode):
    v.save(tmp_path/'right_view_pose.json', {})
    calls = []
    model_entered, return_finished, release_model, model_finished = [threading.Event() for _ in range(4)]
    runtime = SimpleNamespace(cancel=threading.Event())
    folder = tmp_path/'round'; folder.mkdir()
    class Tour:
        def __init__(self, *a): pass
        def capture(self):
            calls.append('side_capture')
            if mode == 'capture_error':
                raise RuntimeError('camera failed')
        def return_to_perception(self):
            assert model_entered.wait(2), 'model must start before return finishes'
            calls.append('return_started')
            if mode == 'return_error':
                raise RuntimeError('controller fault')
            if mode == 'interrupt':
                raise KeyboardInterrupt()
            if mode == 'early_model':
                assert model_finished.wait(2)
            if mode == 'cancel':
                runtime.cancel.set()
            calls.append('return_verified')
            return_finished.set()
    monkeypatch.setattr(v, 'ViewTour', Tour)
    def request():
        calls.append('model'); model_entered.set()
        try:
            if mode in ('return_error', 'interrupt', 'cancel'):
                assert release_model.wait(2)
            elif mode != 'early_model':
                assert return_finished.wait(2), 'return must run while model is pending'
            if mode == 'model_error': raise ValueError('bad model JSON')
            if mode == 'model_timeout': raise TimeoutError('HTTP timed out')
            return 'proposal', {'wall_latency_s': 1.}
        finally:
            model_finished.set()
    timing = {}
    errors = {'model_error': ValueError, 'model_timeout': TimeoutError, 'return_error': v.ViewError,
              'interrupt': KeyboardInterrupt, 'cancel': KeyboardInterrupt, 'capture_error': v.ViewError}
    try:
        if mode in errors:
            with pytest.raises(errors[mode]):
                v.plan_with_right_view(runtime, tmp_path, folder, None, request, timing)
        else:
            assert v.plan_with_right_view(runtime, tmp_path, folder, None, request, timing)[0] == 'proposal'
            calls.append('use_proposal')
            assert calls.index('return_verified') < calls.index('use_proposal')
        evidence = v.read(folder/'view_planning.json')
        assert evidence['model_result_usable'] == (mode not in errors)
        if mode in ('return_error', 'interrupt', 'cancel'):
            assert not model_finished.is_set()  # hardware stop does not wait for HTTP.
        if mode == 'capture_error':
            assert calls == ['side_capture']
        else:
            assert timing['view_return_s'] >= timing['planner_return_overlap_s'] >= 0
    finally:
        release_model.set()
        if mode != 'capture_error':
            assert model_finished.wait(2)


def test_return_failure_never_captures_or_accepts_pose(tmp_path, monkeypatch):
    from cloth_agent import trajectory_preview
    monkeypatch.setattr(trajectory_preview, 'record_preview', lambda *a: None)
    runtime, record = runtime_and_record()
    arrivals = []
    runtime.arrival = lambda q: arrivals.append(q)
    runtime._execute_preparation = lambda *a: {'status': 'STOPPED'}
    before = SimpleNamespace(meta={'joints_deg': {'left': [0]*6}}, validate_for=lambda *a, **k: None)
    tour = v.ViewTour(runtime, before, tmp_path, record)
    with pytest.raises(v.ViewError, match='transition failed'):
        tour.return_to_perception()
    assert len(arrivals) == 2  # initial preflight and return-start check, no final arrival.
