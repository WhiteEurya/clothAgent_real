"""Single-camera consent wiring, without opening cameras or robot connections."""
import json
import time
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline
from cloth_agent.fold_exploration_pipeline import _capture_grasp_check_rgb, _grasp_check_images
from PIL import Image
import numpy as np


def pipeline_for(tmp_path, *, real=True, confirmed=True, mode="single_camera_rgbd", cameras=("A",)):
    (tmp_path / "run_metadata.json").write_text(json.dumps({
        "last_perception_mode": mode, "last_active_cameras": list(cameras),
    }))
    pipeline = FoldExplorationPipeline.__new__(FoldExplorationPipeline)
    pipeline.real = real
    pipeline.confirm_real = confirmed
    pipeline.record_video = False
    pipeline.observer_camera_serial = None
    pipeline.session = SimpleNamespace(run_dir=tmp_path, run_experiment=Mock(return_value={"status": "SUCCESS"}))
    pipeline.session.runner = SimpleNamespace(preflight=lambda _: SimpleNamespace(actions=[
        {'name': 'move'}, {'name': 'open_gripper'}, {'name': 'move'}, {'name': 'close_gripper'}]))
    pipeline._debug = Mock()
    pipeline._debug_exception = Mock()
    return pipeline


def test_single_camera_fold_passes_confirmation_to_session(tmp_path):
    pipeline = pipeline_for(tmp_path)
    config = SimpleNamespace(active_camera_labels=("A",))
    result, recording = pipeline._execute(tmp_path / "plan.py", config, tmp_path, label="fold")
    assert result["status"] == "SUCCESS"
    kwargs = pipeline.session.run_experiment.call_args.kwargs
    assert kwargs["single_view_confirmed"] is True
    assert kwargs["real"] is True
    assert kwargs["confirmed"] is True
    assert recording["status"] == "disabled"


@pytest.mark.parametrize("configured,observed,mode", [
    (("A", "B"), ("A",), "single_camera_rgbd"),
    (("A",), ("B",), "single_camera_rgbd"),
    (("A",), ("A", "B"), "dense_ab_rgbd_fusion"),
    (("A",), (), "single_camera_rgbd"),
    (("A",), ("A",), None),
])
def test_camera_mismatch_rejected_before_recording_or_execution(tmp_path, monkeypatch, configured, observed, mode):
    pipeline = pipeline_for(tmp_path, mode=mode, cameras=observed)
    pipeline.record_video = True
    recorder = Mock(side_effect=AssertionError("recorder must not start"))
    monkeypatch.setattr("cloth_agent.fold_exploration_pipeline.DualRealSenseRolloutRecorder", recorder)
    config = SimpleNamespace(active_camera_labels=configured)
    with pytest.raises(PermissionError, match="camera configuration"):
        pipeline._single_view_execution_confirmation(config, {
            "perception_mode": mode, "active_cameras": list(observed),
        })
    with pytest.raises(PermissionError, match="camera configuration"):
        pipeline._execute(tmp_path / "plan.py", config, tmp_path, label="fold")
    recorder.assert_not_called()
    pipeline.session.run_experiment.assert_not_called()


def test_single_camera_real_run_still_requires_real_confirmation(tmp_path):
    pipeline = pipeline_for(tmp_path, confirmed=False)
    with pytest.raises(PermissionError, match="--confirm-real"):
        pipeline._execute(tmp_path / "plan.py", SimpleNamespace(active_camera_labels=("A",)), tmp_path, label="fold")
    pipeline.session.run_experiment.assert_not_called()


@pytest.mark.parametrize("real,mode,cameras", [
    (False, "single_camera_rgbd", ("A",)),
    (True, "dense_ab_rgbd_fusion", ("A", "B")),
])
def test_dry_run_and_dual_camera_do_not_claim_single_view_consent(tmp_path, real, mode, cameras):
    pipeline = pipeline_for(tmp_path, real=real, mode=mode, cameras=cameras)
    pipeline._execute(tmp_path / "plan.py", SimpleNamespace(active_camera_labels=cameras), tmp_path, label="fold")
    assert pipeline.session.run_experiment.call_args.kwargs["single_view_confirmed"] is False


def test_cam_a_photo_wait_does_not_block_transport_without_observer(tmp_path, monkeypatch):
    pipeline = pipeline_for(tmp_path)
    order = []
    transported = threading.Event()
    def capture(config, recorder, path, after_ns):
        assert recorder is None
        assert not transported.is_set(), 'photo must precede transport'
        order.append(path.stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (4, 4)).save(path)
        return {'status': 'CAPTURED', 'image': str(path), 'frame_monotonic_ns': after_ns + 1}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb', capture)
    def run(*args, **kwargs):
        callback = kwargs['action_callback']
        callback(2, {'name': 'move', 'args': {'z': 20}})
        order.append('closure_feedback_confirmed')
        callback(3, {'name': 'close_gripper', 'success': True})
        order.append('lift_motion')
        callback(4, {'name': 'move', 'args': {'z': 80}, 'success': True})
        order.append('transport_motion')
        transported.set()
        callback(5, {'name': 'move', 'args': {'z': 80}, 'success': True})
        return {'status': 'SUCCESS'}
    pipeline.session.run_experiment = run
    _, recording = pipeline._execute(tmp_path / 'plan.py', SimpleNamespace(active_camera_labels=('A',)),
                                     tmp_path, label='fold', hold_action_index=4)
    assert order == ['camera_A_grasp_before_lift', 'closure_feedback_confirmed', 'lift_motion', 'camera_A_grasp_after_lift', 'transport_motion']
    assert len(_grasp_check_images(recording)) == 2
    manifest = json.loads((tmp_path / 'hold_check' / 'grasp_snapshots.json').read_text())
    assert 'after_close' not in manifest
    assert manifest['after_lift']['action_index'] == 4
    assert manifest['after_lift']['requested_lift_mm'] == 60


def test_acquisition_probe_only_captures_moves_at_least_30mm_above_contact(tmp_path, monkeypatch):
    pipeline = pipeline_for(tmp_path)
    captured = []
    def capture(config, recorder, path, after_ns):
        captured.append(path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (4, 4), 'white').save(path)
        return {'status': 'CAPTURED', 'image': str(path), 'frame_monotonic_ns': after_ns + 1}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb', capture)
    def run(*args, **kwargs):
        callback = kwargs['action_callback']
        callback(2, {'name': 'move', 'args': {'z': 20}})
        callback(3, {'name': 'close_gripper'})
        for index, z in enumerate((30, 40, 50, 20), 4):
            callback(index, {'name': 'move', 'args': {'z': z}})
        callback(8, {'name': 'open_gripper'})
        callback(9, {'name': 'move', 'args': {'z': 90}})
        return {'execution_completed': True}
    pipeline.session.run_experiment = run
    _, recording = pipeline._execute(tmp_path / 'probe.py', SimpleNamespace(active_camera_labels=('A',)),
                                     tmp_path, label='probe', hold_action_index=4, acquisition_probe=True)
    assert len(recording['lift_snapshots']) == 1
    assert [item['action']['args']['z'] for item in recording['lift_snapshots']] == [50]
    assert captured == ['camera_A_grasp_before_lift.png', 'camera_A_lift_checkpoint_01.png']
    assert (tmp_path / 'lift_checkpoints/snapshots.json').is_file()


def test_cam_a_snapshot_uses_fresh_active_recorder_and_rejects_stale(tmp_path, monkeypatch):
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline.capture_observer_rgb',
                        Mock(side_effect=AssertionError('must not reopen recorder-owned camera')))
    now = time.monotonic_ns()
    frame = SimpleNamespace(rgb=np.zeros((3, 5, 3), dtype=np.uint8), host_monotonic_ns=now + 1,
                            color_frame_number=10, host_utc='timestamp')
    recorder = Mock()
    recorder.wait_for_latest_rgbd.return_value = {'A': frame}
    path = tmp_path / 'after_lift.png'
    result = _capture_grasp_check_rgb(SimpleNamespace(active_camera_labels=('A',)), recorder, path, now)
    recorder.wait_for_latest_rgbd.assert_called_once_with(after_monotonic_ns=now, timeout_s=3.0, labels=('A',))
    assert result['source'] == 'active_camera_A_recorder' and path.is_file()
    frame.host_monotonic_ns = now
    with pytest.raises(RuntimeError, match='stale'):
        _capture_grasp_check_rgb(SimpleNamespace(active_camera_labels=('A',)), recorder, tmp_path / 'stale.png', now)
    assert not (tmp_path / 'stale.png').exists()


def test_pre_lift_frame_is_frozen_before_motion_and_saved_off_callback(tmp_path, monkeypatch):
    pipeline = pipeline_for(tmp_path)
    pipeline.record_video = True
    pipeline.recording_native = False
    pipeline.recording_codec = 'mp4v'
    recorder = Mock()
    stopped = threading.Event()
    transported = threading.Event()
    recorder.record.side_effect = lambda: stopped.wait(3) or {}
    recorder.request_stop.side_effect = lambda *a: stopped.set()
    def freeze(**kwargs):
        assert not transported.is_set()
        return SimpleNamespace(rgb=np.full((4, 4, 3), 60, dtype=np.uint8),
            host_monotonic_ns=kwargs['before_ns'] - 1, color_frame_number=10, host_utc='test')
    recorder.latest_pre_lift_rgb.side_effect = freeze
    factory = Mock(return_value=recorder)
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline.DualRealSenseRolloutRecorder', factory)
    save = Image.Image.save
    def slow_save(image, path, *args, **kwargs):
        if str(path).endswith('before_lift.png'):
            assert not transported.is_set(), 'PNG save must finish before motion'
        return save(image, path, *args, **kwargs)
    monkeypatch.setattr(Image.Image, 'save', slow_save)
    def lift(config, rec, path, after_ns):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (4, 4), 'white').save(path)
        return {'status': 'CAPTURED', 'image': str(path), 'frame_monotonic_ns': after_ns + 1}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb', lift)
    def run(*args, **kwargs):
        callback = kwargs['action_callback']
        callback(2, {'name': 'move', 'args': {'z': 20}})
        callback(3, {'name': 'close_gripper'})
        callback(4, {'name': 'move', 'args': {'z': 60}})
        transported.set()
        return {'status': 'SUCCESS'}
    pipeline.session.run_experiment = run
    _, recording = pipeline._execute(tmp_path / 'plan.py', SimpleNamespace(active_camera_labels=('A',)),
                                     tmp_path, label='fold')
    assert factory.call_args.kwargs['record_composite'] is False
    assert [p.name for p in _grasp_check_images(recording)] == [
        'camera_A_grasp_before_lift.png', 'camera_A_grasp_after_lift.png']
    assert recording['grasp_snapshots']['before_lift']['status'] == 'CAPTURED'


def test_cam_a_one_shot_uses_configured_serial_when_video_disabled(tmp_path, monkeypatch):
    def capture(serial, directory, **kwargs):
        assert serial == 'wrist-camera'
        assert kwargs['label'] == 'A'
        directory.mkdir(parents=True)
        raw = directory / 'camera_A_observer_rgb.png'
        Image.new('RGB', (3, 3)).save(raw)
        return {'rgb_image': str(raw)}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline.capture_observer_rgb', capture)
    spec = SimpleNamespace(label='A', serial='wrist-camera', color_exposure=700, color_white_balance=3800)
    config = SimpleNamespace(active_camera_labels=('A',), cameras=[spec], width=640, height=480,
                             fps=15, warmup_frames=20)
    path = tmp_path / 'camera_A_grasp_after_close.png'
    result = _capture_grasp_check_rgb(config, None, path, time.monotonic_ns())
    assert result['status'] == 'CAPTURED' and path.is_file()
    assert len(list(tmp_path.rglob('*.png'))) == 1


def test_snapshot_failure_logged_without_claiming_grasp_success(tmp_path, monkeypatch):
    pipeline = pipeline_for(tmp_path)
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb',
                        Mock(side_effect=TimeoutError('no fresh frame')))
    def run(*args, **kwargs):
        kwargs['action_callback'](2, {'name': 'move', 'args': {'z': 20}})
        kwargs['action_callback'](3, {'name': 'close_gripper', 'success': True})
        kwargs['action_callback'](4, {'name': 'move', 'args': {'z': 50}})
        return {'status': 'SUCCESS'}
    pipeline.session.run_experiment = run
    _, recording = pipeline._execute(tmp_path / 'plan.py', SimpleNamespace(active_camera_labels=('A',)),
                                     tmp_path, label='fold')
    assert recording['grasp_snapshots']['after_lift']['status'] == 'FAILED'
    assert _grasp_check_images(recording) == []
