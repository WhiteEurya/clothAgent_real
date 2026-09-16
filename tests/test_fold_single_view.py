"""Single-camera consent wiring, without opening cameras or robot connections."""
import json
import time
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


def test_cam_a_grasp_stills_precede_lift_and_transport_without_observer(tmp_path, monkeypatch):
    pipeline = pipeline_for(tmp_path)
    order = []
    def capture(config, recorder, path, after_ns):
        assert recorder is None
        order.append(path.stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (4, 4)).save(path)
        return {'status': 'CAPTURED', 'image': str(path)}
    monkeypatch.setattr('cloth_agent.fold_exploration_pipeline._capture_grasp_check_rgb', capture)
    def run(*args, **kwargs):
        callback = kwargs['action_callback']
        order.append('closure_feedback_confirmed')
        callback(3, {'name': 'close_gripper', 'success': True})
        order.append('lift_motion')
        callback(4, {'name': 'move', 'args': {'z': 80}, 'success': True})
        order.append('transport_motion')
        callback(5, {'name': 'move', 'args': {'z': 80}, 'success': True})
        return {'status': 'SUCCESS'}
    pipeline.session.run_experiment = run
    _, recording = pipeline._execute(tmp_path / 'plan.py', SimpleNamespace(active_camera_labels=('A',)),
                                     tmp_path, label='fold', hold_action_index=4)
    assert order == ['closure_feedback_confirmed', 'camera_A_grasp_after_close', 'lift_motion',
                     'camera_A_grasp_after_lift', 'transport_motion']
    assert len(_grasp_check_images(recording)) == 2
    manifest = json.loads((tmp_path / 'hold_check' / 'grasp_snapshots.json').read_text())
    assert manifest['after_close']['action_index'] == 3
    assert manifest['after_lift']['action_index'] == 4


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
        kwargs['action_callback'](3, {'name': 'close_gripper', 'success': True})
        return {'status': 'SUCCESS'}
    pipeline.session.run_experiment = run
    _, recording = pipeline._execute(tmp_path / 'plan.py', SimpleNamespace(active_camera_labels=('A',)),
                                     tmp_path, label='fold')
    assert recording['grasp_snapshots']['after_close']['status'] == 'FAILED'
    assert _grasp_check_images(recording) == []
