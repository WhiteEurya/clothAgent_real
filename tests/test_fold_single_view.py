"""Single-camera consent wiring, without opening cameras or robot connections."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cloth_agent.fold_exploration_pipeline import FoldExplorationPipeline


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
