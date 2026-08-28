from __future__ import annotations

import numpy as np
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import pytest

from cloth_agent.rollout_recorder import (
    DualRealSenseRolloutRecorder,
    RolloutRGBDFrame,
    build_rollout_phase_timeline,
    compose_four_panel,
    depth_to_bgr,
    finalize_mp4_h264,
    label_iteration_mp4,
)


def _rollout_frame(label: str, timestamp_ns: int, value: int) -> RolloutRGBDFrame:
    return RolloutRGBDFrame(
        label=label,
        serial=f"{label}-serial",
        rgb=np.full((2, 3, 3), value, dtype=np.uint8),
        depth_m=np.full((2, 3), value / 100.0, dtype=np.float32),
        host_utc="2026-08-23T00:00:00+00:00",
        host_monotonic_ns=timestamp_ns,
        color_frame_number=value,
        depth_frame_number=value,
        valid_depth_fraction=1.0,
    )


def test_latest_rgbd_waits_for_both_fresh_camera_frames(tmp_path: Path) -> None:
    recorder = DualRealSenseRolloutRecorder(
        SimpleNamespace(warmup_frames=0),  # type: ignore[arg-type]
        tmp_path / "recording",
    )
    recorder._latest_rgbd = {
        "A": _rollout_frame("A", 10, 1),
        "B": _rollout_frame("B", 10, 1),
    }
    result: dict[str, RolloutRGBDFrame] = {}
    started = threading.Event()

    def wait_for_snapshot() -> None:
        started.set()
        result.update(
            recorder.wait_for_latest_rgbd(
                after_monotonic_ns=10,
                timeout_s=1.0,
            )
        )

    thread = threading.Thread(target=wait_for_snapshot)
    thread.start()
    assert started.wait(timeout=1.0)
    with recorder._snapshot_condition:
        recorder._latest_rgbd = {
            "A": _rollout_frame("A", 11, 2),
            "B": _rollout_frame("B", 12, 3),
        }
        recorder._snapshot_condition.notify_all()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert set(result) == {"A", "B"}
    assert result["A"].host_monotonic_ns == 11
    assert result["B"].host_monotonic_ns == 12
    result["A"].rgb[0, 0] = 99
    assert np.all(recorder._latest_rgbd["A"].rgb[0, 0] == 2)


def test_depth_to_bgr_uses_fixed_scale_and_masks_invalid() -> None:
    depth = np.array([[np.nan, 0.1, 0.2, 1.0, 1.9, 2.1]], dtype=np.float32)
    image = depth_to_bgr(depth, min_depth_m=0.15, max_depth_m=2.0)
    assert image.shape == (1, 6, 3)
    assert np.all(image[0, 0] == 0)
    assert np.all(image[0, 1] == 0)
    assert np.all(image[0, 5] == 0)
    # Near valid depth is red in RGB, hence a dominant BGR red channel at index 2.
    assert int(image[0, 2, 2]) > int(image[0, 2, 0])
    # Far valid depth is blue in RGB, hence a dominant BGR blue channel at index 0.
    assert int(image[0, 4, 0]) > int(image[0, 4, 2])


def test_compose_four_panel_layout() -> None:
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
    result = compose_four_panel(*frames)
    assert result.shape == (4, 6, 3)
    assert np.all(result[:2, :3] == 10)
    assert np.all(result[:2, 3:] == 20)
    assert np.all(result[2:, :3] == 30)
    assert np.all(result[2:, 3:] == 40)


def test_compose_four_panel_rejects_mismatched_frames() -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="same HxWx3"):
        compose_four_panel(frame, frame, frame, np.zeros((3, 3, 3), dtype=np.uint8))


def test_finalize_mp4_h264_replaces_only_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"mp4v-source")
    monkeypatch.setattr("cloth_agent.rollout_recorder.shutil.which", lambda _: "/usr/bin/ffmpeg")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"h264-output")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("cloth_agent.rollout_recorder.subprocess.run", fake_run)
    result = finalize_mp4_h264(source)
    assert source.read_bytes() == b"h264-output"
    assert result["codec"] == "h264"
    assert result["faststart"] is True


def test_label_iteration_mp4_burns_iteration_into_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "composite.mp4"
    output = tmp_path / "labelled.mp4"
    source.write_bytes(b"source-video")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "cloth_agent.rollout_recorder.shutil.which", lambda _: "/usr/bin/ffmpeg"
    )

    def fake_run(command, **kwargs):
        seen["command"] = command
        Path(command[-1]).write_bytes(b"labelled-video")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("cloth_agent.rollout_recorder.subprocess.run", fake_run)

    result = label_iteration_mp4(
        source,
        output,
        iteration=7,
        phase_timeline=[
            {"start_s": 1.0, "end_s": 2.5, "label": "STEP 01/02 | APPROACH TARGET"}
        ],
    )

    command = seen["command"]
    assert isinstance(command, list)
    filter_graph = command[command.index("-vf") + 1]
    assert "ITER 007" in filter_graph
    assert "drawbox=" in filter_graph
    assert "STEP 01/02 | APPROACH TARGET" in filter_graph
    assert "between(t\\,1.000000\\,2.500000)" in filter_graph
    assert output.read_bytes() == b"labelled-video"
    assert result["iteration"] == 7
    assert result["label"] == "ITER 007"
    assert result["phase_timeline"][0]["start_s"] == 1.0


def test_rollout_phase_timeline_tracks_hold_check_and_abort_flow() -> None:
    def action(name: str, start_s: int, end_s: int) -> dict[str, object]:
        return {
            "name": name,
            "args": {},
            "requested_at": f"2026-08-23T00:00:{start_s:02d}+00:00",
            "completed_at": f"2026-08-23T00:00:{end_s:02d}+00:00",
        }

    execution = {
        "actual_robot_actions": [
            action("open_gripper", 2, 3),
            action("move", 3, 5),
            action("close_gripper", 5, 6),
            action("move", 6, 8),
            action("move", 12, 14),
            action("open_gripper", 14, 15),
            action("home", 15, 18),
        ],
        "checkpoint_action_index": 3,
        "checkpoint": {"executed_branch": "ABORT_RELEASE"},
    }
    manifest = {
        "created_at": "2026-08-23T00:00:00+00:00",
        "composite_encoded_frame_count": 600,
        "fps": 30,
    }

    timeline = build_rollout_phase_timeline(execution, manifest)

    labels = [item["label"] for item in timeline]
    assert labels[0] == "WAITING FOR ROBOT EXECUTION"
    assert "STEP 04/07 | LIFT TO HOLD CHECK" in labels
    hold = next(item for item in timeline if item["label"] == "HOLD CHECK | CLAUDE EVALUATION")
    assert hold["start_s"] == 8.0
    assert hold["end_s"] == 12.0
    assert "STEP 05/07 | ABORT | DESCEND TO ORIGIN" in labels
    assert "STEP 06/07 | ABORT | RELEASE AT ORIGIN" in labels
    assert labels[-1] == "ROLLOUT COMPLETE"
