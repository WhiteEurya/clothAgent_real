from __future__ import annotations

import numpy as np
from pathlib import Path
import subprocess
import pytest

from cloth_agent.rollout_recorder import (
    compose_four_panel,
    depth_to_bgr,
    finalize_mp4_h264,
)


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
