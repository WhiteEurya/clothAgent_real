from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

import pytest

from cloth_agent.auto_exploration import (
    AutoExplorationError,
    CameraAWebMonitor,
    ExplorationEvaluation,
    _depth_preview,
    _json_default,
    validate_evaluation_payload,
)


def test_evaluation_contract_is_strict():
    evaluation = validate_evaluation_payload(
        {
            "useful": True,
            "confidence": 0.8,
            "observed_change": "The raised fold moved and more fabric is visible.",
            "next_objective": "Inspect the newly exposed sleeve edge.",
            "stop": False,
            "reason": "The first action produced a useful change.",
        }
    )
    assert isinstance(evaluation, ExplorationEvaluation)
    assert evaluation.useful is True
    assert evaluation.stop is False
    assert evaluation.as_dict()["confidence"] == pytest.approx(0.8)


def test_evaluation_contract_accepts_optional_caveats():
    evaluation = validate_evaluation_payload(
        {
            "useful": False,
            "confidence": 0.7,
            "observed_change": "The fold was unchanged.",
            "next_objective": "Re-observe from the same viewpoint.",
            "stop": True,
            "reason": "The action was not justified.",
            "caveats": ["Depth is noisy near the sleeve edge."],
        }
    )
    assert evaluation.caveats == ("Depth is noisy near the sleeve edge.",)
    assert evaluation.as_dict()["caveats"] == ["Depth is noisy near the sleeve edge."]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "useful": "yes",
            "confidence": 0.8,
            "observed_change": "x",
            "next_objective": "y",
            "stop": False,
            "reason": "z",
        },
        {
            "useful": True,
            "confidence": 1.5,
            "observed_change": "x",
            "next_objective": "y",
            "stop": False,
            "reason": "z",
        },
    ],
)
def test_evaluation_contract_rejects_hard_failures(payload):
    with pytest.raises(AutoExplorationError):
        validate_evaluation_payload(payload)


def test_auto_module_requires_explicit_real_flag(tmp_path: Path):
    from cloth_agent.auto_exploration import run_auto_exploration_viewer

    with pytest.raises(PermissionError, match="real-execution only"):
        run_auto_exploration_viewer(  # type: ignore[arg-type]
            None,
            enable_real=False,
        )


def test_web_camera_monitor_uses_dedicated_viser_label():
    from cloth_agent.perception import CameraSpec, MolmoConfig, PerceptionConfig

    dummy = Path("/tmp/cam_a_extrinsics.yaml")
    spec = CameraSpec("A", "serial-A", dummy)
    config = PerceptionConfig(
        cameras=(spec, CameraSpec("B", "serial-B", dummy)),
        molmo=MolmoConfig(Path("/bin/true")),
    )
    monitor = CameraAWebMonitor(
        Path("/tmp/project"),
        Path("/tmp/project/perception.json"),
        spec,
        config,
        lambda: None,
    )
    assert monitor.window_name == "CamA live monitor (serial-A)"


def test_depth_preview_is_rgb_grayscale_and_masks_invalid_depth():
    import numpy as np

    preview = _depth_preview(
        np.asarray([[0.1, 0.5, 1.0, 2.1]], dtype=np.float32),
        min_depth_m=0.15,
        max_depth_m=2.0,
    )
    assert preview.shape == (1, 4, 3)
    assert preview.dtype == np.uint8
    assert np.all(preview[0, 0] == 0)
    assert np.all(preview[0, 3] == 0)
    assert int(preview[0, 1, 0]) > int(preview[0, 2, 0])


def test_continuous_iteration_defaults_and_json_serialization():
    import numpy as np
    from pathlib import Path

    assert _json_default(Path("results/x.json")) == "results/x.json"
    assert _json_default(np.asarray([1, 2])) == [1, 2]


def test_web_monitor_serializes_duplicate_camera_stop_calls():
    from cloth_agent.perception import CameraSpec, MolmoConfig, PerceptionConfig

    dummy = Path("/tmp/cam_a_extrinsics.yaml")
    spec = CameraSpec("A", "serial-A", dummy)
    config = PerceptionConfig(
        cameras=(spec, CameraSpec("B", "serial-B", dummy)),
        molmo=MolmoConfig(Path("/bin/true")),
    )
    monitor = CameraAWebMonitor(
        Path("/tmp/project"),
        Path("/tmp/project/perception.json"),
        spec,
        config,
        lambda: None,
    )

    class FakeCamera:
        def __init__(self):
            self.started = True
            self.stop_calls = 0

        def stop(self):
            if not self.started:
                return
            self.started = False
            self.stop_calls += 1

    camera = FakeCamera()
    threads = [
        threading.Thread(target=monitor._stop_camera, args=(camera,))  # type: ignore[arg-type]
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert camera.stop_calls == 1
