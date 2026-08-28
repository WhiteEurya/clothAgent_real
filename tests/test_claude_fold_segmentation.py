from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.claude_fold_segmentation import segment_with_claude


def test_claude_fold_segmentation_validates_and_renders_polylines(tmp_path: Path, monkeypatch) -> None:
    perception = tmp_path / "runs" / "run" / "workspace" / "perception_views"
    perception.mkdir(parents=True)
    rgb = np.full((60, 80, 3), 240, dtype=np.uint8)
    rgb[10:50, 15:65] = (20, 30, 70)
    Image.fromarray(rgb).save(perception / "camera_0_A.png")
    np.save(perception / "camera_A_garment_mask.npy", np.pad(np.ones((40, 50), dtype=bool), ((10, 10), (15, 15))))
    np.save(perception / "camera_A_height_above_table_mm.npy", np.zeros((60, 80), dtype=np.float32))

    payload = {
        "status": "READY",
        "outer_contour": [[15, 10], [65, 10], [65, 50], [15, 50]],
        "fold_boundaries": [
            {"name": "fold_1", "points": [[20, 30], [40, 24], [60, 30]], "confidence": 0.8, "evidence": "continuous visible edge"}
        ],
        "regions": [
            {"name": "left_panel", "polygon": [[15, 10], [40, 10], [40, 50], [15, 50]], "confidence": 0.7, "evidence": "visible panel"}
        ],
        "notes": ["visible boundaries only"],
    }

    monkeypatch.setattr("scripts.claude_fold_segmentation.shutil.which", lambda _: "/usr/bin/claude")
    class FakeProcess:
        returncode = 0

        def __init__(self, command, *, stdout, **kwargs):
            stdout.write(json.dumps({"structured_output": payload}))
            stdout.flush()

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr("scripts.claude_fold_segmentation.subprocess.Popen", FakeProcess)

    result = segment_with_claude(perception)

    assert result["status"] == "READY"
    assert len(result["fold_boundaries"]) == 1
    assert Path(result["overlay"]).is_file()
