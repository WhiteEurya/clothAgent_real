from __future__ import annotations

import numpy as np

from cloth_agent.perception import (
    CameraSpec,
    PerceptionConfig,
    RGBDFrame,
    _estimate_camera_table_z_offset_mm,
)


def _config(tmp_path):
    # Only the fields read by _estimate_camera_table_z_offset_mm are relevant;
    # construct a valid config through the project fixture shape instead of
    # relying on any robot or camera hardware.
    extrinsics = tmp_path / "A.yaml"
    extrinsics.write_text("", encoding="utf-8")
    camera = CameraSpec("A", "A1", extrinsics)
    camera_b = CameraSpec("B", "B1", extrinsics)
    return PerceptionConfig(
        cameras=(camera, camera_b),
        active_camera_labels=("A", "B"),
        width=30,
        height=30,
        fps=30,
        warmup_frames=1,
        temporal_median_frames=1,
        depth_window_radius_px=1,
        min_depth_m=0.1,
        max_depth_m=2.0,
    )


def test_table_offset_ignores_saturated_depth_bias(tmp_path, monkeypatch) -> None:
    import cloth_agent.perception as perception

    height = width = 30
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # Most near-table pixels are unsaturated neutral table at luma 220 with
    # zero residual; a large saturated tail has a +8 mm residual.
    rgb[:, :] = [220, 220, 220]
    rgb[:5, :] = [254, 254, 254]
    frame = RGBDFrame(
        "A",
        "A1",
        rgb,
        np.ones((height, width), dtype=np.float32),
        np.asarray([[100.0, 0.0, 14.5], [0.0, 100.0, 14.5], [0.0, 0.0, 1.0]]),
        np.eye(4),
    )
    residual = np.zeros((height, width), dtype=np.float64)
    residual[:5, :] = 8.0
    xyz = np.zeros((height, width, 3), dtype=np.float64)
    yy, xx = np.indices((height, width))
    xyz[..., 0] = xx
    xyz[..., 1] = yy
    xyz[..., 2] = residual
    monkeypatch.setattr(
        perception,
        "camera_base_xyz_map_mm",
        lambda _frame, _config: (xyz, np.ones((height, width), dtype=bool)),
    )
    offset, diagnostics = _estimate_camera_table_z_offset_mm(
        frame,
        _config(tmp_path),
        np.asarray([0.0, 0.0, 0.0]),
    )
    assert diagnostics["method"] == "unsaturated_near_table_median_residual"
    assert offset == 0.0
    assert diagnostics["saturation_luma_cut"] == 252.0
