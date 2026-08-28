from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.segment_garment_folds import segment


def test_segment_garment_folds_writes_visible_boundary_and_region_artifacts(tmp_path: Path) -> None:
    height, width = 80, 100
    rgb = np.full((height, width, 3), 245, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[15:65, 20:85] = True
    rgb[mask] = (25, 35, 80)
    rgb[38:42, 30:75] = (70, 85, 140)
    surface = np.zeros((height, width), dtype=np.float32)
    surface[mask] = 2.0
    surface[38:42, 30:75] = 12.0
    Image.fromarray(rgb).save(tmp_path / "camera_0_A.png")
    np.save(tmp_path / "camera_A_garment_mask.npy", mask)
    np.save(tmp_path / "camera_A_height_above_table_mm.npy", surface)

    report = segment(tmp_path, camera="A", edge_percentile=85.0, rgb_weight=0.0)

    assert report["status"] == "READY"
    assert report["fold_edge_pixel_count"] > 0
    assert Path(report["overlay"]).is_file()
    assert Path(report["visible_regions"]).is_file()
    assert (tmp_path / "fold_segmentation" / "camera_A_fold_segmentation.json").is_file()
