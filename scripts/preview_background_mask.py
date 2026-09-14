#!/usr/bin/env python3
"""Preview RGB background separation on a saved photo; no camera or robot IO.

This only tests appearance. It cannot validate depth, calibration or workspace.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.perception import PerceptionConfig, _estimate_camera_table_appearance, _camera_table_appearance_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/perception.free_exploration.json"))
    parser.add_argument("--output", type=Path, default=Path("results/background_preview"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = PerceptionConfig.load(root, args.config)
    rgb = np.array(Image.open(args.image).convert("RGB"))
    # No depth evidence is available: allow all pixels only for this explicitly
    # non-executable appearance preview. Runtime uses measured table heights.
    heights = np.zeros(rgb.shape[:2])
    valid = np.ones(rgb.shape[:2], dtype=bool)
    appearance = _estimate_camera_table_appearance(
        rgb, heights, valid, minimum_color_distance=24., mode=config.table_appearance_mode,
        table_roi_xyxy=config.table_roi_xyxy,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"mode": "rgb_only_appearance_preview", "depth_validated": False,
              "background": appearance}
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not appearance["confident"]:
        raise SystemExit(f"Background estimate rejected: {appearance['reason']}")
    mask, _ = _camera_table_appearance_mask(rgb, heights, valid, minimum_color_distance=24.,
                                           table_appearance=appearance)
    Image.fromarray((mask*255).astype(np.uint8)).save(args.output / "appearance_mask.png")
    overlay = rgb.copy()
    overlay[mask] = (overlay[mask]*0.5 + np.array([0, 255, 0])*0.5).astype(np.uint8)
    overlay = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay)
    h, w = mask.shape
    roi = config.table_roi_xyxy
    draw.rectangle((int(roi[0]*w), int(roi[1]*h), int(roi[2]*w)-1, int(roi[3]*h)-1), outline="yellow", width=3)
    draw.text((12, 12), "RGB ONLY: green=appearance; yellow=work surface ROI", fill="yellow", stroke_fill="black", stroke_width=2)
    overlay.save(args.output / "appearance_overlay.png")
    print(json.dumps(report, indent=2))
    print(f"Saved RGB-only preview to {args.output}")


if __name__ == "__main__":
    main()
