# Wrist camera background segmentation

The current configuration uses `table_appearance_mode: "border_background"`.
It estimates the dominant color in the inner border of a visible work surface,
using valid depth within 15 mm of the fitted table plane. Both the fused cloud
and camera mask compare against that color with a noise-dependent RGB distance
threshold. Light cloth on a dark mat and dark cloth on a light table use the same
rule. Missing or ambiguous background evidence stops perception.

`table_roi_xyxy: [0.18, 0.03, 0.77, 0.98]` is a normalized rectangle
`[left, top, right, bottom]` in the **original, unrotated camera image**. It was
chosen from the supplied 1280 × 720 wrist photograph to exclude the white rails,
floor and people. It is not a calibration matrix or a robot workspace bound.
Cloth must be inside this rectangle, with bare background visible around it.
If the observation pose or framing changes, check the yellow rectangle and
adjust it in `config/perception.free_exploration.json`.

Preview a saved photograph without camera/robot access:

```bash
python scripts/preview_background_mask.py test.png
```

Inspect `results/background_preview/appearance_overlay.png` (green foreground,
yellow ROI), `appearance_mask.png`, and `diagnostics.json`. This is an RGB-only
appearance check, not a depth or workspace validation. The image's pixel frame
and resolution are preserved. The preview does not produce an executable plan.

To test the complete perception pipeline while the robot is already stationary
at the calibrated observation pose:

```bash
python scripts/test_height_map_pipeline.py
```

Live wrist capture reads the joints for the camera transform; it sends no motion
commands and does not invoke Claude. The script saves a `raw_capture` directory
that can be passed back with `--input-capture` for hardware-free replay.

During fold runs the diagnostics are saved in
`runs/<run>/results/perception/center_<timestamp>/`, including on later workspace
validation failure (they are not in `before_raw`):

- `camera_A_background_diagnostics.json`: background RGB, noise threshold and sample support.
- `camera_A_appearance_mask.png`, `camera_A_appearance_overlay.png`: appearance selection before the final geometry gates.
- `camera_A_mask_diagnostics.json`, `camera_A_garment_only.png`: final mask and gate counts.
- `center_workspace_diagnostics.json`, `camera_A_center_debug.png`: center and workspace evidence.

The geometric table fit, depth consistency checks, workspace checks and robot
execution checks still apply. The supplied photograph alone has no depth and
cannot establish whether its final 3-D center is inside the robot workspace.
Historical configurations without a mode retain `bright_table` behaviour.
