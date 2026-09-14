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

## Flat table perpendicular to the camera

The current wrist configuration sets `table_plane_mode: "camera_parallel"`.
This fixes the table normal to the camera optical axis and estimates its distance
from valid bare-background patches inside the ROI. It transforms this
plane into robot-base coordinates with the capture's saved extrinsics. It does
not assume the robot base Z axis is perfectly aligned with the camera.

An RGB garment silhouette is computed before the plane fit (so it does not depend
on the fitted heights). Enclosed dark print is included in that exclusion, and
`table_reference_clearance_px` expands it by 12 pixels by default. The remaining
background is divided into a 6×8 grid. Each cell contributes at most one 3×3
depth patch, entirely outside the excluded region. Patch depth medians receive
equal weight in the table-distance estimate. At least six consistent patches
are required; there is no fallback to points on the garment.

Samples must cover at least three ROI quadrants, and their 95th percentile
depth residual must be at most 15 mm. Inconsistent data stops the fit rather
than falling back to unrelated picture corners. `camera_A_table_references.png`
now displays the actual patches considered by this fit: green for used patches,
orange for depth outliers. A red tint shows the excluded silhouette and clearance.
`camera_A_table_references.json` records each grid cell, patch depth, `used_in_fit`,
and the estimated plane. `camera_A_table_garment_exclusion.png` and
`camera_A_table_background_candidates.png` show the exact exclusion and available
background masks. These are appearance-based exclusions, not a semantic guarantee
when garment and background colors are indistinguishable. If the camera is no longer
perpendicular to the table, this mode's assumption does not apply; check the
observation pose before using it. Historical configurations default to
`reference_fit`.

Missing depth is never replaced by table height or interpolated into grounding.
`camera_A_mask_rejection_legend.png` explains black holes in the height image:

| Color | Meaning |
| --- | --- |
| Green | Accepted garment pixels |
| Magenta | Depth missing or outside the configured range |
| Blue | RGB foreground with depth, outside projected garment silhouette |
| Red | Depth disagrees with projected surface |
| Yellow | Height outside the permitted garment envelope |
| Purple | Appearance filter rejection |
| Orange | Disconnected component removed |
| White | Fixture filter rejection |

The unlabelled map preserves RGB dimensions in `camera_A_mask_rejection_map.png`.
Exact boolean masks are in `camera_A_mask_rejections.npz`; stage counts, colors
and nonfinite/zero/out-of-range depth counts are in `camera_A_mask_diagnostics.json`.
The contact sheet includes the legend image. Background outside the diagnostic
domain stays black; black is not itself a rejection classification.

Replay an existing capture without any camera or robot connection:

```bash
python scripts/test_height_map_pipeline.py --input-capture path/to/raw_capture
```

The directory must contain `capture_manifest.json`, RGB and depth arrays from
the standalone perception test. The configured mode and saved camera transform
are used. The new output contains the table references and hole diagnostics.
