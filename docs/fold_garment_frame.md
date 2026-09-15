# Folding side convention

The fold loop uses `CLAUDE_FOLD_AUTHORITY_V1`: Claude decides semantic targets
from current RGB, using Molmo annotations as optional evidence. Left/right mean
viewer-left/right with the garment's
collar imagined above its hem, without mirroring. This is not wearer anatomy
or robot/world left/right. A shirt whose collar points left and hem points
right has its **left sleeve below** the axis in the displayed image.

The existing clockwise-90 Camera A display rotation does not align the garment.
Images, Rxxx IDs, depth maps, camera transforms and robot grounding retain their
existing coordinates. Legacy generic callers still support `GARMENT_FRAME_V1`
region checks; the fold loop no longer uses those bands to veto Claude's choice.

Each before/after capture queries Molmo for the current collar and hem before
supervisor reasoning. These are two additional point queries. A missing,
low-confidence, out-of-image or degenerate axis is recorded as an unavailable
hint; Claude still inspects the current RGB and decides. This also
applies with `--no-molmo-sleeve-grounding`, which disables only the optional
sleeve hint. Model confidence is not proof that a landmark is correct.

The frame is bound to the current RGB by a content digest. Supervisor, Molmo
sleeve query and both remote planner stages can receive the tentative frame.
Old sleeve hints are not reused across captures; stale generated frame sidecars
are cleared before a new attempt. An opposite-side Molmo point is displayed as
a hypothesis, never mirrored into an invented point. Claude may accept, correct
or ignore it. The current garment mask, measured geometry, workspace, grasp
height, trajectory and IK checks remain mandatory.

For each capture inspect:

- `before_raw/garment_frame.json` (or the corresponding after/retry directory):
  current collar, hem and basis vectors in displayed pixels.
- `before_raw/camera_A_garment_frame.png`: yellow collar-to-hem line, cyan LEFT
  direction, magenta RIGHT direction when Molmo provides a valid tentative axis.
- `before_raw/camera_A_molmo_frame_hint.png`: RGB annotation explicitly labeled
  as a hint; supplied to the supervisor/planner, including an unavailable label
  when no valid axis exists.
- `molmo_sleeve_locator/camera_A_molmo_hint_upright.png`: current RGB with raw
  sleeve point/confidence annotations. No XYZ/depth is sent in this image.
- `before_raw/garment_axis_locator/molmo_keypoints_raw.json`: model outputs and
  confidences, before geometric checks.
- `planning_attempt_*/.../reference_prevalidation.json`: executable reference
  checks. Claude owns the semantic region; current cloth mask/geometry and
  workspace still constrain the selected grasp.
- `planning_attempt_*/.../pixel_source_resolution.json`: source image/view,
  source pixels, host-mapped floating-point pixels and final rounded pixels.

Every remote transport move must name its `image_id` and `pixel_xy` in that
exact source view. Only the current canonical Cam-A RGB and verified derivatives
from this same request are eligible. Static references, Rxxx overlays, Molmo
annotations, stale/unknown view IDs and rotation padding cannot supply transport
coordinates. The host walks the full transform chain and rounds once to the
nearest pixel center (half rounds upward), then performs normal depth/workspace
validation. A `target=grasp` move uses the selected Rxxx with null image_id and
pixel_xy. Claude can still use map_point to inspect a mapping, but must not pair
already-mapped coordinates with a derived view ID.

Viser shows only the two newest iterations; when iteration 3 appears, iteration
1's image, panel and trajectory handles are removed. Files remain on disk.

Normal startup is unchanged. Start a fresh run to avoid reusing completion
labels from the earlier, incorrect screen-axis convention. Reference collections
still default to `data/reference/fold_states/active`. Their state_01 must mean
garment-frame left sleeve folded, and state_02 both sleeves folded. Existing
collections are not silently relabeled; inspect them if captured using raw
screen-left instead. Newly captured manifests record this convention.

Offline verification:

```bash
python -m pytest -q tests/test_fold_frame.py tests/test_remote_fold.py \
  tests/test_fold_exploration_pipeline.py tests/test_auto_exploration.py
```
