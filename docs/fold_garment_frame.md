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

The remote fold path uses this order for each sleeve step:

1. Supervisor decides the current step using the same garment-relative convention.
2. Claude Reads the current RGB, chooses rotation/crop/resize as needed, and Reads
   its selected collar-up, hem-down view. An already aligned original can be selected.
3. The host verifies the image hash, current-source ancestry, successful Read and
   declared collar/hem alignment (within 14 degrees of vertical). It stages that exact
   RGB for Molmo. This alignment check validates Claude's declaration, not the semantic
   correctness of the landmarks. Ambiguity, timeout, missing Read, invalid transform or
   invalid selection stops the handoff; no guessed orientation is used.
4. Molmo sees only this selected RGB and a literal IMAGE LEFT/RIGHT sleeve request.
   This RGB-only query does not load depth, transform geometry maps or install grasp
   references. It does not read static reference images. A hidden/ambiguous sleeve
   should produce no point, rather than selecting the other sleeve.
5. The host maps the hint back through the verified transforms to current canonical
   and raw Camera-A pixels. Claude receives both processed and canonical RGB overlays
   and decides whether to accept, correct or ignore the hint.

The orientation call has a hard shared budget of **6 edit attempts** across rotation,
crop and resize (invalid attempts count too). Claude may iterate and correct its views,
but should finish as soon as a suitable view exists. Tool replies and Viser show used
and remaining edits. At zero, all further edits are rejected; Read, image_info and
map_point remain available to inspect/select existing images (the general call/time
limits still apply). Budget state persists inside the remote job across MCP restarts.
An orientation failure, UNCERTAIN response or timeout is non-retriable, including in
unattended mode: it stops before Molmo/robot execution rather than starting another
Claude call with fresh budget. Successful selection after using all 6 edits is allowed.
Other Claude stages retain their existing limits. Six edits bounds image mutations,
not total model latency or the number of Read calls.

The early Molmo collar/hem pass is disabled for the remote fold path. Generated stale
frame sidecars are still cleared. `--no-molmo-sleeve-grounding` skips the entire optional
orientation-to-Molmo handoff. Legacy local planner mode retains its old Molmo axis/hint
flow; this image-tool handoff is implemented through `RemoteFoldClient`.

The current garment mask, measured geometry, workspace, grasp height, trajectory and
IK checks remain mandatory. No transformed image gets paired with an untransformed
depth map, and an opposite-side hint is never automatically mirrored.

For each iteration inspect:

- `claude_image_tools/molmo_orientation_*/`: prompt, all operations/Read events,
  original and replayed images, timings, hashes and failures.
- `claude_molmo_orientation/selection.json`: exact selected view, source chain,
  collar/hem coordinates, validation status and Claude's reasoning.
- `claude_molmo_orientation/molmo_input/camera_0_A.png`: exact RGB given to Molmo.
- `claude_molmo_orientation/claude_orientation_debug.png`: collar/hem annotations
  for humans; this annotated copy is not given to Molmo.
- `molmo_handoff.json`: actual input path/digest, current step and literal Molmo prompt.
- `molmo_sleeve_locator/camera_A_molmo_hint_collar_up.png`: Molmo's point in the
  selected collar-up view; also sent to Claude.
- `molmo_sleeve_locator/camera_A_molmo_hint_upright.png`: the same hint mapped to
  the canonical fixed camera display; also sent to Claude.
- `molmo_sleeve_locator/camera_A_molmo_hint_raw.png`: the same hint in raw Cam A.
- `molmo_sleeve_locator/pixel_mapping.json`: floating and rounded pixel mappings.
- `molmo_sleeve_locator/molmo_keypoints_raw.json`: original worker outputs/confidences.
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
  tests/test_fold_exploration_pipeline.py tests/test_auto_exploration.py \
  tests/test_claude_molmo_view.py tests/test_molmo_keypoint_pipeline.py
```
