# Folding side convention

The fold loop uses `GARMENT_FRAME_V1`: viewer-left/right with the garment's
collar imagined above its hem, without mirroring. This is not wearer anatomy
or robot/world left/right. A shirt whose collar points left and hem points
right has its **left sleeve below** the axis in the displayed image.

The existing clockwise-90 Camera A display rotation does not align the garment.
Images, Rxxx IDs, depth maps, camera transforms and robot grounding retain their
existing coordinates. Only semantic region tests use the measured garment axes.

Each before/after capture queries Molmo for the current collar and hem before
supervisor reasoning. These are two additional point queries. A missing,
low-confidence, out-of-image or degenerate axis blocks planning. This also
applies with `--no-molmo-sleeve-grounding`, which disables only the optional
sleeve hint. Heavy folds hiding the collar/hem may therefore require a clearer
observation. Model confidence is not proof that a landmark is correct.

The frame is bound to the current RGB by a content digest. Supervisor, Molmo
sleeve query, local candidate checks and both remote planner stages receive
the same frame. Old sleeve hints are not reused across captures. An opposite-side
Molmo hint is discarded, never mirrored. The original workspace/IK checks remain.

For each capture inspect:

- `before_raw/garment_frame.json` (or the corresponding after/retry directory):
  current collar, hem and basis vectors in displayed pixels.
- `before_raw/camera_A_garment_frame.png`: yellow collar-to-hem line, cyan LEFT
  direction, magenta RIGHT direction. Viser discovers this diagnostic image.
- `before_raw/garment_axis_locator/molmo_keypoints_raw.json`: model outputs and
  confidences, before geometric checks.
- `planning_attempt_*/.../reference_prevalidation.json`: per-reference side
  and longitudinal region decisions. The outer-side and sleeve-band tests
  remain coarse heuristics; they are now evaluated in garment axes.

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
