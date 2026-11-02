# Original garment reference

These are small, frozen inputs used by existing live experiment entrypoints.
They are not a goal for every garment or a calibration for another installation.

- `goal.png`: original `reports/goal_only_full_episode_v1_restart02/goal/target.png`.
- `ggo_goal.png`: original collection-aligned round 006 after RGB.
- `observation/`: matching RGB, world XYZ and original observation metadata.
- `ggo_annotations.json`: original reference self annotations v2. These were
  assistant-marked, not human-verified ground truth; provenance is preserved.

Observation hashes and original metadata are unchanged. Historical source paths
inside metadata describe provenance and are not relocated evidence. New garments
need their own goal; GGO needs matching depth, calibration and reference labels.
See `docs/deployment.md`.
