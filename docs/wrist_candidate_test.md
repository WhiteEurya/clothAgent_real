# Wrist calibration and measured table-plane configuration

The candidate was promoted at the user's request on 2026-09-23.
Both `config/perception.free_exploration.json` and `config/perception.example.json`
now use `background_fit`; their shared `config/extrinsics_A.yaml` contains the
candidate transform. The previous calibration is preserved as
`config/extrinsics_A.pre_candidate_20260923.yaml`.
`config/perception.wrist_candidate_test.json` remains available with the explicit
candidate filename. SDK camera intrinsics remain unchanged.

The temporary 20 mm descent settings in `config/robot.example.json` have been
reduced to 8 mm for both grasp compression and support-layer press defaults and
their configured maxima. The physical support thickness remains 20 mm.
These configuration changes take effect when a new process loads them;
already-running sessions and saved run configurations are not rewritten.

The candidate hand-eye transform comes from
`results/wrist_camera_board/20260923T073249789451Z/handeye_candidate_comparison.json`:
15 training poses and 5 held-out poses. Held-out board-center RMS decreases from
6.64 to 4.36 mm; this is a consistency check, not independently measured absolute
position accuracy.

`background_fit` samples bare-table depth patches outside the garment clearance,
transforms them into robot coordinates, then fits `z = ax + by + c` using RANSAC
and least squares. Patches are 15x15 at 1280-pixel width (scaled with resolution),
and the entire patch must contain valid bare-background pixels outside the garment
clearance. This replaces the initial noise-sensitive 3x3 sampling; the legacy
`camera_parallel` sampler is unchanged. For the black sponge, the shipped configs
use 24 px garment clearance and `background_plane_inlier_threshold_mm=6.0`.
The generic default remains 3 mm; configurable values must be finite and in
(0, 10] mm. It requires at least 6 inliers, 70% agreement within the configured tolerance,
coverage of 3 ROI quadrants and 40% of each image ROI dimension, and plane slope
at most 0.12. Insufficient support raises an error without substituting a constant
depth or horizontal plane. Reference coordinates, residuals, inlier flags and
plane coefficients are saved with the perception artifacts.

Offline checks on 2026-09-23:

- 14 targeted tests pass, including an oblique camera viewing a horizontal table,
  a large depth outlier, and missing table evidence.
- Three saved RGB-D captures, both modes using the candidate extrinsics: a right
  tabletop patch has median height residual about -9 to -9.5 mm with
  `camera_parallel`, versus -1.0 to -1.9 mm with `background_fit`.
  Details: `results/wrist_camera_board/20260923T073249789451Z/table_fit_validation.json`.
- Full offline `ClothCenterPerception.locate(..., frames=[saved_frame])` succeeds;
  artifacts: `/tmp/cloth_candidate_full_perception_2`.

This fits the observed table; it does not correct depth distortion or prove that
the observed plane matches the physical table in robot coordinates. Grasp surface
coordinates still come from measured depth. A subsequent live capture at the
perception pose confirmed visibly reduced tilt in the raw point cloud; it is saved
under `results/wrist_candidate_cloud/20260923T075804Z`. This is not a physical
grasp accuracy measurement.

## Sponge support adjustment (2026-09-24)

The fitted plane describes the measured sponge top as an overall reference, not
an impenetrable hard tabletop. The 6 mm fit tolerance allows local surface
variation; it neither flattens raw points nor adds any grasp descent. Local depth,
8 mm default press, 20 mm support thickness, robot workspace limits and existing
support-floor checks are unchanged. Spatial coverage and slope checks still fail
on insufficient or implausible evidence.

All three saved failing RGB-D captures pass full offline perception with these
settings (31/31 reference inliers each). Failed captures did not save K/T, so the
replay used the earlier capture at the same perception pose. This is an offline
regression result, not a guarantee for every future surface.
