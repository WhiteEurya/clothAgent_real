# host_grasp_geometry_audit_v2

Independent offline audit of the 12 saved one-shot rounds, including user-authorized continuation segments. Does not import a robot runtime, use a model/API, train a model, change Graph, or mutate any source observation/configuration/planner. Algebraic local surface fitting is diagnostic only.

```bash
MPLCONFIGDIR=/tmp/host_grasp_geometry_audit_v2_mpl /home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.host_grasp_geometry_audit_v2
```

Output defaults to `reports/host_grasp_geometry_audit_v2`; existing output is refused. `--output` supports a separate reproduction directory. `--report-only` regenerates Markdown from saved metrics and optional `interpretation.json` without re-encoding, refitting, or searching.

The initial all-round gate checks observation hashes/calibration and the original config hash, then calls the actual read-only `Observation.sample` and independently reproduces its arithmetic. A mismatch writes STOPPED summary and prevents all fitting/search. The exact gate is a 5x5 patch, >=18 finite XYZ points, finite center XYZ, and world-Z max−min <=8 mm. No additional secondary discontinuity gate exists in the current sample function. Target Pb is sampled separately.

Frozen geometric methods:

- Primary 5x5 and contextual 31x31 patches, centered on the original Pa. World X/Y predictors, world Z response. Huber IRLS k=1.345, MAD scale multiplier=1.4826, 50 iterations maximum, minimum 18 finite XYZ points, rank 3 required. Centered/scaled coordinates are converted back to world plane coefficients. Numerical scale floor 1e-6 mm is not a safety threshold.
- Report signed residuals and absolute summary statistics for every finite point, including points downweighted by the robust fit. Missing values stay missing. Round 7's 16-point primary patch is not fit; its 31x31 context is fit. No new residual threshold or automatic classifier is introduced.
- Search inclusive Euclidean radii 3/5/10/15 px, ordering distance then y then x. Candidate must pass all existing grasp depth checks and preserve valid Pb plus the existing 10–150 mm XY translation range. The original Pa is included, so EXECUTED references have distance 0.
- Candidate continuity uses a measured 4-connected pixel path within the radius with each adjacent world-Z difference <= the already existing 8 mm number. This is an explicitly offline screen, not a deployed safety rule. RGB path and neighborhood differences are recorded without choosing an RGB threshold. Missing-origin paths begin at nearest measured anchors in the original 5x5; the unknown bridge is reported, never asserted observed.
- When original center XYZ is missing, its actual world XY displacement is null. A separately named original-valid-patch-median proxy is reported. No invented center XYZ.
- “Host-valid” means the saved-data depth gate in this audit, not full robot executability. No IK, workspace, collision, contact, or acquisition certification. Local RGB review cannot prove fabric-layer correspondence.
- Round 11 is SEPARATE_IK_FAILURE, retained with Pa/Pb/motion/geometry and recorded residual/tolerances; excluded from plane fitting and snap search.

`interpretation.json` contains the post-hoc qualitative reading of stored plots (local SMOOTH_SLOPE for rounds 9/10), not a learned classifier or tuned threshold. True-discontinuity ground truth and confirmed acquisition labels are absent. Eight EXECUTED references mean past Host authorization only.

Artifacts include the five requested audit/metrics/summary files, per-point residual CSVs, NPZ arrays/masks, original-threshold candidate lists, diagnostic and profile plots, focused snap zooms and 3-D world-coordinate planes, isolated IK evidence, frozen protocol/input hashes, and verification. Executed calculation source is retained separately from later report-only code additions. No further experiment is automatically started.
