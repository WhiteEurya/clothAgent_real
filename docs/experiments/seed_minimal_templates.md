# Seed → per-edge minimal Host templates

No robot is started by either compiler. Capture a new seed through the existing seed command separately. After capture, run:

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_policy compile \
  --run /absolute/path/to/new_seed_run --model gpt-6-astra --no-cache
```

This compiles the ordinary experience graph and automatically writes `compiled_policy/`:

- `policy_graph.json`: all original edges, default replay eligibility, per-edge minimal status/spec/template.
- `host_templates.json`: executable templates only, signed with SHA-256.
- `edge_experiences.json`: every attempt's semantics, exact command, outcome, seed anchors, normalized path, heights, orientation and unsupported reasons. Unrecoverable values remain null.
- `summary.json` and `report.md`: edge counts, unsupported reasons and each edge's grasp/target spec.

Visual graph consolidation (`graph_merge`) also recompiles templates for the resulting graph, into `<graph_stem>_compiled_policy/`. No hard-coded node count, edge ID or family taxonomy is used. First eligible successful/useful-progress seed edge per source becomes preferred; other experience is retained. Self-loops, failures, NO_PROGRESS, unknown results and absent after observations cannot become the default. No terminal completion is inferred just because a trajectory ends.

## Contact-based TCP Z (default compiler mode)

Default `--frame-mode action_local_contact` retains action-local v2 XY and rotation, but separates visual anchor depth from TCP height. Each edge stores its executed seed TCP waypoint height relative to the confirmed open-gripper contact reference. Runtime requires the same recorded contact calibration and restores those TCP offsets; RGB-D residuals are recorded for diagnosis, not treated as cloth thickness. All waypoint Z values use this policy, preserving the seed vertical profile. Recorded executor lowering still applies exactly once. No fixed extra 13.5/15 mm correction is introduced.

This is an empirical same-state/table seed-transfer hypothesis, not a general conversion from optical cloth height to finger contact. It does not adapt to changed stack thickness or derive new orientation-dependent fingertip geometry. Missing/unconfirmed/changed contact metadata rejects compilation or replay. Existing templates are immutable and keep their old policy; use a newly compiled graph. `--frame-mode action_local` explicitly selects the legacy visual-Pa Z mode described below.

## Evidence and limits

The action-local compiler uses `action-local-host-template-v2`: only measured grasp Pa, target Pb and world up are needed. Collar, hem, print and SIFT are not used by this runtime. Each edge is compiled from its own actual effective seed command. Reference images remain audit provenance, not runtime inputs.

Project Pb−Pa onto world XY, normalize it to u, take v = z × u, and B = [u,v,z]. The compiler stores each waypoint's along/across fractions relative to the seed horizontal anchor distance, fixed height above measured Pa, and full relative rotation `R_rel = B_seed.T @ R_tool`. Runtime scales XY fractions by the current anchor distance and restores rotation as `B_current @ R_rel`. Height is fixed millimetres relative to current Pa; Pb depth does not tilt the basis or adapt the trajectory to the current receiving surface. Anchor separation below 10 mm is rejected.

Seed pixels may be reconstructed from recorded XYZ rather than original GPT pixel outputs. Exact seed XYZ is retained in the local coefficients, including subpixel sampling residuals. Recorded grasp lowering is undone at compile time and applied once by the existing runtime. This preserves the original command on seed reconstruction.

Supported skeleton: one right-arm grasp followed by move_tcp/move_transfer operations. Unsupported primitives, invalid depth or ambiguous seed reconstruction remain MINIMAL_UNSUPPORTED. Failures and NO_PROGRESS retain experience geometry but cannot become default replayable templates.

The legacy garment-frame compiler remains available with `--frame-mode garment`, including its collar/hem and SIFT checks. It is not the default. Historical coverage reports made with that mode are not current action-local coverage.

Action-local mode cannot independently check sleeve-side/hem-direction semantics without garment landmarks. It records `NOT_INDEPENDENTLY_EVALUATED`; graph matching and minimal GPT anchors must supply those semantics. Existing physical workspace, IK, collision and safety code is unchanged. Offline numerical reconstruction does not establish grasp correctness, surface clearance or physical success.

## Replay modes

Offline static check (no model and no robot):

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_policy_replay \
  --graph /absolute/path/to/new_seed_run/compiled_policy/policy_graph.json \
  --grounding-mode minimal --max-fallbacks 0 --dry-run \
  --output /absolute/path/to/unused_preflight_directory
```

The new graph automatically supplies its own bundle path. Existing `--grounding-templates` remains available for legacy bundles. `--grounding-mode full` selects full grounding for debugging or unsupported edges. The default frozen-Graph runner now permits one handle-recovery manipulation after a grasp-gate rejection; it does not redesign a minimal template or add Graph nodes. See [action robustness](action_robustness.md). `--max-fallbacks 0` disables the separate legacy node-mismatch recovery, while `--max-recovery-attempts 0` also disables the new handle recovery. Static checks do not establish physical safety or successful live registration.

For physical replay, only when explicitly requested by the operator, replace `--dry-run` with `--execute` and use a fresh output directory. Runtime GPT receives current RGB and that edge's short grasp/target spec only; it returns grasp_px, target_px, confidence. Host anchor depth sampling, local orientation/waypoint expansion and the existing dispatcher handle the rest. Full mode retains the existing garment semantic checks; action-local minimal explicitly logs that those checks are not independently evaluated. IK, workspace, collision and safety remain in the existing physical dispatch path.

An existing frozen experience graph can be audited without a model or robot:

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_template_compiler \
  --graph /absolute/path/to/graph_v1.json --output /absolute/path/to/new_compiled_directory
```

The historical graph_v1 audit under `reports/seed_minimal_template_compiler/legacy_seed_audit` retains 14 edges and compiles zero executable templates: eight eligible edges lack source garment-frame/registration annotations, five completed edges are not success/progress candidates, and one did not complete execution. This is a legacy metadata limitation, not a new seed result; no seed or GPT call was run for this implementation.

## Hand-eye calibration migration

The contact contract includes the optical support plane. A hand-eye update that
re-expresses that plane therefore invalidates old contact templates even when the
physical contact TCP height is unchanged. Ordinary graph dry-run does not check
the live contact contract.

`cloth_agent.experiments.replay_calibration_migration` performs an offline migration
using `--graph`, `--templates`, `--activation`, and a fresh `--output` directory.
It checks the activation file hashes, runtime FK, old/new camera frames, transformed
plane and anchor, unchanged physical contact evidence and unchanged limits. It
rejects active empirical XY patches and changes to the contact baseline. It writes
a new frozen bundle and verifies every edge expands to identical commands at
identical measured anchors. Existing motion profiles and repair provenance are
retained. Historical files and runtime contract rejection remain unchanged.

For the controller-fit Camera B activation on 2026-10-16, the migrated bundle is
`runs/replay_unified_calibration_20261016/host_templates.json`, based on the last
validated repair bundle for
`runs/seed_fresh_20261015_153150/graph_replay_v2_compiled_policy/policy_graph.json`.
Add `--grounding-templates runs/replay_unified_calibration_20261016/host_templates.json`
to that graph's minimal replay command, including when using `--repair-self-loop`.
Explicit selection takes precedence over the old repair registry. Numerical checks
are in `migration.json`; graph-only preflight is in `dry_run/`. This migration does
not establish physical alignment or resolve missing live depth. Fresh observation
and all normal motion checks still apply.

## Automatic grasp debug (opt-in)

Use `--debug-grasp-retry --grounding-mode minimal --max-fallbacks 0 --max-recovery-attempts 0`. No operator prompt.

Empty-grasp detection and downward retry have been removed. An unchanged source node stops strict replay without extra visual review or a second physical grasp.

Suspected obstruction requires valid stopped/grasp telemetry stable for 3 seconds and >=80% of initial opening remaining. This does not prove sponge contact. Before any dispatch, the existing planner certifies an alternate +2 mm vertical lift, reclose, original transport, release and return route. If that branch cannot be planned, no initial motion is dispatched. After stopped confirmation the branch lifts and sends close again. Still-wide or unknown feedback stops before transport. Maximum one unstick lift per run. Continuously moving jaws are NOT classified as obstructed solely by elapsed time. Controller faults and invalid feedback do not trigger a lift.

Only opt-in debug closes permit 3-second stopped confirmation without measurable additional closure (needed to recognize an entirely stalled close and to recheck after lifting). Default close waiting is unchanged. This changes experimental completion handling, not workspace, IK, collision, speed or controller-fault checks. The gripper is stopped when the lift begins; this is lift-then-reclose, not asynchronous motion while actively closing. Cloth can still be misclassified as obstruction; 80% is a conservative unvalidated debug threshold.

Evidence: `auto_unstick_plan.json`, `auto_close_*.json`, gripper traces, controller events. Original policy/templates/global depth patch remain unchanged. `--max-steps 1` can include the bounded in-action unstick branch.

## Grounding recapture retry

Non-null anchors with confidence below the existing localization threshold retain the one-time recapture retry. It captures fresh RGB-D, re-matches the source, and repeats minimal grounding with cache disabled. Null anchors now represent unavailable robust anchors and go directly to the bounded handle-recovery path, as do explicit `NO_ROBUST_ACTION` responses; they are not recaptured and re-grounded first. No threshold is lowered and no old anchor is reused. Depth, calibration, IK and collision failures do not trigger grounding retries or the handle-recovery fallback.

Original anchors remain in the step directory; new images, match, anchors and outcome are under `grounding_retry_001/`. Metrics record recapture attempts/successes. This refreshes evidence but does not guarantee that a physically unchanged view resolves ambiguity. No new CLI switch is required.
