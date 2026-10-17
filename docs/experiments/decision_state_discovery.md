# Offline decision-state discovery

This experiment imports no robot runtime, captures no new images, trains no model, changes no folding policy and publishes no approved skill. “Offline” means analysis of historical records; `--execute` sends images to GPT-6 through the project's existing Codex CLI multimodal command interface. The default dry-run makes zero model requests.

## First review extraction

Use the project environment (Pillow, jsonschema and pytest are installed):

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.decision_state_discovery \
  --runs /mnt/newssd/sja/clothAgent_real/runs /mnt/newssd/sja/clothAgent_real/local_results \
  --subgoal right_sleeve --model gpt-6-astra --batch-size 12 --seed 0 \
  --output results/decision_state_discovery_20261012_goal_v2 --dry-run
```

Inspect `manifest.json`, `extraction_summary.json` and `report.md`. The CLI prints a manifest hash. To authorize model analysis, repeat the same data/model/config arguments with `--execute --accept-manifest <that hash>` in place of `--dry-run`. This explicit hash is an extraction review gate, not a robot execution authorization.

`--subgoal auto` deterministically selects the label with the most eligible before images (lexical tie-break). Keep the same subgoal option when resuming; to change the dataset/config, use a new output directory. The initial local audit selected `right_sleeve`. Auto-selection uses only explicit task metadata, never a model's visual guess.

## Explicit task goal (protocol v2)

The first image-only experiment returned `INSUFFICIENT`: a `right_sleeve` label did not specify the side convention or desired terminal condition. Version 2 supplies the same explicit `task_goal` to discovery, aggregation, classification, action review and outcome review. It does not add historical plans or evaluations to Stage 1.

The currently reviewed mappings are `left_sleeve` and `right_sleeve`. The offline adapter parses static Python literals (without importing robot modules) from `fold_frame.FRAME_RULE`, `fold_exploration_pipeline.FOLD_STEPS` and its static sleeve completion rubric:

- Left/right is viewer-left/right after mentally rotating collar above hem, without mirroring, not wearer anatomy or fixed camera sides.
- The goal is sleeve inward: its distal lobe no longer protrudes outward and lies over/inboard on the torso. The existing task rubric does not require a perfectly flat cuff.
- These are desired conditions, never claims that an observation is complete. An already-complete/no-further-manipulation decision is allowed. Runtime GPT still discovers all factors/classes.

`manifest.json:task_contract` and `task_goal.json` freeze the goal, its hash, source excerpts, symbols, lines and file hashes. Only the semantic goal reaches GPT, not source paths or historical record fields. This is the repository's static contract at extraction time, not proof that every historical run had identical prompt wording. Images remain before RGB only in Stage 1; no historical target/after image is added.

Unsupported/missing static goals are marked `UNAVAILABLE` during dry-run and block model calls. Add a reviewed, source-backed mapping before analyzing another subgoal; do not silently fall back to label-only analysis. Do not insert observation-dependent labels, action choices, or evaluations into a task definition.

Use a **new output directory** for v2. Old v1 output is rejected before writes or model calls, preserving the earlier negative result. Goal changes alter the prompt/cache key; the manifest hash also binds the task contract. Review the new extraction summary and `task_goal.json`, then use its new manifest hash with `--execute --stage image_only --max-calls 12`.

## Historical adapter and missing evidence

The `fold-record-v1` adapter scans `iteration_*` and `round_*` directories, preserving every candidate in the manifest. Eligible input requires:

- `record.json` (or `partial_record.json`) `planned_step`, falling back to `supervisor_before.current_step`, with exact-label identity mapping and source provenance.
- Explicit `action_mode=FOLD` (or `mode=FOLD`). Repairs and acquisition probes are not folded into that label.
- An existing readable clean Camera A RGB from `before_raw/camera_A_rgb_upright.png`, `camera_0_A.png`, `camera_A_rgb.png`, or the same known filenames in `before_images`. No overlays or segmentation previews.

Actions come only from `execution_proposal.actions` or `execution_plan.json:actions`. Only numeric `move(x,y,z,yaw)` and named open/close/home commands are accepted. Observation descriptions, planning reasons, success fields and other plan metadata are stripped. Raw action commands are preserved alongside Pa/Pb/delta, if a unique close→release cycle makes those endpoints unambiguous. Pa is the last move before closure; Pb is the last held move before first release. These are commanded Cartesian coordinates in the legacy API's millimetres, not measured contacts; cross-run frame registration is not established. No bbox/mask normalization is fabricated.

After images use the same Camera A filename whitelist in `after_raw`/`after_images`; no next-iteration image is silently substituted. Stage 3 requires `physical_execution=true` plus after imagery or historical evaluation. Evaluator judgments and host policy labels are separately identified as non-ground-truth. In particular UNCHANGED/ACQUISITION does not prove empty jaws, Z error or mechanism. Missing after imagery can leave a sample eligible using recorded evaluation, but its absence is explicitly exposed to GPT.

Unknown format / missing subgoal Codex rounds stay in the manifest with skip reasons. IDs are opaque hashes of resolved source iteration paths; run and iteration provenance are host-only. Duplicate before-image hash groups are reported and not silently removed. No claim of independent trials or generalization to other garments is made.

## Model discovery and leakage boundary

Stage 1 requests contain only opaque sample IDs, subgoal, the static `task_goal` and before RGB attachments. Batch discovery proposes factors before provisional classes. Three-way hierarchical aggregation merges structured summaries with bounded pages of representative/ambiguous thumbnails; all covered IDs must survive. Missing representatives are explicitly listed and never treated as evidence of absence. A separate classification pass applies frozen `taxonomy_v0` to every before image, and predicts an image-only next-action family. Unknown/new-class responses are legal.

Stage 2 adds sanitized raw commands and normalized displacement, compares action equivalence, seeks both forms of visual/action counterexample, and emits `taxonomy_v1` plus revised assignments. Stage 3 adds after images, execution telemetry and evaluator/policy evidence, reviews transitions and aliasing, and emits `taxonomy_v2`. Factor/class names and all merge/split decisions come from runtime GPT, never host heuristics. No state count is enforced. Q1–Q6, confidence, negative results and limitations are required in the final schema. A history-conditioned representation may be proposed, but this adapter does not provide pre-before history; the model must not claim such a hypothesis has been validated.

Every request is a NEW ephemeral Codex invocation. It never resumes the robot planner session. Reuses `codex_astra_pipeline.build_command` and real `--image` attachments, with a separate experiment-only subprocess wrapper. Bubblewrap mounts only a fresh request workspace, the static Codex binary, selected system TLS/DNS files and the CLI authentication file. Host repository, original runs, earlier outputs, user instructions, session history and MCP configuration are not mounted. No shell/Python executable or robot devices are provided; shell feature and web search are disabled; subprocess environment is whitelisted. Auth is mounted read-only and is never copied into artifacts. The approved skill library is neither read nor written.

Isolation fails closed: missing/unsupported bubblewrap or a non-static Codex deployment is an error, with no unsafe host-workspace fallback. This host's static `codex-cli 0.162.1` startup was checked inside the namespace without any API call. Actual authenticated GPT inference still needs to be tested after reviewing extraction. If the deployment requires a proxy or custom model provider, this strict adapter does not inherit it automatically; configure a separately reviewed allowlist rather than weakening filesystem isolation.

## Resume, versions and cost control

```bash
# After dry-run review; same --runs/--subgoal/--model/--batch-size/--seed/--output:
... --execute --accept-manifest <hash> --stage image_only --max-calls 12
... --execute --accept-manifest <hash> --stage action --max-calls 12
... --execute --accept-manifest <hash> --stage outcome --max-calls 20
# Or --stage all (default).
```

All prompt/schema hashes, image content hashes, taxonomy hashes, model, structured responses, elapsed time, available usage, retry count and cache hits are recorded in `requests/`. Cache keys include the protocol, stage, model, prompt/schema, ordered image IDs/content and taxonomy version hash. Successful structured responses are validated on cache reuse. Default format repair limit is 1; `--format-retries` allows 0–2. Transport failures are logged and stop; no hidden infinite retries. `--max-calls` bounds actual invocations per CLI run including repairs. Re-running resumes through cached requests, without calling GPT again for matching inputs.

`--no-cache` forces model calls rather than using response cache; frozen taxonomy/assignment files still cannot be overwritten with a different value. Use a new output directory for an intentional fresh experimental replicate. A resumed run keeps its original config and manifest; source images are checked against extraction hashes. The seed fixes batch order only; it does not guarantee deterministic model generation. Model inputs strip image metadata; originals are never modified. Generated montages annotate only sample IDs.

## Artifacts

```text
manifest.json / manifest.sha256.json / extraction_summary.json / config.json / task_goal.json
invocations.jsonl
stage1/batch_*.json / aggregate*.json / taxonomy_v0.json / assignments.json
stage2/action_review.json / taxonomy_v1.json / assignments.json / skipped.json
stage3/outcome_review.json / taxonomy_v2.json / assignments.json / skipped.json
stage*/montages/              # opaque class filenames, labels sidecar + index
images/                      # RGB-only copies; before/after not globally exposed
cache/ / requests/
report.md / report.json
```

Model-produced versions and assignments have immutable hash sidecars. Missing prerequisites or hash changes stop later stages. Empty evidence cohorts produce UNAVAILABLE status rather than an invented review. Reports on a dry-run explicitly show NOT_RUN for all three stages and no final taxonomy. Model results are candidate research artifacts only.

Validation:

```bash
/home/sja/miniconda3/envs/cali/bin/python -m pytest tests/test_decision_state_discovery.py -q
```

## Prompt-only execution-variation comparison

Use `--stage1-focus execution` to ask about execution-relevant variation **within the same objective**, rather than merely grouping observations by their next manipulation objective. Every proposed factor must connect visual evidence to a changed execution choice; shared objectives alone do not establish execution equivalence. No factor vocabulary or number of classes is prescribed. The discovery, aggregation and classification instructions change, but the schema, task goal, multimodal attachments and action/outcome disclosure rules stay the same. The existing assignment field `next_action_family_without_historical_action` now asks for a prospective execution approach and its distinguishing conditions, or an explicit lack of evidence. Stage 2/3 instructions are unchanged.

For a controlled dataset comparison, use `--reuse-manifest <baseline>/manifest.json`. Its SHA256 sidecar is required. The manifest, selected IDs, image hashes and task contract are copied exactly; extraction/selection is never rerun. Keep model, batch size, seed and other settings identical to the baseline. The reuse path/hash and focus are frozen in config. Model randomness and model-chosen representative images can still differ; this is not deterministic causal proof.

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.decision_state_discovery \
  --runs /mnt/newssd/sja/clothAgent_real/runs /mnt/newssd/sja/clothAgent_real/local_results \
  --subgoal right_sleeve --model gpt-6-astra --batch-size 12 --seed 0 \
  --reuse-manifest results/decision_state_discovery_20261012_goal_v2/manifest.json \
  --stage1-focus execution \
  --output results/decision_state_discovery_20261012_execution_v3 \
  --stage image_only --execute \
  --accept-manifest 4b668cdc30b1f5531e0797ea1d536cf02ea8a4f69c4e36df8f6a3bec90090e00 \
  --max-calls 12
```

Use `--dry-run` instead of `--execute --accept-manifest ...` to prepare without model calls. Do not reuse the baseline output directory or expose its model responses to the new image-only requests.

## Frozen taxonomy action-overlay audit (no outcomes)

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.action_overlay_review \
  --source results/decision_state_discovery_20261012_execution_v3 \
  --output results/decision_state_discovery_20261012_action_overlay \
  --execute --max-calls 16
```

Omit `--execute` for projection preparation only. This independent audit reuses the baseline manifest, taxonomy, image-only assignments, model and seed. Only the action-eligible cohort participates. It never regenerates Stage 1 or emits taxonomy_v1. Source hashes are checked before and after analysis; proposed extra factors are suggestions only.

For each current before RGB, the adapter follows its recorded upright mapping to the corresponding perception's Camera A intrinsics and `X_base_camera`. It verifies exact pixel equality of the rotated raw RGB and checks recorded guide XYZ-to-pixel residuals (max 1 px permitted). Pa/Pb use the historical command endpoints and original Z, transformed from base millimetres to camera metres, then projected and rotated clockwise. Recorded extrinsics already include manual base offsets. No replacement with nearest surface point, invented normalization, or off-image clamping is allowed. Missing/invalid projection is marked UNAVAILABLE. This consistency check is not an independent test of physical calibration accuracy.

Overlays mark Pa cyan, Pb magenta, and endpoint displacement yellow. This is not a measured grasp, proof of execution, or the full trajectory. Each batch receives clean before RGB plus overlay, bounded to 12 total attachments by default. All-sample sheets support cross-batch comparison; the model's proposed counterexample pairs are checked again at full resolution, then consolidated. The schema requires per-sample grasp/target/displacement explanations, common families, supporting evidence, both counterexample types, unexplained variation and proposed-factor status. Original Stage 1 taxonomy stays frozen, including when action evidence contradicts it.

The input manifest is the existing extraction artifact; only explicitly whitelisted action/before/projection/Stage1 fields are copied to this audit. The audit does not open evaluation files, execution results or the current sample's after observation. Some before files reuse a previous iteration's capture; their role remains the current before. Model calls remain isolated ephemeral multimodal requests. Reports and every request include audited cache/elapsed/retry information. Resume with the same command reuses validated cache; incomplete runs stop at the finite call budget or repair limit.

Artifacts: `manifest.json`, `frozen_taxonomy.json` (+hash), `overlays/*.png`, `montages/*.png`, `stage2/batch_*.json`, `aggregate_initial.json`, `pair_check_*.json`, `action_review.json`, `requests/`, `cache/`, `report.md`, `report.json`.

## Stage 3A: controlled outcome pair audit

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.controlled_outcome_audit \
  --source results/decision_state_discovery_20261012_action_overlay \
  --execute --max-calls 48 --workers 2
```

Omit `--execute` for evidence extraction only. Outputs are under `<source>/stage3a/` (or `--output <directory>/stage3a/`): `manifest.json`, `samples/<sample_id>.json`, `pair_reviews/<pair_id>.json`, `pairs.json`, `report.md`, `status.json`, images, cache and request audits. No Stage 1/2 rerun, taxonomy revision, training or policy artifact is produced. Source taxonomy/assignments and Stage 2 review hashes are checked.

Pairs are selected **before reading outcomes**, from Stage 2 appearance-similar/action-different evidence. The four user-requested depth comparisons are prioritized. Numeric candidate grouping uses grasp XY distance <=2 mm and release XYZ distance <=1 mm for Z pairs (Z difference >=0.5 mm); grasp XY <=2 mm with target XY difference >=10 mm for target pairs; grasp XY difference >=10 mm for subregion candidates. These thresholds do not establish matched appearance or isolated variation. Within-panel changes and multiple-variable changes remain candidates for GPT to qualify or reject. Every changed command variable, including yaw and lift sequence, is supplied. No outcome-based pair selection occurs.

Each sample is assessed in a separate fresh multimodal call without pairing information or other samples' assessments. Before, after, pre-close and post-lift evidence are supplied with source IDs and provenance hashes. Extra lift captures, if numerous, use a labeled sheet; unavailable evidence is explicitly recorded. Historical evaluation and supervisor/host labels are fallible claims. Future experiment suggestions are excluded. Simulated trajectory/preflight feedback is not physical evidence: only non-simulated `execution.actual_robot_actions` and snapshot events are retained. Controller success/contact is not proof of cloth acquisition. Fixed-perception before/after and moving-wrist grasp views must not be conflated.

The sample schema separates acquisition, task progress, completion, side effect and actual execution reach. Every status, including UNKNOWN, requires evidence citations validated against the supplied source registry. Each pair then receives a fresh GPT call with original evidence, independent assessments, Stage 2 pair rationale and full command differences. It rechecks visual comparability, confounds, actual execution reach and both outcome dimensions. NO_OUTCOME_DIFFERENCE cannot substitute for UNKNOWN acquisition/progress. Parameter-effect labels require supported retrospective association; no single failure proves causation, deeper Z is not automatically better, and UNCHANGED is not empty jaws. Target-distance effects cannot be tested if the transport was withheld.

The report groups all three pair types and lists strongest supported parameter associations (or explicitly none), multiple-action feasibility, unresolved cases and possible RGB aliasing. Ranking uses the runtime model's evidence-strength rating, not outcome desirability. Positive feasibility requires positive acquisition and task-progress evidence for both actions. No hidden-state or causal conclusion is forced.

`--workers 1|2` bounds concurrent independent calls. A shared lock enforces `--max-calls`; finite format repair, prompt/image/schema/taxonomy caching and request audits reuse the existing interface. On failure, completed immutable sample/pair files remain available; repeat the same command to resume through validated cache. Lower the remaining per-invocation budget as needed. Freeze hashes protect prior stages; changing evidence/config requires a new output directory. These are retrospective comparisons, not randomized causal experiments.
