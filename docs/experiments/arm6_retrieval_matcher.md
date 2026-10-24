# Arm6 DINO retrieval + GPT matcher

`retrieval_matcher.py` is an optional state matcher for the existing Arm6 replay runner. Each new RGB is encoded once with DINOv2-small; the precomputed FAISS HNSW index returns Top-10 reference images; one existing GPT node-matching request compares those images with the query. Recognition returns real graph node IDs to the original replay code.

References in Top-10 that belong to the same node are grouped under that node. This is the production adaptation from the benchmark's singleton-image candidates: two representative images of one graph state should not become two competing states. The node descriptions, image-based matcher prompt, model, reasoning effort and confidence gates are retained. The matcher does not receive edges, the expected next node, action history, or grasp coordinates. It does not select actions or plan motion.

## Start on this machine

The complete archive graph is indexed: **148 references / 77 observed nodes**, including the earlier folding and newer unfolding states. Of those embeddings, 120 were reused from the verified benchmark cache and 28 were encoded offline. This preserves the previously selected complete-graph replay scope. Unsupported trajectory actions remain unavailable.

```bash
cd /home/sja/clothAgent_real
conda activate cali
python -m cloth_agent.experiments.graph_policy_replay \
  --graph reports/arm6_archive_graph_20261019/merged/graph.json \
  --arm6-expanded-graph \
  --matcher retrieval_gpt_rerank \
  --retrieval-memory reports/arm6_retrieval_matcher_20261019/memory \
  --execute --max-steps 20 --no-cache \
  --output "reports/arm6_retrieval_replay_$(date +%Y%m%d_%H%M%S)"
```

This command physically runs **Arm6 / Camera A** using the existing CollectionRuntime. It still observes after each action, verifies the result, grounds the selected demonstration against current RGB-D and uses the existing Arm6 motion/gripper implementation. Initial matching, post-action matching and depth-view-recovery rematching all use the selected matcher.

The old matcher remains the default when the new options are omitted. To explicitly select it, use `--matcher current` and omit `--retrieval-memory` / `--retrieval-python`.

AMBIGUOUS, UNKNOWN (legacy replay status `NO_MATCH`), low-confidence/contradictory matches and worker/model failures do not get converted to matches. There is no hidden full-gallery GPT fallback, and a broken GPT response is not retried inside the retrieval matcher. Grounding/action-selection requests retain their existing behavior. The shared `--max-calls` budget includes retrieval reranking and those other model requests.

## ML environment

The robot runs in `cali`, which remains free of Torch/FAISS installs. A persistent local subprocess runs DINO and FAISS in:

```
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python
```

It loads the model and index once, then handles query-only requests until replay exits. No gallery encoding happens in replay. Worker initialization and graph/index checks happen before camera/controller construction. Worker exceptions stop the replay; cleanup closes the worker as well as the existing robot resources.

The worker interpreter is selected from `--retrieval-python`, otherwise the current interpreter if it has Torch/Transformers/FAISS, otherwise the installed cache environment above. `--retrieval-timeout-s` defaults to 120 for local startup/query IPC, separate from the GPT timeout. `--retrieval-top-k` defaults to 10.

To rebuild this environment on the same host:

```bash
/home/sja/miniconda3/envs/molmo/bin/python -m venv --system-site-packages \
  /home/sja/.cache/cloth_agent/visual_retrieval_env
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python -m pip install faiss-cpu==1.15.1
```

It inherits the existing Torch 2.11.0+cu128 / Transformers 4.57.6 installation. On another machine, provide an environment with compatible Torch, Transformers, NumPy, Pillow, jsonschema and FAISS; pass its Python explicitly. The model checkpoint is local, not downloaded during replay.

## Prepare a different graph

Graph, RGB, embedding and index hashes are checked. A changed graph requires a new memory, even if node IDs look similar. Preparation covers every observed graph node rather than silently reusing the benchmark's unfolding-only subset.

```bash
/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python \
  -m cloth_agent.experiments.visual_retrieval_index prepare \
  --graph reports/arm6_archive_graph_20261019/merged/graph.json \
  --model-path /mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small \
  --reuse-memory reports/unfold_visual_retrieval_benchmark_20261019/memory \
  --output reports/my_new_retrieval_memory
```

Use a new output directory. `--reuse-memory` is optional and reuses only vectors whose image IDs, image hashes and encoder contract agree; it does not reuse an incomplete subset index. A fresh full-graph HNSW index is built offline. No synthetic scaling vectors are used in replay.

## Check without moving the robot

Static replay preflight validates graph templates and retrieval memory. It makes no model or robot call and does not start the GPU worker:

```bash
python -m cloth_agent.experiments.graph_policy_replay \
  --graph reports/arm6_archive_graph_20261019/merged/graph.json \
  --arm6-expanded-graph --matcher retrieval_gpt_rerank \
  --retrieval-memory reports/arm6_retrieval_matcher_20261019/memory \
  --dry-run --output reports/my_retrieval_preflight
```

Exercise actual DINO retrieval on a saved image:

```bash
python -m cloth_agent.experiments.retrieval_matcher \
  --graph reports/arm6_archive_graph_20261019/merged/graph.json \
  --memory reports/arm6_retrieval_matcher_20261019/memory \
  --image /absolute/path/to/saved/rgb.png \
  --exclude-self --output reports/my_retrieval_image_check
```

Add `--gpt` for one actual GPT rerank, still without camera/robot access. Offline `--exclude-self` removes gallery references with the same image file hash; live replay uses its fresh observation and does not need benchmark self-exclusion rules. This CLI is an integration smoke test, not a new latency/accuracy benchmark.

## Evidence and tests

Each replay saves `matcher_config.json`, worker stderr, per-step `matching/retrieval.json` (ranked references, real node IDs and local timing), and `matching/matching_attempts.json` (shortlist, GPT result, timing and errors). Existing `model/requests.jsonl`, `result.json`, timing and replay video remain available.

`matching_metrics` includes local decode, DINO preprocessing/forward, ANN search, retrieval total and GPT rerank wall time. The production GPT timer includes attachment preparation and request logging; unlike the instrumented benchmark, it does not separate temporary PNG generation into a distinct stage. Do not compare that field directly with the benchmark's narrower GPT stage. Total matcher wall time is separately recorded.

The prior exploratory benchmark's data/config are unchanged. Its six original matcher source files were archived in `reports/unfold_visual_retrieval_benchmark_20261019/run/original_matcher_sources/` before this integration, because the benchmark's original source hashes must not silently refer to changed replay code.

```bash
python -m pytest tests/test_retrieval_matcher.py tests/test_arm6_traj_replay.py \
  tests/test_graph_policy_matching.py tests/test_replay_cli_failure.py -q
```

Offline results from this integration are in `reports/arm6_retrieval_matcher_20261019/`. They validate wiring and a saved-image match; they do not establish live robot task success or unseen-state matching accuracy.

### Fixed-route assisted unfolding trial

```bash
bash scripts/arm6_unfold_trial.sh
```

This runs the existing seven edges `e103,e104,e105,e106,e107,e108,e109`, from
`n092` through `n093,n094,n095,n096,n097,n098` to `n077`. The graph and its image
index remain frozen. Completion requires all seven actions and confirmation of
the final state; outgoing edges at `n077` are not executed afterward.

The launcher enables `--arm6-live-start`: after moving to the perception pose,
the arm stays stationary while the left panel streams Camera A and the right
panel shows the `n092` source reference. Arrange the cloth, remove your hands,
then click **Recognize**. This captures a fresh RGB-D observation and freezes the
preview while the existing matcher runs. **Continue** becomes available only
when that snapshot reliably matches `n092` at the configured threshold. Other
matches, ambiguity and unknown states do not qualify. **Resume live** discards
the previous match and lets you rearrange and try again. Recognition is explicitly
triggered, not called on every video frame. Failed and accepted attempts are
saved under `start_placement/`. The accepted RGB-D and match are reused for the
first step, without a duplicate model call or stale initial depth. Do not move
the cloth after recognition; subsequent anchor confirmation is still required.
Stopping during recognition waits for the current request to finish before
releasing the shared camera; no grasp is dispatched from this window.

The opt-in `--arm6-route` argument validates a continuous path of supported
actions before connecting hardware. Without `--arm6-review`, normal confidence
checks still apply and a different recognized node stops the fixed-route trial.
With `--arm6-review`, each step shows the current image beside the demonstration
with its original clicked anchors. The operator must explicitly confirm the
expected state and accept or correct the current two anchors. Closing the window
stops. A new depth-recovery image requires a new review. Depth, IK and execution
checks remain in the existing Arm6 path.

Original model matches and confidences are retained. Operator decisions, edited
points, source-image hashes and review time are saved separately under each step's
`anchor_review/` (and `goal_review/` at the end). `result.json` identifies an
operator-assisted trial; this is not reported as autonomous success.

Outputs under `reports/arm6_unfold_trial_<timestamp>/` include the before/after
RGBs, `timing.json`, a timed side-by-side `replay_summary.mp4`, and chronological
`replay_sequence.mp4` at 0.5 seconds per unique consecutive captured RGB. These
are keyframe videos, not continuous recordings of robot motion. A stopped run
retains its available frames and stop status.

### Local appearance retrieval baseline (superseded for structural matching)

This crop-based baseline does **not** implement explicit cloth-layer morphology
matching. It remains available as a comparison; do not use its launch script to
test the structural prototype below.

`arm6_local_action_policy` retrieves actions from local image regions. It does
not invoke graph node matching, follow graph edges, or assume the cloth belongs
to the demonstration's whole-image state. The existing graph replay is unchanged.

```bash
bash scripts/arm6_local_unfold.sh
```

This starts a maximum of three reviewed Arm6 actions. Before every action, the
operator reviews local applicability and accepts or edits current grasp/target
pixels. Model confidence is recorded without a score gate in this reviewed
pilot; it is not raised by operator approval. STOP/DONE produce no motion. Depth
and IK planning use the existing Arm6 demonstration executor. Missing depth
stops this first pilot rather than changing the camera and reusing old pixels.

The prepared memory at `reports/arm6_local_action_20261020/memory` contains 86
supported actions from the explicitly selected `20261019` unfolding collection
cohort. Inward moves that can expose hidden cloth remain included; neither
movement direction nor the historical action name establishes success. Earlier
folding sessions and unsupported actions are excluded, with reasons recorded.
Every action retains its exact trajectory BEFORE/AFTER, original anchors and
template. Merged-node representative images are never substituted for outcomes.

Offline reference crops span 22% and 44% of the shorter image dimension around
the demonstrated grasp, augmented by rotations of 0/90/180/270 degrees. All 688
DINOv2 crop embeddings and their HNSW index are precomputed. At query time,
overlapping windows at the same scales scan the whole RGB. The best crop pair
per action supplies a Top-6 action shortlist. These uncalibrated similarities
are retrieval hints, not action confidence or success estimates. Ordinary
texture/background matches remain possible.

One GPT request normally selects an action and transfers both anchors together,
using the full current image plus candidate boards showing the demonstration
before/after and the matched local neighborhoods. A contract/transport failure
can invoke the existing Requests repair retry; actual calls are logged. The
chosen grasp must lie inside its retrieved region; target placement uses current
global context. Subsequent decisions see the previous before image and action,
and the last two action IDs are temporarily excluded to discourage immediate
repetition. This exclusion is a pilot heuristic, not a learned progress policy.

```bash
# Offline query; never opens a camera or robot connection.
python -m cloth_agent.experiments.arm6_local_action_policy \
  --library reports/arm6_local_action_20261020/memory \
  --image path/to/current.png --output reports/local_action_test
# Add --retrieval-only to skip GPT.
```

Artifacts include `retrieval.json`, per-candidate visual boards,
`proposal.json`, `proposal_preview.png`, operator review, dispatch/execution
records, per-phase timing and before/after recap videos. Each search currently
starts a fresh ML subprocess; setup time is reported separately and included in
its wall time. This first implementation is a pilot, not an accuracy or latency
benchmark. Offline pixel proposals do not validate depth, IK or physical effect.

### Offline local structure prototype

`cloth_agent.experiments.local_structure_matcher` independently parses reference
BEFORE images and the current RGB into visible surfaces, free edges, fold lines,
occlusion edges, opening rims and exposed table regions. Directed relations
record boundary membership, fold connections, coverage and adjacent regions.
Reference AFTER images describe the observed effect, not hidden BEFORE topology.

Typed, injective subgraph matching requires the same relation directions and
entity types; extra relations among mapped entities reject that correspondence.
No image embeddings, crop similarity or whole-garment state IDs enter this stage.
Only matched candidates can reach a separate visual review/anchor transfer call.
All commands are offline and have no hardware execution option.

```bash
python -m cloth_agent.experiments.local_structure_matcher build \
  --source-library reports/arm6_local_action_20261020/memory \
  --edges e085,e103,e104 --output reports/structure_library_test
python -m cloth_agent.experiments.local_structure_matcher inspect \
  --image path/to/current.png --output reports/structure_scene_test
python -m cloth_agent.experiments.local_structure_matcher query \
  --library reports/structure_library_test/structure_library.json \
  --image path/to/current.png --scene-dir reports/structure_scene_test \
  --output reports/structure_query_test
```

Outputs include original-pixel structure overlays, extraction evidence and
uncertainties, reference rejection reasons, and entity/relationship mappings.
When candidates exist, each has a side-by-side `comparison.png`; a successful
offline transfer also produces `proposal_preview.png`. `--no-transfer` skips the
final model call. Saved scene reuse checks image and annotation hashes.

The first pilot at `reports/arm6_structure_pilot_20261020` tested e085, e103 and
e104. All three were rejected: extracted structures contained exposed edges but
no sufficiently established fold/coverage context and unfolding effect. The
saved tangled query therefore returned `NO_STRUCTURAL_MATCH`, with zero robot
calls. This is an extraction/eligibility failure, not evidence that the current
cloth has no usable action. Synthetic relation tests pass, but successful real
action transfer has not been demonstrated. Exact relation requirements and the
0.65 annotation-confidence cutoff are prototype choices, not validated criteria
or calibrated probabilities. The existing graph runtime is unchanged.
