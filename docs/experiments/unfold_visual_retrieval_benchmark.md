# Independent unfolding visual retrieval benchmark

This experiment measures the existing matcher before any optimization. It does not construct a robot runtime, connect to cameras/controllers, dispatch actions, or change the graph/replay implementation.

## Environment and commands

The first run uses a separate environment inheriting the installed PyTorch/Transformers packages from `molmo`; the collection environment `cali` is unchanged:

```bash
cd /home/sja/clothAgent_real
/home/sja/miniconda3/envs/molmo/bin/python -m venv --system-site-packages /tmp/unfold_visual_retrieval_benchmark_env
/tmp/unfold_visual_retrieval_benchmark_env/bin/python -m pip install faiss-cpu==1.15.1 matplotlib==3.11.2 scipy==1.17.1
```

Other first-run versions: Python 3.11, torch 2.11.0+cu128, transformers 4.57.6, numpy 2.4.6. The environment also needs Pillow and jsonschema. GPT conditions use the same `codex` binary, authentication, bubblewrap isolation and model transport as existing replay. Model calls incur the normal service usage; `retrieval_only` requires no model service.

Precompute the reference gallery and all scaling indexes **once**, before timing:

```bash
/tmp/unfold_visual_retrieval_benchmark_env/bin/python -m cloth_agent.experiments.unfold_visual_retrieval_benchmark prepare \
  --graph reports/arm6_archive_graph_20261019/merged/graph.json \
  --session-prefix 20261019 \
  --model-path /mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small \
  --output reports/unfold_visual_retrieval_benchmark_20261019/memory
```

`--session-prefix 20261019` selects the recent unfolding collection sessions in this archive, not earlier folding sessions. For a different archive, choose appropriate repeated `--session SESSION_ID` arguments rather than assuming the date establishes task intent. Only COMPLETE, operator-accepted trajectories are included; both saved before and after RGB observations are used. Frozen memory records the selected sessions, source hashes, individual image IDs, provenance and graph proxy groups.

Run all three modes, paired across 20 different queries, after a complete warmup for each mode:

```bash
/tmp/unfold_visual_retrieval_benchmark_env/bin/python -u -m cloth_agent.experiments.unfold_visual_retrieval_benchmark run \
  --memory reports/unfold_visual_retrieval_benchmark_20261019/memory \
  --output reports/unfold_visual_retrieval_benchmark_20261019/run \
  --repeats 20 --warmup 1 --model gpt-6-astra --timeout-s 600
```

Output directories must be new. To continue an interrupted run, use the exact same arguments with `--resume`; completed trials, including failed attempts, are retained. Resume performs fresh warmups and checks the frozen configuration. To run only the local pipeline, use `--modes retrieval_only` and a separate output directory; it cannot provide speedup against an unmeasured global baseline.

Regenerate CSV, statistics, Markdown and plots from completed trial records:

```bash
/tmp/unfold_visual_retrieval_benchmark_env/bin/python -m cloth_agent.experiments.unfold_visual_retrieval_benchmark report \
  --output reports/unfold_visual_retrieval_benchmark_20261019/run
```

## Conditions and measurement boundaries

- `retrieval_only`: decode RGB → unmodified DINOv2-small processor → FP32 CLS embedding → HNSW Top-20 (self and pixel-identical duplicates excluded). No candidate image loading is necessary for this mode.
- `retrieval_gpt_rerank`: same encoding → HNSW Top-10 → one invocation with query and ten candidate RGB images. The existing Arm6 common prompt, `node_match` prompt, schema and validation are reused. Each candidate reference becomes a singleton node ID; its existing visual signature is retained. MATCH returns that image ID; NO_MATCH maps to UNKNOWN. AMBIGUOUS/UNKNOWN has no fabricated winning candidate. No repair retry, planning, graph edge selection or action execution.
- `current_global_match`: existing `topology_match(..., pending=None)` global path, existing Requests and bounded contract repair. Same unfolding gallery, excluding the query and pixel-identical copies, with original grouped nodes. It does not attach unrelated earlier folding references. This is the current matching entrypoint on a controlled dataset, not timing the entire robot replay loop.

The timed runtime loads frozen gallery embeddings/indexes only; it never re-encodes gallery images. Query encoding is counted and asserted per trial (one for retrieval, zero for global). Accuracy evaluation separately encodes each image as a query; it does not reuse that image's gallery embedding as its query vector.

All wall-clock stages use `perf_counter`. CUDA is synchronized around DINO forward. CPU preprocessing is reported separately; forward includes transfer, network forward, normalization and return to CPU. Candidate loading measures graph/candidate preparation plus reference RGB decoding and temporary PNG copies already performed by the existing transport. GPT timing is client-observed invocation wall time minus image preparation, including CLI/isolation/network overhead; it is not pure server inference time. Query transport copies, image hashing, JSON serialization and request logging are in `other_host_s`. Stage means add to mean total; independently computed stage medians generally do not add to the median total.

No application response cache is used. Provider prompt caching and warm OS filesystem caches are not controlled. Conditions execute serially in rotating order. Do not run simultaneous GPU/model benchmarks if comparing these measurements. Warmups are recorded separately and excluded. Failed attempts remain in CSV; summary tables distinguish attempts/successes and statistics also retain all-attempt total latency.

## Accuracy and scaling interpretation

Labels are model-compiled graph groups approved by the operator as **temporary proxy labels**, not human-verified state truth. Recall@K is the fraction of eligible queries with at least one other same-group reference among Top-K. Mean fractional positive-set recall is additionally reported to remove ambiguity about the recall convention. Queries without another valid positive are listed but excluded from this denominator. Exact RGB duplicates and self IDs are excluded; near-identical consecutive captures can remain and inflate the score. A cross-session-positive hit-rate diagnostic is reported, but its gallery still contains same-session distractors and it is not an independently labelled test split.

HNSW scaling covers N=100,500,1000,5000,10000. Cyclic repeated embeddings are permitted **only for search latency**. At N=100 this is a truncated subset; above the real gallery size the vectors repeat. These indexes do not contribute to proxy recall, and do not model the accuracy or diversity of a future 10,000-image dataset. Scaling searches precomputed query vectors and excludes DINO/image decode time.

Index parameters are fixed for the first round: normalized inner product, HNSW M=32, efConstruction=200, efSearch=64, one FAISS CPU thread. No model/matcher tuning is performed. Model, processor and source hashes are frozen for reproducibility.

## Artifacts

- `memory/memory.json`, reference embeddings, reference HNSW index and five scaling indexes.
- `run/benchmark.csv`: every measured main-mode/scaling attempt and stage latency.
- `run/summary.md`, `statistics.json`: median / mean / p95, counts and speedup against the measured global median.
- `run/latency_breakdown.png` and `.svg`: mean stage breakdown and total median/mean/p95; `ann_scaling.png` for search-only scaling.
- `run/proxy_recall.json`, `recall_summary.json`: per-query labels, rankings, exclusions and proxy accuracy.
- `run/trials/`: actual request/response evidence and per-trial results; `warmup.jsonl` is excluded from benchmark.csv.
- `run/config.json`: paired query schedule, frozen memory hash, model and measurement definitions.

Protocol tests (no external model or hardware):

```bash
/tmp/unfold_visual_retrieval_benchmark_env/bin/python -m pytest tests/test_unfold_visual_retrieval_benchmark.py -q
```

References: [DINOv2-small model](https://huggingface.co/facebook/dinov2-small), [FAISS metrics](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances), [FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## First-run exploratory early stop

During the 2026-10-19 run, the operator changed the stopping decision after seeing a stable large latency difference: complete the ongoing tenth paired group, then stop all expensive GPT calls. Thus **n=10 is an outcome-dependent early stop, not the original design**. The original frozen `config.json` still specifies 20 paired groups and retains all twenty query IDs. `early_stop.json` records the decision, its time and the original config hash; `benchmark_source_at_run.py` preserves the timed implementation. Reporting code was subsequently changed only to disclose the exploratory status and enlarge the local-stage plot.

All raw trials and warmups remain available. Five ANN scaling conditions still receive 20 measurements each. Median/mean/p95 and speedup from the ten paired observations are descriptive; the stopping decision can bias effect-size estimates, and ten samples provide a particularly weak estimate of tail latency.

For **future authorized supplementation**, the original `run` command with `--resume` retains the original 20-repeat plan and skips the first ten completed groups. It performs fresh warmups and then measures the remaining ten query groups. Do not use `run --resume` merely to regenerate the early-stop report, because it starts additional GPT calls. Use `report` instead. The historical early-stop disclosure is retained even if supplementation later reaches twenty samples.

The tenth global call terminated with a transport/completion-validation error after 258.798 seconds (an unrecovered reconnect timeout), so the final dataset contains **ten completed paired attempts, but nine pairs with valid responses in all modes**. No replacement GPT request was issued. Primary n=10 tables/plots include all time-to-terminal observations, including that failure; `paired_success_statistics.json` and a separate table compare the same nine successful query pairs in all modes. The early-stop decision and boundary record are preserved separately from this failure-accounting annotation. Accuracy evaluation still covers all 120 real query images and is not reduced to the ten latency queries.
