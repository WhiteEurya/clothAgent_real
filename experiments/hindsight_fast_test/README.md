# Hindsight Skill Discovery — End-to-End Fast Test

Offline only. This experiment never imports robot controllers or dispatches an action.

```bash
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -u experiments/hindsight_fast_test/run.py
```

Output: `reports/hindsight_fast_test_20261029/review.html`.

The frozen index contains 532 provenance-valid physical Before/After pairs. Grouping
uses the complete top-level run, keeping episodes from a batch together. Before
opening images, a fixed SHA-256 seeded ordering reserves three complete runs.
Only one Before from each reserved run is used after skill freezing. No held-out
After or executed action is sent to any discovery or validation model.

Contact sheets show the remaining pairs without policy rationales or CV scores.
Each fresh API session screens at most three sheets and nominates at most three
IDs. A frozen evidence ranking with run diversity selects at most ten pairs for
full-resolution analysis. Original intentions are then read from raw dispatch
records. One case is selected automatically using the frozen evidence order.
Provisional evidence is permitted, but cannot become confirmed physical success.

One induction request proposes competing explanations and observable action rules.
One synthesis request writes `generate_action(rgb, target_corner)` and its frozen
inference prompt. The generated function may use an injected, current-image-only
`VISION_MODEL`; therefore this is potentially a VLM skill, not necessarily a
deterministic OpenCV algorithm. The code is checked before execution. Three held-out
RGBs are then passed through the generated function. Pixel bounds are checked;
robot feasibility and actual reveal success remain untested.

The existing generic transport is reused. It uses ephemeral `codex -p api`
sessions, gpt-6-astra/high, two workers, no tools, one attempt per request, a
480-second request timeout and a hard cap of 20 attempts. Normal maximum: 16 calls.
All requests, images, hashes, responses, usage and latencies are preserved.

No prior discovery conclusions, hand-selected example IDs, expert skills or
conversation history are supplied to the models. The host assistant retains its
conversation context; this limitation is disclosed. Historical executed actions
may themselves reflect expert policies. Model visual judgments are not independent
physical ground truth. The experiment ends after this single offline run.
