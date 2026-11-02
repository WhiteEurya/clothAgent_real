# End-to-End Hindsight-Driven Skill Discovery v0

Start the fixed six-stage raw-data pipeline (resumable):

```bash
cd /mnt/newssd/sja/clothAgent_real
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -u \
  experiments/end_to_end_hindsight_v0/run.py \
  --root reports/end_to_end_hindsight_v0_20261029 --prepare-live
```

The first invocation freezes Python source hashes and protocol. Do not edit frozen Python files during a run. Reusing the root resumes successful API calls; each failed call has at most one preserved retry. Different code requires a new root.

Progress: `progress.json`; raw API attempts: `calls/`; final browser report: `review.html`.

A long offline run does not assume an operator remains present. The live preparation adapter captures a stationary RGB-D, invokes the frozen candidate on only that RGB, freezes its prediction, checks historical-image similarity, and uses the existing planner. It does not reset cloth or silently retry a rejected action. A matching historical scene is PENDING_NEW_STATE.

When an independent scene and the project's existing onsite execution permission are actually available, the same validation adapter accepts the existing execution flags:

```bash
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -u \
  experiments/end_to_end_hindsight_v0/validate.py \
  --root reports/end_to_end_hindsight_v0_20261029 \
  --prepare-live --execute --onsite-confirmed --operator sja
```

There is an exclusive one-dispatch guard. No second action/reset. Before/After observer does not receive skill, prediction, goal or expected verdict; the frozen target is compared only after blind observation. Any refinement is saved as unvalidated, not promoted.

Tests:

```bash
PYTHONPATH=experiments/end_to_end_hindsight_v0 \
  /home/sja/miniconda3/envs/cali/bin/python -m unittest discover \
  -s experiments/end_to_end_hindsight_v0 -p 'test_*.py'
```

## Boundaries

- Sources are raw native execution records and image pairs; no earlier discovery report or expert prompt is loaded into discovery calls.
- The orchestrator retains earlier conversation context. Model discovery requests are independent ephemeral sessions with stored inputs and tool-use audit. This is input-level isolation, not a claim that model pretraining has no garment knowledge.
- Historical actions may already reflect human-designed policy. Raw planner intentions are legitimate experience, not expert-free data.
- `CONFIRMED_SEMANTIC_REVEAL` is a model visual label, independently re-reviewed for selection, not instrumented material ground truth.
- All qualifying experience rows are observed; the one strongest reveal per row is indexed, additional effects may remain in evidence text. Candidate counts are experience-level, not an exhaustive count of every endpoint event.
- Incomplete/failed requests are reported, never silently counted as negatives.
- Existing scene adapter is reused without edits, including prior operator-established contact boundary and absent Arm7. No contact commissioning or safety override is introduced.
