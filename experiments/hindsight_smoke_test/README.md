# Hindsight Skill Discovery v0 — Smoke Test

```bash
cd /mnt/newssd/sja/clothAgent_real
PYTHONPATH=. /home/sja/miniconda3/envs/cali/bin/python -u experiments/hindsight_smoke_test/run.py
```

Output: `reports/hindsight_smoke_test/review.html`.

The full experiment remains paused. This runner has no robot connection or execution entry. It reads only the existing raw-experience index and raw images/intentions, never the full run's model results or previous discovery reports.

Frozen selection: reserve four different runs by seeded hash before CV; measure all indexed pairs; choose 4 high, 3 medium, 3 low deformation and 2 seeded random control pairs, preferring distinct runs and contour shape buckets. CV scores are sampling diagnostics, not semantic reveal labels. Held-out CV and After data never enter learning or action generation; validation selection is fixed independently of CV scores.

Normally at most 14 calls (6 observation batches, intention comparison, review of at most two cases, one induction, one synthesis, four dry-run calls). The transport enforces a 20-call cap, including recorded attempts. This smoke transport makes one attempt per logical call. Errors are retained and reported rather than counted as negative effects.

If independent review confirms no unplanned semantic reveal, the run ends with NO_VALID_DISCOVERY; hypothesis/synthesis/dry-run stages are explicitly untriggered. A generated candidate, if any, is an LLM-grounded Python function with learned strategy content, not a trained vision model or proven physical Skill. Dry-run tests the generated function on only held-out Before RGB.

Test:

```bash
PYTHONPATH=experiments/hindsight_smoke_test /home/sja/miniconda3/envs/cali/bin/python -m unittest discover -s experiments/hindsight_smoke_test -p 'test_*.py'
```

Do not modify frozen Python code or prompts mid-run. Existing successful requests are reusable only under the same protocol. Every model input, source image hash and raw output is retained in `calls/`.
