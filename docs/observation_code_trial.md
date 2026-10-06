# Observation → code → observation pilot

`python -m cloth_agent.harness.observation_code_trial` tests one small offline loop:

1. Claude compiles one pure `prepare(request, source, available)` module from saved experience plus review corrections (text only).
2. The restricted interpreter checks the program and runs its generated recipe tests.
3. Claude sees current images and binds an information need, source image, ROI, rotation and enlargement request.
4. Host runs the fixed module and validates image identities, alignment and coordinate lineage.
5. Claude sees the resulting images, assesses whether the information was obtained, and chooses another binding or stops.
6. A text-only reflection summarizes the actual trace, code/binding limitations and a future revision suggestion.

The module cannot perform file IO or robot actions. Only Host crop/rotate/resize recipes are allowed.
A trial defaults to two module executions and twelve image operations; a longer trial can use
`--max-executions 5 --max-host-ops 24`. These are upper limits, not required iterations: stop when the
information is sufficient or the available method cannot resolve the gap. Each meaningful code batch is
preceded by a model observation and followed by another observation; individual deterministic crop/resize
operations inside one batch do not require separate model calls. The module source hash remains fixed.
Success means the scoped information task was judged sufficient, not physical correctness or generalization.
The trial does not run the full grasp planner or compare latency against baseline. UNKNOWN is a valid result.
Compilation may result in a module not needed in the current scene; NOT_EXERCISED must not be called success.

Artifacts include exact input/returned JSON per call, module.py/module.json, module_tests.json, observation
images and lineage, the ordered report.json trace, feedback.json, timing and token metrics. No production
skills are modified. Feedback proposes future edits; it does not silently alter the running module.

Compilation includes at most one repair call after a local validation failure. To continue with a
previously generated module, add `--module-from PATH/compile_returned.json` (or
`PATH/compile_repair_returned.json`). The module is validated again; it is not regenerated if valid.
The exact reused artifact and any repair request/response are retained. Pure code supports bounded
list concatenation for constructing recipes; sequence multiplication and arbitrary execution remain
unavailable. Compiler tests check interface/recipe consistency, not whether the visual method is useful.

`--initial-observation-from PREVIOUS_RUN` can reuse its first returned observation after an interrupted
trial. The exact prompt (including image identities, goal, module spec and limits), image count and
module implementation hash must match. Subsequent observations are fresh. Reuse is explicit in the
report and excluded from new call costs; consult the original run for the cached call's cost/time.
Before any operation, an UNKNOWN assessment with no cited images is normalized to null and logged;
this compatibility rule never turns UNKNOWN into a successful information result.
`--observations-from PREVIOUS_RUN` applies the same exact-match checks to all available responses,
replaying deterministic Host edits locally; missing responses are obtained through fresh calls.
This permits completing feedback without repeating expensive observations after an interface error.
If DONE/UNKNOWN contains a leftover request, the stop decision takes precedence: the request is retained
as an `unexecuted_suggestion`, never executed. Raw responses remain intact. DONE still cannot override
an insufficient/unknown last result.

Example:

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.harness.observation_code_trial \
  --evidence results/reasoning_k3_prepared_20261003/evidence/evidence.json \
  --experience results/experience_review_20261005_094428/experience.json \
  --review-notes results/experience_review_20261005_094428/review_notes.md \
  --output "results/observation_code_trial_$(date +%Y%m%d_%H%M%S)" \
  --backend remote --ssh-host company-planner \
  --call-timeout 600 --max-executions 2 --max-host-ops 12
```
