# Execution-robustness planning

`cloth_agent/action_robustness.py` supplies the shared short planning policy. The
current physical action loop (`codex_loop.loop`, including seed, free-plan recovery,
unfold and IK code repair) appends it after the existing task/coordinate contract.
Replay full/minimal grounding uses the same preferences within the selected edge's
immutable primitive/template constraints. Minimal grounding does not redesign motion.

As of 2026-10-17, candidate ordering is: reliably move the intended cloth region;
carry that region fully across the fold line into a supported receiving region;
tolerate small Pa/Pb and trajectory errors; then minimize unnecessary disturbance.
For sleeves and torso inward folds, favor lift-and-transfer / arc-like motion,
an interior Pb with margin from the fold line and receiving boundary, and moderate
overshoot within the receiving region and existing Host constraints. Plan supported
placement before release to avoid a suspended bridge and rebound. Existing release
and retreat remain Host-owned; choose placement/pose to avoid pushing the fold back.
These are planning objectives, not a claim that support/contact has been verified.

Normal multilayer folding is allowed. Exact layer identity and a visible lower-layer
edge are not prerequisites. Local overlap or layer ambiguity alone must not reject
a candidate that can reasonably carry the target region, including adjacent layers.
`NO_ROBUST_ACTION` is limited to no viable candidate due to: inability to confirm
the grasp is on cloth; clearly inadequate jaw closure space; potentially grasping
unrelated structures; no handle useful for the target manipulation; or an actual
Host safety/workspace/IK/collision rejection. Consider alternative useful handles
before refusing. Inability to establish Host safety from RGB is not a Host rejection.
When refusal is necessary, use `motion.blocked(reason="NO_ROBUST_ACTION: ...")`,
without Pa/Pb or motion commands.
Replay uses its existing null/zero-confidence anchors or
`NEED_MORE_INFORMATION` result with empty slots. The gate adds no confidence
threshold, scores or explanation fields; diagnostic tags cannot authorize dispatch.
The historical `low_layer_ambiguity` tag stays schema-compatible for old logs but is
not required or a dispatch criterion. Existing anchor-localization validation remains
unchanged; the prompt does not treat uncertain layer identity as a missing parameter.
Static RGB need not establish true `safe_retreat`: omit that tag when unsupported;
its absence does not fail the grasp gate. In the seed/free-action loop, a blocked
action still skips that round. Frozen-Graph replay instead uses the bounded
recovery described below. Unknown future
cloth effects are not invented visual violations. Host validation remains mandatory.

The Python answer may prepend one `# ACTION_ROBUSTNESS {...}` comment containing
`tags`, `confidence`, and a short `reason`. In unfold mode it lives inside the
existing JSON `code` string. Grounding adds one optional `robustness` object with
the same fields, preserving compatibility with historical responses.

The host saves `action_robustness.json` and removes this metadata before passing
code/candidates to execution, iteration history, or Graph consumers. The raw model
answer remains an audit artifact. Logs explicitly mark `source=model_self_assessment`
and `host_validated=false`; missing or malformed comment metadata is recorded as
`MISSING`/`INVALID`, never fabricated or used as execution authorization. These tags
are not Graph state, new predicates, safety checks or controller parameters.

For a paired offline comparison on saved pre-change real observations:

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.robustness_planner_audit \
  --round runs/seed_fresh_unified_20261016_211319/seed/round_001 \
  --round runs/seed_fresh_unified_20261016_211319/seed/round_002 \
  --round runs/seed_fresh_20261015_153150/seed/round_012 \
  --output "reports/robustness_planner_$(date +%Y%m%d_%H%M%S)"
```

This invokes the same GPT model/settings with fresh independent sessions and each
saved request's task/images. Only the robustness block differs within a pair.
It reads saved RGB-D, generates proposals, and renders Pa/Pb and commanded TCP
waypoints. It never initializes a robot runtime or dispatches motion. Images and
clouds are hashed; each snapshot's own geometry is used for projection, including
its support-height affine correction. Dashed transfer links, if present, represent
endpoints only. Projection self-consistency does not validate physical calibration.

The first comparison is in `reports/robustness_planner_20261016/report.md`.

## One recovery manipulation in frozen-Graph replay

The `cloth_agent.experiments.graph_policy_replay` CLI now defaults to
`--max-recovery-attempts 1 --max-fallbacks 0`. This changes Graph replay, not
`graph_policy seed`: seed has no selected Graph edge to recover.

Normal accepted grounding takes the existing execution path. A full grounding
`NO_ROBUST_ACTION` or minimal null anchor raises a distinct gate rejection; it
does not use the old grounding-recapture/applicability retries. The host calls
the existing full GPT-6 Astra planner once on the rejected observation, supplying
the selected edge/skill and refusal reasons. Its objective is one minimal,
low-risk manipulation that exposes a reliable handle for the skill, not forced
completion of the fold. The recovery grasp must itself pass the same gate.

The answer uses the existing single-grasp/held-motion schema. It passes through
the unchanged executor and workspace/IK/collision/safety preflight. Refusal,
invalid/multiple recovery actions, model failure, or rejected preflight returns
`RECOVERY_UNRESOLVED`. A successful recovery is followed by a fresh observation
and normal matching against existing nodes, without an expected-node hint. A
reliable match may return to the same node or another existing node; its own
normal preferred edge is selected. No match/ambiguous match stops unresolved.

The budget is **one recovery attempt per run**, including attempts that fail to
produce an action. A further gate rejection stops unresolved. Reusing the same
edge on byte-identical RGB-D is blocked before grounding; capture timestamps and
paths do not make old evidence new. This content check does not claim to detect
all physically unchanged cloth states through camera noise. Recovery consumes
one step of `--max-steps`, with a final matching turn even at that step limit.

This mode neither creates `R_` nodes nor extends/repairs Graph edges/templates.
It cannot combine with legacy extending fallback, applicability recovery, edge
repair or grasp retries. CLI defaults disable those paths. Explicit legacy
Python callers that select these older modes without the new budget retain their
old behavior; explicitly requesting both modes is rejected. Set the new budget
to 0 when intentionally using legacy CLI modes. `--online-growth` remains a
separate explicitly requested runner, not this recovery path.

Audit files are `step_*/gate_rejection.json` and
`robustness_recovery/audit.json`, plus `attempt_001/context.json`, the full
planner request/answer/events, and `after_observation/`. They preserve original
node/edge, gate reasons, current RGB-D references/hashes, recovery code and
execution result, before/after images, fresh match, whether the originally
rejected edge was physically dispatched, subsequent controller outcomes and
after-image matching. A controller success is not recorded as proof of cloth
success. If no safe after capture completed, its absence is explicit.
