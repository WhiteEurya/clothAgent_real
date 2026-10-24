# Continuous one-shot unfolding collection

Run in the supervised robot terminal:

```bash
bash scripts/run_one_shot_goal_continuous.sh --execute
# Or collect at most ten episodes:
bash scripts/run_one_shot_goal_continuous.sh --execute --episodes 10
```

Default `--episodes 0` means unlimited episodes, subject to operator stop or a
non-DONE termination. Each episode has its own `--max-rounds 12` limit.
`--between-episodes-s 5` provides an interruptible pause before the next reset.
Keep the existing emergency stop and human supervision in place.

```text
Existing center-region pickup/drop initialization
  → perception RGB → right view RGB → FastPlanner (both current views + goal, concurrent slow return)
  → ACTION: shadow gate → Host checks → one manipulation → observe
  → allowlisted pre-dispatch error: fresh observation → Codex replan → same Host checks
  → DONE: close/save episode → wait → next independent initialization
  → ABSTAIN / ROUND_LIMIT / unrecoverable error / operator stop: stop the whole batch
```

Only a completed child process with return code 0, termination reason DONE,
finished summary and no cleanup/report errors permits another initialization.
The existing episode runner also returns 0 for some other stop reasons, so the
supervisor checks the saved reason explicitly. A rejected proposal is never
dispatched or converted into a reset. The batch does not restart interrupted
episodes and does not automatically resume old batches.

Codex recovery is enabled by default (`--no-codex-recovery` disables it).
Known pre-dispatch missing-depth, proposal-geometry, planned IK/workspace and
Fast response errors can hand off to a separate Codex Agent session. It receives
a fresh observation, the frozen goal and rejected proposals/errors; it can inspect
the current RGB-D using isolated tools. It receives no trajectory memory or Graph.
The unchanged Fast request is not retried or repaired.

Defaults: `--max-replans-per-incident 2`, `--max-codex-replans 4`,
`--codex-timeout-s 600`. Recovery consumes normal round slots. Before recovery,
after the Agent and before dispatch, live controller/fault/pose checks must pass.
Recovery never clears STOP or controller faults. The replacement must pass the
same Host checks; repeating both rejected Pa/Pb within 8 px is rejected.
A successful replacement returns the following round to FastPlanner.
Gate remains shadow-only and never triggers this branch.

Initialization failures, hardware/collision/feedback faults, errors after physical
dispatch, unknown errors, recovery-Agent failures, exhausted budgets and operator
interrupts stop collection. There is no automatic retry of physical execution.
The original failed round remains unexecuted with `recovery_eligibility.json` and
`recovery_handoff.json`. Recovery rounds save `recovery_request.json`,
`deliberative_agent_plan.json`, `deliberative_plan.json`, Agent events and
`deliberative_metrics.json`; they do not fabricate FastPlanner or gate calls.
Codex latency/call counts are reported separately from FastPlanner and gate.

Each episode runs in a separate subprocess using the unchanged one-shot entrypoint.
There is no trajectory/history transfer between episodes, no memory insertion,
no Graph write, and no training. The current FastPlanner, shadow gate, table-only
Pb depth fallback and manipulation primitive stay in place. Initialization
retains the current Home Y/yaw release policy.

The goal image is copied once for the entire batch. Policy, calibration, config
and goal hashes are checked before every child launch. Each child retains its
own request/response, source snapshot, before/after, execution, timing and review
artifacts. No child launches after Ctrl+C/SIGTERM: the supervisor forwards the
signal to the active child and waits for its existing stop/cleanup routine.
It does not use SIGKILL.

```text
reports/one_shot_goal_continuous_v1_<timestamp>/
  goal/target.png
  protocol.json
  events.jsonl
  batch_summary.json
  episodes.csv
  report.md
  episode_0001.run.log
  episode_0001.completion.json
  episode_0001/
    initialization/
    round_000/ ...
    episode_summary.json
    episode_contact_sheet.png
    review.html
    ...
  episode_0002/ ...
```

The batch report links each episode's review page. It aggregates rounds,
initialization actions, unfolding manipulation dispatches, safety rejections,
FastPlanner/gate latency and robot execution time. Dispatch counts may include
partial execution; completed initialization does not prove acquisition.
Human FOLD_READY remains pending until review. Model DONE, including a DONE
with zero unfolding actions, is retained as a model judgment and never promoted
to confirmed experimental success. Human review exports remain separate from
execution records.

The old `run_one_shot_goal_episode.sh` still runs exactly one episode.
The new wrapper with no arguments or `--help` does not connect hardware.

2026-10-21: each child episode uses the dual-view observation flow described in
[the episode guide](one_shot_goal_full_episode_v1.md). Coordinates remain in perception.
Optional Codex recovery also receives both views, after a synchronous return.
