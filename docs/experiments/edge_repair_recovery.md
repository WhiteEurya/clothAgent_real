# Edge repair recovery

`--repair-self-loop` enables both post-action self-loop repair and **pre-dispatch structured IK failure** repair in minimal strict replay (`--max-fallbacks 0`). It takes precedence over downward missed-grasp retries. It does not run hardware unless the existing `--execute` path is used.

Each edge has one repair budget per run, shared between the two triggers. IK eligibility requires `PLANNING_FAILED`, `physical_execution=false`, `failure_kind=IK_FAILED`, and `ik_failure.kind=IK_NO_ACCEPTABLE_SOLUTION`. Collision/controller/partial-execution failures stop. All original IK tolerances, workspace, collision and controller checks remain unchanged. A byte-different but operation-identical proposed repair is rejected before dispatch.

The repair planner receives current RGB, the edge's semantic action role, rejected command and diagnostics (or self-loop visual evidence). It repairs the same edge; it is not authorized to pursue an unrelated action family. Self-loop is not proof of missed contact or a Z error. IK rejection is recorded as a pre-dispatch failure, never fabricated as a physical self-loop.

After execution, a fresh observation must reliably match the expected target. Only then is the effective executed command compiled into an action-local contact template using the existing compiler checks. Unsupported geometry or missing provenance produces `MINIMAL_UNSUPPORTED`, not a reusable revision. Unexpected/unmatched outcomes and repeated self-loops do not publish templates.

Per-run artifacts:

- `edge_repairs/edge_experience.json`: edge failure attributes and all candidate revisions.
- `edge_repairs/host_templates_revision_NNN.json` with checksum: successful compiled revision bundle.
- `step_XXX/ik_repair/planner/` or `step_XXX/edge_repair/planner/`: planning and native execution evidence.

Successful bundles are published by an atomic manifest under `<graph directory>/edge_repair_registry/<graph SHA256>/latest.json`. On the next CLI run with `--repair-self-loop`, minimal mode and no explicit `--grounding-templates`, the newest published bundle is loaded and checked against the exact graph and bundle hashes. The loaded path is printed and recorded in runner configuration. Prior edge experience is carried forward when that exact published bundle is selected. The frozen seed graph and original bundle are never overwritten. An explicit `--grounding-templates ORIGINAL_PATH` selects the original bundle instead.

Failed and interrupted candidates cannot replace the published bundle. A publication I/O failure is recorded separately from successful compilation. These mechanisms validate bookkeeping and control flow; real-world grasp/replanning reliability still requires physical evaluation.
