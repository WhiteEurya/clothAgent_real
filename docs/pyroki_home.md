# Unified PyRoki Home

The current ZED configuration sets `home_path.max_height_mm` to 500 mm in the
shared world frame, for both arms' Home TCP paths and checked command envelopes.
This absolute ceiling replaces the endpoint-relative `extra_height_mm` policy
only when explicitly configured. Other workspaces and collision checks remain
unchanged; configurations without this key retain the original relative policy.

Offline validation from a recorded actual state (not a fabricated Home start):

```bash
python -m cloth_agent.experiments.cloth_execution home \
  --robot-state outputs/home_request_20261104_retry_01/robot_state.json \
  --output outputs/pyroki_home_new_run
```

PyRoki optimizes joint waypoints with joint-limit constraints, endpoint penalties
and smoothness costs, using the installed URDF joint-order adapter. It is a
candidate generator, not a replacement collision authority or an RRT planner.
The current objective does not optimize obstacle clearance; rejected candidates
are reported, not automatically converted into an unverified detour.

Each sequential leg retains the other arm at its actual start or previously
planned Home state. Existing `joint_motion`, `validate_sample`, and
`prepare_segments` check both arms, tool geometry, joint intervals, and Home
corridor limits. The exact mirrored workspace halfspace is also enforced.

`cloth_execution.execute_home(coordinator, confirmed=...)` reads fresh snapshots,
plans the whole program, then delegates to `execute_sequential_home`. That
executor rebuilds and checks command segments, verifies initial and inactive-arm
positions, monitors feedback, and stops on failure. Position-controller
interpolation is covered by existing complete joint-box checks, not assumed to
follow the optimized line. The offline artifact alone never authorizes dispatch.

The CLI intentionally has no real execution option. No acceptance flags are
modified; the existing executor's real-execution restrictions still apply.
Passing offline planning is not evidence that those restrictions have passed.

Replay's separate Home command bypasses graph matching and historical single-arm
camera preparation, not safety checks:

```bash
python -m cloth_agent.experiments.graph_policy_replay --home-only --execute \
  --config config/dual_arm.zed_dual_xarm7_20261103.json \
  --output outputs/replay_home_new_run
```

This calls the existing `gripper_home` entry. Its prerequisites remain intact;
after those pass, 7+7 configurations now route to the same checked PyRoki Home
planner. Live snapshots, sequential execution and failure stopping are retained.
Without `--execute`, this replay command is simulated, not live-state validation.
