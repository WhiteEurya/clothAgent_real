#!/usr/bin/env bash
# Frozen Arm6 / Camera A folding graph. Offline unless --execute is explicit.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"
exec "${CLOTH_PYTHON:-python}" -m cloth_agent.experiments.graph_policy_replay \
  --graph "$REPO_ROOT/data/graphs/arm6_replay_goal_n007/graph.json" \
  --config "$REPO_ROOT/config/dual_arm.local.json" \
  --grounding-mode minimal --max-recovery-attempts 0 \
  --max-steps 8 --max-calls 32 --graph-port 8090 \
  --output "$REPO_ROOT/reports/arm6_folding_$(date +%Y%m%d_%H%M%S_%N)" \
  "$@"
