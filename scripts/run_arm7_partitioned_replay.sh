#!/usr/bin/env bash
# Current dual-xArm7 / fixed ZED; defaults to offline graph validation.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"
export CLOTH_REPLAY_EVIDENCE_ROOT="${CLOTH_REPLAY_EVIDENCE_ROOT:-$REPO_ROOT/data/reproduction_arm7}"
exec "${CLOTH_PYTHON:-python}" -m cloth_agent.experiments.graph_policy_replay \
  --config "$REPO_ROOT/config/dual_arm.zed_dual_xarm7_20261103.json" \
  --graph "$CLOTH_REPLAY_EVIDENCE_ROOT/workspace/runs/graph_recover_20261017_134332/episode_0001/growth_000096.json" \
  --grounding-mode full \
  --max-steps 20 --max-calls 128 --max-recovery-attempts 0 \
  --output "$REPO_ROOT/reports/arm7_partitioned_$(date +%Y%m%d_%H%M%S_%N)" \
  "$@"
