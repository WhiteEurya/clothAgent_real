#!/usr/bin/env bash
# Independent episodes; only a clean model DONE starts the next initialization.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"
CLOTH_PYTHON="${CLOTH_PYTHON:-/home/sja/miniconda3/envs/cali/bin/python}"
if [[ $# -eq 0 ]]; then
    cat <<'USAGE'
Usage:
  bash scripts/run_one_shot_goal_continuous.sh --execute
  bash scripts/run_one_shot_goal_continuous.sh --execute --episodes 10

Default: unlimited independent episodes, 12 rounds each.
Clean model DONE -> save -> wait 5 s -> new pickup/drop initialization.
Pre-dispatch proposal errors -> fresh observation + bounded Codex recovery.
Default recovery budget: 2 calls per incident, 4 per episode; --no-codex-recovery disables it.
ABSTAIN, ROUND_LIMIT, unrecoverable error or Ctrl+C stops the whole batch.
Keep the existing emergency-stop and operator-supervision procedure.
No robot connection without --execute. --help is offline.
USAGE
    exit 0
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
exec "${CLOTH_PYTHON}" -u -m cloth_agent.experiments.one_shot_goal_continuous_v1 \
    --output "${REPO_ROOT}/reports/one_shot_goal_continuous_v1_${RUN_STAMP}" "$@"
