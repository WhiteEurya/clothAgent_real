#!/usr/bin/env bash
# One initialization followed by the existing one-shot closed-loop policy.
# No hardware is connected unless --execute is explicitly passed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"

CLOTH_PYTHON="${CLOTH_PYTHON:-/home/sja/miniconda3/envs/cali/bin/python}"
if [[ ! -x "${CLOTH_PYTHON}" ]]; then
    echo "Python interpreter unavailable: ${CLOTH_PYTHON}" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    cat <<'USAGE'
Usage:
  bash scripts/run_one_shot_goal_episode.sh --execute
  bash scripts/run_one_shot_goal_episode.sh --execute --max-rounds 6
  bash scripts/run_one_shot_goal_episode.sh --execute --goal-image /path/to/goal.png

Runs one center-region pickup/drop initialization, then FastPlanner + SHADOW gate
and real execution. Gate cannot block/replace actions; slow planners are disabled.
Default: 12 rounds. Ctrl+C stops the run. --gate-timeout-s defaults to 120.
Each invocation creates a new reports/one_shot_goal_full_episode_v1_<timestamp> directory.
All arguments are passed to the existing episode runner. --help does not connect hardware.
USAGE
    exit 0
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
OUTPUT_DIR="${REPO_ROOT}/reports/one_shot_goal_full_episode_v1_${RUN_STAMP}"

# exec preserves foreground Ctrl+C/SIGTERM handling in the existing runner.
# Explicit user --output/--goal-image arguments override these defaults.
exec "${CLOTH_PYTHON}" -u -m cloth_agent.experiments.one_shot_goal_full_episode_v1 \
    --output "${OUTPUT_DIR}" \
    --goal-image "${REPO_ROOT}/reports/goal_only_full_episode_v1_restart02/goal/target.png" \
    "$@"
