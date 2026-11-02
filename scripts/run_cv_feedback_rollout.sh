#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/run_cv_feedback_rollout.sh --execute [--episodes 1] [--no-reset]'
    echo 'Manual initial state: --execute --no-reset --episodes 1'
    echo 'Single-call GPT + CV geometry + actual outcome feedback; 12 actions per episode.'
    exit 0
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
exec "${CLOTH_PYTHON:-python}" -u -B \
    -m cloth_agent.experiments.automatic_top5_batch \
    --output "${REPO_ROOT}/reports/cv_feedback_rollout_${RUN_STAMP}" \
    --visible-sleeve-hem --stateful-context --episodes 1 --max-rounds 13 --max-actions 12 \
    --initialize-center-radius-range-mm 20 40 --port 8769 --cv-feedback --auto-next \
    --goal-image "${REPO_ROOT}/data/reference/deployment_example/ggo_goal.png" \
    "$@"
