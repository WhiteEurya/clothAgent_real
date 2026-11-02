#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/run_ggo_manual_single_steps.sh --execute [--wait-before-first]'
    echo '3 episodes, reset radius 20 mm, one action per episode; automatic anchors and next episode; control :8769'
    exit 0
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
exec "${CLOTH_PYTHON:-python}" -u -B \
    -m cloth_agent.experiments.automatic_top5_batch \
    --output "${REPO_ROOT}/reports/ggo_single_steps_${RUN_STAMP}" \
    --visible-sleeve-hem --stateful-context --episodes 3 --max-rounds 1 --ggo-auto-anchors --auto-next \
    --initialize-center-radius-mm 20 --port 8769 --ggo-port 8781 \
    --goal-image "${REPO_ROOT}/data/reference/deployment_example/ggo_goal.png" \
    --ggo-reference-annotations "${REPO_ROOT}/data/reference/deployment_example/ggo_annotations.json" "$@"
