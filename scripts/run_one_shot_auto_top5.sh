#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/run_one_shot_auto_top5.sh --execute [--episodes 3] [--max-rounds 12] [--initialize-center-radius-mm 20] [--stateful-context]'
    echo 'Control: http://127.0.0.1:8769 ; stop current / stop batch / ready for next reset.'
    exit 0
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
exec "${CLOTH_PYTHON:-/home/sja/miniconda3/envs/cali/bin/python}" -u -B \
    -m cloth_agent.experiments.automatic_top5_batch \
    --output "${REPO_ROOT}/reports/automatic_top5_batch_${RUN_STAMP}" "$@"
