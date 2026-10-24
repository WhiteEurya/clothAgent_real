#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/run_reset_radius_sweep.sh --execute [--wait-before-first] [--port 8775]'
    echo 'Control: http://127.0.0.1:8775 ; stop current / stop batch / ready for next reset.'
    exit 0
fi
RUN_STAMP="$(date +%Y%m%d_%H%M%S_%N)"
exec "${CLOTH_PYTHON:-/home/sja/miniconda3/envs/cali/bin/python}" -u -B \
    -m cloth_agent.experiments.reset_radius_sweep \
    --output "${REPO_ROOT}/reports/reset_radius_sweep_${RUN_STAMP}" "$@"
