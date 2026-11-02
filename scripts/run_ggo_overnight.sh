#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
    echo 'Usage: bash scripts/run_ggo_overnight.sh --execute [--max-hours 8] [--wait-before-first]'
    echo 'Current GGO; continuous unfolding; reset radius uniform 0–40 mm; 12 actions; stop on errors.'
    exit 0
fi
# A conservative episode cap supplements the wall-time limit. Neither bypasses Host checks.
exec bash "${SCRIPT_DIR}/run_ggo_rollout.sh" --episodes 1000 --max-hours 8 \
    --initialize-center-radius-range-mm 0 40 --min-free-gb 20 --systematic-report "$@"
