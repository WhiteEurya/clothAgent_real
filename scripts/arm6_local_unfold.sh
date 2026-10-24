#!/usr/bin/env bash
# Independent local action pilot. Every physical action requires local/point review.
set -euo pipefail
cd "$(dirname "$0")/.."
exec /home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.arm6_local_action_policy \
  --library reports/arm6_local_action_20261020/memory \
  --execute --max-steps 3 "$@"
