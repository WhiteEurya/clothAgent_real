#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ARGS=("$@")
has_existing_run=0
for arg in "${ARGS[@]}"; do
  case "$arg" in
    --run-id|--run-id=*|--run-dir|--run-dir=*) has_existing_run=1 ;;
  esac
done
if [[ "$has_existing_run" -eq 0 ]]; then
  # A fresh run is required so the effective robot/tool configuration is
  # snapshotted from the current config instead of an older workspace copy.
  RUN_ID="fold_$(date -u +%Y%m%dT%H%M%S%NZ)"
  ARGS+=(--run-id "$RUN_ID")
  echo "[fold-launch] creating new run: $RUN_ID" >&2
fi
exec "$PYTHON" "$ROOT/scripts/claude_fold_exploration.py" "${ARGS[@]}"
