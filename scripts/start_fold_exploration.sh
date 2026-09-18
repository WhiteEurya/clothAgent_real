#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$ROOT/scripts/manage_runs.py" check
# Restore the complete saved RGB preset before any capture or robot action.
# Override CAMERA_PRESET / CAMERA_PYTHON when using a different camera or SDK env.
CAMERA_PRESET="${CAMERA_PRESET:-$ROOT/config/realsense_rgb_317222073552.json}"
CAMERA_PYTHON="${CAMERA_PYTHON:-$PYTHON}"
"$CAMERA_PYTHON" "$ROOT/scripts/apply_realsense_preset.py" "$CAMERA_PRESET"
ARGS=("$@")
has_existing_run=0
has_recovery_mode=0
for arg in "${ARGS[@]}"; do
  case "$arg" in
    --run-id|--run-id=*|--run-dir|--run-dir=*) has_existing_run=1 ;;
    --unattended|--continue-on-error|--no-unattended) has_recovery_mode=1 ;;
  esac
done
if [[ "$has_recovery_mode" -eq 0 ]]; then
  ARGS+=(--unattended)
fi
if [[ "$has_existing_run" -eq 0 ]]; then
  # A fresh run is required so the effective robot/tool configuration is
  # snapshotted from the current config instead of an older workspace copy.
  RUN_ID="fold_$(date -u +%Y%m%dT%H%M%S%NZ)"
  ARGS+=(--run-id "$RUN_ID")
  echo "[fold-launch] creating new run: $RUN_ID" >&2
fi
exec "$PYTHON" "$ROOT/scripts/claude_fold_exploration.py" "${ARGS[@]}"
