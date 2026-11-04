#!/usr/bin/env bash
# Current ZED Mini / dual xArm7 formal loop.
# The runtime safety gates remain authoritative; this wrapper only supplies
# the current execution config and portable local result storage.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export CLOTH_RUN_STORAGE_CONFIG="${CLOTH_RUN_STORAGE_CONFIG:-$REPO_ROOT/config/run_storage.portable.json}"
if [[ "$*" == *"--help"* || "$*" == *" -h"* ]]; then
  exec "${CLOTH_PYTHON:-python}" scripts/codex_astra_loop.py --help
fi
active_arm_count=0
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  arg="${args[i]}"
  if [[ "$arg" == "--active-arm" ]]; then
    i=$((i+1))
    value="${args[i]-}"
    [[ "$value" == "left" || "$value" == "right" ]] && ((active_arm_count+=1))
  elif [[ "$arg" == --active-arm=* ]]; then
    value="${arg#--active-arm=}"
    [[ "$value" == "left" || "$value" == "right" ]] && ((active_arm_count+=1))
  fi
done
if [[ "$active_arm_count" -ne 1 ]]; then
  echo 'error: choose exactly one active arm: --active-arm left|right' >&2
  exit 2
fi
exec "${CLOTH_PYTHON:-python}" scripts/codex_astra_loop.py \
  --config "$REPO_ROOT/config/dual_arm.zed_dual_xarm7_20261103.json" \
  "$@"
