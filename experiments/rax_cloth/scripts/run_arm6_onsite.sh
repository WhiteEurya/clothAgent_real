#!/usr/bin/env bash
# Starting this foreground command authorizes this supervised, time-limited run.
set -euo pipefail
RAX_CLOTH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAX_REPO_ROOT="$(cd -- "$RAX_CLOTH_ROOT/../.." && pwd)"
cd "$RAX_REPO_ROOT"
RAX_RUN_DIR="$RAX_CLOTH_ROOT/runs/arm6_onsite_$(date +%Y%m%d_%H%M%S)_$$"
exec bash "$RAX_CLOTH_ROOT/scripts/run_rax_codex.sh" \
  --profile chatgpt --mode real_supervised --onsite --initial-reset \
  --robot-config "$RAX_REPO_ROOT/config/dual_arm.local.json" \
  --session "$RAX_RUN_DIR" --max-model-calls 32 --max-seconds 1200 \
  --progress-interval 5 "$@"
