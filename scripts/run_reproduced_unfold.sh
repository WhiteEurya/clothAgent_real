#!/usr/bin/env bash
# Same formal pipeline, pinned to the shared 2026-11-14 experiment configuration.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode=()
if [[ "${1:-}" == --dry-run || "${1:-}" == --help ]]; then
  mode+=("$1")
  shift
fi
exec bash scripts/run_arm7_unfold_dual_wrist.sh "${mode[@]}" \
  --runs 1 --manual-reset --max-actions 20 --reset-profile body-drop \
  --planner-transport openai-cli \
  --goal-image data/reference/white_black_cat_20261113_160203/goal.png \
  --topdown-config config/topdown.dual_wrist_rgb_stitched.json "$@"
