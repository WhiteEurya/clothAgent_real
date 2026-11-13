#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# One manual trial by default; pass --runs N for repeated manual setups.
cloth_mode_args=()
if [[ "${1:-}" == --dry-run || "${1:-}" == --help ]]; then
  cloth_mode_args=("$1")
  shift
fi
exec bash scripts/run_white_unfold.sh "${cloth_mode_args[@]}" --runs 1 "$@" \
  --goal-image data/reference/pink_kirby_goal_20261109/goal.png \
  --topdown-config config/topdown.dual_wrist_rgb_stitched.json \
  --manual-reset
