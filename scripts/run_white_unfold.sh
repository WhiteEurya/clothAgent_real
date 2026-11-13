#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Defaults can be overridden, e.g. --runs 5 --goal-image saved/target.png.
# With no explicit goal, capture the unfolded garment once before the first reset.
cloth_goal_args=(--capture-goal)
for cloth_arg in "$@"; do
  case "$cloth_arg" in
    --goal-image|--goal-image=*|--continue-from|--continue-from=*|--continue-batch|--continue-batch=*) cloth_goal_args=();;
  esac
done
# The lower-level launcher recognizes --dry-run/--help as its first argument.
cloth_mode_args=()
if [[ "${1:-}" == --dry-run || "${1:-}" == --help ]]; then
  cloth_mode_args=("$1")
  shift
fi
exec bash scripts/run_arm7_unfold_dual_wrist.sh "${cloth_mode_args[@]}" \
  --runs 6 --reset-profile body-drop --max-actions 20 \
  "${cloth_goal_args[@]}" "$@"
