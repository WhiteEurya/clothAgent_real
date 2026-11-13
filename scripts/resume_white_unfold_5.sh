#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Continue the five remaining trials using the original unfolded white-shirt goal.
# The first trial's execution failure remains in its original logs; the user
# accepted its garment outcome as successful.
cloth_goal="reports/formal_experiments/arm7_unfold_20261111_010925_000575636/arm7_unfold_20261111_010925_000575636_run_01/goal/target.png"
if [[ ! -f "$cloth_goal" ]]; then
  echo "缺少首轮目标图：$cloth_goal" >&2
  exit 1
fi
cloth_mode_args=()
if [[ "${1:-}" == --dry-run || "${1:-}" == --help ]]; then
  cloth_mode_args=("$1")
  shift
fi
exec bash scripts/run_white_unfold.sh "${cloth_mode_args[@]}" \
  --runs 5 --goal-image "$cloth_goal" "$@"
