#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
cloth_batch=arm7_unfold_20261113_103910_478310057
cloth_resume="reports/formal_experiments/${cloth_batch}/${cloth_batch}_run_01"
if [[ ! -f "${cloth_resume}/episode_summary.json" ]]; then
  echo "找不到待续跑记录：${cloth_resume}/episode_summary.json" >&2
  exit 2
fi
exec bash scripts/run_pink_manual_unfold.sh "$@" --continue-from "$cloth_resume"
