#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
cloth_python="${CLOTH_PYTHON:-}"
if [[ -z "$cloth_python" ]]; then
  if [[ -x /home/CNS2026330003/miniconda3/envs/cloth-real/bin/python ]]; then
    cloth_python=/home/CNS2026330003/miniconda3/envs/cloth-real/bin/python
  else
    cloth_python=python3
  fi
fi
args=(
  --manual-reset
  --camera-mode dual-wrist
  --config config/dual_arm.dual_wrist_xarm7_20261110.json
  --fixed-wrist-config config/calibration/dual_wrist_relaxed_20261110/extrinsics.json
  --topdown-config config/topdown.dual_wrist_rgb.json
)
cloth_output_group="formal_experiments"
if [[ "${1:-}" == --dry-run ]]; then
  shift
  cloth_output_group="startup_checks"
elif [[ "${1:-}" != --help ]]; then
  args+=(--execute)
fi
cloth_run_id="arm7_unfold_$(date +%Y%m%d_%H%M%S_%N)"
args+=(--output "reports/${cloth_output_group}/${cloth_run_id}" --video-dir "videos/formal_experiments")
exec "$cloth_python" -u -m cloth_agent.experiments.unfold_batch "${args[@]}" "$@"
