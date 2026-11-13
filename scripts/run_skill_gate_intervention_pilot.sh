#!/usr/bin/env bash
# Four-run intervention pilot. Physical execution requires explicit --execute.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
pilot_python="${CLOTH_PYTHON:-/home/CNS2026330003/miniconda3/envs/cloth-real/bin/python}"
output="${CLOTH_INTERVENTION_PILOT_OUTPUT:-$PWD/reports/skill_gate_intervention_online_$(date +%Y%m%d_%H%M%S)}"
args=("$@")
has_output=0
for arg in "${args[@]}"; do
  [[ "$arg" == "--output" ]] && has_output=1
done
if (( ! has_output )); then args+=(--output "$output"); fi
if (( ${#args[@]} == 0 )); then args+=(--check); fi
if [[ "${args[0]}" == "--check" ]]; then
  exec "$pilot_python" -u -m cloth_agent.experiments.skill_gate_intervention_pilot_20261107 "${args[@]}"
fi
printf -v command '%q ' env PYTHONUTF8=1 LANG="$LANG" LC_ALL="$LC_ALL" "$pilot_python" -u -m cloth_agent.experiments.skill_gate_intervention_pilot_20261107 "${args[@]}"
exec sg zed -c "exec $command"
