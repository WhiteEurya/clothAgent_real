#!/usr/bin/env bash
# check/status never access hardware; start explicitly executes the six-run pilot.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
pilot_python="${CLOTH_PYTHON:-/home/CNS2026330003/miniconda3/envs/cloth-real/bin/python}"
mode="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  check)
    for arg in "$@"; do
      if [[ "$arg" == --execute ]]; then
        printf 'check不执行真机；请显式使用start。\n' >&2
        exit 2
      fi
    done
    exec "$pilot_python" -u -m cloth_agent.experiments.skill_gate_online_batch "$@"
    ;;
  status)
    batch="${1:-latest}"
    if [[ $# -gt 0 ]]; then shift; fi
    exec "$pilot_python" -u -m cloth_agent.experiments.skill_gate_online_batch --status "$batch" "$@"
    ;;
  start)
    mkdir -p reports
    console="reports/skill_gate_console_$(date +%Y%m%d_%H%M%S)_$$.log"
    printf '控制台日志：%s\n' "$PWD/$console"
    printf -v command '%q ' "$pilot_python" -u -m cloth_agent.experiments.skill_gate_online_batch --execute "$@"
    set +e
    sg zed -c "exec $command" 2>&1 | tee "$console"
    result=${PIPESTATUS[0]}
    exit "$result"
    ;;
  *)
    printf '用法：bash scripts/run_skill_gate_online.sh {check|start|status [latest或批次目录]}\n' >&2
    exit 2
    ;;
esac
