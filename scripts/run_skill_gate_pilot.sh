#!/usr/bin/env bash
# Isolated pilot. No default command and no implicit physical execution.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
pilot_python="${CLOTH_PYTHON:-/home/CNS2026330003/miniconda3/envs/cloth-real/bin/python}"
pilot_root="${CLOTH_GATE_PILOT_ROOT:-$PWD/reports/skill_gate_pilot_20261106}"
case "${1:-}" in
    capture-goal|capture-before|trial)
        printf -v pilot_command '%q ' "$pilot_python" -u -m cloth_agent.experiments.skill_gate_pilot --root "$pilot_root" "$@"
        exec sg zed -c "$pilot_command"
        ;;
    *) exec "$pilot_python" -u -m cloth_agent.experiments.skill_gate_pilot --root "$pilot_root" "$@" ;;
esac
