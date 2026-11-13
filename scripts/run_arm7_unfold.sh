#!/usr/bin/env bash
# Real unfolding by default; --dry-run performs offline configuration checks only.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
export CLOTH_PYTHON="${CLOTH_PYTHON:-/home/CNS2026330003/miniconda3/envs/cloth-real/bin/python}"
if [[ "${1:-}" == '--dry-run' ]]; then
    shift
    exec "$CLOTH_PYTHON" -m cloth_agent.experiments.partitioned_unfold "$@"
fi
if [[ "${1:-}" == '--help' ]]; then
    exec "$CLOTH_PYTHON" -m cloth_agent.experiments.partitioned_unfold --help
fi
printf -v unfold_command '%q ' "$CLOTH_PYTHON" -u -m cloth_agent.experiments.partitioned_unfold --execute "$@"
exec sg zed -c "$unfold_command"
