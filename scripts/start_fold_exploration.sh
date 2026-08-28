#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/CNS2026330003/miniconda3/envs/cali/bin/python}"
exec "$PYTHON" "$ROOT/scripts/claude_fold_exploration.py" "$@"
