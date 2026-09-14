#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
exec "$PYTHON" "$ROOT/scripts/claude_fold_exploration.py" "$@"
