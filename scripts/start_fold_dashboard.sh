#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SOURCE="${1:-}"
if [[ -z "$SOURCE" ]]; then
  SOURCE="$(find "$ROOT/runs" -type d -path '*/results/fold_exploration/*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
fi
if [[ -z "$SOURCE" || ! -d "$SOURCE" ]]; then
  echo "No fold_exploration output directory found. Pass one explicitly." >&2
  exit 2
fi
exec "$PYTHON" -m cloth_agent.fold_exploration_viser "$SOURCE" "${@:2}"
