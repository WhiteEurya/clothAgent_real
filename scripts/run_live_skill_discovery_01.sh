#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}/.."
exec "${CLOTH_PYTHON:-python}" -u -B experiments/live_skill_discovery_01/run.py "$@"
