#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GAMEPAD_PYTHON="${GAMEPAD_PYTHON:-/home/sja/miniconda3/envs/cali/bin/python}"
exec "$GAMEPAD_PYTHON" -m cloth_agent.gamepad_check "$@"
