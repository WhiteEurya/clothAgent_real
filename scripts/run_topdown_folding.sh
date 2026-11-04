#!/usr/bin/env bash
# Current ZED / dual-xArm7 observation-to-proposal workflow. No robot execution.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_ROOT"
exec "${CLOTH_PYTHON:-python}" -m cloth_agent.experiments.topdown_folding "$@"
