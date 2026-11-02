#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd -- "${REPO_ROOT}"
exec "${CLOTH_PYTHON:-python}" -u -B \
  -m cloth_agent.experiments.minimal_one_shot_6iter "$@"
