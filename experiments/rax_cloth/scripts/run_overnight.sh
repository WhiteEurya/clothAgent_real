#!/usr/bin/env bash
set -euo pipefail
RAX_CLOTH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAX_REPO_ROOT="$(cd -- "$RAX_CLOTH_ROOT/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RAX_CLOTH_ROOT:$RAX_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${RAX_CLOTH_PYTHON:-/home/sja/miniconda3/envs/cali/bin/python}" -m rax_cloth.cli "$@"
