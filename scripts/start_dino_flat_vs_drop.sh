#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/start_dino_correspondence.sh \
  --cases reports/dino_flat_vs_drop_v1/cases.json \
  --output reports/dino_flat_vs_drop_queries_v1 "$@"
