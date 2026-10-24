#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dino_python="${CLOTH_DINO_PYTHON:-/home/sja/.cache/cloth_agent/visual_retrieval_env/bin/python}"
if [ ! -x "$dino_python" ]; then
  dino_python=python
fi
if [ "$#" -eq 0 ]; then
  exec "$dino_python" -m cloth_agent.dino_correspondence_tool ui \
    --cases reports/dino_correspondence_cases_v1/cases.json \
    --output reports/dino_correspondence_v1
fi
exec "$dino_python" -m cloth_agent.dino_correspondence_tool ui "$@"
