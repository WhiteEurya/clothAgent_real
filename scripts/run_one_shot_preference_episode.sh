#!/usr/bin/env bash
# One-shot with a human Top10 ranking, fixed alpha and learned residual correction.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
    cat <<'USAGE'
Usage:
  bash scripts/run_one_shot_preference_episode.sh --execute
  bash scripts/run_one_shot_preference_episode.sh --execute --max-rounds 1

Per round: perception current + goal -> Ps -> fixed directions -> Pt -> base + R -> Top10.
Open http://127.0.0.1:8768; it stays open and shows new rounds automatically.
Reorder all 10 actions, select one to execute and confirm once.
Alpha stays fixed. Pre-feedback evaluation is saved before training R for later rounds.
Includes the existing random pickup/drop initialization before candidate review.
Defaults: reuse saved alpha, zero residual initially, 12 rounds, 30-minute review timeout.
Ctrl+C or the page's Stop button ends the run. --help does not connect hardware.
USAGE
    exit 0
fi
# The portal owns no robot/camera and remains available between rounds and runs.
# Help/usage never starts a service or connects hardware.
if [[ " $* " == *" --execute "* && " $* " != *" --help "* && " $* " != *" -h "* ]]; then
    (cd -- "${SCRIPT_DIR}/.." && "${CLOTH_PYTHON:-/home/sja/miniconda3/envs/cali/bin/python}" -B \
        -m cloth_agent.preference_dashboard --ensure)
fi
exec bash "${SCRIPT_DIR}/run_one_shot_goal_episode.sh" --human-preference --ranking-correction "$@"
