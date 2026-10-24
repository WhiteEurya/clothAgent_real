#!/usr/bin/env bash
# Fixed existing n092 -> n077 path. Opens explicit state/anchor review before motion.
set -euo pipefail
cd "$(dirname "$0")/.."
exec /home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_policy_replay \
  --graph reports/arm6_graph_incremental_20261019_v3/final/graph.json \
  --arm6-expanded-graph \
  --arm6-route e103,e104,e105,e106,e107,e108,e109 \
  --arm6-review \
  --arm6-live-start \
  --matcher retrieval_gpt_rerank \
  --retrieval-memory reports/arm6_graph_incremental_20261019_v3/memory \
  --max-steps 7 \
  --output "reports/arm6_unfold_trial_$(date +%Y%m%d_%H%M%S)" \
  --execute "$@"
