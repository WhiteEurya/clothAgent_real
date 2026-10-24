#!/usr/bin/env bash
# Operator-reviewed handoff to the existing folding demonstrations.
# n027/n077 do not have a graph edge to n000; fresh state/anchor review is required.
set -euo pipefail
cd "$(dirname "$0")/.."
exec /home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.graph_policy_replay \
  --graph reports/arm6_graph_incremental_20261019_v3/final/graph.json \
  --arm6-expanded-graph \
  --arm6-route e000,e001,e002,e003,e004,e005,e006 \
  --arm6-review \
  --matcher retrieval_gpt_rerank \
  --retrieval-memory reports/arm6_graph_incremental_20261019_v3/memory \
  --max-steps 7 \
  --output "reports/arm6_fold_trial_$(date +%Y%m%d_%H%M%S)" \
  --execute "$@"
