#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# A single two-drop trial, never the unattended unfold batch.
exec bash scripts/run_arm7_unfold_dual_wrist.sh "$@" --runs 1 --reset-only --reset-repeats 2
