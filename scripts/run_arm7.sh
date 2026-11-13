#!/usr/bin/env bash
# Short entrypoint for the current site's real, independent graph replay.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CLOTH_PYTHON="${CLOTH_PYTHON:-/home/CNS2026330003/miniconda3/envs/cloth-real/bin/python}"
export PATH="/home/CNS2026330003/miniconda3/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin:$PATH"
printf -v replay_command '%q ' bash scripts/run_arm7_partitioned_replay.sh --execute "$@"
exec sg zed -c "$replay_command"
