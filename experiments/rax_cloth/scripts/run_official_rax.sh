#!/usr/bin/env bash
# Only delegate to the pinned upstream entrypoint. No custom agent, relay or service.
set -euo pipefail
RAX_CLOTH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAX_UPSTREAM="$RAX_CLOTH_ROOT/vendor/rax"
RAX_EXPECTED=36c926ca3cf15921c6fc746a4bf309c26b0aa3c4
[[ "$(git -C "$RAX_UPSTREAM" rev-parse HEAD)" == "$RAX_EXPECTED" ]] || { echo 'Upstream commit mismatch' >&2; exit 2; }
[[ -z "$(git -C "$RAX_UPSTREAM" status --porcelain)" ]] || { echo 'Upstream checkout has unreviewed modifications' >&2; exit 2; }
export RAX_PYTHON="${RAX_PYTHON:-$RAX_CLOTH_ROOT/.venv-host/bin/python}"
exec bash "$RAX_UPSTREAM/scripts/start_agent.sh" "$@"
