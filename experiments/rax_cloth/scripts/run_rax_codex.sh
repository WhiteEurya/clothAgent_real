#!/usr/bin/env bash
# RAX workflow + Codex CLI adapter. Robot creation remains in the official backend registry.
set -euo pipefail
RAX_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$RAX_SCRIPT_DIR/run_agent_offline.sh" --workflow upstream "$@"
