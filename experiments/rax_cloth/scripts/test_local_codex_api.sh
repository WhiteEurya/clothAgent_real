#!/usr/bin/env bash
# Direct CLI/provider diagnostic: no RAX relay, robot service or garment workspace.
set -euo pipefail
RAX_TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAX_TEST_LOG="$(mktemp -d "$RAX_TEST_ROOT/validation/direct-api-XXXXXXXX")"
RAX_TEST_WORK="$(mktemp -d)"
trap 'rm -rf -- "$RAX_TEST_WORK"' EXIT
printf 'Direct codex -p api test; limit 180 seconds. Logs: %s\n' "$RAX_TEST_LOG"
RAX_TEST_START=$SECONDS
set +e
timeout --signal=TERM --kill-after=5s 180s codex exec -p api \
  --ephemeral --sandbox read-only --skip-git-repo-check --ignore-rules \
  --cd "$RAX_TEST_WORK" --json --color never -c 'approval_policy="never"' \
  'Reply with exactly RAX_API_OK. Do not use tools, read files, or run commands.' \
  </dev/null 2>"$RAX_TEST_LOG/stderr.txt" | tee "$RAX_TEST_LOG/events.jsonl"
RAX_TEST_RC=${PIPESTATUS[0]}
set -e
printf 'elapsed_seconds=%s exit_code=%s\n' "$((SECONDS-RAX_TEST_START))" "$RAX_TEST_RC" | tee "$RAX_TEST_LOG/result.txt"
if [[ "$RAX_TEST_RC" != 0 ]]; then tail -n 12 "$RAX_TEST_LOG/stderr.txt"; fi
exit "$RAX_TEST_RC"
