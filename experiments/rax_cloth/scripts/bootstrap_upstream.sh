#!/usr/bin/env bash
# Run when GitHub is reachable. No hardware or agent launch; source checkout only.
set -euo pipefail
RAX_CLOTH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RAX_DEST="$RAX_CLOTH_ROOT/vendor/rax"
if [[ -e "$RAX_DEST" ]]; then
  echo 'Destination exists; refusing to overwrite or silently update pinned source.' >&2
  exit 1
fi
mkdir -p "$RAX_CLOTH_ROOT/vendor"
git clone --branch real/rainbow --single-branch https://github.com/robot-agent-exploration/rax.git "$RAX_DEST"
RAX_COMMIT="$(git -C "$RAX_DEST" rev-parse HEAD)"
git -C "$RAX_DEST" checkout --detach "$RAX_COMMIT"
printf '%s\n' "$RAX_COMMIT" > "$RAX_CLOTH_ROOT/vendor/UPSTREAM_COMMIT"
echo "Pinned real/rainbow at $RAX_COMMIT; source only, no execution enabled."
