#!/usr/bin/env bash
# Pull a subtree of the LAN node's NEW work root back into the local exp/ tree.
#
#   bash exp/remote/lan2_pull.sh results/pivot/e3/idrid
#   DEST=exp/results_remote_lan bash exp/remote/lan2_pull.sh runs/lan_e3
#
# Default destination is the local exp/ tree itself (same relative path), which
# is what the E3 results are meant for. Set DEST to stage them elsewhere.
# Files are overwritten; nothing is deleted on either side (user rule: the node
# keeps its copy).

set -euo pipefail

SSH_HOST="${SSH_HOST:-user@LAN_HOST}"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6
          -o Compression=no -c aes128-gcm@openssh.com)
REMOTE_DIR="${REMOTE_ROOT:-/mnt/data/Programming/research_ws/medical1}/exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${DEST:-$EXP_DIR}"
cd "$EXP_DIR"

[ "$#" -ge 1 ] || { echo "usage: lan2_pull.sh <remote_subpath> [...]" >&2; exit 1; }

for SUB in "$@"; do
  echo "[lan2_pull] $SUB  ->  $DEST/$SUB"
  mkdir -p "$DEST/$(dirname "$SUB")"
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cd '$REMOTE_DIR' && tar cf - $(printf '%q' "$SUB")" \
  | tar xf - -C "$DEST"
  R=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "find '$REMOTE_DIR/$SUB' -type f | wc -l")
  L=$(find "$DEST/$SUB" -type f | wc -l)
  echo "[lan2_pull]   remote files: $R   local files: $L"
  [ "$R" = "$L" ] || echo "[lan2_pull]   WARNING: file counts differ" >&2
done
