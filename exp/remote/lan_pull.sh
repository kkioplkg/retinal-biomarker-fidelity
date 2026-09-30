#!/usr/bin/env bash
# Pull a subpath of /home/user/medical1/exp back to a local directory.
#
# Usage:
#   bash exp/remote/lan_pull.sh <remote_subpath> [local_dir]
#
# Examples:
#   bash exp/remote/lan_pull.sh results
#       -> pulls exp/results (remote) into exp/results_remote_lan/results/
#   bash exp/remote/lan_pull.sh runs/seg/drive/seed0 exp/results_remote_lan
#       -> pulls runs/seg/drive/seed0 into exp/results_remote_lan/runs/seg/drive/seed0

set -euo pipefail

SSH_HOST="user@LAN_HOST"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/home/user/medical1/exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBPATH="${1:?usage: lan_pull.sh <remote_subpath> [local_dir]}"
DEST="${2:-$EXP_DIR/results_remote_lan}"

mkdir -p "$DEST/$(dirname "$SUBPATH")"

echo "[lan_pull] pulling $SUBPATH -> $DEST/$SUBPATH"
ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
  "cd '$REMOTE_DIR' && tar czf - '$SUBPATH'" \
| tar xzf - -C "$DEST"

echo "[lan_pull] done. Local contents:"
find "$DEST/$SUBPATH" -maxdepth 1 | head -20
echo "[lan_pull] local file count: $(find "$DEST/$SUBPATH" -type f | wc -l)"
echo "[lan_pull] remote file count: $(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "find '$REMOTE_DIR/$SUBPATH' -type f | wc -l")"
