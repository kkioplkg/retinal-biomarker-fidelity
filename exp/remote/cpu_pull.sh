#!/usr/bin/env bash
# Pull a subtree of the CPU node's work root back into the local exp/ tree.
#
#   bash exp/remote/cpu_pull.sh results/pivot/e3/aptos2019
#   DEST=exp/results_remote_cpu bash exp/remote/cpu_pull.sh runs/cpu_e4
#
# Default destination is the local exp/ tree itself (same relative path), which
# is what the E4 bio.csv files and the E3 clf outputs are meant for.  Files are
# overwritten; nothing is deleted on either side.

set -euo pipefail

SSH_HOST="${SSH_HOST:-user@CPU_HOST}"
SSH_PORT="${SSH_PORT:-SSH_PORT}"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30
          -o ServerAliveCountMax=6 -o Compression=no -c aes128-gcm@openssh.com)
REMOTE_DIR="${REMOTE_ROOT:-/path/to/workdir/medical1}/exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${DEST:-$EXP_DIR}"
cd "$EXP_DIR"

[ "$#" -ge 1 ] || { echo "usage: cpu_pull.sh <remote_subpath> [...]" >&2; exit 1; }

for SUB in "$@"; do
  if ! ssh "${SSH_OPTS[@]}" "$SSH_HOST" "test -e '$REMOTE_DIR/$SUB'"; then
    echo "[cpu_pull] SKIP (absent remotely): $SUB" >&2
    continue
  fi
  echo "[cpu_pull] $SUB  ->  $DEST/$SUB"
  mkdir -p "$DEST/$(dirname "$SUB")"
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cd '$REMOTE_DIR' && tar cf - $(printf '%q' "$SUB")" \
  | tar xf - -C "$DEST"
  R=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "find '$REMOTE_DIR/$SUB' -type f | wc -l")
  L=$(find "$DEST/$SUB" -type f | wc -l)
  echo "[cpu_pull]   remote files: $R   local files: $L"
  [ "$R" = "$L" ] || echo "[cpu_pull]   WARNING: file counts differ" >&2
done
