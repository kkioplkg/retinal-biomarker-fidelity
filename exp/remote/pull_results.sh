#!/usr/bin/env bash
# Pull /path/to/workdir/medical1/exp/results (and optional runs/ subdirs) back
# to a local directory.
#
# Usage:
#   bash exp/remote/pull_results.sh [local_dest_dir] [runs_subdir1 runs_subdir2 ...]
#
# Examples:
#   bash exp/remote/pull_results.sh
#       -> pulls results/ into exp/results_remote/results/
#   bash exp/remote/pull_results.sh exp/results_remote runs/c1 runs/gateA
#       -> also pulls runs/c1 and runs/gateA into exp/results_remote/runs/...

set -euo pipefail

SSH_PORT=SSH_PORT
SSH_HOST="user@GPU_HOST"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/path/to/workdir/medical1/exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEST="${1:-$EXP_DIR/results_remote}"
shift || true
RUNS_SUBDIRS=("$@")

mkdir -p "$DEST/results"

echo "[pull_results] pulling results/ -> $DEST/results"
ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
  "cd '$REMOTE_DIR' && tar czf - results" \
| tar xzf - -C "$DEST" --strip-components=0

if [ "${#RUNS_SUBDIRS[@]}" -gt 0 ]; then
  for sub in "${RUNS_SUBDIRS[@]}"; do
    echo "[pull_results] pulling $sub -> $DEST/$sub"
    mkdir -p "$DEST/$(dirname "$sub")"
    ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
      "cd '$REMOTE_DIR' && tar czf - '$sub' 2>/dev/null" \
    | tar xzf - -C "$DEST"
  done
fi

echo "[pull_results] done. Local contents:"
find "$DEST" -maxdepth 2
