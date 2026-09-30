#!/usr/bin/env bash
# Re-push exp/src + plan files to the remote GPU server.
# Excludes __pycache__ and *.pt checkpoint files. Safe to re-run any time.
#
# Usage: bash exp/remote/sync_code.sh
# Run from anywhere; it cd's to the exp/ directory relative to this script.

set -euo pipefail

SSH_PORT=SSH_PORT
SSH_HOST="user@GPU_HOST"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/path/to/workdir/medical1/exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$EXP_DIR"

echo "[sync_code] pushing src/ + plan files -> ${SSH_HOST}:${REMOTE_DIR}"

tar czf - \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  src EXPERIMENT_PLAN.md DECISIONS.md \
| ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
  "mkdir -p '$REMOTE_DIR' && cd '$REMOTE_DIR' && tar xzf -"

echo "[sync_code] done. Verifying file count..."
LOCAL_COUNT=$(find src -type f ! -path '*__pycache__*' ! -name '*.pt' ! -name '*.pth' | wc -l)
REMOTE_COUNT=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "find '$REMOTE_DIR/src' -type f ! -path '*__pycache__*' | wc -l")
echo "[sync_code] local src files: $LOCAL_COUNT   remote src files: $REMOTE_COUNT"
if [ "$LOCAL_COUNT" != "$REMOTE_COUNT" ]; then
  echo "[sync_code] WARNING: file counts differ!" >&2
fi
