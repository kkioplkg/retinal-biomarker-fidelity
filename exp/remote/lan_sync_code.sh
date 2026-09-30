#!/usr/bin/env bash
# Re-push exp/src + plan files + configs to the LAN GPU node.
# Excludes __pycache__ and checkpoint files. Safe to re-run any time.
#
# Usage: bash exp/remote/lan_sync_code.sh

set -euo pipefail

SSH_HOST="user@LAN_HOST"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/home/user/medical1/exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$EXP_DIR"

echo "[lan_sync_code] pushing src/ + configs/ + plan files -> ${SSH_HOST}:${REMOTE_DIR}"

tar czf - \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  src configs EXPERIMENT_PLAN.md DECISIONS.md S4_RUNBOOK.md \
| ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
  "mkdir -p '$REMOTE_DIR' && cd '$REMOTE_DIR' && tar xzf -"

echo "[lan_sync_code] done. Verifying file count..."
LOCAL_COUNT=$(find src -type f ! -path '*__pycache__*' ! -name '*.pt' ! -name '*.pth' | wc -l)
REMOTE_COUNT=$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" "find '$REMOTE_DIR/src' -type f ! -path '*__pycache__*' | wc -l")
echo "[lan_sync_code] local src files: $LOCAL_COUNT   remote src files: $REMOTE_COUNT"
if [ "$LOCAL_COUNT" != "$REMOTE_COUNT" ]; then
  echo "[lan_sync_code] WARNING: file counts differ!" >&2
fi
