#!/usr/bin/env bash
# Run a command on the LAN GPU node, with cwd = /home/user/medical1/exp
# and the venv at /home/user/medical1/venv activated.
#
# Usage:
#   bash exp/remote/lan_remote.sh "python -m src.topo.tests_synthetic"
#   bash exp/remote/lan_remote.sh --bg myjob "python -m src.train.foo --epochs 50"
#
# --bg <name> : run detached via nohup, logging to
#               /home/user/medical1/exp/runs_remote/logs/<name>.log
#               and return immediately (prints the remote PID).
#
# Node has a single RTX 3080 10GB (not 20GB like the old cloud box) --
# halve any seg training batch sizes (bs 4 -> 2) when driving jobs through
# this script.

set -euo pipefail

SSH_HOST="user@LAN_HOST"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/home/user/medical1/exp"
VENV_ACTIVATE="source /home/user/medical1/venv/bin/activate"

BG_MODE=0
BG_NAME=""

while [ "${1:-}" == "--bg" ]; do
  BG_MODE=1; BG_NAME="$2"; shift 2
done

CMD="$*"
if [ -z "$CMD" ]; then
  echo "usage: lan_remote.sh [--bg name] \"<command>\"" >&2
  exit 1
fi

if [ "$BG_MODE" -eq 1 ]; then
  LOG="$REMOTE_DIR/runs_remote/logs/${BG_NAME}.log"
  echo "[lan_remote] launching in background as '$BG_NAME', log: $LOG"
  REMOTE_SCRIPT=$(cat <<EOF
mkdir -p "$REMOTE_DIR/runs_remote/logs"
$VENV_ACTIVATE
cd "$REMOTE_DIR"
nohup bash -c $(printf '%q' "$CMD") > "$LOG" 2>&1 &
echo REMOTE_PID=\$!
disown
EOF
)
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_SCRIPT"
else
  REMOTE_SCRIPT=$(cat <<EOF
$VENV_ACTIVATE
cd "$REMOTE_DIR"
bash -c $(printf '%q' "$CMD")
EOF
)
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_SCRIPT"
fi
