#!/usr/bin/env bash
# Run a command on the remote GPU server, with cwd = /path/to/workdir/medical1/exp.
#
# Usage:
#   bash exp/remote/remote.sh "python -m src.topo.tests_synthetic"
#   bash exp/remote/remote.sh --env medical1 "python -m src.topo.tests_synthetic"
#   bash exp/remote/remote.sh --bg myjob "python -m src.train.foo --epochs 50"
#   bash exp/remote/remote.sh --env c1venv --bg c1job "python -m src.c1.run"
#
# --env medical1|c1venv : which Python environment to use (default: medical1).
#                          medical1 is a conda env (/root/miniconda3/envs/medical1).
#                          c1venv   is a plain venv   (/path/to/workdir/c1venv).
# --bg <name>            : run detached via nohup, logging to
#                          /path/to/workdir/medical1/exp/runs_remote/logs/<name>.log
#                          and return immediately (prints the remote PID).
#
# All remote commands run under nohup and the ssh connection uses
# ServerAliveInterval=30, because the host drops long-lived idle ssh sessions.

set -euo pipefail

SSH_PORT=SSH_PORT
SSH_HOST="user@GPU_HOST"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/path/to/workdir/medical1/exp"

BG_MODE=0
BG_NAME=""
ENV_NAME="medical1"

while [ "${1:-}" == "--bg" ] || [ "${1:-}" == "--env" ]; do
  case "$1" in
    --bg)  BG_MODE=1; BG_NAME="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
  esac
done

CMD="$*"
if [ -z "$CMD" ]; then
  echo "usage: remote.sh [--env medical1|c1venv] [--bg name] \"<command>\"" >&2
  exit 1
fi

case "$ENV_NAME" in
  medical1)
    ENV_SETUP='source /root/miniconda3/bin/activate medical1'
    ;;
  c1venv)
    ENV_SETUP='source /path/to/workdir/c1venv/bin/activate'
    ;;
  *)
    echo "unknown --env '$ENV_NAME' (expected medical1 or c1venv)" >&2
    exit 1
    ;;
esac

if [ "$BG_MODE" -eq 1 ]; then
  LOG="$REMOTE_DIR/runs_remote/logs/${BG_NAME}.log"
  echo "[remote.sh] env=$ENV_NAME launching in background as '$BG_NAME', log: $LOG"
  REMOTE_SCRIPT=$(cat <<EOF
mkdir -p "$REMOTE_DIR/runs_remote/logs"
$ENV_SETUP
cd "$REMOTE_DIR"
nohup bash -c $(printf '%q' "$CMD") > "$LOG" 2>&1 &
echo REMOTE_PID=\$!
disown
EOF
)
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_SCRIPT"
else
  REMOTE_SCRIPT=$(cat <<EOF
$ENV_SETUP
cd "$REMOTE_DIR"
nohup bash -c $(printf '%q' "$CMD")
EOF
)
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_SCRIPT"
fi
