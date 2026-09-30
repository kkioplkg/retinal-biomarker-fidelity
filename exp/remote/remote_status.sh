#!/usr/bin/env bash
# Quick health check of the remote GPU server: GPU usage, uptime, running
# python processes, disk usage.
#
# Usage: bash exp/remote/remote_status.sh

set -euo pipefail

SSH_PORT=SSH_PORT
SSH_HOST="user@GPU_HOST"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

ssh "${SSH_OPTS[@]}" "$SSH_HOST" '
echo "===== nvidia-smi ====="
nvidia-smi
echo
echo "===== uptime ====="
uptime
echo
echo "===== python processes ====="
ps aux | grep -i "[p]ython"
echo
echo "===== screen sessions ====="
screen -ls 2>&1 || true
echo
echo "===== disk usage ====="
df -h / /path/to/workdir
echo
echo "===== medical1/exp size ====="
du -sh /path/to/workdir/medical1/exp 2>/dev/null
du -sh /path/to/workdir/medical1/exp/* 2>/dev/null
'
