#!/usr/bin/env bash
# Quick health check of the LAN GPU node: GPU usage, uptime, running python
# processes, disk usage.
#
# Usage: bash exp/remote/lan_status.sh

set -euo pipefail

SSH_HOST="user@LAN_HOST"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

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
echo "===== screen/tmux sessions ====="
screen -ls 2>&1 || true
tmux ls 2>&1 || true
echo
echo "===== disk usage ====="
df -h / /home
echo
echo "===== medical1/exp size ====="
du -sh /home/user/medical1/exp 2>/dev/null
du -sh /home/user/medical1/exp/* 2>/dev/null
'
