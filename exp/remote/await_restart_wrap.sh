#!/usr/bin/env bash
cd /e/Programming/research_ws/medical1/exp || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH"
mkdir -p logs
if ! mkdir remote/.await_restart.lock 2>/dev/null; then
  echo "[wrap] another await_restart holds the lock $(date -Is)" >> logs/await_restart.log; exit 0
fi
trap 'rm -rf remote/.await_restart.lock' EXIT
bash remote/await_restart_pull_shutdown.sh
