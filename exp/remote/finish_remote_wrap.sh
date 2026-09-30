#!/usr/bin/env bash
# Wrapper: keeps ONE finish_remote.sh alive across transient ssh failures.
# The mkdir lock is atomic, so a second wrapper exits instead of racing the
# first -- the same guard the remote H launch needed.
cd /e/Programming/research_ws/medical1/exp || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH"
mkdir -p logs
if ! mkdir remote/.finish_wrap.lock 2>/dev/null; then
  echo "[wrap] $(date -Is) another wrapper holds remote/.finish_wrap.lock; exiting" >> logs/finish_remote.log
  exit 0
fi
echo $$ > remote/.finish_wrap.lock/pid
trap 'rm -rf remote/.finish_wrap.lock' EXIT
echo "[wrap] start $(date -Is) pid $$" >> logs/finish_remote.log
for i in $(seq 1 60); do
  [ -f remote/.remote_shutdown_issued ] && break
  [ -f remote/.finish_remote.stop ] && break
  [ -f remote/.finish_remote.ALERT ] && { echo "[wrap] ALERT present; not restarting" >> logs/finish_remote.log; break; }
  bash remote/finish_remote.sh >> logs/finish_remote_wrap_stderr.log 2>&1
  sleep 120
done
echo "[wrap] exit $(date -Is)" >> logs/finish_remote.log
