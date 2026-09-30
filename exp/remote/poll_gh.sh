#!/usr/bin/env bash
# Detached poller: pull G/H artefacts from the remote mirror every 15 minutes
# until both are complete (or --once / a stop file).  Safe to run twice: the
# pull itself is idempotent.
#   bash exp/remote/poll_gh.sh            # loop
#   bash exp/remote/poll_gh.sh --once
set -u
cd "$(dirname "$0")/.."
STOP=remote/.poll_gh.stop
INTERVAL=${POLL_INTERVAL:-900}
once=0; [ "${1:-}" = "--once" ] && once=1
[ "$once" = 1 ] || echo $$ > "remote/.poll_gh.pid"   # only a looping instance claims the pid file
rm -f "$STOP"
while :; do
  echo "===== $(date -Is) poll ====="
  bash remote/pull_gh.sh
  # done when all four LODO runs and the Exp2 table have landed
  n=$(ls -d runs/rigr/lodo_*/risk runs/rigr/lodo_*/uniform 2>/dev/null | wc -l)
  if [ "$n" -ge 8 ] && [ -f results/tab1_exp2.csv ]; then
    echo "[poll] G (8/8 lodo run dirs) and H (results/tab1_exp2.csv) are local; stopping."
    break
  fi
  echo "[poll] have $n/8 lodo run dirs; tab1_exp2.csv: $([ -f results/tab1_exp2.csv ] && echo yes || echo no)"
  [ "$once" = 1 ] && break
  [ -f "$STOP" ] && { echo "[poll] stop file present; exiting."; break; }
  sleep "$INTERVAL"
done
echo "[poll] exit $(date -Is)"
