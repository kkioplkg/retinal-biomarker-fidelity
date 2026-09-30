#!/usr/bin/env bash
# NODE side of the E4 MAPLES-DR offload: poll for newly arrived, fully synced
# prediction directories and measure their biomarkers.  Launch once, detached:
#
#   ssh -p SSH_PORT root@... "cd /path/to/workdir/medical1/exp && \
#     nohup setsid bash remote/cpu_e4_watch.sh > runs/cpu_e4_watch.log 2>&1 < /dev/null &"
#
# Exits by itself once all EXPECT directories have a bio.csv.

set -uo pipefail
R="${R:-/path/to/workdir/medical1}"
cd "$R/exp"
MAPLES="runs/pivot/e4/maples"
EXPECT="${EXPECT:-24}"
SLEEP_S="${SLEEP_S:-120}"

while true; do
  PAR="${PAR:-4}" PROCS="${PROCS:-8}" bash remote/cpu_e4_bio.sh
  n=$(ls -d $MAPLES/*/bio.csv 2>/dev/null | wc -l)
  echo "[watch] $(date +%F' '%T)  bio.csv done: $n / $EXPECT"
  [ "$n" -ge "$EXPECT" ] && { echo "[watch] all done -- exiting"; exit 0; }
  sleep "$SLEEP_S"
done
