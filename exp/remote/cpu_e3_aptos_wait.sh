#!/usr/bin/env bash
# Wait for APTOS-2019's twelve FIVES biomarker tables to land in the LOCAL tree
# (the LAN run rsyncs them into results/pivot/e3/aptos2019/), then push them
# to the CPU node and start the locked classification stage there.
#
#   nohup bash exp/remote/cpu_e3_aptos_wait.sh > runs/cpu_e3_aptos_wait.log 2>&1 &
#
# It refuses to start on a partial cohort: every one of the twelve bio.csv must
# exist AND have exactly 3662 rows AND carry a meta.json (the E3 bio stage
# appends every 10 images, so a row count is the only safe completeness test).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

SSH_HOST="${SSH_HOST:-user@CPU_HOST}"
SSH_PORT="${SSH_PORT:-SSH_PORT}"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30
          -o ServerAliveCountMax=6 -o Compression=no -c aes128-gcm@openssh.com)
REMOTE_EXP="${REMOTE_ROOT:-/path/to/workdir/medical1}/exp"
DS=aptos2019
ROWS="${ROWS:-3662}"
ROUND_S="${ROUND_S:-600}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

while true; do
  ready=0; missing=()
  : > "$TMP/list.txt"
  for a in baseline continued reliseg cfloss; do
    for s in 0 1 2; do
      t="e3_fives_s${s}_${a}"
      f="results/pivot/e3/$DS/$t/bio.csv"
      if [ -f "$f" ] && [ -f "results/pivot/e3/$DS/$t/meta.json" ]; then
        n=$(( $(wc -l < "$f") - 1 ))
        if [ "$n" -eq "$ROWS" ]; then
          ready=$((ready + 1)); echo "results/pivot/e3/$DS/$t" >> "$TMP/list.txt"; continue
        fi
        missing+=("$t(rows=$n)")
      else
        missing+=("$t(absent)")
      fi
    done
  done
  echo "[aptos] $(date +%F' '%T) ready $ready/12 ${missing[*]:-}"
  if [ "$ready" -eq 12 ]; then
    echo "[aptos] all 12 tables complete -- pushing to node"
    LIST="$TMP/list.txt" bash remote/cpu_sync.sh 2>&1 | sed 's/^/[aptos] /'
    echo "[aptos] launching clf on node"
    ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
      "cd '$REMOTE_EXP' && nohup setsid env NJOBS=12 SEEDPAR=2 PAR=1 bash remote/cpu_e3_clf.sh $DS >> runs/cpu_e3_driver.log 2>&1 < /dev/null & sleep 2; echo launched"
    echo "[aptos] done -- exiting $(date +%F' '%T)"
    exit 0
  fi
  sleep "$ROUND_S"
done
