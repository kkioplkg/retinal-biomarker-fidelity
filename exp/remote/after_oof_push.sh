#!/usr/bin/env bash
# When the FIVES OOF backfill finishes: push the updated build_data.py (it now
# reports OOF found-vs-expected) and restart ONLY the three chains whose pi fit
# had started against the incomplete set.  lodo_fives is left alone -- FIVES is
# its held-out domain, so it never reads a FIVES OOF map.
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=25 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
echo "[after_oof] $(date -Is) waiting for the backfill"
for i in $(seq 1 240); do
  n=$($SSH "ls $R/runs/seg_oof/fives/pred/prob 2>/dev/null | wc -l" 2>/dev/null || echo 0)
  case "$n" in ''|*[!0-9]*) n=0;; esac
  if [ "$n" -ge 510 ]; then echo "[after_oof] $(date -Is) mirror has $n/600 -- proceeding"; break; fi
  [ $((i % 6)) -eq 1 ] && echo "[after_oof] $(date +%H:%M) $n/600"
  sleep 60
done
if [ "${n:-0}" -lt 510 ]; then echo "[after_oof] gave up: only $n maps"; exit 1; fi

echo "[after_oof] pushing build_data.py (OOF found-vs-expected warning)"
for a in 1 2 3 4 5; do
  tar czf - src/rigr/build_data.py | $SSH "cd $R && tar xzf - -C . && echo OK" | grep -q OK && { echo "[after_oof] pushed"; break; }
  sleep 20
done
suffix="ooffix$(date +%H%M)"
echo "[after_oof] restarting lodo_drive / lodo_chasedb1 / lodo_hrf ($suffix)"
$SSH "cd $R && bash restart_fives_source_chains.sh '$suffix'" 2>&1 | sed 's/^/    /'
echo "[after_oof] done $(date -Is)"
