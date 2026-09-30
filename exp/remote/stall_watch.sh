#!/usr/bin/env bash
# Stall detector for the remote step-G / E:train logs.
#
# WHY.  At 09:50 an fd exhaustion killed three head trainings.  lodo_hrf exited
# rc=1 and my step-event monitor reported it.  lodo_drive and lodo_chasedb1
# instead HUNG -- the main process survived, wedged on dead DataLoader workers,
# writing nothing.  A hang emits no START, no DONE and no FAILED, so an
# event-driven monitor is blind to it; both sat there for 3 h 52 m holding
# ~1.4 GB of GPU each and were tenants in the rnca_fives OOM at 12:47.
#
# This watches log MTIME instead of log content: any tracked log that has not
# advanced in STALL_MIN minutes while its process is still alive is reported.
# It only reports -- it never terminates anything.
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 user@GPU_HOST"
R=/path/to/workdir/medical1
STALL_MIN=${STALL_MIN:-25}
INTERVAL=${STALL_INTERVAL:-600}
STOP=remote/.stall_watch.stop
once=0; [ "${1:-}" = "--once" ] && once=1
[ "$once" = 1 ] || echo $$ > remote/.stall_watch.pid
rm -f "$STOP"

while :; do
  echo "===== $(date -Is) stall check (threshold ${STALL_MIN} min) ====="
  out=$($SSH "cd $R && now=\$(date +%s); for f in exp/runs/s4_logs/G_head_*.log exp/runs/s4_logs/G_data_*.log logs/Etrain_*.log; do
          [ -f \"\$f\" ] || continue
          m=\$(stat -c %Y \"\$f\"); age=\$(( (now - m) / 60 ))
          if [ \"\$age\" -ge $STALL_MIN ]; then
            tail -1 \"\$f\" | grep -qE 'finished|rc=|done:|DONE' && continue   # finished, not stalled
            echo \"STALL \$age min  \$f  | \$(tail -1 \"\$f\" | cut -c1-70)\"
          fi
        done; echo PROBE_OK" 2>/dev/null) || out=""
  if ! printf '%s' "$out" | grep -q PROBE_OK; then
    echo "[stall] probe did not complete (ssh failure) -- no conclusion drawn"
  else
    hits=$(printf '%s\n' "$out" | grep '^STALL' || true)
    if [ -n "$hits" ]; then printf '%s\n' "$hits" | sed 's/^/  /'
    else echo "  no stalled logs"; fi
  fi
  [ "$once" = 1 ] && break
  [ -f "$STOP" ] && { echo "[stall] stop file present; exiting."; break; }
  sleep "$INTERVAL"
done
