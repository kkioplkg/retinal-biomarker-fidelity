#!/usr/bin/env bash
# Wait for the cloud instance to come back up (the user is restarting it),
# recover the three step-D outputs that were stranded when it was shut down at
# 06:00, prove nothing else on it is unaccounted for, and shut it down again.
#
# Sequence:
#   1. ssh probe every 5 min, up to 6 h
#   2. bash remote/pull_lost_D.sh   (retries, verifies 61 / 31 / 61 rows)
#   3. FAIL-CLOSED SWEEP: list every runs/rigr/**/summary.json on the remote and
#      require a local counterpart for each.  This is the check whose absence
#      let the 06:00 shutdown strand three finished results: a gate that
#      enumerates the REMOTE side cannot be defeated by someone forgetting to
#      add a new output to a hand-written list.  Quarantined snapshots
#      (*_partial*, *_sigma_pre*, *_bdfix*, *_ooffix*, *_failed*, *_btr_pre*,
#      *_lamdedup*, *_sweepfix*) are excluded -- they are deliberately dead.
#   4. only if 2 and 3 both pass: shutdown -h now
#
# Anything unexplained writes remote/.finish_remote.ALERT and stops. The box is
# never shut down on a failed or partial check.  Terminates nothing.
set -u
cd /e/Programming/research_ws/medical1/exp || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH"
LOG=logs/await_restart.log
mkdir -p logs
exec >>"$LOG" 2>&1
echo "===== await_restart start $(date -Is) pid $$ ====="
echo $$ > remote/.await_restart.pid
STOP=remote/.await_restart.stop
ALERT=remote/.finish_remote.ALERT
rm -f "$STOP"

R=/path/to/workdir/medical1/exp
SSHOPT="-p SSH_PORT -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15"
HOST=user@GPU_HOST

# ---------------------------------------------------------------- 1. wait
up=0
for i in $(seq 1 72); do            # 72 x 5 min = 6 h
  [ -f "$STOP" ] && { echo "stop file; exiting"; exit 0; }
  if timeout 60 ssh $SSHOPT "$HOST" 'echo UP; uptime' 2>/dev/null | grep -q UP; then
    echo "[1] $(date -Is) instance is back (probe $i)"; up=1; break
  fi
  echo "[1] $(date -Is) probe $i: still down"
  sleep 300
done
if [ "$up" != 1 ]; then
  echo "[1] instance did not come back within 6 h; nothing pulled, nothing shut down."
  exit 1
fi

# ---------------------------------------------------------------- 2. pull
echo "[2] $(date -Is) recovering the three stranded step-D outputs"
if ! bash remote/pull_lost_D.sh; then
  echo "[2] pull_lost_D.sh FAILED"
  { echo "$(date -Is)"; echo "pull_lost_D.sh failed after the instance came back";
    echo "box left RUNNING on purpose; nothing was shut down."; } > "$ALERT"
  exit 1
fi
echo "[2] recovered and verified"

# ------------------------------------------------- 3. fail-closed sweep
echo "[3] $(date -Is) fail-closed sweep: every remote runs/rigr summary.json must have a local counterpart"
REMOTE_LIST=.remote_rigr_summaries.txt
got=0
for i in $(seq 1 20); do
  if timeout 120 ssh $SSHOPT "$HOST" \
      "cd $R && find runs/rigr -name summary.json 2>/dev/null | sort; echo SWEEP_OK" \
      > "$REMOTE_LIST" 2>/dev/null && grep -q SWEEP_OK "$REMOTE_LIST"; then
    got=1; break
  fi
  sleep 15
done
if [ "$got" != 1 ]; then
  echo "[3] could not enumerate the remote tree -- FAIL CLOSED, not shutting down"
  { echo "$(date -Is)"; echo "remote enumeration did not complete; cannot prove nothing is stranded";
    echo "box left RUNNING on purpose."; } > "$ALERT"
  exit 1
fi
sed -i '/SWEEP_OK/d' "$REMOTE_LIST"
missing=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in
    *_partial*|*_sigma_pre*|*_bdfix*|*_ooffix*|*_failed*|*_btr_pre*|*_lamdedup*|*_sweepfix*)
      echo "  skip (quarantined) $f"; continue ;;
  esac
  if [ -f "$f" ]; then echo "  OK   $f"
  else echo "  MISSING LOCALLY  $f"; missing=$((missing+1)); fi
done < "$REMOTE_LIST"
echo "[3] remote summary.json files: $(grep -c . "$REMOTE_LIST"), missing locally: $missing"
if [ "$missing" -ne 0 ]; then
  echo "[3] FAIL CLOSED -- $missing remote result(s) have no local counterpart; NOT shutting down"
  { echo "$(date -Is)"; echo "$missing remote runs/rigr result(s) are not present locally";
    echo "see logs/await_restart.log and $REMOTE_LIST; box left RUNNING."; } > "$ALERT"
  exit 1
fi

# ---------------------------------------------------------------- 4. shutdown
# DISABLED 2026-09-06 15:35 by explicit user instruction, which overrides the
# earlier "release the billed server once nothing more can run there" directive:
# the box is now to be used for the remaining stage-3 work (offloaded D / F /
# LODO risk re-runs), and it is shut down ONLY on an explicit instruction.
# Set ALLOW_SHUTDOWN=1 in the environment to re-enable this block.
if [ "${ALLOW_SHUTDOWN:-0}" = "1" ]; then
  echo "[4] $(date -Is) everything accounted for; shutting the instance down"
  for i in $(seq 1 8); do
    timeout 60 ssh $SSHOPT "$HOST" "nohup sh -c 'sleep 3; shutdown -h now' >/dev/null 2>&1 & exit 0" 2>/dev/null && break
    sleep 15
  done
  date -Is > remote/.remote_shutdown_issued_2
  echo "[4] shutdown issued $(date -Is)"
else
  echo "[4] $(date -Is) AUTOMATIC SHUTDOWN IS DISABLED (user instruction "
  echo "    2026-09-06 15:35).  Everything is accounted for, but the box is "
  echo "    left RUNNING for the remaining stage-3 work.  Re-enable with "
  echo "    ALLOW_SHUTDOWN=1 only on an explicit instruction."
fi
echo "===== await_restart end $(date -Is) ====="
