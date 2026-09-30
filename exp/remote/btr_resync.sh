#!/usr/bin/env bash
# Watch runs/btr/habs_trainonly/ for the C1 agent's regeneration (FIVES train
# events added), re-push it to the remote mirror, and repair any LODO run that
# was already priced by the OLD heads.
#
# Only `G:run:<D>:risk` consumes the BTR heads (`_rigr_cmd` adds --btr_* only
# for mode == risk), so a `uniform` run is never invalidated.
#
# Repair policy (the coordinator's: "a re-run is cheap (run_rigr only) and
# preferred"): the stale output directory is *moved aside*, never deleted, and
# the per-dataset G orchestrator is relaunched -- head / data / fit are already
# DONE on disk so only run_rigr re-executes.
#
#   bash exp/remote/btr_resync.sh --once
#   bash exp/remote/btr_resync.sh            # loop every 5 min
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
SRC=runs/btr/habs_trainonly
FP=remote/.btr_habs.fingerprint
VLOG=remote/btr_versions.log
STOP=remote/.btr_resync.stop
INTERVAL=${BTR_INTERVAL:-300}
once=0; [ "${1:-}" = "--once" ] && once=1
[ "$once" = 1 ] || echo $$ > "remote/.btr_resync.pid"   # only a looping instance claims the pid file
rm -f "$STOP"

fingerprint() {
  ( cd "$SRC" 2>/dev/null && for f in *.joblib; do
      printf '%s %s %s\n' "$f" "$(stat -c %s "$f")" "$(stat -c %Y "$f")"
    done ) | sort
}

# The C1 agent rewrites these files in place.  At 10:33 btr_{miss,false}_all
# were 1019/1020 bytes -- LightGBM heads mid-write, not finished heads.  Pushing
# one of those would put a stub on the remote and every risk run after it would
# be priced by garbage.  So: all ten files must exist, each must clear a size
# floor, and the fingerprint must be IDENTICAL to the previous cycle before
# anything is pushed.  A set still in motion is reported and left alone.
MIN_BYTES=${BTR_MIN_BYTES:-100000}
PREV=""
sane() {
  local n bad=0 f sz
  n=$(ls "$SRC"/*.joblib 2>/dev/null | wc -l)
  if [ "$n" -ne 10 ]; then echo "  [btr] only $n/10 .joblib present -- still being written"; return 1; fi
  for f in "$SRC"/*.joblib; do
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -lt "$MIN_BYTES" ]; then
      echo "  [btr] $(basename "$f") is only $sz bytes (< $MIN_BYTES) -- partial"; bad=1
    fi
  done
  return $bad
}

while :; do
  now=$(fingerprint)
  old=$(cat "$FP" 2>/dev/null || true)
  if [ -n "$now" ] && [ "$now" != "$old" ] && { ! sane || [ "$now" != "$PREV" ]; }; then
    echo "$(date -Is) habs_trainonly is CHANGING -- waiting for it to settle (no push)"
    PREV="$now"
  elif [ -n "$now" ] && [ "$now" != "$old" ]; then
    PREV="$now"
    ts=$(date -Is)
    echo "===== $ts habs_trainonly CHANGED ====="
    diff <(printf '%s\n' "$old") <(printf '%s\n' "$now") | sed 's/^/    /' || true

    # 1. which risk runs are already on the remote?  Those used the old heads.
    #    FAIL CLOSED: an ssh that dies mid-probe must not look like "nothing to
    #    repair".  Abort the trigger instead and retry with the fingerprint
    #    still un-advanced.
    if ! stale=$($SSH "cd $R && ls -d runs/rigr/lodo_*/risk 2>/dev/null; echo PROBE_OK" 2>/dev/null)        || ! printf '%s' "$stale" | grep -q PROBE_OK; then
      echo "[btr] stale-list probe did not complete (ssh failure?) -- aborting this trigger, will retry"
      [ "$once" = 1 ] && break
      sleep 60; continue
    fi
    stale=$(printf '%s
' "$stale" | grep -v PROBE_OK | grep -v '^$' || true)

    # 2. push the new heads
    if tar czf - "$SRC" | $SSH "tar xzf - -C $R"; then
      echo "[btr] pushed $SRC  $ts"
    else
      echo "[btr] PUSH FAILED $ts -- not updating the fingerprint, will retry"
      [ "$once" = 1 ] && break
      sleep 60; continue
    fi
    {
      echo "## $ts  runs/btr/habs_trainonly pushed to the remote mirror"
      printf '%s\n' "$now" | sed 's/^/    /'
      echo "    risk runs that existed BEFORE this push (priced by the previous heads):"
      if [ -n "$stale" ]; then printf '%s\n' "$stale" | sed 's/^/      /'; else echo "      (none)"; fi
    } >> "$VLOG"
    printf '%s\n' "$now" > "$FP"

    # 3. repair anything that was already priced by the old heads
    if [ -n "$stale" ]; then
      suffix="btr_pre$(date +%H%M)"
      for d in $stale; do
        ds=$(printf '%s\n' "$d" | sed -n 's|runs/rigr/lodo_\([^/]*\)/risk|\1|p')
        [ -n "$ds" ] || continue
        echo "[btr] re-running G:run:$ds:risk (old output -> ${d%/risk}/risk_$suffix)"
        $SSH "cd $R && mv '$d' '${d%/risk}/risk_$suffix'"
        $SSH "cd $R && mkdir -p runs/s4_logs && setsid nohup env PATH=/root/miniconda3/envs/medical1/bin:\$PATH \
                PYTHONPATH=$R PYTHONUNBUFFERED=1 \
                python -m src.pipeline.s4_all --only G --datasets $ds --seeds 0 --jobs 2 \
                --config s4_remote_cfg_gpu1.json \
                > runs/s4_logs/orch_G_rerun_${ds}_$suffix.log 2>&1 < /dev/null & sleep 2; exit 0"
        echo "[btr] relaunched G for $ds (head/data/fit are DONE on disk, so only run_rigr re-executes)"
      done
    else
      echo "[btr] no risk run had completed yet -- nothing to repair, the new heads are simply in place."
    fi
  else
    PREV="$now"
    echo "$(date -Is) habs_trainonly unchanged"
  fi
  [ "$once" = 1 ] && break
  [ -f "$STOP" ] && { echo "[btr] stop file present; exiting."; break; }
  sleep "$INTERVAL"
done
echo "[btr] watcher exit $(date -Is)"
