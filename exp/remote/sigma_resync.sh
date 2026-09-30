#!/usr/bin/env bash
# sigma_resync -- watch results/gateA_biomarker_scales_train.csv for CONTENT
# change and rebuild every step-G chain against the new table.
#
# Decision 2026-09-03 10:40: the 10:13 table is being replaced wholesale by
# an all-native one (DRIVE/CHASE unchanged, HRF re-measured at native rather
# than C1's 0.4384 work scale, FIVES added from the first 120 training masks).
# That invalidates EVERY G:data / G:fit built against 10:13 -- not only the
# FIVES and HRF chains, because a chain's R_false rows involve HRF source
# images whatever its held-out dataset is.  Heads are sigma-independent and are
# never retrained.  G:run:*:{risk,uniform} stay held back until the new table is
# on the remote AND the orchestrator preflight passes there.
#
# The trigger is a CONTENT HASH, not an mtime: the replacement is a rewrite of
# the same path, and biomarker_eval.py will not change again.
#
# PROCESS SAFETY (incident 09:56): this script terminates nothing.  Every stop
# happens inside the remote sigma_rebuild_all.sh, by explicit PID, through
# stop_pid.sh, which re-verifies /proc/<pid>/cmdline before signalling.
#
#   bash exp/remote/sigma_resync.sh --once
#   bash exp/remote/sigma_resync.sh              # loop every 3 min
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
CSV=results/gateA_biomarker_scales_train.csv
BASE=remote/.sigma.csvhash
VLOG=remote/sigma_versions.log
STOP=remote/.sigma_resync.stop
PIDF=remote/.sigma_resync.pid
INTERVAL=${SIGMA_INTERVAL:-180}
WANT_DS="DRIVE CHASE_DB1 HRF FIVES"
once=0; [ "${1:-}" = "--once" ] && once=1
[ "$once" = 1 ] || echo $$ > "$PIDF"   # only a looping instance claims the pid file
rm -f "$STOP"

hm() { stat -c '%y' "$1" 2>/dev/null || echo "(absent)"; }

if [ ! -f "$BASE" ]; then
  sha256sum "$CSV" | awk '{print $1}' > "$BASE" 2>/dev/null || echo none > "$BASE"
  echo "[sigma] baseline hash $(cat "$BASE") for $CSV ($(hm $CSV))"
fi

while :; do
  echo "===== $(date -Is) sigma check ====="
  if [ ! -f "$CSV" ]; then
    echo "[sigma] waiting: $CSV absent"
  else
    NOW=$(sha256sum "$CSV" | awk '{print $1}')
    if [ "$NOW" = "$(cat "$BASE")" ]; then
      echo "[sigma] unchanged ($NOW, $(hm $CSV))"
    else
      ts=$(date -Is); suffix="sigma_pre$(date +%H%M)"
      echo "[sigma] TABLE REPLACED $ts"
      echo "[sigma]   $CSV  $(hm $CSV)"
      echo "[sigma]   hash $(cat "$BASE") -> $NOW"

      # completeness check -- an incomplete table must not release G:run
      missing=""
      for d in $WANT_DS; do
        cut -d, -f1 "$CSV" | grep -qx "$d" || missing="$missing $d"
      done
      if [ -n "$missing" ]; then
        echo "[sigma] NOTE the new table still has no rows for:$missing"
        echo "[sigma] the remote preflight will decide; G:run stays held back if it is fatal"
      else
        echo "[sigma] all four datasets present in the new table"
      fi
      awk -F, 'NR==1 || $1=="HRF" && $2=="FD_skan" {print "    " $0}' "$CSV" | head -3

      # push the new table + the whole src tree (carries src/rigr/candidates.py,
      # the port-clustering rule flagged in review)
      if tar czf - --exclude='__pycache__' --exclude='*.pyc' "$CSV" src \
           | $SSH "cd $R && rm -rf src/__pycache__ src/*/__pycache__ && tar xzf - -C $R && echo PUSH_OK" \
           | grep -q PUSH_OK; then
        echo "[sigma] pushed $CSV + src/ (incl. src/rigr/candidates.py)"
      else
        echo "[sigma] PUSH FAILED -- baseline not advanced, retrying"
        [ "$once" = 1 ] && break
        sleep 60; continue
      fi

      {
        echo "## $ts  sigma_B table replaced (all-native)"
        echo "    $CSV  $(hm $CSV)  sha256 $NOW"
        echo "    src/rigr/candidates.py  $(hm src/rigr/candidates.py)"
        echo "    datasets in table: $(cut -d, -f1 "$CSV" | sort -u | grep -v '^dataset$' | tr '\n' ' ')"
      } >> "$VLOG"

      # WHICH chains are actually invalidated?  Adding STARE rows does not move
      # a single DRIVE/CHASE_DB1/HRF/FIVES sigma, so a blanket rebuild would
      # destroy four running G:data runs for nothing.  (That is exactly what the
      # unpatched version did at 18:24 when the STARE rows landed.)  Compare the
      # per-dataset rows against the table we last pushed and rebuild only the
      # affected chains: lodo_<D> reads the other three datasets, so it is
      # affected by a change in any dataset EXCEPT D.
      PUSHED=remote/.sigma.table.pushed
      changed=""
      if [ -f "$PUSHED" ]; then
        for d in DRIVE CHASE_DB1 HRF FIVES; do
          a=$(grep "^$d," "$PUSHED" | sort | md5sum | cut -d" " -f1)
          b=$(grep "^$d," "$CSV"    | sort | md5sum | cut -d" " -f1)
          [ "$a" = "$b" ] || changed="$changed $d"
        done
      else
        changed=" DRIVE CHASE_DB1 HRF FIVES"
      fi
      added=$(comm -13 <(cut -d, -f1 "$PUSHED" 2>/dev/null | sort -u) <(cut -d, -f1 "$CSV" | sort -u) | tr "\n" " ")
      echo "[sigma] datasets whose sigma CHANGED:${changed:- none}"
      echo "[sigma] datasets newly ADDED:        ${added:- none}"
      if [ -z "$changed" ]; then
        echo "[sigma] no existing dataset moved -- rebuilding NOTHING (a pure addition unblocks new work, it invalidates none)."
      else
        aff=""
        for d in $changed; do
          case "$d" in DRIVE) k=drive;; CHASE_DB1) k=chasedb1;; HRF) k=hrf;; FIVES) k=fives;; *) k="";; esac
          for t in drive chasedb1 hrf fives; do
            [ "$t" = "$k" ] && continue
            case " $aff " in *" $t "*) ;; *) aff="$aff $t";; esac
          done
        done
        echo "[sigma] chains to rebuild:$aff"
        for t in $aff; do
          $SSH "cd $R && bash sigma_rebuild_one.sh '$t' '$suffix'" 2>&1 | sed "s/^/    /"
        done
      fi
      cp "$CSV" "$PUSHED"
      printf '%s\n' "$NOW" > "$BASE"
    fi
  fi
  [ "$once" = 1 ] && break
  [ -f "$STOP" ] && { echo "[sigma] stop file present; exiting."; break; }
  sleep "$INTERVAL"
done
rm -f "$PIDF"
echo "[sigma] watcher exit $(date -Is)"
