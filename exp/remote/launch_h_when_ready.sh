#!/usr/bin/env bash
# Wait for the LOCAL step-A / step-C seed-0 artefacts that H:exp2 consumes,
# push whatever is on disk AT LAUNCH TIME, and start `--only H` on the remote.
#
# H:exp2 reads, for every dataset at seed 0:
#   runs/rigr_head/<ds>/seed0/best.pt          (A:<ds>:s0)
#   runs/rigr_models/<ds>/seed0/scorer.joblib  (C:<ds>:s0)
#   runs/rigr_models/<ds>/seed0/btr_false_deployed.joblib   (optional)
#   runs/seg/<ds>/seed0/pred                   (already on the mirror)
#
# GATING (team decision, 2026-09-03 09:35): the scorer.joblib files that phase-1b
# produced are the OLD hdep fits and WILL be overwritten by a forced B/C re-run.
# Their mere existence therefore means nothing.  The gate is the marker file
#     runs/rigr_models/.phase2_refit_done
# which the local phase-2 chain creates after the refit.  Only then do we push.
#
# Nothing is cached here: the tar is built from the live tree at launch, and
# every pushed file's size + mtime is appended to remote/h_inputs_mtimes.txt and
# echoed into this log, so the exact version H consumed is recoverable.
# PROCESS-SAFETY (incident 2026-09-03 09:56): this script terminates nothing,
# and identifies no process by pattern.  The two `pgrep` probes it used to carry
# (to size the worker pool and to confirm the launch) are gone: the pool is
# sized from the remote load average, and the launch is confirmed from the
# orchestrator's own log.  No pkill, no taskkill, no `pgrep -f`, ever.
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
DATASETS="drive chasedb1 hrf fives"
INTERVAL=${H_INTERVAL:-600}
STOP=remote/.launch_h.stop
MLOG=remote/h_inputs_mtimes.txt
[ "$once" = 1 ] || echo $$ > "remote/.launch_h_when_ready.pid"   # only a looping instance claims the pid file
rm -f "$STOP"

MARKER=runs/rigr_models/.phase2_refit_done

ready() {
  local ds ok=0
  # The phase-2 refit marker is the gate.  Without it the scorers on disk are
  # the stale hdep fits, and H would be run on models that are about to change.
  if [ ! -f "$MARKER" ]; then
    echo "  waiting: $MARKER (phase-2 B/C refit not finished; any scorer.joblib on disk now is the OLD hdep fit)"
    ok=1
  fi
  # once the marker is there the artefacts themselves must also be complete
  for ds in $DATASETS; do
    [ -f "runs/rigr_head/$ds/seed0/best.pt" ]         || { echo "  waiting: runs/rigr_head/$ds/seed0/best.pt"; ok=1; }
    [ -f "runs/rigr_models/$ds/seed0/scorer.joblib" ] || { echo "  waiting: runs/rigr_models/$ds/seed0/scorer.joblib"; ok=1; }
  done
  return $ok
}

while :; do
  echo "===== $(date -Is) launch_h check ====="
  if $SSH "test -f $R/results/tab1_exp2.csv"; then
    echo "[H] remote results/tab1_exp2.csv already exists; nothing to do."
    break
  fi
  if ready; then
    ts=$(date -Is)
    echo "[H] phase-2 refit marker present and all four seed-0 heads + scorers are local;"
    echo "[H] marker mtime: $(stat -c '%y' "$MARKER" 2>/dev/null)"
    echo "[H] building the push list from the live tree."
    # whatever is actually present, right now
    LIST=$(ls -d runs/rigr_head/*/seed0 runs/rigr_models/*/seed0 2>/dev/null | grep -vE '/(lodo_|smoke)' || true)
    if [ -z "$LIST" ]; then echo "[H] nothing to push?!"; sleep 60; continue; fi
    {
      echo "## $ts  step-H inputs pushed to the remote mirror"
      echo "    gate: $MARKER  mtime $(stat -c '%y' "$MARKER" 2>/dev/null)"
      for d in $LIST; do
        find "$d" -maxdepth 1 -type f \( -name '*.pt' -o -name '*.joblib' -o -name '*.json' \) \
          -printf '    %-62p %10s bytes  %TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null
      done
    } | tee -a "$MLOG"
    if tar czf - --exclude='__pycache__' $LIST | $SSH "tar xzf - -C $R"; then
      echo "[H] push OK $ts"
    else
      echo "[H] push FAILED $ts"; sleep 60; continue
    fi
    # Size the pool from the remote 1-minute load average rather than by
    # looking for another job's processes by name: a pattern that matches
    # someone else's job is exactly the failure mode we are avoiding, and load
    # is the thing we actually care about.
    LOAD=$($SSH "awk '{printf \"%d\", \$1}' /proc/loadavg" 2>/dev/null || echo 99)
    case "$LOAD" in ""|*[!0-9]*) LOAD=99 ;; esac
    if [ "$LOAD" -ge 16 ]; then W=8; else W=20; fi
    echo "[H] remote 1-min load average $LOAD -> exp2_workers=$W"
    echo "[H] launching --only H with exp2_workers=$W"
    $SSH "cd $R && printf '{\n  \"exp2_workers\": %d,\n  \"cpu_jobs\": 2\n}\n' $W > s4_remote_cfg_h.json && \
          mkdir -p runs/s4_logs && \
          setsid nohup env PATH=/root/miniconda3/envs/medical1/bin:\$PATH \
              PYTHONPATH=$R PYTHONUNBUFFERED=1 \
              python -m src.pipeline.s4_all --only H --seeds 0 --jobs 2 \
              --config s4_remote_cfg_h.json \
              > runs/s4_logs/orch_H.log 2>&1 < /dev/null & sleep 2; exit 0"
    sleep 25
    # confirm from the orchestrator's own log, not from a process search
    if $SSH "grep -q 'H:exp2. START' $R/runs/s4_logs/orch_H.log"; then
      echo "[H] launched -- orch_H.log shows H:exp2 START"
    else
      echo "[H] launch not confirmed; orch_H.log says:"
      $SSH "tail -20 $R/runs/s4_logs/orch_H.log" 2>/dev/null || true
    fi
    break
  fi
  [ -f "$STOP" ] && { echo "[H] stop file present; exiting."; break; }
  sleep "$INTERVAL"
done
echo "[H] watcher exit $(date -Is)"
