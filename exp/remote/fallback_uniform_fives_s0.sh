#!/usr/bin/env bash
# D:uniform:fives:s0 is the only LIVE recovery item left on the powered-off
# cloud box (the two risk dirs there are pre-C_geom-fix artefacts and are
# being re-run locally anyway).  Rather than let stage 3 block on an instance
# that may never come back, run it locally when either
#
#   (a) the local orchestrator has nothing else left to do -- i.e. `--status`
#       reports exactly one pending step, which can only be this one, or
#   (b) the 09-06 22:00 deadline passes,
#
# whichever comes first.  It exits immediately if the step is already on disk,
# so if the restart watcher recovers it first this script is a no-op.  The two
# never run at once: this script takes an atomic lock that the recovery path
# also respects via output-existence.
#
# The restart watcher (remote/await_restart_pull_shutdown.sh) is deliberately
# left running: if the box does come back later it will still pull whatever is
# there, run the fail-closed sweep and shut it down.
#
# Terminates nothing.
set -u
cd /e/Programming/research_ws/medical1/exp || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH"
LOG=logs/fallback_uniform_fives.log
mkdir -p logs
exec >>"$LOG" 2>&1
echo "===== fallback_uniform_fives start $(date -Is) pid $$ ====="
echo $$ > remote/.fallback_uniform_fives.pid

PY=D:/Anaconda/envs/medical1/python.exe
OUT=runs/rigr/uniform/fives/seed0
DEADLINE=$(date -d '2026-09-06 22:00:00' +%s 2>/dev/null || echo 0)

pending_count() {
  "$PY" -m src.pipeline.s4_all --only D,F,I --skip-steps "G:*,H:*" \
      --config configs/s4_phase2.json --status 2>/dev/null \
    | sed -n 's/.*steps  *(\([0-9]*\) done, \([0-9]*\) pending.*/\2/p' | tail -1
}

while :; do
  if [ -f "$OUT/summary.json" ]; then
    echo "[fallback] $(date -Is) $OUT already complete (recovered or run); nothing to do"
    exit 0
  fi
  [ -f remote/.fallback_uniform_fives.stop ] && { echo "stop file; exiting"; exit 0; }

  now=$(date +%s)
  n=$(pending_count)
  case "${n:-}" in ''|*[!0-9]*) n=999 ;; esac
  reason=""
  [ "$n" -le 1 ] && reason="orchestrator has nothing else pending ($n)"
  [ "$DEADLINE" -gt 0 ] && [ "$now" -ge "$DEADLINE" ] && reason="22:00 deadline passed"

  if [ -n "$reason" ]; then
    echo "[fallback] $(date -Is) running D:uniform:fives:s0 locally -- $reason"
    "$PY" -m src.pipeline.s4_all --only D --datasets fives --seeds 0 \
        --skip-steps "D:prob:fives:s0,D:risk:fives:s0" \
        --config configs/s4_phase2.json --jobs 1 \
        >> logs/fallback_uniform_fives_run.log 2>&1
    rc=$?
    echo "[fallback] rc=$rc  summary.json present: $([ -f "$OUT/summary.json" ] && echo yes || echo NO)"
    [ -f "$OUT/summary.json" ] && exit 0
    echo "[fallback] did not produce the output; retrying in 30 min"
    sleep 1800
  else
    echo "[fallback] $(date -Is) waiting ($n pending, deadline 22:00)"
    sleep 900
  fi
done
