#!/usr/bin/env bash
# Pull back the stage-3 work offloaded to the remote on 2026-09-06 16:12, and
# guard against the one stale-data hazard this round created.
#
#   chain 1  runs/rigr/prob/hrf/seed0            (the local critical path)
#   chain 2  runs/rigr/ablation_{+failcond,+btr_utility,frangi_vs_learned}/*/seed0
#   chain 3  runs/rigr/lodo_<D>/risk             (corrected-utility LODO risk)
#
# It NEVER powers the box off -- user instruction 2026-09-06: shutdown only on
# an explicit instruction.  It only pulls, verifies and reports.
#
# GUARD: `pull_lost_D.sh` extracted the remote PRE-C_geom-fix risk dirs straight
# into the live tree while two local re-runs were writing there, so
# runs/rigr/risk/{hrf,fives}/seed0 currently hold pre-fix per_image.csv /
# summary.json.  The local re-runs overwrite them when they finish (same
# dataset, same seed, same image set -> every file name is rewritten).  This
# script checks the end state: a summary.json without `geom_scale` is pre-fix
# and gets quarantined so the orchestrator re-runs it instead of silently
# treating it as DONE.
set -u
cd /e/Programming/research_ws/medical1/exp || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH"
LOG=logs/pull_offload2.log
mkdir -p logs
exec >>"$LOG" 2>&1
echo "===== pull_offload2 start $(date -Is) pid $$ ====="
if ! mkdir remote/.pull_offload2.lock 2>/dev/null; then
  echo "another instance holds the lock; exiting"; exit 0
fi
trap 'rm -rf remote/.pull_offload2.lock' EXIT

R=/path/to/workdir/medical1/exp
SSHOPT="-p SSH_PORT -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15"
HOST=user@GPU_HOST
STOP=remote/.pull_offload2.stop

rssh() { local n="$1"; shift; local i
  for i in $(seq 1 "$n"); do timeout 120 ssh $SSHOPT "$HOST" "$@" && return 0; sleep 10; done; return 1; }

rpull() { local n="$1" spec="$2" i
  for i in $(seq 1 "$n"); do
    if timeout 1200 ssh $SSHOPT "$HOST" "cd $R && tar czf - --exclude=__pycache__ $spec" > .off2.tgz 2>/dev/null \
       && [ -s .off2.tgz ] && tar tzf .off2.tgz >/dev/null 2>&1; then
      tar xzf .off2.tgz && rm -f .off2.tgz && return 0
    fi
    rm -f .off2.tgz; sleep 15
  done; return 1; }

# --------------------------------------------------- wait for the three chains
for cycle in $(seq 1 200); do          # 200 x 10 min = ~33 h ceiling
  [ -f "$STOP" ] && { echo "stop file; exiting"; exit 0; }
  n=$(rssh 4 "cd $R && ls -d runs/.offload2_lock 2>/dev/null | wc -l; ps -eo cmd= | grep -c '[s]4_all --only'" | tail -1)
  case "${n:-}" in ''|*[!0-9]*) n=1 ;; esac
  if [ "$n" -eq 0 ]; then echo "[wait] $(date -Is) all remote chains finished"; break; fi
  echo "[wait] $(date -Is) $n remote orchestrator(s) still running"
  sleep 600
done

# --------------------------------------------------------------- pull each set
echo "[pull] $(date -Is)"
rpull 30 "runs/rigr/prob/hrf/seed0"           && echo "  chain1 OK" || echo "  chain1 FAILED"
rpull 30 "runs/rigr/ablation_*"               && echo "  chain2 OK" || echo "  chain2 FAILED"
rpull 30 "runs/rigr/lodo_drive/risk runs/rigr/lodo_chasedb1/risk runs/rigr/lodo_hrf/risk runs/rigr/lodo_fives/risk" \
                                              && echo "  chain3 OK" || echo "  chain3 FAILED"
rpull 10 "runs/s4_logs" && echo "  remote logs OK" || true

# --------------------------------------------- fail-closed enumeration (report)
echo "[sweep] $(date -Is) remote runs/rigr summary.json without a local counterpart"
if rssh 8 "cd $R && find runs/rigr -name summary.json | sort" > .remote_rigr_summaries.txt 2>/dev/null; then
  miss=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    case "$f" in *_partial*|*_sigma_pre*|*_bdfix*|*_ooffix*|*_failed*|*_btr_pre*|*_lamdedup*|*_sweepfix*|*_etascale_pre*) continue;; esac
    if [ ! -f "$f" ]; then echo "  MISSING LOCALLY $f"; miss=$((miss+1)); fi
  done < .remote_rigr_summaries.txt
  echo "[sweep] missing locally: $miss"
else
  echo "[sweep] enumeration FAILED -- no conclusion drawn"
fi

# ------------------------------------------------- guard: no pre-fix risk data
echo "[guard] $(date -Is) every live risk summary.json must carry geom_scale"
bad=0
for d in $(ls -d runs/rigr/risk/*/seed* runs/rigr/lodo_*/risk 2>/dev/null | grep -v _etascale_pre); do
  s="$d/summary.json"
  [ -f "$s" ] || continue
  if grep -q '"geom_scale": *[0-9]' "$s"; then
    echo "  OK        $d"
  else
    echo "  PRE-FIX   $d  -> quarantining so it is re-run"
    mv "$d" "${d}_etascale_pre0906_late$(date +%H%M)" && bad=$((bad+1))
  fi
done
echo "[guard] quarantined $bad pre-fix risk dir(s)"
echo "[done] the box is left RUNNING -- shutdown only on an explicit instruction"
echo "===== pull_offload2 end $(date -Is) ====="
