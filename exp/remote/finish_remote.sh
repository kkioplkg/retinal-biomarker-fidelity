#!/usr/bin/env bash
# Detached finisher for the remote cloud box.  Every remote call is retried,
# because ssh to this host currently fails ~60% of attempts (DNS + banner
# timeouts).  Terminates nothing on either box except the final, fully gated
# shutdown.  Idempotent: each phase is skipped once its artefact is local.
#
#   A  push the refit habs_trainonly BTR heads + the plan_H fix, relaunch H
#   B  pull the remaining STARE artefacts
#   C  wait for results/tab1_exp2.csv, pull it
#   D  wait for runs/repair/ckpt/evapore_hrf.pt, pull it + the remote logs
#   E  verify EVERY shutdown-checklist item locally, by byte count / row count
#   F  tar remote logs/ + exp/results + exp/runs/s4_logs -> results_remote_final/
#   G  shutdown -h now   (ONLY if E and F both passed)
set -u
cd /e/Programming/research_ws/medical1/exp || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH"
LOG=logs/finish_remote.log
mkdir -p logs results_remote_final
exec >>"$LOG" 2>&1
echo "===== finish_remote start $(date -Is) pid $$ ====="
echo $$ > remote/.finish_remote.pid
STOP=remote/.finish_remote.stop
rm -f "$STOP"

R=/path/to/workdir/medical1/exp
RROOT=/path/to/workdir/medical1
SSHOPT="-p SSH_PORT -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
HOST=user@GPU_HOST
TR_LOGS="s|^runs/s4_logs|runs/s4_logs_remote|"
TR_ETRAIN="s|^logs|runs/s4_logs/remote_Etrain|"

rssh() {   # rssh <tries> <remote command string>
  local n="$1"; shift; local i
  for i in $(seq 1 "$n"); do
    if timeout 90 ssh $SSHOPT "$HOST" "$@"; then return 0; fi
    sleep 10
  done
  return 1
}

rpush() {  # rpush <tries> <localfile> <remotepath>
  local n="$1" lf="$2" rp="$3" i
  for i in $(seq 1 "$n"); do
    if timeout 900 ssh $SSHOPT "$HOST" "cat > $rp" < "$lf"; then return 0; fi
    sleep 10
  done
  return 1
}

rpull_tar() {  # rpull_tar <tries> <remote tar spec> [extra local tar args...]
  local n="$1" spec="$2"; shift 2; local i
  for i in $(seq 1 "$n"); do
    if timeout 1200 ssh $SSHOPT "$HOST" "cd $R && tar czf - --exclude=__pycache__ $spec" > .rpull.tgz 2>/dev/null \
       && [ -s .rpull.tgz ] && tar tzf .rpull.tgz >/dev/null 2>&1; then
      if tar xzf .rpull.tgz "$@"; then rm -f .rpull.tgz; return 0; fi
    fi
    rm -f .rpull.tgz
    sleep 10
  done
  return 1
}

# ---------------------------------------------------------------- A
if [ ! -f remote/.h_relaunched ]; then
  echo "[A] $(date -Is) push habs_trainonly heads + plan_H fix, relaunch H"
  if [ -f .h_fix.tgz ] && rpush 30 .h_fix.tgz "$R/h_fix.tgz"; then
    if rssh 30 "cd $R && tar xzf h_fix.tgz && rm -f h_fix.tgz && ls -la runs/btr/habs_trainonly/ | head -20 && grep -c btr_runs_dir src/pipeline/s4_all.py"; then
      echo "[A] extracted OK; relaunching step H"
      rssh 60 "cd $R && { mkdir runs/.h_launch_lock 2>/dev/null || { echo '[A] remote H launch lock already held -- not launching a second H'; exit 0; }; } && rm -f results/tab1_exp2.csv && printf '{\"exp2_workers\": 20, \"cpu_jobs\": 2}\n' > s4_remote_cfg_h.json && mkdir -p runs/s4_logs && setsid nohup env PATH=/root/miniconda3/envs/medical1/bin:\$PATH PYTHONPATH=$R PYTHONUNBUFFERED=1 python -m src.pipeline.s4_all --only H --seeds 0 --jobs 20 --config s4_remote_cfg_h.json > runs/s4_logs/orch_H.log 2>&1 < /dev/null & sleep 3; exit 0"
      sleep 90
      rssh 10 "tail -8 $R/runs/s4_logs/orch_H.log; echo ---; tail -4 $R/runs/s4_logs/H_exp2.log 2>/dev/null" || true
      touch remote/.h_relaunched
      echo "[A] done $(date -Is)"
    else
      echo "[A] EXTRACT FAILED"
    fi
  else
    echo "[A] PUSH FAILED"
  fi
fi

# ---------------------------------------------------------------- B
if [ ! -f remote/.stare_pulled ]; then
  echo "[B] $(date -Is) pull remaining STARE artefacts"
  ok=1
  rpull_tar 30 "runs/rigr/prob/stare runs/rigr/risk/stare runs/rigr/uniform/stare" || ok=0
  rpull_tar 30 "results/seg_per_image_stare.csv" || ok=0
  # only the predictions the checklist needs -- the whole runs/seg/stare is
  # 627 MB, almost all of it segmenter checkpoints, and the link is flaky.
  rpull_tar 30 "runs/seg/stare/seed0/pred runs/seg/stare/seed1/pred runs/seg/stare/seed2/pred" || ok=0
  # the STARE segmenter + cross-fit checkpoints, best-effort (not a gate)
  rpull_tar 8 "runs/seg/stare/seed0/best.pt runs/seg/stare/seed1/best.pt runs/seg/stare/seed2/best.pt runs/seg/stare/crossfit2_fold0_seed0/best.pt runs/seg/stare/crossfit2_fold1_seed0/best.pt" || echo "[B] STARE seg checkpoints not pulled (best-effort)"
  rpull_tar 30 "runs/seg_oof/stare runs/rigr_head/stare runs/rigr_models/stare" || ok=0
  if [ "$ok" = 1 ]; then touch remote/.stare_pulled; echo "[B] OK"; else echo "[B] incomplete"; fi
fi

# ---------------------------------------------------------------- C
while [ ! -s results/tab1_exp2.csv ]; do
  if [ -f "$STOP" ]; then echo "[C] stop file present; exiting"; exit 0; fi
  if rssh 6 "test -s $R/results/tab1_exp2.csv"; then
    echo "[C] $(date -Is) tab1_exp2.csv present remotely; pulling"
    rpull_tar 30 "results/tab1_exp2.csv" || true
  else
    rssh 4 "tail -3 $R/runs/s4_logs/H_exp2.log 2>/dev/null; tail -3 $R/runs/s4_logs/orch_H.log 2>/dev/null" || true
    sleep 300
  fi
done
echo "[C] tab1_exp2.csv local: $(stat -c%s results/tab1_exp2.csv) bytes, $(wc -l < results/tab1_exp2.csv) lines"

# ---------------------------------------------------------------- D
# Wait for evapore_hrf.pt WITH A FAILURE DETECTOR.  Polling only for the output
# file is what let the 17:23 OOM go unnoticed for 2.4 h: the job was dead and
# the watcher happily kept waiting.  ``etrain_status`` reads the log, never a
# process table, so it never identifies anybody's process by pattern:
#   DONE          the checkpoint exists remotely
#   FAILED:<rc>   an "rc=<non-zero>" line appeared after the last start marker
#   STALL:<min>   no checkpoint, no rc line, and the log has not advanced in
#                 STALL_MIN minutes (pool lines land every ~15 min, epoch lines
#                 every ~8 min, so 90 min is far outside normal)
#   RUNNING       otherwise
STALL_MIN=${STALL_MIN:-90}
ALERT=remote/.finish_remote.ALERT

etrain_status() {   # $1 = log tag, $2 = remote ckpt path (relative to exp/)
  rssh 6 "bash $R/etrain_status.sh '$1' '$2' $STALL_MIN"
}

raise_alert() {   # $1 = short reason
  echo "[ALERT] $(date -Is) $1"
  { echo "$(date -Is)"; echo "$1";
    echo "the finisher has STOPPED polling; nothing was shut down."; } > "$ALERT"
}

while [ ! -s runs/repair/ckpt/evapore_hrf.pt ]; do
  if [ -f "$STOP" ]; then echo "[D] stop file present; exiting"; exit 0; fi
  st=$(etrain_status evapore_hrf runs/repair/ckpt/evapore_hrf.pt | tail -1)
  case "$st" in
    DONE)
      echo "[D] $(date -Is) evapore_hrf.pt present remotely; pulling"
      rpull_tar 30 "runs/repair/ckpt/evapore_hrf.pt" || true ;;
    FAILED:*|STALL:*)
      rssh 4 "tail -25 $RROOT/logs/Etrain_evapore_hrf.log" || true
      raise_alert "evapore_hrf $st -- no checkpoint was produced"
      exit 1 ;;
    *)
      rssh 4 "tail -2 $RROOT/logs/Etrain_evapore_hrf.log" || true
      sleep 600 ;;
  esac
done
echo "[D] evapore_hrf.pt local: $(stat -c%s runs/repair/ckpt/evapore_hrf.pt) bytes"

# --------------------------------------------------------------- D2
# Decision 2026-09-05 20:30: once evapore_hrf is done the box would
# otherwise idle until shutdown, so run AMP retrains of the CHEAP datasets as a
# supplementary consistency check -> runs/repair/ckpt/evapore_<ds>_amp.pt.
# FIVES is deliberately EXCLUDED: its fp32 training alone took 101155 s (28.1 h,
# 55 epochs), which blows the agreed ~10 h budget on its own.  drive
# (5625 s) + chasedb1 (3438 s) are ~2.5-3.5 h together.
# Tab.2's EVAPORE rows still come from the fp32 checkpoints for these three.
AMP_DS="${AMP_DS:-drive chasedb1}"
if [ ! -f remote/.amp_retrain_launched ]; then
  echo "[D2] $(date -Is) launching AMP retrains: $AMP_DS"
  rssh 20 "cd $R && mkdir runs/.amp_retrain_lock 2>/dev/null || { echo 'lock held'; exit 0; };     setsid nohup env PATH=/root/miniconda3/envs/medical1/bin:\$PATH       PYTHONPATH=$R PYTHONUNBUFFERED=1 AMP_DS='$AMP_DS'       bash run_evapore_amp_retrain.sh > $RROOT/logs/amp_retrain_wrap.log 2>&1 < /dev/null & sleep 3; exit 0"     && touch remote/.amp_retrain_launched
fi
for ds in $AMP_DS; do
  while [ ! -s "runs/repair/ckpt/evapore_${ds}_amp.pt" ]; do
    if [ -f "$STOP" ]; then echo "[D2] stop file present; exiting"; exit 0; fi
    st=$(etrain_status "evapore_${ds}_amp" "runs/repair/ckpt/evapore_${ds}_amp.pt" | tail -1)
    case "$st" in
      DONE) rpull_tar 30 "runs/repair/ckpt/evapore_${ds}_amp.pt" || true ;;
      FAILED:*|STALL:*)
        rssh 4 "tail -25 $RROOT/logs/Etrain_evapore_${ds}_amp.log" || true
        raise_alert "evapore_${ds}_amp $st -- no checkpoint was produced"
        exit 1 ;;
      *) sleep 600 ;;
    esac
  done
  echo "[D2] evapore_${ds}_amp.pt local: $(stat -c%s runs/repair/ckpt/evapore_${ds}_amp.pt) bytes"
done

echo "[D] pulling remote step logs + E:train logs"
rpull_tar 10 "runs/s4_logs" --transform "$TR_LOGS" || true
for i in $(seq 1 10); do
  if timeout 600 ssh $SSHOPT "$HOST" "cd $RROOT && tar czf - logs" > .etl.tgz 2>/dev/null \
     && [ -s .etl.tgz ] && tar tzf .etl.tgz >/dev/null 2>&1; then
    tar xzf .etl.tgz --transform "$TR_ETRAIN"; rm -f .etl.tgz; break
  fi
  rm -f .etl.tgz; sleep 20
done

# ---------------------------------------------------------------- E
echo "[E] $(date -Is) shutdown-checklist verification"
FAIL=0
chk() {
  local sz
  if [ -e "$1" ]; then
    sz=$(du -sb "$1" 2>/dev/null | cut -f1)
    if [ "${sz:-0}" -ge "$2" ]; then printf "  OK   %-56s %s bytes\n" "$1" "$sz"; return 0; fi
    printf "  FAIL %-56s %s bytes (want >= %s)\n" "$1" "${sz:-0}" "$2"; FAIL=1; return 1
  fi
  printf "  FAIL %-56s MISSING\n" "$1"; FAIL=1; return 1
}
rows() {
  local n
  n=$(wc -l < "$1" 2>/dev/null || echo 0)
  if [ "$n" = "$2" ]; then printf "  OK   %-56s %s lines\n" "$1" "$n"
  else printf "  FAIL %-56s %s lines (want %s)\n" "$1" "$n" "$2"; FAIL=1; fi
}

for m in rnca evapore; do
  for d in drive chasedb1 hrf fives; do chk "runs/repair/ckpt/${m}_${d}.pt" 60000; done
done
rows runs/rigr/lodo_drive/risk/per_image.csv 21
rows runs/rigr/lodo_drive/uniform/per_image.csv 21
rows runs/rigr/lodo_chasedb1/risk/per_image.csv 9
rows runs/rigr/lodo_chasedb1/uniform/per_image.csv 9
rows runs/rigr/lodo_hrf/risk/per_image.csv 31
rows runs/rigr/lodo_hrf/uniform/per_image.csv 31
rows runs/rigr/lodo_fives/risk/per_image.csv 61
rows runs/rigr/lodo_fives/uniform/per_image.csv 61
for d in drive chasedb1 hrf fives; do
  chk "runs/rigr_head/lodo_$d/seed0" 1000000
  chk "runs/rigr_models/lodo_$d/seed0" 1000
  chk "runs/rigr_data/lodo_$d" 1000
done
chk results/tab1_exp2.csv 1000
# supplementary AMP consistency checkpoints (decision 2026-09-05 20:30)
for d in $AMP_DS; do chk "runs/repair/ckpt/evapore_${d}_amp.pt" 60000; done
for m in prob risk uniform; do rows "runs/rigr/$m/stare/seed0/per_image.csv" 11; done
chk runs/rigr_head/stare 1000000
chk runs/rigr_models/stare 1000
chk runs/seg_oof/stare 100000
chk results/seg_per_image_stare.csv 1000
for s in 0 1 2; do chk "runs/seg/stare/seed$s/pred" 100000; done
for d in drive chasedb1 hrf fives stare; do chk "results/fig4a_stability_${d}_none_per_image.csv" 1000; done
chk results/fig4b_resolution_hrf_per_image.csv 1000
chk runs/s4_logs_remote 10000
echo "[E] FAIL=$FAIL"

# ---------------------------------------------------------------- F
ARCH_OK=0
if [ "$FAIL" = 0 ]; then
  echo "[F] $(date -Is) building + pulling the final remote archive"
  if rssh 30 "cd $RROOT && tar czf /path/to/workdir/final_remote_archive.tgz logs exp/results exp/runs/s4_logs && ls -la /path/to/workdir/final_remote_archive.tgz"; then
    for i in $(seq 1 30); do
      if timeout 1800 ssh $SSHOPT "$HOST" "cat /path/to/workdir/final_remote_archive.tgz" > results_remote_final/final_remote_archive.tgz 2>/dev/null \
         && [ -s results_remote_final/final_remote_archive.tgz ] \
         && tar tzf results_remote_final/final_remote_archive.tgz >/dev/null 2>&1; then
        echo "[F] archive OK $(stat -c%s results_remote_final/final_remote_archive.tgz) bytes, $(tar tzf results_remote_final/final_remote_archive.tgz | wc -l) entries"
        ARCH_OK=1
        break
      fi
      sleep 30
    done
  fi
fi

# ---------------------------------------------------------------- G
if [ "${ALLOW_SHUTDOWN:-0}" != "1" ]; then
  echo "[G] $(date -Is) AUTOMATIC SHUTDOWN IS DISABLED (user instruction 2026-09-06 15:35)."
elif [ "$FAIL" = 0 ] && [ "$ARCH_OK" = 1 ]; then
  echo "[G] $(date -Is) every checklist item verified local; shutting the box down"
  rssh 6 "nohup sh -c 'sleep 3; shutdown -h now' >/dev/null 2>&1 & exit 0" || true
  date -Is > remote/.remote_shutdown_issued
  echo "[G] shutdown issued $(date -Is)"
else
  echo "[G] NOT shutting down: FAIL=$FAIL ARCH_OK=$ARCH_OK -- inspect $LOG"
fi
echo "===== finish_remote end $(date -Is) ====="
