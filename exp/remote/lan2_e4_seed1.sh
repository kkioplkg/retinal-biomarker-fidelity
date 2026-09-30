#!/usr/bin/env bash
# CMIG review point 16 -- E4 extra fine-tune seed, run on the LAN node.
#
# One additional fine-tune seed (default 1) for the two arms of the key null
# (`reliseg`, `continued`) in all five folds, started from the SAME per-fold
# base checkpoint `runs/pivot/e4/fold<k>/base/best.pt` as the original seed-0
# fine-tunes, with every other hyper-parameter identical -- only
# `p5_finetune --seed` changes (see e4_replication.E4_EXTRA_SEEDS).
#
# Each unit is train -> infer on that fold's 20 held-out images.  Units whose
# `summary.json` / `infer_meta.json` already exists are skipped, so the script
# is restartable and can be split with the local machine.
#
#   nohup setsid bash remote/lan2_e4_seed1.sh > runs/pivot/r2/logs/lan_e4_s1.log 2>&1 &
#   FOLDS="0 1 2" bash remote/lan2_e4_seed1.sh     # subset
set -uo pipefail

R=/mnt/data/Programming/research_ws/medical1
cd "$R/exp"
source "$R/venv/bin/activate"

SEED="${SEED:-1}"
GPU="${GPU:-0}"
FOLDS="${FOLDS:-0 1 2 3 4}"
ARMS="${ARMS:-continued reliseg}"
LOG="$R/exp/runs/pivot/r2/logs"
mkdir -p "$LOG"

echo "[lan-e4] start $(date -Is)  seed=$SEED gpu=$GPU folds='$FOLDS' arms='$ARMS'"
for k in $FOLDS; do
  for arm in $ARMS; do
    d="runs/pivot/e4/fold${k}/${arm}_s${SEED}"
    if [ -f "$d/summary.json" ]; then
      echo "[lan-e4] skip train f$k $arm (done)"
    else
      echo "[lan-e4] TRAIN f$k $arm seed$SEED  $(date -Is)"
      python -m src.pivot.e4_replication finetune --fold "$k" --arm "$arm" \
        --seed "$SEED" --gpu "$GPU" >> "$LOG/e4_ft_f${k}_${arm}_s${SEED}.log" 2>&1
      rc=$?
      echo "[lan-e4] TRAIN f$k $arm rc=$rc $(date -Is)"
      [ "$rc" -eq 0 ] || continue
    fi
    if [ -f "$d/pred/infer_meta.json" ]; then
      echo "[lan-e4] skip infer f$k $arm (done)"
    else
      echo "[lan-e4] INFER f$k $arm seed$SEED  $(date -Is)"
      python -m src.pivot.e4_replication infer --fold "$k" --arm "$arm" \
        --seed "$SEED" --gpu "$GPU" >> "$LOG/e4_infer_f${k}_${arm}_s${SEED}.log" 2>&1
      echo "[lan-e4] INFER f$k $arm rc=$? $(date -Is)"
    fi
  done
done
echo "[lan-e4] all done $(date -Is)"
