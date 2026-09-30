#!/usr/bin/env bash
# CMIG review point 16, locked plan 2026-09-18: E4 needs FIVE fine-tune seeds
# per arm per fold for the key null.  Seed 0 already exists; this runs the
# extra seeds on the LAN node's 10 GB card.
#
# The recipe is UNCHANGED -- measured 8599 MiB of 10240 MiB at batch 4 /
# patch 768 / longest 1536, so there is no batch-size halving and no gradient
# accumulation, i.e. no protocol deviation.  Only `p5_finetune --seed` differs
# between a unit and its seed-0 twin; the per-fold base checkpoint is the same
# `fold<k>/base/best.pt`.
#
# Units whose summary.json / infer_meta.json already exist are skipped, so the
# script is restartable and the work can be split with the local machine: run
# it with a subset and the local lane with the complement.
#
#   SEEDS="2 3 4" nohup setsid bash remote/lan2_e4_seeds.sh \
#       > runs/pivot/r2/logs/lan_e4_s234.log 2>&1 &
#
# WAIT_FOR=<pattern> makes it idle until no process matching that pattern is
# left (used to chain it behind the seed-1 driver on the one GPU).
set -uo pipefail

R=/mnt/data/Programming/research_ws/medical1
cd "$R/exp"
source "$R/venv/bin/activate"

SEEDS="${SEEDS:-2 3 4}"
GPU="${GPU:-0}"
FOLDS="${FOLDS:-0 1 2 3 4}"
ARMS="${ARMS:-continued reliseg}"
WAIT_FOR="${WAIT_FOR:-}"
LOG="$R/exp/runs/pivot/r2/logs"
mkdir -p "$LOG"

if [ -n "$WAIT_FOR" ]; then
  echo "[lan-e4] waiting for '$WAIT_FOR' to finish ..."
  while pgrep -f "$WAIT_FOR" > /dev/null 2>&1; do sleep 60; done
  echo "[lan-e4] '$WAIT_FOR' gone at $(date -Is); starting"
fi

echo "[lan-e4] start $(date -Is)  seeds='$SEEDS' folds='$FOLDS' arms='$ARMS'"
for s in $SEEDS; do
  for k in $FOLDS; do
    for arm in $ARMS; do
      d="runs/pivot/e4/fold${k}/${arm}_s${s}"
      if [ -f "$d/summary.json" ]; then
        echo "[lan-e4] skip train f$k $arm s$s (done)"
      else
        echo "[lan-e4] TRAIN f$k $arm s$s  $(date -Is)"
        python -m src.pivot.e4_replication finetune --fold "$k" --arm "$arm" \
          --seed "$s" --gpu "$GPU" >> "$LOG/e4_ft_f${k}_${arm}_s${s}.log" 2>&1
        rc=$?
        echo "[lan-e4] TRAIN f$k $arm s$s rc=$rc $(date -Is)"
        [ "$rc" -eq 0 ] || continue
      fi
      if [ -f "$d/pred/infer_meta.json" ]; then
        echo "[lan-e4] skip infer f$k $arm s$s (done)"
      else
        echo "[lan-e4] INFER f$k $arm s$s  $(date -Is)"
        python -m src.pivot.e4_replication infer --fold "$k" --arm "$arm" \
          --seed "$s" --gpu "$GPU" >> "$LOG/e4_infer_f${k}_${arm}_s${s}.log" 2>&1
        echo "[lan-e4] INFER f$k $arm s$s rc=$? $(date -Is)"
      fi
    done
  done
done
echo "[lan-e4] all done $(date -Is)"
