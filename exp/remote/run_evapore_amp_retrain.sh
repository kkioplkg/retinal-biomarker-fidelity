#!/usr/bin/env bash
# Supplementary AMP consistency check (decision 2026-09-05 20:30).
#
# Re-trains EVAPORE with --amp on the CHEAP datasets, one at a time, writing
# runs/repair/ckpt/evapore_<ds>_amp.pt.  The fp32 checkpoints are untouched and
# remain the source of Tab.2's EVAPORE rows for these datasets; these are for a
# supplementary "AMP changes the numbers negligibly" table.
#
# FIVES is deliberately NOT in the default list: its fp32 training alone ran
# 101155 s (28.1 h, 55 epochs before early stop), which exceeds the whole ~10 h
# budget by itself.  drive (5625 s) + chasedb1 (3438 s) are ~2.5-3.5 h together.
#
# Runs on GPU 0 only after evapore_hrf has released it, one job at a time, so it
# can never co-tenant an EVAPORE with anything else -- co-tenancy is what caused
# four of the five HRF OOMs.  Terminates nothing.
set -u
cd /path/to/workdir/medical1/exp
export PATH=/root/miniconda3/envs/medical1/bin:$PATH
export PYTHONPATH=/path/to/workdir/medical1/exp
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
ulimit -n 65536 2>/dev/null || true

LOGD=/path/to/workdir/medical1/logs
mkdir -p "$LOGD" runs/repair/ckpt
DS="${AMP_DS:-drive chasedb1}"

for ds in $DS; do
  out="runs/repair/ckpt/evapore_${ds}_amp.pt"
  log="$LOGD/Etrain_evapore_${ds}_amp.log"
  [ -s "$out" ] && { echo "[amp] $ds already done"; continue; }
  echo "=== AMP retrain evapore $ds start $(date) pid $$ ===" >> "$log"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$log"
  python -m src.baselines.evapore.evapore_adapter --dataset "$ds" --gpu 0 \
      --epochs 150 --seed 0 --out "$out" \
      --pred_dir "runs/seg_oof/${ds}/pred" \
      --patience 25 --monitor val_auc --amp >> "$log" 2>&1
  rc=$?
  echo "=== AMP retrain evapore $ds rc=$rc $(date) ===" >> "$log"
  [ "$rc" -eq 0 ] && [ -s "$out" ] && touch "$LOGD/Etrain_evapore_${ds}_amp.done"
done
rm -rf runs/.amp_retrain_lock
echo "[amp] all requested AMP retrains finished $(date)"
