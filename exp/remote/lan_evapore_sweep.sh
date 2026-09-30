#!/usr/bin/env bash
# EVAPORE end-to-end acceptance-threshold sweep (DECISIONS.md 2026-09-15 21:30
# item 2), run ON the LAN node (cwd = exp, venv active).
#
# `run_baseline` already has the override: `repair_e2e(..., accept_p=tau)` is
# their per-path acceptance threshold (their `max_f1_threshold` otherwise), and
# `--opt accept_p=<tau>` reaches it through the generic method-kwargs parser.
# No code change was needed.
#
# Usage:
#   bash remote/lan_evapore_sweep.sh <device> <dataset> [<dataset> ...]
#     device: cuda:0 | cpu
#
# One dataset at a time, taus sequentially: EVAPORE's U-Net runs on the FULL
# resolution image, so concurrent taus would multiply peak memory.

set -uo pipefail

DEV="${1:?usage: lan_evapore_sweep.sh <device> <dataset>...}"
shift
# TAUS env override lets several workers split the sweep (each skips a tau whose
# per_image.csv is already on disk, so give disjoint lists to avoid a race).
read -r -a TAUS <<< "${TAUS:-0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
LOGDIR="runs_remote/logs"
mkdir -p "$LOGDIR"

limit_args() { if [ "$1" == "fives" ]; then echo "--limit 60"; fi; }

for ds in "$@"; do
  ck="runs/repair/ckpt/evapore_${ds}.pt"
  if [ ! -f "$ck" ]; then echo "[sweep] MISSING $ck -- skipping $ds" >&2; continue; fi
  for tau in "${TAUS[@]}"; do
    out="runs/repair/evapore_e2e_sweep/${ds}/seed0/tau${tau}"
    if [ -f "$out/per_image.csv" ]; then
      echo "[sweep] $ds tau=$tau already on disk -- skip" >&2
      continue
    fi
    mkdir -p "$out"
    echo "=== [$(date '+%F %T')] evapore_e2e $ds tau=$tau dev=$DEV -> $out" >&2
    t0=$SECONDS
    OMP_NUM_THREADS=4 python -m src.baselines.run_baseline \
      --method evapore_e2e --dataset "$ds" \
      --pred_dir "runs/seg/${ds}/seed0/pred" \
      --out "$out" --seed 0 --bio both \
      --ckpt "$ck" --device "$DEV" \
      --opt "accept_p=${tau}" \
      $(limit_args "$ds") --no_save_masks
    echo "=== [$(date '+%F %T')] $ds tau=$tau rc=$? $((SECONDS-t0))s" >&2
  done
done
echo "[sweep] done"
