#!/usr/bin/env bash
# E3 external-inference driver for the rented 2x RTX 2080 Ti node
# (work root /path/to/workdir/medical1).
#
# Runs a list of `python -m src.pivot.e3_external bio` jobs with bounded
# parallelism on ONE card. The stage is CPU-bound (the skan biomarker pipeline;
# user-time == real-time, i.e. ~1 core per job) and GPU-light, so N concurrent
# jobs give a near-linear speed-up while the 11 GB card stays far from full.
#
# GPU memory per worker at --resize-longest 1536 (measured on the LAN RTX 3080,
# same code path):  --sw-batch 4 -> 2134 MiB ;  --sw-batch 1 -> 712 MiB, at the
# same 3.9 s/img.  So always --sw-batch 1 and cap concurrency by cores, not VRAM.
#
# Usage (on the node), one driver per card:
#   nohup setsid bash remote/gpu2_e3_driver.sh remote/gpu2_jobs_odir5k_g0.txt 7 0 \
#         > runs/gpu2_e3/driver_g0.log 2>&1 < /dev/null &
#   nohup setsid bash remote/gpu2_e3_driver.sh remote/gpu2_jobs_odir5k_g1.txt 7 1 \
#         > runs/gpu2_e3/driver_g1.log 2>&1 < /dev/null &
#
# JOBS.txt lines:  <dataset> <ckpt-path> <tag>
# A job whose results/pivot/e3/<ds>/<tag>/meta.json already exists is skipped
# (meta.json is written only after a whole pass completes -- bio.csv is appended
# every 10 images and is NOT a completion marker).

set -uo pipefail

ROOT="${ROOT:-/path/to/workdir/medical1}"
EXP="$ROOT/exp"
PY="$ROOT/venv/bin/python"
JOBS="${1:?usage: gpu2_e3_driver.sh <jobs.txt> [parallel] [gpu]}"
PAR="${2:-7}"
GPU="${3:-0}"
RESIZE="${RESIZE:-1536}"
SW_BATCH="${SW_BATCH:-1}"

cd "$EXP"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/gpu2_e3

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[driver-g$GPU $(stamp)] jobs=$JOBS parallel=$PAR gpu=$GPU resize=$RESIZE sw_batch=$SW_BATCH"

run_one() {
  local ds="$1" ckpt="$2" tag="$3"
  local log="runs/gpu2_e3/${ds}__${tag}.log"
  if [ -f "results/pivot/e3/$ds/$tag/meta.json" ]; then
    echo "[driver-g$GPU $(stamp)] SKIP (done) $ds $tag"; return 0
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[driver-g$GPU $(stamp)] MISSING CKPT $ckpt -- skipping $ds $tag"; return 0
  fi
  echo "[driver-g$GPU $(stamp)] START $ds $tag  ($ckpt)"
  "$PY" -m src.pivot.e3_external bio --dataset "$ds" --ckpt "$ckpt" \
        --tag "$tag" --device "cuda:$GPU" --resize-longest "$RESIZE" \
        --sw-batch "$SW_BATCH" --resolution-record >> "$log" 2>&1
  local rc=$?
  echo "[driver-g$GPU $(stamp)] END   $ds $tag rc=$rc  rows=$(($(wc -l < "results/pivot/e3/$ds/$tag/bio.csv" 2>/dev/null || echo 1) - 1))"
  return 0
}

n=0
while read -r ds ckpt tag; do
  case "$ds" in ""|\#*) continue;; esac
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  run_one "$ds" "$ckpt" "$tag" &
  n=$((n + 1))
done < "$JOBS"
wait
echo "[driver-g$GPU $(stamp)] ALL DONE ($n jobs dispatched)"
