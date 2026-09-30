#!/usr/bin/env bash
# E3 external-inference driver for the LAN node (new /dev/sda1 work root).
#
# Runs a list of `python -m src.pivot.e3_external bio` jobs with bounded
# parallelism. The stage is GPU-light (measured 0-6% utilisation; the cost is
# the CPU-side skan biomarker pipeline, and a probe showed user-time == real-
# time, i.e. one core per job), so N concurrent jobs on a 20-core box give a
# near-linear speed-up while the 10 GB card stays far from full.
#
# Usage (on the node):
#   nohup bash remote/lan2_e3_driver.sh JOBS.txt 5 > runs/lan_e3/driver.log 2>&1 &
#
# JOBS.txt lines:  <dataset> <ckpt-path> <tag>
# A job whose results/pivot/e3/<ds>/<tag>/meta.json already exists is skipped
# (meta.json is written only after a whole pass completes -- bio.csv is
# appended every 10 images and is NOT a completion marker).

set -uo pipefail

ROOT="${ROOT:-/mnt/data/Programming/research_ws/medical1}"
EXP="$ROOT/exp"
PY="$ROOT/venv/bin/python"
JOBS="${1:?usage: lan2_e3_driver.sh <jobs.txt> [parallel]}"
PAR="${2:-9}"
RESIZE="${RESIZE:-1536}"
# GPU memory per worker, measured on this node at resize-longest 1536:
#   --sw-batch 4 -> 2134 MiB (only 4 workers fit in 9.65 GiB; the 5th OOMs)
#   --sw-batch 1 ->  712 MiB, same 3.9 s/img (the stage is CPU-bound)
SW_BATCH="${SW_BATCH:-1}"

cd "$EXP"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/lan_e3

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[driver $(stamp)] jobs=$JOBS parallel=$PAR resize=$RESIZE sw_batch=$SW_BATCH"

run_one() {
  local ds="$1" ckpt="$2" tag="$3"
  local log="runs/lan_e3/${ds}__${tag}.log"
  if [ -f "results/pivot/e3/$ds/$tag/meta.json" ]; then
    echo "[driver $(stamp)] SKIP (done) $ds $tag"; return 0
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[driver $(stamp)] MISSING CKPT $ckpt -- skipping $ds $tag"; return 0
  fi
  echo "[driver $(stamp)] START $ds $tag  ($ckpt)"
  "$PY" -m src.pivot.e3_external bio --dataset "$ds" --ckpt "$ckpt" \
        --tag "$tag" --device cuda:0 --resize-longest "$RESIZE" \
        --sw-batch "$SW_BATCH" --resolution-record >> "$log" 2>&1
  local rc=$?
  echo "[driver $(stamp)] END   $ds $tag rc=$rc  rows=$(($(wc -l < "results/pivot/e3/$ds/$tag/bio.csv" 2>/dev/null || echo 1) - 1))"
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
echo "[driver $(stamp)] ALL DONE ($n jobs dispatched)"
