#!/usr/bin/env bash
# CPU-node driver for the E4 MAPLES-DR biomarker units (runs ON the node).
#
# For every runs/pivot/e4/maples/<tag>/ that has a complete manifest.csv + mask/
# and no bio.csv yet: re-root the manifest's Windows paths, then measure the
# biomarkers through the project's own estimator path
# (``e4_replication bio`` -> ``p5_eval.cmd_bio`` -> ``p5_eval._bio_one``, i.e.
# compute_all(fd_rotations=5) with the optic disc found from the fundus image
# alone).  Nothing about the estimator is reimplemented here.
#
#   PAR=4 PROCS=8 bash exp/remote/cpu_e4_bio.sh
#
# PAR   directories measured concurrently   (default 4)
# PROCS worker processes inside one directory (default 8)
# PAR*PROCS should be <= nproc (32 on this box).

set -uo pipefail

R="${R:-/path/to/workdir/medical1}"
cd "$R/exp"
PY="$R/venv/bin/python"
MAPLES_ROOT="runs/pivot/e4/maples"
LOGDIR="runs/cpu_e4"
PAR="${PAR:-4}"
PROCS="${PROCS:-8}"
mkdir -p "$LOGDIR"

# BLAS/OpenMP must stay single-threaded: PAR*PROCS processes each spawning a
# thread pool would oversubscribe 32 cores several times over and run slower.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

[ -d "$MAPLES_ROOT" ] || { echo "[cpu_e4] no $MAPLES_ROOT yet -- nothing to do"; exit 0; }

todo=()
for d in "$MAPLES_ROOT"/*/; do
  tag="$(basename "$d")"
  [ -f "$d/bio.csv" ] && continue
  [ -f "$d/manifest.csv" ] || continue
  # a manifest is only complete once the pull-loop marked it so; the marker is
  # written by cpu_e4_loop.sh after the whole directory transferred
  [ -f "$d/.synced" ] || { echo "[cpu_e4] $tag: manifest present but not marked .synced -- skipping this round"; continue; }
  nm=$(ls "$d/mask" 2>/dev/null | wc -l)
  nr=$(( $(wc -l < "$d/manifest.csv") - 1 ))
  if [ "$nm" -ne "$nr" ] || [ "$nr" -lt 1 ]; then
    echo "[cpu_e4] $tag: manifest rows=$nr but mask/ has $nm -- incomplete, skipping"
    continue
  fi
  todo+=("$tag")
done

echo "[cpu_e4] $(date +%F' '%T)  ${#todo[@]} directories to measure (PAR=$PAR PROCS=$PROCS)"
[ "${#todo[@]}" -gt 0 ] || exit 0

run_one() {
  local tag="$1"
  local d="$MAPLES_ROOT/$tag"
  "$PY" remote/cpu_fix_manifest.py "$d/manifest.csv" >> "$LOGDIR/$tag.log" 2>&1 || {
    echo "[cpu_e4] $tag: manifest re-root FAILED (masks missing) -- not measuring" | tee -a "$LOGDIR/$tag.log"
    return 1
  }
  echo "[cpu_e4] $tag: start $(date +%T)" >> "$LOGDIR/$tag.log"
  "$PY" -m src.pivot.e4_replication bio --maples --skip-gt \
        --procs "$PROCS" --only "$tag" >> "$LOGDIR/$tag.log" 2>&1
  local rc=$?
  if [ -f "$d/bio.csv" ]; then
    echo "[cpu_e4] $tag: DONE rows=$(( $(wc -l < "$d/bio.csv") - 1 )) rc=$rc $(date +%T)"
  else
    echo "[cpu_e4] $tag: FAILED rc=$rc (see $LOGDIR/$tag.log)"
  fi
}

n=0
for tag in "${todo[@]}"; do
  run_one "$tag" &
  n=$((n + 1))
  if [ "$n" -ge "$PAR" ]; then wait -n 2>/dev/null || wait; n=$((n - 1)); fi
done
wait
echo "[cpu_e4] $(date +%F' '%T)  round complete"
