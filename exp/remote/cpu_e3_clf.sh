#!/usr/bin/env bash
# CPU-node driver for the locked E3 classification stage (runs ON the node).
#
#   bash exp/remote/cpu_e3_clf.sh idrid messidor2 aptos2019
#
# Per cohort it runs `python -m src.pivot.e3_external clf` with every
# pre-registered default untouched -- n_boot 1000, SEED 0, 5x3 nested grouped CV,
# image-level splitting for these three cohorts, the Messidor-2 162-image
# MAPLES-DR exclusion applied at clf time, logreg + gbdt, bio + bio+cov.
# The ONLY flag passed besides --dataset/--tags is `--n-jobs`, which is pure
# compute parallelism inside GridSearchCV and cannot change a number.
#
# Two families of run, because `delta_vs_ref` is always measured against
# `tags[0]`:
#
#   A. canonical  -- all 12 FIVES tags in one call, reference
#      `e3_fives_s0_baseline`, written to the canonical
#      results/pivot/e3/<cohort>/{summary,delta}.csv.
#   B. per-seed   -- the pre-registration reports the contrast PER SEED, so for
#      each seed k a 4-tag call is made twice, once with the seed's `baseline`
#      as reference and once with its `continued`, giving paired-bootstrap CIs
#      for both `ReliSeg - baseline` and `ReliSeg - continued` within the seed.
#      These land under results/pivot/e3_perseed/<ref>_s<k>/<cohort>/ , whose
#      tag directories are symlinks to the canonical bio.csv files -- no
#      biomarker row is copied, recomputed or modified.

set -uo pipefail
R="${R:-/path/to/workdir/medical1}"
cd "$R/exp"
PY="$R/venv/bin/python"
NJOBS="${NJOBS:-8}"
PAR="${PAR:-3}"
E3="results/pivot/e3"
PERSEED="results/pivot/e3_perseed"
LOGDIR="runs/cpu_e3"
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

SEEDS=(0 1 2)
ARMS=(baseline continued reliseg cfloss)

tags_all() {           # canonical order: baseline(ref) first, then per seed
  local out=()
  for a in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do out+=("e3_fives_s${s}_${a}"); done; done
  (IFS=,; echo "${out[*]}")
}

have_all() {           # $1 = cohort ; true when all 12 bio.csv exist
  local ds="$1" s a
  for a in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do
    [ -f "$E3/$ds/e3_fives_s${s}_${a}/bio.csv" ] || { echo "  missing e3_fives_s${s}_${a}"; return 1; }
  done; done
  return 0
}

run_cohort() {
  local ds="$1"
  echo "[e3] === $ds canonical (12 tags, ref=e3_fives_s0_baseline) $(date +%T)"
  "$PY" -m src.pivot.e3_external clf --dataset "$ds" --tags "$(tags_all)" \
        --n-jobs "$NJOBS" > "$LOGDIR/${ds}_canonical.log" 2>&1
  echo "[e3] $ds canonical rc=$? $(date +%T)"

  # The six per-seed runs are independent of each other, so run SEEDPAR of them
  # at a time.  On APTOS (3,662 rows) the sequential tail is what would collide
  # with the E4 biomarker lane starting on the same box.
  local s ref others t nseed=0
  for s in "${SEEDS[@]}"; do
    for ref in baseline continued; do
     (
      local outroot="$PERSEED/ref${ref}_s${s}"
      mkdir -p "$outroot/$ds"
      # symlink the four tag directories of this seed into the private out-root
      for a in "${ARMS[@]}"; do
        t="e3_fives_s${s}_${a}"
        [ -e "$outroot/$ds/$t" ] || ln -s "$(readlink -f "$E3/$ds/$t")" "$outroot/$ds/$t"
      done
      others=("e3_fives_s${s}_${ref}")
      for a in "${ARMS[@]}"; do
        [ "$a" = "$ref" ] && continue
        others+=("e3_fives_s${s}_${a}")
      done
      echo "[e3] --- $ds seed$s ref=$ref $(date +%T)"
      "$PY" -m src.pivot.e3_external clf --dataset "$ds" \
            --tags "$(IFS=,; echo "${others[*]}")" --out-root "$outroot" \
            --n-jobs "$NJOBS" > "$LOGDIR/${ds}_s${s}_ref${ref}.log" 2>&1
      echo "[e3] $ds seed$s ref=$ref rc=$? $(date +%T)"
     ) &
      nseed=$((nseed + 1))
      if [ "$nseed" -ge "${SEEDPAR:-1}" ]; then wait -n 2>/dev/null || wait; nseed=$((nseed - 1)); fi
    done
  done
  wait
  echo "[e3] === $ds COMPLETE $(date +%T)"
}

n=0
for ds in "$@"; do
  if ! have_all "$ds"; then
    echo "[e3] SKIP $ds -- not all 12 FIVES tags present yet"
    continue
  fi
  run_cohort "$ds" &
  n=$((n + 1))
  if [ "$n" -ge "$PAR" ]; then wait -n 2>/dev/null || wait; n=$((n - 1)); fi
done
wait
echo "[e3] all requested cohorts done $(date +%F' '%T)"
