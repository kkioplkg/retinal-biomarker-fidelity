#!/usr/bin/env bash
# Local-box driver for the locked E3 classification stage.
#
#   bash exp/remote/local_e3_clf.sh odir5k
#
# A faithful port of remote/cpu_e3_clf.sh to the Windows workstation (Git Bash
# + the conda interpreter).  The ONLY differences from the CPU-node script are
# the interpreter path, the repo root, and the parallelism knobs.  Every
# pre-registered default of `e3_external clf` is untouched -- n_boot 1000,
# SEED 0, 5x3 nested grouped CV, the cohort's own splitting unit from
# COHORT_ROLE, the Messidor-2 MAPLES-DR exclusion, logreg + gbdt, bio + bio+cov.
# `--n-jobs` is pure compute parallelism inside GridSearchCV and cannot change
# a number; BLAS threads are pinned to 1 so that parallelism is not nested.
#
# Two families of run, identical in structure to the CPU node's:
#   A. canonical -- the 12 FIVES tags in one call, reference
#      `e3_fives_s0_baseline`, into results/pivot/e3/<cohort>/{summary,delta}.csv
#   B. per-seed  -- for each seed k, a 4-tag call twice (ref = that seed's
#      `baseline`, then its `continued`), into
#      results/pivot/e3_perseed/ref<ref>_s<k>/<cohort>/ , whose tag directories
#      are symlinks to the canonical bio.csv files -- no biomarker row is
#      copied, recomputed or modified.
#
# Note on cohort scope: like the CPU node, only the 12 FIVES tags enter the
# classification stage.  Cohorts that also carry HRF-trained tags (odir5k has
# e3_hrf_s0_{baseline,reliseg}) keep them as biomarker tables only; they are a
# separate secondary comparison and are deliberately not mixed into the
# pre-registered FIVES contrast.

set -uo pipefail
R="${R:-/e/Programming/research_ws/medical1}"
cd "$R/exp" || exit 1
PY="${PY:-D:/Anaconda/envs/medical1/python.exe}"
NJOBS="${NJOBS:-6}"
SEEDPAR="${SEEDPAR:-3}"
PAR="${PAR:-1}"
E3="results/pivot/e3"
PERSEED="results/pivot/e3_perseed"
LOGDIR="runs/local_e3_clf"
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

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

# Per-seed family FIRST, canonical second -- the reverse of the CPU node's
# order. The two families are independent (separate --out-root, no shared
# state, they only read the same bio.csv), so the order changes no number. It
# is reversed because the pre-registration reports the contrast PER SEED, so
# the per-seed runs carry the reportable delta-AUC while the 12-tag canonical
# run is roughly 3x longer; running canonical first delays the headline by
# hours for no gain.
run_cohort() {
  local ds="$1"
  local s ref others t a nseed=0
  for s in "${SEEDS[@]}"; do
    for ref in baseline continued; do
     (
      outroot="$PERSEED/ref${ref}_s${s}"
      mkdir -p "$outroot/$ds"
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
      "$PY" -W ignore -m src.pivot.e3_external clf --dataset "$ds" \
            --tags "$(IFS=,; echo "${others[*]}")" --out-root "$outroot" \
            --n-jobs "$NJOBS" > "$LOGDIR/${ds}_s${s}_ref${ref}.log" 2>&1
      echo "[e3] $ds seed$s ref=$ref rc=$? $(date +%T)"
     ) &
      nseed=$((nseed + 1))
      if [ "$nseed" -ge "$SEEDPAR" ]; then wait -n 2>/dev/null || wait; nseed=$((nseed - 1)); fi
    done
  done
  wait
  echo "[e3] --- $ds PER-SEED FAMILY COMPLETE (reportable delta-AUC ready) $(date +%T)"

  echo "[e3] === $ds canonical (12 tags, ref=e3_fives_s0_baseline) $(date +%T)"
  "$PY" -W ignore -m src.pivot.e3_external clf --dataset "$ds" --tags "$(tags_all)" \
        --n-jobs "$NJOBS" > "$LOGDIR/${ds}_canonical.log" 2>&1
  echo "[e3] $ds canonical rc=$? $(date +%T)"
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
