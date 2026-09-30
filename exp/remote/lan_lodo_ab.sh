#!/usr/bin/env bash
# LODO follow-ups from DECISIONS.md 2026-09-15 21:30, run ON the LAN node
# (cwd = /home/user/medical1/exp, venv already active).
#
#   (A) re-run G:run:<D>:risk with the DEPLOYED R_false head
#       runs/rigr_models/lodo_<D>/seed0/btr_false_deployed.joblib
#       (fitted by step G:fit from the three source domains' rfalse_train.csv,
#        target habs) instead of the C1 bridge head btr_false_wo_<D>.joblib.
#       geom_scale.json is unchanged: m_R is a median over R_MISS, and only the
#       R_false head changes here.
#
#   (B) LODO seeds 1 and 2: same seed-0 head / scorer / BTR heads, only the
#       test predictions change (runs/seg/<D>/seed<k>/pred).  Written to
#       runs/rigr/lodo_<D>/{risk,uniform}_seed<k>/ and annotated in
#       summary.json with models_from_seed = 0.
#
# One chain per dataset, four chains in parallel.  Usage:
#   bash remote/lan_lodo_ab.sh            # all four
#   bash remote/lan_lodo_ab.sh drive hrf  # a subset

set -uo pipefail

LOGDIR="runs_remote/logs"
mkdir -p "$LOGDIR"

btr_variant() {
  case "$1" in
    drive)     echo "wo_DRIVE" ;;
    chasedb1)  echo "wo_CHASE_DB1" ;;
    hrf)       echo "wo_HRF" ;;
    fives)     echo "wo_FIVES" ;;
    *) echo "unknown dataset $1" >&2; exit 2 ;;
  esac
}

limit_args() {
  # image_caps.test in src/pipeline/s4_all.py: only FIVES is capped
  if [ "$1" == "fives" ]; then echo "--limit 60"; fi
}

# annotate a finished run's summary.json: the models are the seed-0 LODO ones
annotate() {
  local out="$1" seed="$2"
  python - "$out" "$seed" <<'PY'
import json, os, sys
out, seed = sys.argv[1], int(sys.argv[2])
p = os.path.join(out, "summary.json")
if not os.path.exists(p):
    print("[annotate] no summary.json in %s" % out, flush=True)
    raise SystemExit(0)
with open(p, encoding="utf-8") as f:
    d = json.load(f)
d["models_from_seed"] = 0
d["test_pred_seed"] = seed
d["models_note"] = (
    "LODO head, pair scorer, deployed R_false and geom_scale are the seed-0 "
    "LODO models (runs/rigr_head|rigr_models/lodo_<D>/seed0); only the test "
    "segmentation predictions come from seed %d.  DECISIONS.md 2026-09-15 "
    "21:30 item (3)." % seed)
with open(p, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, default=float)
print("[annotate] %s -> models_from_seed=0 test_pred_seed=%d" % (p, seed), flush=True)
PY
}

run_one() {
  # run_one <dataset> <seed> <mode> <out>
  local ds="$1" seed="$2" mode="$3" out="$4"
  local var; var="$(btr_variant "$ds")"
  local extra=()
  if [ "$mode" == "risk" ]; then
    extra=(--btr_false "runs/rigr_models/lodo_${ds}/seed0/btr_false_deployed.joblib"
           --btr_variant "$var"
           --btr_runs_dir runs/btr/habs_trainonly)
  fi
  echo "=== [$(date '+%F %T')] $ds seed=$seed mode=$mode -> $out" >&2
  local t0=$SECONDS
  python -m src.rigr.run_rigr \
    --dataset "$ds" --seed "$seed" --mode "$mode" \
    --pred_dir "runs/seg/${ds}/seed${seed}/pred" \
    --out "$out" \
    --head_ckpt "runs/rigr_head/lodo_${ds}/seed0/best.pt" \
    --scorer "runs/rigr_models/lodo_${ds}/seed0/scorer.joblib" \
    --evidence auto --orientation learned \
    --lam 1.0 --eta 0.1 --tau 0.5 --bio both --gpu 0 \
    $(limit_args "$ds") "${extra[@]}"
  local rc=$?
  echo "=== [$(date '+%F %T')] $ds seed=$seed mode=$mode rc=$rc  $((SECONDS-t0))s" >&2
  return $rc
}

chain() {
  local ds="$1"
  # ---- (A) seed 0 risk, deployed R_false
  run_one "$ds" 0 risk "runs/rigr/lodo_${ds}/risk"
  # ---- (B) seeds 1 and 2, risk + uniform
  for k in 1 2; do
    for mode in risk uniform; do
      local out="runs/rigr/lodo_${ds}/${mode}_seed${k}"
      run_one "$ds" "$k" "$mode" "$out" && annotate "$out" "$k"
    done
  done
  echo "=== [$(date '+%F %T')] chain $ds finished" >&2
}

DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(drive chasedb1 hrf fives)
fi

pids=()
for ds in "${DATASETS[@]}"; do
  chain "$ds" > "$LOGDIR/lodo_ab_${ds}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "[lan_lodo_ab] $ds chain PID=$pid  log=$LOGDIR/lodo_ab_${ds}.log"
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done
echo "[lan_lodo_ab] all chains done, rc=$rc"
exit $rc
