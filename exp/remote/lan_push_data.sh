#!/usr/bin/env bash
# One-time bootstrap push of data + checkpoints/results the LAN node needs to
# reproduce local pipelines. Safe to re-run (tar overwrite); large (~16 GB).
#
# Usage: bash exp/remote/lan_push_data.sh

set -euo pipefail

SSH_HOST="user@LAN_HOST"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/home/user/medical1/exp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

ssh "${SSH_OPTS[@]}" "$SSH_HOST" "mkdir -p '$REMOTE_DIR'"

push() {
  local label="$1"; shift
  echo "[lan_push_data] pushing $label ..."
  tar czf - --exclude='__pycache__' "$@" \
  | ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cd '$REMOTE_DIR' && tar xzf -"
}

# 1. datasets (raw + fov), ~3.68 GB
push "data/{drive,chasedb1,hrf,fives,stare}" \
  data/drive data/chasedb1 data/hrf data/fives data/stare

# 2. gateA results csvs
push "results/gateA_*.csv" \
  results/gateA_biomarker_scales.csv \
  results/gateA_biomarker_scales_train.csv \
  results/gateA_biomarker_scales_train_vs_all.csv \
  results/gateA_biomarkers_gt.csv \
  results/gateA_pipeline_agreement.csv \
  results/gateA_pipeline_agreement_train.csv

# 3. model artifacts
push "runs/btr/habs_trainonly, runs/rigr_models, runs/rigr_head" \
  runs/btr/habs_trainonly runs/rigr_models runs/rigr_head

# 4. seg predictions + best.pt only (not full runs/seg, ~14G total on disk)
SEG_PATHS=()
while IFS= read -r p; do SEG_PATHS+=("$p"); done < <(find runs/seg -path '*/seed*/pred' -o -name best.pt)
if [ "${#SEG_PATHS[@]}" -gt 0 ]; then
  push "runs/seg/*/seed*/pred + best.pt" "${SEG_PATHS[@]}"
else
  echo "[lan_push_data] WARNING: no runs/seg pred/best.pt paths found locally" >&2
fi

# 5. seg_oof predictions (masks+prob)
push "runs/seg_oof/*/pred" runs/seg_oof/drive/pred runs/seg_oof/chasedb1/pred \
  runs/seg_oof/hrf/pred runs/seg_oof/fives/pred runs/seg_oof/stare/pred

# 6. repair checkpoints
push "runs/repair/ckpt/*.pt" runs/repair/ckpt

echo "[lan_push_data] all pushes issued. Run lan_status.sh / a verify pass to confirm counts."
