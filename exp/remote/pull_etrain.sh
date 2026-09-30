#!/usr/bin/env bash
# Pull finished S4 step-E baseline-training checkpoints (rNCA / EVAPORE) back
# from the cloud remote into the EXACT local paths src/pipeline/s4_all.py
# expects, so a subsequent
#   python -m src.pipeline.s4_all --status --only E
# recognises them as DONE (Step.done() is pure output-existence, see
# src/pipeline/s4_all.py's Step.done()/missing()).
#
# Idempotent: re-running only copies a checkpoint that is missing locally or
# whose remote size differs from the local copy (a partially-written local
# file from an earlier interrupted pull will be re-pulled). Safe to run every
# few minutes while training jobs are still in flight -- it just finds
# nothing new to do for datasets that have not finished yet.
#
# Checkpoints are self-contained: `torch.save(dict(state_dict=..., cfg=...,
# meta=...))` in both src/baselines/rnca/rnca_adapter.py:save_ckpt and
# src/baselines/evapore/evapore_adapter.py:save_ckpt. There is currently no
# separate .json/.meta sidecar file for step E (unlike step A's
# best.pt+history.json pair) -- METHODS_WITH_SIDECARS below is empty for E,
# but the pull loop still checks for a same-stem .json next to the .pt in
# case a future adapter version adds one, so this script does not need to
# change if that happens.
#
# Usage:
#   bash exp/remote/pull_etrain.sh              # one pull, then exit
#   bash exp/remote/pull_etrain.sh --loop 900    # pull every 15 min forever
#   bash exp/remote/pull_etrain.sh --loop 900 &  # ... in the background
#   disown                                       # ... survives shell exit

set -uo pipefail

SSH_PORT=SSH_PORT
SSH_HOST="user@GPU_HOST"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=20 \
          -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/path/to/workdir/medical1/exp"
REMOTE_LOG_DIR="/path/to/workdir/medical1/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_CKPT_DIR="$EXP_DIR/runs/repair/ckpt"
LOCAL_LOG_DIR="$EXP_DIR/runs/s4_logs/remote_Etrain"

METHODS=(rnca evapore)
DATASETS=(drive chasedb1 hrf fives)

mkdir -p "$LOCAL_CKPT_DIR" "$LOCAL_LOG_DIR"

pull_once() {
  local pulled=0 pending=0 upToDate=0

  # one remote `stat` call for every checkpoint we might want, instead of
  # one ssh round-trip per file
  local remote_sizes
  remote_sizes="$(ssh "${SSH_OPTS[@]}" "$SSH_HOST" '
    cd '"$REMOTE_DIR"'/runs/repair/ckpt 2>/dev/null || exit 0
    for f in *.pt; do
      [ -e "$f" ] || continue
      stat -c "%n %s" "$f"
    done
  ' 2>/dev/null)"

  if [ -z "$remote_sizes" ]; then
    echo "[pull_etrain] $(date -u +%FT%TZ)  no checkpoints on remote yet"
    return 0
  fi

  for method in "${METHODS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      local name="${method}_${ds}.pt"
      local rsize
      rsize="$(awk -v n="$name" '$1==n{print $2}' <<<"$remote_sizes")"
      if [ -z "$rsize" ]; then
        pending=$((pending+1))
        continue
      fi
      local lpath="$LOCAL_CKPT_DIR/$name"
      local lsize=-1
      [ -f "$lpath" ] && lsize=$(stat -c%s "$lpath" 2>/dev/null || stat -f%z "$lpath" 2>/dev/null)
      if [ "$lsize" = "$rsize" ]; then
        upToDate=$((upToDate+1))
        continue
      fi
      echo "[pull_etrain] pulling $name ($rsize bytes) ..."
      scp -P "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=30 \
          "$SSH_HOST:$REMOTE_DIR/runs/repair/ckpt/$name" "$lpath.part" \
        && mv -f "$lpath.part" "$lpath" \
        && pulled=$((pulled+1)) \
        || { echo "[pull_etrain] FAILED to pull $name" >&2; rm -f "$lpath.part"; }

      # future-proofing: pull a same-stem sidecar if one ever exists
      local sidecar_name="${method}_${ds}.json"
      local rsc
      rsc="$(awk -v n="$sidecar_name" '$1==n{print $2}' <<<"$remote_sizes")"
      if [ -n "$rsc" ]; then
        scp -P "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=30 \
            "$SSH_HOST:$REMOTE_DIR/runs/repair/ckpt/$sidecar_name" \
            "$LOCAL_CKPT_DIR/$sidecar_name" 2>/dev/null
      fi
    done
  done

  # logs + .done markers, best-effort, purely informational (S4 never reads
  # these -- it only checks the .pt path)
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
    "cd '$REMOTE_LOG_DIR' 2>/dev/null && tar czf - Etrain_*.log Etrain_*.done 2>/dev/null" \
    2>/dev/null | tar xzf - -C "$LOCAL_LOG_DIR" 2>/dev/null

  echo "[pull_etrain] $(date -u +%FT%TZ)  pulled=$pulled up_to_date=$upToDate pending=$pending"
  return 0
}

if [ "${1:-}" = "--loop" ]; then
  interval="${2:-900}"
  echo "[pull_etrain] looping every ${interval}s -- Ctrl+C or 'kill \$(pgrep -f pull_etrain.sh)' to stop"
  while true; do
    pull_once
    sleep "$interval"
  done
else
  pull_once
fi
