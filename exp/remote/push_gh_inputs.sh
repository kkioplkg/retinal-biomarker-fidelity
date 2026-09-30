#!/usr/bin/env bash
# Push the inputs steps G and H need onto the remote mirror.  Idempotent:
# tar overwrites, and every list is derived from the same capped subsets the
# orchestrator's image_caps produce, so re-running only re-copies.
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
L="${1:?need list dir}"

push() {   # $1 = label, rest = tar args
  local label="$1"; shift
  echo "[push] $label  $(date -Is)"
  tar czf - "$@" | $SSH "mkdir -p $R && tar xzf - -C $R" \
    && echo "[push] $label OK $(date -Is)" || echo "[push] $label FAILED $(date -Is)"
}

push "fives-oof-58"   -T "$L/oof_missing.lst"
push "seg-small"      runs/seg/drive/seed0/pred runs/seg/chasedb1/seed0/pred
push "seg-hrf"        runs/seg/hrf/seed0/pred
push "seg-fives-60"   runs/seg/fives/seed0/pred/manifest.csv -T "$L/seg_fives60.lst"
echo "[push] all done $(date -Is)"
