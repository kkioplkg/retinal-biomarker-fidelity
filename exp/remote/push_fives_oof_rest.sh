#!/usr/bin/env bash
# Push the FIVES OOF training prob maps the remote is missing.
# Coordinator 2026-09-03 18:40: pi must match the local main experiment, which
# fits it on ALL 510 FIVES OOF training images.  The remote had 178 (the capped
# 120 plus 58 pushed at 09:00), so build_data was fitting pi on a different set
# than the local B step -- a silent divergence.
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=25 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
L="${1:?need list file}"
n=$(wc -l < "$L")
echo "[push] $(date -Is) $n FIVES OOF prob files"
for attempt in 1 2 3 4 5; do
  if tar czf - -T "$L" | $SSH "mkdir -p $R && tar xzf - -C $R && echo PUSH_OK" | grep -q PUSH_OK; then
    echo "[push] OK $(date -Is)"
    $SSH "ls $R/runs/seg_oof/fives/pred/prob | wc -l" | sed 's/^/[push] remote now has /'
    exit 0
  fi
  echo "[push] attempt $attempt failed, retrying"; sleep 20
done
echo "[push] FAILED after 5 attempts"; exit 1
