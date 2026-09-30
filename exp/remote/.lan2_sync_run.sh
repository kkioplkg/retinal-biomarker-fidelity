#!/usr/bin/env bash
# Resumable push of arbitrary exp/ subtrees to the LAN node's NEW work root
# (/mnt/data/Programming/research_ws/medical1, /dev/sda1 NTFS).
#
# Windows Git Bash has no rsync, so this emulates `rsync -a --size-only
# --partial`: it diffs a (path,size) manifest against the node, then ships only
# the missing / size-mismatched files, in ~400 MB tar batches. Interrupting it
# costs at most one batch -- re-run and it picks up where it stopped.
#
# Usage:
#   bash exp/remote/lan2_sync.sh <subpath> [<subpath> ...]
#   BATCH_MB=800 bash exp/remote/lan2_sync.sh data/external/idrid/raw
#
# Subpaths are relative to exp/ (files or directories).

set -euo pipefail

SSH_HOST="${SSH_HOST:-user@LAN_HOST}"
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6
          -o Compression=no -c aes128-gcm@openssh.com)
# GZIP=0 (default) streams plain tar: the payload is JPEG/PNG/.pt, already
# compressed, and gzip is a single-core bottleneck on the Windows side
# (measured 2.8 MB/s with -z vs ~60 MB/s without).
GZIP="${GZIP:-0}"
if [ "$GZIP" = "1" ]; then TAR_C=(czf -); TAR_X=(xzf -); else TAR_C=(cf -); TAR_X=(xf -); fi
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/Programming/research_ws/medical1}"
REMOTE_DIR="$REMOTE_ROOT/exp"
BATCH_MB="${BATCH_MB:-400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

[ "$#" -ge 1 ] || { echo "usage: lan2_sync.sh <subpath> [...]" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---- local manifest: "<size>\t<path>" ------------------------------------
: > "$TMP/local.tsv"
for sp in "$@"; do
  if [ ! -e "$sp" ]; then echo "[lan2_sync] MISSING LOCAL: $sp" >&2; exit 1; fi
  find "$sp" -type f ! -path '*__pycache__*' ! -name '*.pyc' -printf '%s\t%p\n' >> "$TMP/local.tsv"
done
sort -k2 "$TMP/local.tsv" -o "$TMP/local.tsv"
LN=$(wc -l < "$TMP/local.tsv")
LB=$(awk -F'\t' '{s+=$1} END{printf "%.2f", s/1073741824}' "$TMP/local.tsv")
echo "[lan2_sync] local: $LN files, ${LB} GB  ->  ${SSH_HOST}:${REMOTE_DIR}"

# ---- remote manifest ------------------------------------------------------
REMOTE_FIND="mkdir -p '$REMOTE_DIR' && cd '$REMOTE_DIR' && for sp in $(printf "%q " "$@"); do [ -e \"\$sp\" ] && find \"\$sp\" -type f -printf '%s\t%p\n'; done 2>/dev/null; true"
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_FIND" | sort -k2 > "$TMP/remote.tsv" || true
RN=$(wc -l < "$TMP/remote.tsv")
echo "[lan2_sync] remote already has: $RN files"

# ---- diff: local lines whose exact (size,path) pair is absent remotely -----
# comm needs both sides sorted the same way -- plain full-line sort
sort "$TMP/local.tsv" -o "$TMP/local.tsv"
sort "$TMP/remote.tsv" -o "$TMP/remote.tsv"
comm -23 "$TMP/local.tsv" "$TMP/remote.tsv" | cut -f2- | sort -u > "$TMP/todo.txt"
TN=$(wc -l < "$TMP/todo.txt")
if [ "$TN" -eq 0 ]; then echo "[lan2_sync] nothing to do -- in sync."; exit 0; fi
# sizes come from the manifest, never from a per-file `stat`: spawning one
# process per file costs minutes on Windows Git Bash at a few thousand files.
awk -F'\t' 'NR==FNR{want[$0]=1; next} ($2 in want){print $1"\t"$2}' \
    "$TMP/todo.txt" "$TMP/local.tsv" > "$TMP/todo.tsv"
TB=$(awk -F'\t' '{s+=$1} END{printf "%.2f", s/1073741824}' "$TMP/todo.tsv")
echo "[lan2_sync] to send: $TN files, ${TB} GB (batches of ${BATCH_MB} MB)"

# ---- ship in batches ------------------------------------------------------
BATCH_BYTES=$((BATCH_MB * 1024 * 1024))
acc=0; n=0; batch=0
: > "$TMP/batch.txt"
flush() {
  [ -s "$TMP/batch.txt" ] || return 0
  batch=$((batch + 1))
  local mb=$((acc / 1048576))
  echo "[lan2_sync]   batch $batch: $n files, ${mb} MB ... $(date +%H:%M:%S)"
  tar "${TAR_C[@]}" --files-from="$TMP/batch.txt" \
  | ssh "${SSH_OPTS[@]}" "$SSH_HOST" "mkdir -p '$REMOTE_DIR' && cd '$REMOTE_DIR' && tar ${TAR_X[*]}"
  : > "$TMP/batch.txt"; acc=0; n=0
}
while IFS=$'\t' read -r sz f; do
  printf '%s\n' "$f" >> "$TMP/batch.txt"
  acc=$((acc + sz)); n=$((n + 1))
  [ "$acc" -ge "$BATCH_BYTES" ] && flush
done < "$TMP/todo.tsv"
flush

# ---- verify ---------------------------------------------------------------
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_FIND" | sort -k2 > "$TMP/remote2.tsv" || true
sort "$TMP/remote2.tsv" -o "$TMP/remote2.tsv"
LEFT=$(comm -23 "$TMP/local.tsv" "$TMP/remote2.tsv" | wc -l)
echo "[lan2_sync] done. remote files now: $(wc -l < "$TMP/remote2.tsv") ; still differing: $LEFT"
[ "$LEFT" -eq 0 ] || echo "[lan2_sync] WARNING: $LEFT file(s) still missing/mismatched -- re-run to resume." >&2
