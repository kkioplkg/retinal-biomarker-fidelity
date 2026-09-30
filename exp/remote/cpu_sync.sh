#!/usr/bin/env bash
# Resumable push of exp/ subtrees to the rented CPU node
# (user@CPU_HOST:SSH_PORT, work root /path/to/workdir/medical1).
#
# Same manifest-diff-then-tar design as lan2_sync.sh (Windows Git Bash has no
# rsync): it diffs a (size,path) manifest against the node and ships only the
# missing / size-mismatched files in ~400 MB tar-over-ssh batches.  Interrupting
# costs at most one batch -- re-run to resume.  Key auth only (BatchMode=yes).
#
# Usage:
#   bash exp/remote/cpu_sync.sh <subpath> [<subpath> ...]
#   LIST=/tmp/files.txt bash exp/remote/cpu_sync.sh          # one path per line
#   BATCH_MB=800 bash exp/remote/cpu_sync.sh data/maplesdr
#
# Subpaths are relative to exp/ (files or directories).

set -euo pipefail

SSH_HOST="${SSH_HOST:-user@CPU_HOST}"
SSH_PORT="${SSH_PORT:-SSH_PORT}"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30
          -o ServerAliveCountMax=6 -o Compression=no -c aes128-gcm@openssh.com)
# TARGZ=0 (default): payload is JPEG/PNG/.pt, and gzip is a single-core
# bottleneck on the Windows side.  This link is WAN, so TARGZ=1 can still pay
# off for text-heavy subtrees (src/, csv) -- set it explicitly there.
TARGZ="${TARGZ:-0}"   # NB: not GZIP -- that name is gzip's own env var and it rejects "1"
if [ "$TARGZ" = "1" ]; then TAR_C=(czf -); TAR_X=(xzf -); else TAR_C=(cf -); TAR_X=(xf -); fi
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/workdir/medical1}"
REMOTE_DIR="$REMOTE_ROOT/exp"
BATCH_MB="${BATCH_MB:-400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---- argument list (positional, or LIST=file) -----------------------------
ARGS=()
if [ -n "${LIST:-}" ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && ARGS+=("$line")
  done < "$LIST"
fi
ARGS+=("$@")
[ "${#ARGS[@]}" -ge 1 ] || { echo "usage: cpu_sync.sh <subpath> [...] | LIST=f cpu_sync.sh" >&2; exit 1; }

# ---- local manifest: "<size>\t<path>" -------------------------------------
: > "$TMP/local.tsv"
for sp in "${ARGS[@]}"; do
  if [ ! -e "$sp" ]; then echo "[cpu_sync] MISSING LOCAL: $sp" >&2; exit 1; fi
  find "$sp" -type f ! -path '*__pycache__*' ! -name '*.pyc' -printf '%s\t%p\n' >> "$TMP/local.tsv"
done
sort "$TMP/local.tsv" -o "$TMP/local.tsv"
LN=$(wc -l < "$TMP/local.tsv")
LB=$(awk -F'\t' '{s+=$1} END{printf "%.2f", s/1073741824}' "$TMP/local.tsv")
echo "[cpu_sync] local: $LN files, ${LB} GB  ->  ${SSH_HOST}:${REMOTE_DIR}"

# ---- remote manifest ------------------------------------------------------
# The arg list can be long (a 162-file explicit list), so ship it as a file the
# remote side reads, instead of interpolating it into the command line.
printf '%s\n' "${ARGS[@]}" > "$TMP/args.txt"
# NB: the remote arg-file name must be interpolated LOCALLY.  Leaving `$$`
# inside the single-quoted remote command made the remote shell expand it to
# its own pid, so the find always read a non-existent file and reported "remote
# has 0 files" -- every run re-sent everything and the verify step never passed.
ARGF="/tmp/.cpu_sync_args.$$"
REMOTE_FIND="mkdir -p '$REMOTE_DIR' && cd '$REMOTE_DIR' && while IFS= read -r sp; do [ -e \"\$sp\" ] && find \"\$sp\" -type f -printf '%s\t%p\n'; done < '$ARGF' 2>/dev/null; true"
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cat > '$ARGF'" < "$TMP/args.txt"
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_FIND" | sort > "$TMP/remote.tsv" || true
RN=$(wc -l < "$TMP/remote.tsv")
echo "[cpu_sync] remote already has: $RN files"

# ---- diff -----------------------------------------------------------------
comm -23 "$TMP/local.tsv" "$TMP/remote.tsv" | cut -f2- | sort -u > "$TMP/todo.txt"
TN=$(wc -l < "$TMP/todo.txt")
if [ "$TN" -eq 0 ]; then
  echo "[cpu_sync] nothing to do -- in sync."
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "rm -f "$ARGF"" || true
  exit 0
fi
awk -F'\t' 'NR==FNR{want[$0]=1; next} ($2 in want){print $1"\t"$2}' \
    "$TMP/todo.txt" "$TMP/local.tsv" > "$TMP/todo.tsv"
TB=$(awk -F'\t' '{s+=$1} END{printf "%.2f", s/1073741824}' "$TMP/todo.tsv")
echo "[cpu_sync] to send: $TN files, ${TB} GB (batches of ${BATCH_MB} MB)"

# ---- ship in batches ------------------------------------------------------
BATCH_BYTES=$((BATCH_MB * 1024 * 1024))
acc=0; n=0; batch=0
: > "$TMP/batch.txt"
flush() {
  [ -s "$TMP/batch.txt" ] || return 0
  batch=$((batch + 1))
  local mb=$((acc / 1048576))
  echo "[cpu_sync]   batch $batch: $n files, ${mb} MB ... $(date +%H:%M:%S)"
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
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_FIND" | sort > "$TMP/remote2.tsv" || true
LEFT=$(comm -23 "$TMP/local.tsv" "$TMP/remote2.tsv" | wc -l)
echo "[cpu_sync] done. remote files now: $(wc -l < "$TMP/remote2.tsv") ; still differing: $LEFT"
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "rm -f "$ARGF"" || true
[ "$LEFT" -eq 0 ] || echo "[cpu_sync] WARNING: $LEFT file(s) still missing/mismatched -- re-run to resume." >&2
