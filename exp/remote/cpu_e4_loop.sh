#!/usr/bin/env bash
# LOCAL side of the E4 MAPLES-DR offload: every ROUND_S seconds, push newly
# finished MAPLES prediction directories to the CPU node and pull back the
# bio.csv files it has produced.  The measuring itself is done by the node's
# own watcher (exp/remote/cpu_e4_watch.sh -> cpu_e4_bio.sh).
#
#   bash exp/remote/cpu_e4_loop.sh            # foreground
#   nohup bash exp/remote/cpu_e4_loop.sh > runs/cpu_e4_loop.log 2>&1 &
#
# What is pushed per directory: manifest.csv + mask/ ONLY.  prob/ is float16
# .npy (~6.6 MB/image, 25 GB over the 24 dirs) and the biomarker step never
# reads it.  A directory is considered finished locally when infer_meta.json
# exists -- p5_eval.cmd_infer writes it last, after the masks and the manifest.
#
# Nothing local is ever deleted or modified; the only local writes are the
# pulled `bio.csv` files, at the exact path the E4 finisher reads.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

SSH_HOST="${SSH_HOST:-user@CPU_HOST}"
SSH_PORT="${SSH_PORT:-SSH_PORT}"
SSH_OPTS=(-p "$SSH_PORT" -o BatchMode=yes -o ServerAliveInterval=30
          -o ServerAliveCountMax=6 -o Compression=no -c aes128-gcm@openssh.com)
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/workdir/medical1}"
REMOTE_EXP="$REMOTE_ROOT/exp"
MAPLES="runs/pivot/e4/maples"
ROUND_S="${ROUND_S:-600}"
EXPECT="${EXPECT:-24}"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

while true; do
  echo "=== [cpu_e4_loop] $(date +%F' '%T) ==="

  # ---- 1. which local dirs are finished and not yet fully on the node? -----
  : > "$TMP/push.txt"
  npend=0
  if [ -d "$MAPLES" ]; then
    for d in "$MAPLES"/*/; do
      tag="$(basename "$d")"
      [ -f "$d/infer_meta.json" ] || continue        # still being written
      [ -f "$d/manifest.csv" ] || continue
      echo "$d/manifest.csv" | sed 's#//*#/#g' >> "$TMP/push.txt"
      echo "$d/mask" | sed 's#//*#/#g' >> "$TMP/push.txt"
      npend=$((npend + 1))
    done
  fi
  echo "[loop] local finished MAPLES dirs: $npend"

  # ---- 2. push (cpu_sync diffs, so already-synced dirs cost one manifest) ---
  if [ -s "$TMP/push.txt" ]; then
    LIST="$TMP/push.txt" bash remote/cpu_sync.sh 2>&1 | sed 's/^/[loop] /'
  fi

  # ---- 3. mark complete directories on the node ---------------------------
  # .synced is what cpu_e4_bio.sh gates on: it is written only when the node's
  # own mask count matches the manifest's row count, so a directory that
  # arrived mid-batch is never measured half-transferred.
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cd '$REMOTE_EXP' && for d in $MAPLES/*/; do
      [ -f \"\$d/manifest.csv\" ] || continue
      nr=\$(( \$(wc -l < \"\$d/manifest.csv\") - 1 ))
      nm=\$(ls \"\$d/mask\" 2>/dev/null | wc -l)
      if [ \"\$nr\" -gt 0 ] && [ \"\$nr\" -eq \"\$nm\" ]; then touch \"\$d/.synced\";
      else rm -f \"\$d/.synced\"; fi
    done; echo \"[node] synced dirs: \$(ls -d $MAPLES/*/.synced 2>/dev/null | wc -l)\"" 2>&1 | sed 's/^/[loop] /'

  # ---- 4. pull back every bio.csv the node has finished --------------------
  mapfile -t done_tags < <(ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
      "cd '$REMOTE_EXP' && ls -d $MAPLES/*/bio.csv 2>/dev/null" | sed "s#^$MAPLES/##; s#/bio.csv\$##")
  got=0
  for tag in "${done_tags[@]:-}"; do
    [ -n "$tag" ] || continue
    local_csv="$MAPLES/$tag/bio.csv"
    if [ -f "$local_csv" ]; then got=$((got + 1)); continue; fi
    mkdir -p "$MAPLES/$tag"
    if ssh "${SSH_OPTS[@]}" "$SSH_HOST" "cat '$REMOTE_EXP/$MAPLES/$tag/bio.csv'" > "$local_csv.part"; then
      rows=$(( $(wc -l < "$local_csv.part") - 1 ))
      if [ "$rows" -eq 162 ]; then
        mv "$local_csv.part" "$local_csv"
        echo "[loop] PULLED $tag  rows=$rows"
        got=$((got + 1))
      else
        echo "[loop] REJECTED $tag: $rows rows, expected 162 -- left as .part" >&2
      fi
    else
      rm -f "$local_csv.part"
    fi
  done
  echo "[loop] bio.csv present locally: $got / $EXPECT"

  if [ "$got" -ge "$EXPECT" ]; then
    echo "[loop] all $EXPECT MAPLES bio.csv delivered -- loop exiting $(date +%F' '%T)"
    exit 0
  fi
  sleep "$ROUND_S"
done
