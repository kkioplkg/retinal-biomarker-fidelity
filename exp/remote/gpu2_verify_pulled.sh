#!/usr/bin/env bash
# Verify E3 tags pulled back from the 2x RTX 2080 Ti node.
#
#   bash exp/remote/gpu2_verify_pulled.sh odir5k 6392
#
# For every tag directory under exp/results/pivot/e3/<ds>/ it checks
#   * bio.csv row count (minus header) == the expected image count
#   * meta.json exists (the completion marker)
#   * meta.json's ckpt_sha256 equals the SHA-256 recomputed from the LOCAL
#     checkpoint file it names -- i.e. the frozen-segmenter identity guard
#   * all ckpt_sha256 values across tags are pairwise distinct
#
# Checkpoint paths inside meta.json are absolute paths ON THE NODE, so they are
# remapped to the local tree by their exp/-relative suffix.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

DS="${1:?usage: gpu2_verify_pulled.sh <dataset> <expected_rows> [tag ...]}"
WANT="${2:?usage: gpu2_verify_pulled.sh <dataset> <expected_rows> [tag ...]}"
shift 2

BASE="results/pivot/e3/$DS"
if [ "$#" -gt 0 ]; then TAGS=("$@"); else mapfile -t TAGS < <(ls "$BASE" 2>/dev/null); fi

ok=0; bad=0
: > /tmp/.gpu2_digests.$$
for tag in "${TAGS[@]}"; do
  d="$BASE/$tag"
  if [ ! -f "$d/meta.json" ]; then echo "FAIL $tag: no meta.json (pass incomplete)"; bad=$((bad+1)); continue; fi
  rows=$(( $(wc -l < "$d/bio.csv") - 1 ))
  ck=$(python -c "import json,sys;m=json.load(open(sys.argv[1]));print(m['ckpt'])" "$d/meta.json")
  dg=$(python -c "import json,sys;m=json.load(open(sys.argv[1]));print(m['ckpt_sha256'])" "$d/meta.json")
  # node path -> local path: keep everything from 'runs/' onwards.
  # Tags produced on the Windows box carry backslash paths, so normalise first.
  ck="${ck//\\//}"
  rel="runs/${ck#*/runs/}"
  if [ ! -f "$rel" ]; then echo "FAIL $tag: local checkpoint $rel not found"; bad=$((bad+1)); continue; fi
  loc=$(sha256sum "$rel" | cut -d' ' -f1)
  st="OK"
  [ "$rows" = "$WANT" ] || { st="FAIL(rows=$rows want=$WANT)"; }
  [ "$dg" = "$loc" ]    || { st="$st FAIL(sha meta=${dg:0:12} local=${loc:0:12})"; }
  echo "$dg" >> /tmp/.gpu2_digests.$$
  printf '%-26s rows=%-6s ckpt=%-44s sha=%s  %s\n' "$tag" "$rows" "$rel" "${dg:0:12}" "$st"
  case "$st" in OK) ok=$((ok+1));; *) bad=$((bad+1));; esac
done
n=$(wc -l < /tmp/.gpu2_digests.$$); u=$(sort -u /tmp/.gpu2_digests.$$ | wc -l)
rm -f /tmp/.gpu2_digests.$$
echo "---- $ok ok, $bad bad; $u distinct ckpt_sha256 over $n tags (must be equal)"
[ "$bad" -eq 0 ] && [ "$u" = "$n" ]
