#!/usr/bin/env bash
# Verify that everything on the LAN node has a local counterpart, archive the
# node's logs, and -- ONLY when explicitly asked and only when nothing is
# missing -- delete /home/user/medical1 (tree + venv) from the node.
#
#   bash exp/remote/lan_cleanup.sh            # (1) + (2): check + pull logs.  SAFE.
#   bash exp/remote/lan_cleanup.sh --check    # same thing, explicit
#   bash exp/remote/lan_cleanup.sh --apply    # (1) + (2) + (3) DESTRUCTIVE
#
# --apply additionally requires typing YES at a prompt, and refuses outright if
# the check found anything at all.  `rm -rf /home/user/medical1` removes the
# venv as well and cannot be undone: the node would have to be bootstrapped from
# scratch (remote/lan_push_data.sh + a fresh venv) to run anything again.
#
# ---------------------------------------------------------------- what (1) does
# It builds ONE manifest on the node -- `size<TAB>path` for every file under
#
#     results/ runs/rigr/ runs/repair/ runs/rigr_data/ runs/rigr_models/
#     runs/rigr_head/ runs_remote/logs/
#
# -- and compares it against the local tree.  Two tiers:
#
#   HARD (a mismatch blocks the delete): everything the coordinator named --
#       results/*.{csv,json,md}
#       per_image.csv and summary.json anywhere under runs/rigr, runs/repair
#       every file inside an edges/ mask/ mask_before/ directory under those two
#     Each must exist locally AND have the same byte size.
#
#   NODE-ONLY (also blocks, by default): any other file under those roots with
#     no local counterpart at all.  This tier exists because the delete is
#     irreversible and the named HARD set does not cover, for instance, the
#     s4 step logs or a model file that was fitted on the node and never pulled
#     (runs/rigr_models/stare/seed0/geom_scale.json was exactly that case).
#     Pass --allow-node-only to downgrade this tier to a warning.
#
# Quarantine copies live only on the local side (risk_c1bridge,
# risk_etascale_pre0906, ...), so a node->local walk never sees them; the
# QUARANTINE_RE below is belt-and-braces in case one is ever created remotely.
#
# ---------------------------------------------------------------- what (2) does
# Pulls the node's logs so they survive the delete:
#     runs_remote/logs -> exp/logs_lan/            (as specified)
#     runs/s4_logs     -> exp/logs_lan/s4_logs/    ) node-only extras that would
#     runs_remote/repro-> exp/logs_lan/repro/      ) otherwise be destroyed
# The two extras are skipped without failing if they do not exist.

set -uo pipefail

SSH_HOST="user@LAN_HOST"
# -n keeps ssh from reading this script's stdin, so a piped confirmation (or a
# heredoc) cannot be swallowed by one of the remote calls.
SSH_OPTS=(-n -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
REMOTE_DIR="/home/user/medical1/exp"
DELETE_TARGET="/home/user/medical1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DEST="$EXP_DIR/logs_lan"

ROOTS=(results runs/rigr runs/repair runs/rigr_data runs/rigr_models
       runs/rigr_head runs_remote/logs)
QUARANTINE_RE='_etascale_pre[0-9]+|_c1bridge|_bdfix[0-9]+|_ooffix[0-9]+|_sigma_pre[0-9]+'

MODE="check"
ALLOW_NODE_ONLY=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --apply) MODE="apply" ;;
    --check) MODE="check" ;;
    --allow-node-only) ALLOW_NODE_ONLY=1 ;;
    --yes) ASSUME_YES=1 ;;
    *) echo "unknown argument: $arg" >&2
       echo "usage: lan_cleanup.sh [--check|--apply] [--yes] [--allow-node-only]" >&2
       exit 2 ;;
  esac
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
MANIFEST="$WORK/manifest.tsv"

echo "============================================================"
echo "[cleanup] mode=$MODE  node=$SSH_HOST"
echo "[cleanup] remote=$REMOTE_DIR   local=$EXP_DIR"
echo "============================================================"

# -------------------------------------------------------------------- step (1)
echo
echo "[cleanup] (1) building the node manifest ..."
REMOTE_FIND=$(cat <<EOF
cd "$REMOTE_DIR" || exit 3
echo "### RUNNING"
ps -e -o cmd | grep -c '[p]ython -m'
echo "### INVENTORY"
for r in ${ROOTS[*]}; do
  if [ -d "\$r" ]; then
    n=\$(find "\$r" -type f ! -path '*__pycache__*' ! -name '*.pyc' | wc -l)
    b=\$(find "\$r" -type f ! -path '*__pycache__*' ! -name '*.pyc' -printf '%s\n' | awk '{s+=\$1} END{printf "%.0f", s+0}')
    printf '%s\t%s\t%s\n' "\$r" "\$n" "\$b"
  else
    printf '%s\t%s\t%s\n' "\$r" "ABSENT" "0"
  fi
done
echo "### MANIFEST"
for r in ${ROOTS[*]}; do
  [ -d "\$r" ] && find "\$r" -type f ! -path '*__pycache__*' ! -name '*.pyc' -printf '%s\t%p\n'
done
echo "### SIZES"
du -sb "$DELETE_TARGET" 2>/dev/null
df -B1 --output=avail /home | tail -1
EOF
)
ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$REMOTE_FIND" > "$WORK/remote.txt" 2>"$WORK/remote.err"
rc=$?
if [ $rc -ne 0 ]; then
  echo "[cleanup][FATAL] ssh/enumeration failed (rc=$rc)" >&2
  cat "$WORK/remote.err" >&2
  exit 1
fi

sed -n '/^### MANIFEST$/,/^### SIZES$/p' "$WORK/remote.txt" \
  | sed '1d;$d' > "$MANIFEST"

RUNNING=$(sed -n '/^### RUNNING$/{n;p;}' "$WORK/remote.txt")
echo "[cleanup] python jobs still running on the node: ${RUNNING:-?}"

echo
echo "[cleanup] node inventory (files, bytes):"
sed -n '/^### INVENTORY$/,/^### MANIFEST$/p' "$WORK/remote.txt" | sed '1d;$d' \
  | awk -F'\t' '{printf "  %-22s %8s files  %14s bytes\n", $1, $2, $3}'
echo "[cleanup] manifest rows: $(wc -l < "$MANIFEST")"

# -------------------------------------------------------------------- step (2)
# Done BEFORE the comparison on purpose: the node's logs have no counterpart in
# the local exp/ tree until they are pulled, and the comparison below resolves
# runs_remote/logs/<f> against logs_lan/<f>.
echo
echo "[cleanup] (2) archiving the node's logs -> $LOG_DEST"
mkdir -p "$LOG_DEST"
pull_dir() {  # pull_dir <remote subpath> <local dir> ; missing remote dir is fine
  local sub="$1" dest="$2"
  if ssh "${SSH_OPTS[@]}" "$SSH_HOST" "test -d '$REMOTE_DIR/$sub'"; then
    mkdir -p "$dest"
    ssh "${SSH_OPTS[@]}" "$SSH_HOST" \
      "cd '$REMOTE_DIR/$sub' && tar czf - ." | tar xzf - -C "$dest"
    echo "  $sub -> $dest  ($(find "$dest" -maxdepth 1 -type f | wc -l) files)"
  else
    echo "  $sub: not present on the node, skipped"
  fi
}
pull_dir "runs_remote/logs"  "$LOG_DEST"
pull_dir "runs/s4_logs"      "$LOG_DEST/s4_logs"
pull_dir "runs_remote/repro" "$LOG_DEST/repro"

echo
echo "[cleanup] comparing against the local tree ..."
python - "$MANIFEST" "$EXP_DIR" "$QUARANTINE_RE" "$WORK" <<'PY'
import os, re, sys

manifest, exp_dir, quar_re, work = sys.argv[1:5]
quar = re.compile(quar_re)

HARD_RESULT = re.compile(r'^results/[^/]+\.(csv|json|md)$')
HARD_NAMED  = re.compile(r'^runs/(rigr|repair)/.*/(per_image\.csv|summary\.json)$')
HARD_DIRS   = re.compile(r'^runs/(rigr|repair)/.*/(edges|mask|mask_before)/[^/]+$')

# Throwaway probe output, never a result: the GPU-timing probe and the two
# EVAPORE OOM probes of 2026-09-15 wrote runs/repair/_probe/<name>/ on the node
# and were deliberately never pulled.  They match HARD_NAMED by shape only.
SCRATCH = re.compile(r'^runs/repair/_probe/')

# The node's job logs are archived to exp/logs_lan/, not to exp/runs_remote/.
LOGMAP = re.compile(r'^runs_remote/logs/(.+)$')

def is_hard(p):
    return bool(HARD_RESULT.match(p) or HARD_NAMED.match(p) or HARD_DIRS.match(p))

def local_path(p):
    m = LOGMAP.match(p)
    rel = ('logs_lan/' + m.group(1)) if m else p
    return os.path.join(exp_dir, *rel.split('/'))

missing_hard, mismatch_hard, node_only, mismatch_soft, scratch = [], [], [], [], []
n_hard = n_soft = n_skip = 0

with open(manifest, encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line or '\t' not in line:
            continue
        size_s, path = line.split('\t', 1)
        path = path.replace('\\', '/').lstrip('./')
        # A quarantine suffix is a LOCAL rename, so a node->local walk never
        # needs the exclusion; a node path that carries one is still real data
        # (the lodo_*_bdfix / _ooffix model variants pushed at bootstrap) and is
        # verified like any other file -- just in the soft tier, never HARD.
        quarantined = bool(quar.search(path))
        if quarantined:
            n_skip += 1
        if SCRATCH.match(path):
            scratch.append(path)
            continue
        try:
            size = int(size_s)
        except ValueError:
            continue
        local = local_path(path)
        hard = is_hard(path) and not quarantined
        n_hard += hard
        n_soft += (not hard)
        if not os.path.exists(local):
            (missing_hard if hard else node_only).append(path)
        else:
            lsz = os.path.getsize(local)
            if lsz != size:
                (mismatch_hard if hard else mismatch_soft).append(
                    '%s  node=%d local=%d' % (path, size, lsz))

def dump(title, items, limit=60):
    print('  %-34s %d' % (title, len(items)))
    for it in items[:limit]:
        print('      %s' % it)
    if len(items) > limit:
        print('      ... and %d more' % (len(items) - limit))

print('  checked HARD files                 %d' % n_hard)
print('  checked other files                %d' % n_soft)
print('  of which quarantine-suffixed       %d (checked, but soft tier)' % n_skip)
print()
dump('MISSING (hard, blocks)', missing_hard)
dump('SIZE MISMATCH (hard, blocks)', mismatch_hard)
dump('NODE-ONLY (no local copy)', node_only)
dump('SIZE MISMATCH (other, warn)', mismatch_soft)
dump('SCRATCH waived (probe output)', scratch)

blocking = len(missing_hard) + len(mismatch_hard)
with open(os.path.join(work, 'verdict.txt'), 'w', encoding='utf-8') as f:
    f.write('%d %d %d %d\n' % (blocking, len(node_only), len(mismatch_soft),
                               n_hard))
PY
pyrc=$?
if [ $pyrc -ne 0 ] || [ ! -f "$WORK/verdict.txt" ]; then
  echo "[cleanup][FATAL] comparison step failed" >&2
  exit 1
fi
read -r BLOCKING NODE_ONLY SOFT_MISMATCH N_HARD < "$WORK/verdict.txt"

# ------------------------------------------------------------------- verdict
DU_LINE=$(sed -n '/^### SIZES$/,$p' "$WORK/remote.txt" | sed '1d' | head -1)
AVAIL=$(sed -n '/^### SIZES$/,$p' "$WORK/remote.txt" | sed '1d' | tail -1)
DEL_BYTES=$(echo "$DU_LINE" | awk '{print $1}')
human() { awk -v b="${1:-0}" 'BEGIN{u="B KiB MiB GiB TiB";split(u,a," ");i=1;
  while(b>=1024&&i<5){b/=1024;i++} printf "%.2f %s", b, a[i]}'; }

echo
echo "============================================================"
echo "[cleanup] VERDICT"
echo "  blocking problems (missing/mismatched hard files): $BLOCKING"
echo "  node-only files with no local copy:                $NODE_ONLY"
echo "  size mismatches outside the hard set (warn):       $SOFT_MISMATCH"
echo "  would delete: $DELETE_TARGET  = $(human "$DEL_BYTES") ($DEL_BYTES bytes)"
echo "  /home free before:              $(human "$AVAIL") ($AVAIL bytes)"
echo "============================================================"

GATE=$BLOCKING
if [ "$ALLOW_NODE_ONLY" -eq 0 ]; then
  GATE=$((GATE + NODE_ONLY))
fi

if [ "$MODE" != "apply" ]; then
  echo
  if [ "$GATE" -eq 0 ]; then
    echo "[cleanup] CHECK PASSED -- nothing missing.  Re-run with --apply to delete."
  else
    echo "[cleanup] CHECK FAILED -- $GATE item(s) would be lost.  Delete refused."
  fi
  exit $([ "$GATE" -eq 0 ] && echo 0 || echo 1)
fi

# -------------------------------------------------------------------- step (3)
if [ "$GATE" -ne 0 ]; then
  echo "[cleanup][REFUSED] $GATE item(s) have no verified local copy; not deleting." >&2
  exit 1
fi
if [ "${RUNNING:-1}" != "0" ]; then
  echo "[cleanup][REFUSED] $RUNNING python job(s) still running on the node." >&2
  exit 1
fi

echo
echo "[cleanup] ABOUT TO RUN: rm -rf $DELETE_TARGET   (tree + venv, irreversible)"
if [ "$ASSUME_YES" -eq 1 ]; then
  echo "[cleanup] --yes given: proceeding without the prompt."
else
  printf "[cleanup] type YES to proceed: "
  read -r CONFIRM
  if [ "$CONFIRM" != "YES" ]; then
    echo "[cleanup] aborted by operator."
    exit 1
  fi
fi

ssh "${SSH_OPTS[@]}" "$SSH_HOST" "rm -rf '$DELETE_TARGET'"
rc=$?
echo "[cleanup] rm rc=$rc"

ssh "${SSH_OPTS[@]}" "$SSH_HOST" "
  if [ -e '$DELETE_TARGET' ]; then echo 'STILL_PRESENT'; else echo 'GONE'; fi
  df -B1 --output=avail /home | tail -1
" > "$WORK/after.txt"
STATE=$(head -1 "$WORK/after.txt")
AVAIL_AFTER=$(tail -1 "$WORK/after.txt")
echo "[cleanup] $DELETE_TARGET: $STATE"
echo "[cleanup] /home free after: $(human "$AVAIL_AFTER") ($AVAIL_AFTER bytes)"
echo "[cleanup] freed: $(human $((AVAIL_AFTER - AVAIL)))"
[ "$STATE" = "GONE" ] || exit 1
echo "[cleanup] done.  The node now has no venv and no tree: re-bootstrap with"
echo "          remote/lan_push_data.sh + a fresh python3.12 venv before reuse."
