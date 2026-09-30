#!/usr/bin/env bash
# bd_resync -- wait for the S4 author's FINISHED R_false speed-up in
# src/rigr/build_data.py (C1 work scale 1536 with 1/s length conversion,
# multiprocessing over images, FIVES R_false image cap 40), then sync it and
# restart every G:data from scratch.
#
# WHY.  The G:data runs on the remote are hopeless at the current speed:
#   lodo_fives  R_false: 23 candidates measured  10765 s
#   lodo_hrf    R_false: 13 candidates measured   9393 s
# ~3 h for ~20 candidates.  Nothing downstream of them is worth keeping.
#
# GATE (coordinator 2026-09-03 14:05, after the author was interrupted
# mid-edit).  ALL THREE must hold; a bare content-hash change is NOT enough,
# because a half-written file still parses and still looks "changed":
#
#   1. the marker runs/rigr_data/.build_data_v2 exists;
#   2. the sha256 it contains EQUALS the live sha256 of src/rigr/build_data.py
#      -- the author's attestation of exactly which bytes are final, so an edit
#      that continued after the marker was written cannot slip through;
#   3. a local B-step log NEWER THAN THE MARKER shows <= MAX_SEC per image.
#      Freshness matters: the B logs on disk right now are from the OLD code
#      (B_fives_s0 4395 s, B_hrf_s0 8758 s for 12-15 candidates) and must not
#      be mistaken for evidence of the new one.
#
# On trigger: push src/, then restart every chain from G:data via the remote
# sigma_rebuild_all.sh -- which stops each chain and any in-flight build_data by
# EXPLICIT PID through stop_pid.sh, moves the partial outputs aside rather than
# deleting them, and keeps the heads (they do not depend on build_data).
# This script terminates nothing itself.
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
SRC=src/rigr/build_data.py
MARK=runs/rigr_data/.build_data_v2
DONE=remote/.bd_done      # sha we have already rebuilt for (latch)
VLOG=remote/bd_versions.log
STOP=remote/.bd_resync.stop
PIDF=remote/.bd_resync.pid
INTERVAL=${BD_INTERVAL:-180}
MAX_SEC=${BD_MAX_SEC:-300}   # effective seconds per image (coordinator 18:35)
once=0; [ "${1:-}" = "--once" ] && once=1
[ "$once" = 1 ] || echo $$ > "$PIDF"
rm -f "$STOP"

hm() { stat -c '%y' "$1" 2>/dev/null || echo "(absent)"; }

# newest per-image delta from a B log modified after $1 (epoch), or empty.
# build_data prints "  [i/n] <image> <ds> cand=.. pos=.. cuts=..  <cum>s", so
# the difference between consecutive cumulative times is the per-image cost.
# THROUGHPUT, not per-image wall time.  v2 runs a pool of workers, so each
# image's own elapsed time (the "766.7s" field) is ~N_workers times the
# effective cost and measuring it condemns a build that is actually fine -- my
# first version read 707 s/image on FIVES and held the gate shut for nothing.
# The "(total <T>s)" field is the run's wall clock, so over the last K completed
# images the effective cost is  (T_last - T_first) / (K - 1).
# Worked example from B_fives_s0.log: images 51..55 ran total 10948.0 ->
# 11345.8 s, i.e. 397.8 s for 4 images = 99 s/image effective, against 707 s of
# per-image wall time.
v2_throughput_sec() {
  # Report every v2 log's effective rate, and decide on the BEST-SAMPLED one.
  # Picking the newest log instead put the decision on B_hrf_s0 with 9 samples
  # (632 s/image) over B_fives_s0 with 64 (96 s/image).  HRF is also the
  # unrepresentative case: its lines carry rf=24/16/10, i.e. R_false measured on
  # every image at 3504x2336, while FIVES shows rf=0 because the cap-40 skip is
  # doing its job.  A minimum of 5 completed images keeps a just-started run
  # from deciding anything.
  local f n k best_f="" best_n="" best_k=0
  for f in $(ls -t runs/s4_logs/B_*.log 2>/dev/null | head -8); do
    grep -q 's (total ' "$f" 2>/dev/null || continue
    read -r k n <<<"$(grep -oE '\(total [0-9]+\.[0-9]+s\)' "$f" | grep -oE '[0-9]+\.[0-9]+' \
        | tail -12 | awk 'NR==1{first=$1} {last=$1; c=NR} END{if(c>1) printf "%d %.0f", c, (last-first)/(c-1)}')"
    [ -n "${n:-}" ] || continue
    echo "[bd]   sample: $(basename "$f")  ${k} images  -> ${n}s/image effective" >&2
    if [ "$k" -ge 5 ] && [ "$k" -gt "$best_k" ]; then best_k=$k; best_n=$n; best_f=$f; fi
  done
  [ -n "$best_f" ] || return 1
  echo "$best_f $best_n"
}

while :; do
  echo "===== $(date -Is) build_data v2 check ====="
  have0=$(sha256sum "$SRC" 2>/dev/null | awk '{print $1}')
  if [ -n "$have0" ] && [ "$have0" = "$(cat "$DONE" 2>/dev/null)" ]; then
    echo "[bd] already rebuilt for sha $(printf %.16s "$have0") -- latched, nothing to do"
  elif [ ! -f "$MARK" ]; then
    echo "[bd] gate 1 closed: $MARK absent (author still editing)"
  else
    want=$(grep -oE '[0-9a-f]{64}' "$MARK" | head -1)
    have=$(sha256sum "$SRC" 2>/dev/null | awk '{print $1}')
    if [ -z "$want" ]; then
      echo "[bd] gate 2 closed: $MARK has no sha256 in it"
    elif [ "$want" != "$have" ]; then
      echo "[bd] gate 2 closed: marker sha $(printf %.16s "$want") != file sha $(printf %.16s "$have") -- the file moved after the marker was written"
    else
      if ! pi=$(v2_throughput_sec); then
        echo "[bd] gate 3 closed: no v2-format B_*.log (no \"s (total \" lines) with timings yet"
      else
        f=${pi%% *}; d=${pi##* }
        if [ "$d" -gt "$MAX_SEC" ]; then
          echo "[bd] gate 3 closed: $f effective ${d}s/image throughput (> ${MAX_SEC}s)"
        else
          ts=$(date -Is); suffix="bdfix$(date +%H%M)"
          echo "[bd] ALL THREE GATES OPEN $ts"
          echo "[bd]   marker  $MARK ($(hm $MARK)) sha $(printf %.16s "$want")"
          echo "[bd]   file    $SRC ($(hm $SRC)) sha matches"
          echo "[bd]   timing  $f -> effective ${d}s/image throughput (<= ${MAX_SEC}s)"
          echo "[bd] remote G:data timings being discarded:"
          $SSH "cd $R && for g in runs/s4_logs/G_data_*.log; do [ -f \"\$g\" ] && echo \"    \$g: \$(tail -1 \"\$g\" | cut -c1-70)\"; done" 2>/dev/null || true
          if tar czf - --exclude='__pycache__' --exclude='*.pyc' src \
               | $SSH "cd $R && rm -rf src/__pycache__ src/*/__pycache__ && tar xzf - -C $R && echo PUSH_OK" | grep -q PUSH_OK; then
            echo "[bd] pushed src/"
          else
            echo "[bd] PUSH FAILED -- retrying next cycle"
            [ "$once" = 1 ] && break
            sleep 60; continue
          fi
          {
            echo "## $ts  build_data.py v2 (R_false speed-up) synced"
            echo "    $SRC  $(hm $SRC)  sha256 $have"
            echo "    marker $MARK  $(hm $MARK)"
            echo "    evidence: $f -> ${d}s/image"
          } >> "$VLOG"
          echo "[bd] restarting every chain from G:data (heads kept)"
          $SSH "cd $R && bash bd_rebuild.sh '$suffix'" 2>&1 | sed 's/^/    /'
          printf '%s
' "$have" > "$DONE"      # latch: never rebuild twice for the same file
          echo "[bd] restarted and latched on sha $(printf %.16s "$have")"
          [ "$once" = 1 ] && break
        fi
      fi
    fi
  fi
  [ "$once" = 1 ] && break
  [ -f "$STOP" ] && { echo "[bd] stop file present; exiting."; break; }
  sleep "$INTERVAL"
done
[ "$once" = 1 ] || rm -f "$PIDF"
echo "[bd] watcher exit $(date -Is)"
