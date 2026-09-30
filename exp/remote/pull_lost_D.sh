#!/usr/bin/env bash
# Recover the three step-D outputs that finished remotely at 03:25:53 on
# 2026-09-06 but were never pulled before the box was shut down at 06:00.
#
# They are NOT in results_remote_final/final_remote_archive.tgz: that archive
# covers logs/ + exp/results + exp/runs/s4_logs, and these are under
# exp/runs/rigr/.  They are on the instance's persistent /path/to/workdir disk,
# so restarting the cloud instance and running this script recovers them.
#
#   bash remote/pull_lost_D.sh
#
# Idempotent, retries the flaky link, and verifies the row counts before
# declaring success.  Terminates nothing.
set -u
cd /e/Programming/research_ws/medical1/exp || exit 1
export PATH="/usr/bin:/bin:/mingw64/bin:/usr/local/bin:$PATH"
R=/path/to/workdir/medical1/exp
SSHOPT="-p SSH_PORT -o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15"
HOST=user@GPU_HOST
SPEC="runs/rigr/uniform/fives/seed0 runs/rigr/risk/hrf/seed0 runs/rigr/risk/fives/seed0"

for i in $(seq 1 40); do
  if timeout 900 ssh $SSHOPT "$HOST" "cd $R && tar czf - --exclude=__pycache__ $SPEC" > .lostD.tgz 2>/dev/null \
     && [ -s .lostD.tgz ] && tar tzf .lostD.tgz >/dev/null 2>&1; then
    tar xzf .lostD.tgz && rm -f .lostD.tgz && echo "[lostD] pulled on attempt $i" && break
  fi
  rm -f .lostD.tgz
  echo "[lostD] attempt $i failed (instance stopped? restart it in the cloud console first)"
  sleep 15
done

# The two RISK dirs were produced BEFORE the C_geom scale fix (DECISIONS.md
# 2026-09-06 12:30), so they are pre-fix artefacts: keep them for the
# supplementary "degenerate utility" record, but land them quarantined so they
# can never be mistaken for the live result or collide with the re-run.
for d in runs/rigr/risk/hrf/seed0 runs/rigr/risk/fives/seed0; do
  if [ -d "$d" ] && [ ! -d "${d}_etascale_pre0906" ]; then
    mv "$d" "${d}_etascale_pre0906" && echo "[lostD] quarantined pre-fix $d"
  fi
done

echo "[lostD] verification (expect 61 / 31 / 61 lines incl. header):"
fail=0
for p in "runs/rigr/uniform/fives/seed0 61" "runs/rigr/risk/hrf/seed0_etascale_pre0906 31" "runs/rigr/risk/fives/seed0_etascale_pre0906 61"; do
  set -- $p
  n=$(wc -l < "$1/per_image.csv" 2>/dev/null || echo 0)
  s=$([ -f "$1/summary.json" ] && echo yes || echo NO)
  if [ "$n" = "$2" ] && [ "$s" = yes ]; then printf "  OK   %-32s %s lines, summary yes\n" "$1" "$n"
  else printf "  FAIL %-32s %s lines (want %s), summary %s\n" "$1" "$n" "$2" "$s"; fail=1; fi
done
[ "$fail" = 0 ] && echo "[lostD] ALL THREE RECOVERED" || echo "[lostD] INCOMPLETE"
exit "$fail"
