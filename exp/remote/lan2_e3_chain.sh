#!/usr/bin/env bash
# Chain E3 waves on the LAN node: wait for any running lan2_e3_driver.sh to
# exit, then run the given job lists one after another, so the GPU never idles
# between datasets.
#
#   nohup setsid bash remote/lan2_e3_chain.sh 9 remote/lan2_jobs_messidor2.txt \
#         remote/lan2_jobs_aptos2019.txt > runs/lan_e3/chain.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="${ROOT:-/mnt/data/Programming/research_ws/medical1}"
cd "$ROOT/exp"
PAR="${1:?usage: lan2_e3_chain.sh <parallel> <jobs.txt> [jobs.txt ...]}"; shift
SELF=$$

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
wait_free() {
  while pgrep -f "lan2_e3_driver.sh" > /dev/null; do sleep 60; done
}

echo "[chain $(stamp)] waiting for the running wave to finish ..."
wait_free
for jf in "$@"; do
  echo "[chain $(stamp)] === starting wave: $jf (parallel=$PAR)"
  bash remote/lan2_e3_driver.sh "$jf" "$PAR" >> "runs/lan_e3/driver_$(basename "$jf" .txt).log" 2>&1
  echo "[chain $(stamp)] === wave finished: $jf"
done
echo "[chain $(stamp)] ALL WAVES DONE"
