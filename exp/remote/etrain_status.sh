#!/usr/bin/env bash
# Health of one E:train job, read from its LOG only -- never from a process
# table, so this never identifies anybody's process by pattern.
#
#   etrain_status.sh <log tag> <ckpt path relative to exp/> [stall minutes]
#
# Prints exactly one token:
#   DONE          the checkpoint exists and is non-empty
#   FAILED:<rc>   an "rc=<non-zero>" line appeared AFTER the last start marker
#   STALL:<min>   no checkpoint, no rc line, log not advanced in <stall> minutes
#   RUNNING       otherwise
#
# Why it exists: the finisher used to poll only for the output file, so when
# evapore_hrf was OOM-killed at 17:23 on 2026-09-05 the watcher kept waiting on
# a dead job for 2.4 h.  Pool lines land every ~15 min and epoch lines every
# ~8 min, so a 90-minute silence is far outside normal.
set -u
R=/path/to/workdir/medical1/exp
RROOT=/path/to/workdir/medical1
tag="${1:?need a log tag}"
ck="${2:?need a checkpoint path}"
stall="${3:-90}"
L="$RROOT/logs/Etrain_${tag}.log"

[ -s "$R/$ck" ] && { echo DONE; exit 0; }
[ -f "$L" ] || { echo RUNNING; exit 0; }

# line number of the most recent launch marker, so a previous run's rc line
# (this log is appended across attempts) can never be read as this run's
last_start=$(awk '/=== (SOLO|queue: starting|AMP retrain) /{n=NR} END{print n+0}' "$L")
rcline=$(awk -v s="$last_start" 'NR>s && /rc=/{print}' "$L" | tail -1)
if [ -n "$rcline" ]; then
  rc=$(printf '%s' "$rcline" | sed -n 's/.*rc=\([0-9-]\{1,\}\).*/\1/p')
  [ "${rc:-1}" != "0" ] && { echo "FAILED:${rc:-unknown}"; exit 0; }
fi

age=$(( ( $(date +%s) - $(stat -c %Y "$L") ) / 60 ))
if [ "$age" -ge "$stall" ]; then echo "STALL:$age"; else echo RUNNING; fi
