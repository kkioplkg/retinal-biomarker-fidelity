#!/usr/bin/env bash
# Pre-build the FOV mask cache for an external dataset, sharded over N cores.
#
#   bash remote/gpu2_fov_prebuild.sh odir5k 24
#
# Why this exists: `src.data.external.ensure_fov` writes
# data/external/<ds>/fov/<image_id>.png lazily, from inside the worker that
# first needs it. With ~14 concurrent E3 workers over the same 6,392 images
# that is a write race on the same file, so the cache MUST be complete before
# any driver starts. `check_external --fov` does it single-threaded (~35 min
# for ODIR on one core); this node has 96 cores, so shard by index modulo --
# each shard writes a disjoint set of files, no lock needed.

set -uo pipefail
ROOT="${ROOT:-/path/to/workdir/medical1}"
EXP="$ROOT/exp"
PY="$ROOT/venv/bin/python"
DS="${1:?usage: gpu2_fov_prebuild.sh <dataset> [shards]}"
N="${2:-24}"

cd "$EXP"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p runs/gpu2_e3

for i in $(seq 0 $((N - 1))); do
  "$PY" - "$DS" "$i" "$N" <<'PYEOF' >> "runs/gpu2_e3/fov_${DS}.log" 2>&1 &
import os, sys, time
sys.path.insert(0, os.getcwd())
from src.data.external import load_external, ensure_fov
ds, shard, nshard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
recs = load_external(ds)
mine = [r for k, r in enumerate(recs) if k % nshard == shard]
t0, n = time.time(), 0
for r in mine:
    if not os.path.exists(r["fov_path"]):
        ensure_fov(r); n += 1
print("shard %d/%d: %d of %d built in %.1fs" % (shard, nshard, n, len(mine), time.time() - t0), flush=True)
PYEOF
done
wait
echo "[fov] $DS done: $(find "data/external/$DS/fov" -type f | wc -l) masks"
