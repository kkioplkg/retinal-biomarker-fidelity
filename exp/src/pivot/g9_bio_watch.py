"""CPU biomarker stage for the review-point-9 second-family predictions.

Polls every prediction directory under ``runs/pivot/r2/<model>/<dataset>/pred``
and runs ``src.pivot.p5_eval bio`` on each one as soon as its inference is
finished (``infer_meta.json`` present, ``bio.csv`` absent).  Same estimator
path as every internal arm -- ``compute_all(fd_rotations=5)`` with the optic
disc detected from the fundus image alone -- so the C1 audit can be recomputed
on the second family with no analysis change at all.

    python -m src.pivot.g9_bio_watch --procs 8 --max-hours 14
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
R2_ROOT = os.path.join("runs", "pivot", "r2")
#: directories that are scaffolding, never arms
SKIP_DIRS = {"logs", "_smoke", "_probe", "_selftest", "_stage"}


def units():
    """(dataset, tag, root) for every r2 prediction directory on disk.

    A directory name may carry a ``_s<k>`` seed suffix (``fives_s1``); the
    biomarker stage must still be pointed at the bare dataset (``fives``),
    while the tag -- which is only a cache key -- keeps the suffix so two
    seeds never share cached per-image measurements.
    """
    out = []
    if not os.path.isdir(R2_ROOT):
        return out
    for model in sorted(os.listdir(R2_ROOT)):
        if model in SKIP_DIRS or not os.path.isdir(os.path.join(R2_ROOT, model)):
            continue
        mdir = os.path.join(R2_ROOT, model)
        for name in sorted(os.listdir(mdir)):
            root = os.path.join(mdir, name, "pred")
            if not os.path.isdir(root):
                continue
            m = re.match(r"^(.*?)_s(\d+)$", name)
            ds = m.group(1) if m else name
            out.append((ds, "r2_%s_%s" % (model, name), root))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--poll", type=int, default=180)
    ap.add_argument("--max-hours", type=float, default=14.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(EXP_ROOT)

    t0 = time.time()
    seen_fail = set()
    while True:
        pending = 0
        for ds, tag, root in units():
            done = os.path.join(root, "bio.csv")
            if os.path.exists(done) or tag in seen_fail:
                continue
            if not os.path.exists(os.path.join(root, "infer_meta.json")):
                pending += 1
                continue
            print("[bio-watch] %s  %s  ->  %s" % (time.strftime("%H:%M:%S"),
                                                  tag, root), flush=True)
            rc = subprocess.call(
                [sys.executable, "-m", "src.pivot.p5_eval", "bio",
                 "--dataset", ds, "--tag", tag, "--root", root,
                 "--procs", str(a.procs)], cwd=EXP_ROOT)
            if rc != 0 or not os.path.exists(done):
                print("[bio-watch] FAILED %s rc=%s" % (tag, rc), flush=True)
                seen_fail.add(tag)
            else:
                try:
                    with open(os.path.join(root, "infer_meta.json"),
                              encoding="utf-8") as f:
                        m = json.load(f)
                    print("[bio-watch] OK %s  dice=%.4f cldice=%.4f"
                          % (tag, m.get("mean_dice", float("nan")),
                             m.get("mean_cldice", float("nan"))), flush=True)
                except Exception:                                # noqa: BLE001
                    print("[bio-watch] OK %s" % tag, flush=True)
        if a.once:
            break
        if (time.time() - t0) / 3600.0 > a.max_hours:
            print("[bio-watch] max-hours reached; exiting", flush=True)
            break
        time.sleep(a.poll)
    print("[bio-watch] done. failures=%s" % sorted(seen_fail), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
