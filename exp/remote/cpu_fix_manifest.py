"""Re-root a prediction manifest written on Windows so it resolves on Linux.

``src.seg.infer.predict_records`` writes ``image_path`` / ``label_path`` /
``fov_path`` / ``prob_path`` / ``mask_path`` as ABSOLUTE paths of the machine
that ran inference.  The E4 MAPLES-DR predictions are produced on the Windows
box (``E:\\Programming\\research_ws\\medical1\\exp\\...``) and measured on the
CPU node, where those paths do not exist -- and ``p5_eval.cmd_bio`` feeds
``manifest.mask_path`` straight to the reader, so a stale path is a hard error
(or, worse, a silent fallback to a wrong record).

This rewrites every path column: anything containing an ``exp`` component is
re-rooted onto this tree's ``exp/`` and normalised to forward slashes.  Columns
that are empty, NaN, or already resolve are left alone.  Idempotent: run it
again on a rewritten manifest and nothing changes.

    python exp/remote/cpu_fix_manifest.py runs/pivot/e4/maples/<tag>/manifest.csv
    python exp/remote/cpu_fix_manifest.py --all runs/pivot/e4/maples
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

PATH_COLS = ("image_path", "label_path", "label2_path", "fov_path",
             "prob_path", "mask_path")

EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
#: matches a leading drive/UNC/posix prefix up to and including an ``exp`` component
_SPLIT = re.compile(r"^.*?[\\/]exp[\\/]", re.IGNORECASE)


def reroot(val: str) -> str:
    v = (val or "").strip()
    if not v or v.lower() == "nan":
        return val
    m = _SPLIT.match(v)
    if not m:
        return val                      # already relative, or unrecognised
    rel = v[m.end():].replace("\\", "/")
    # A remainder that still starts with a separator (or carries doubled ones)
    # would make os.path.join DISCARD EXP_ROOT and return an absolute path that
    # silently does not exist -- collapse and strip before joining.
    rel = re.sub(r"/+", "/", rel).lstrip("/")
    if not rel:
        return val
    return os.path.join(EXP_ROOT, rel).replace("\\", "/")


def fix(path: str) -> tuple:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return (0, 0)
    fields = list(rows[0].keys())
    changed = 0
    for r in rows:
        for c in PATH_COLS:
            if c in r:
                new = reroot(r[c])
                if new != r[c]:
                    r[c] = new
                    changed += 1
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    # verification: every mask the bio step will open must exist now
    missing = [r["mask_path"] for r in rows
               if "mask_path" in r and not os.path.exists(r["mask_path"])]
    return (len(rows), len(missing))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--all", action="store_true",
                    help="treat each path as a directory of <tag>/manifest.csv")
    a = ap.parse_args(argv)

    targets = []
    for p in a.paths:
        if a.all:
            for tag in sorted(os.listdir(p)):
                m = os.path.join(p, tag, "manifest.csv")
                if os.path.exists(m):
                    targets.append(m)
        else:
            targets.append(p)

    bad = 0
    for m in targets:
        n, miss = fix(m)
        flag = "OK" if miss == 0 else "MISSING %d MASKS" % miss
        bad += miss
        print("[fix] %-70s %4d rows  %s" % (m, n, flag), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
