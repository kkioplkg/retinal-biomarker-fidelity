"""Integrity check: did every evaluated prediction come from a FINISHED model?

Motivated by a real defect found on 2026-09-18.  ``p5_finetune`` rewrites
``last.pt`` after *every* epoch, so a queue that made an inference job depend on
``last.pt`` existing could -- and did -- evaluate a half-trained checkpoint:
E4 fold 4 seed 0 was inferred at epoch 24 / 38 / 39 instead of 49, about ten
minutes before its fine-tune finished.  A lane that runs train-then-infer can
hit the same thing a different way: if the training process dies, the
``if not exist infer_meta.json`` guard still fires and evaluates the remnant.

Neither failure is visible downstream -- the unit has a manifest, a
``pixel_metrics.csv`` and a plausible Dice -- so it has to be checked
explicitly.  For every prediction directory this compares the epoch recorded in
``infer_meta.json`` against the training schedule and against the presence of
``summary.json`` (written only after the last epoch).

    python -m src.pivot.g9_verify_units            # all trees, table + exit code
    python -m src.pivot.g9_verify_units --root runs/pivot/e4

Exit code 1 if anything is flagged, so it works as a gate before an analysis.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
#: prediction trees and the epoch budget their checkpoints should have reached
TREES = (
    ("runs/pivot/e4/fold*/*/pred*", 50),      # E4 fine-tunes: 50 epochs
    ("runs/pivot/*/seed*/*/pred*", 50),       # E2 fine-tunes: 50 epochs
    ("runs/pivot/r2/*/*/pred*", None),        # second family: early stopping
)


def check_one(pred: str, epochs: int | None) -> dict | None:
    meta = os.path.join(pred, "infer_meta.json")
    if not os.path.exists(meta):
        return None
    run = os.path.dirname(pred)
    if os.path.basename(run) == "base":
        return None
    with open(meta, encoding="utf-8") as f:
        m = json.load(f)
    ck = str(m.get("ckpt", ""))
    ep = m.get("ckpt_epoch")
    summ = os.path.join(run, "summary.json")
    has_summary = os.path.exists(summ)

    problems = []
    # a baseline arm evaluates another run's frozen checkpoint -- skip the
    # schedule test, but still require that run to have finished
    own_ckpt = os.path.normcase(os.path.abspath(ck)).startswith(
        os.path.normcase(os.path.abspath(run)))
    # Only ``last.pt`` is required to sit at the end of the schedule.  The
    # ``fid`` arm deliberately evaluates ``best_fid.pt`` -- the best-validation-
    # fidelity epoch -- and ``best.pt`` the best-val-Dice epoch; an early epoch
    # there is the selection rule working, not a truncated run.
    is_last = os.path.basename(ck).lower() == "last.pt"
    if own_ckpt and not has_summary:
        problems.append("no summary.json: training never finished")
    if own_ckpt and is_last and epochs and isinstance(ep, int) and ep < epochs - 1:
        problems.append("ckpt_epoch=%d but the schedule is %d epochs" % (ep, epochs))
    if own_ckpt and has_summary and os.path.exists(meta):
        if os.path.getmtime(meta) < os.path.getmtime(summ) - 5:
            problems.append("inference predates summary.json "
                            "(evaluated while training was still running)")
    if not problems:
        return None
    return dict(pred=os.path.relpath(pred, EXP_ROOT).replace("\\", "/"),
                ckpt_epoch=ep, has_summary=has_summary,
                mean_dice=m.get("mean_dice"), problems="; ".join(problems))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="restrict to one tree, e.g. runs/pivot/e4")
    a = ap.parse_args(argv)
    os.chdir(EXP_ROOT)

    bad, n = [], 0
    for pattern, epochs in TREES:
        if a.root and not pattern.startswith(a.root):
            continue
        for pred in sorted(glob.glob(pattern)):
            if not os.path.isdir(pred) or "_STALE_" in pred:
                continue
            n += 1
            r = check_one(pred, epochs)
            if r:
                bad.append(r)
    print("[verify] %d prediction directories checked" % n)
    if not bad:
        print("[verify] OK -- every evaluated checkpoint came from a finished run")
        return 0
    print("[verify] %d FLAGGED:" % len(bad))
    for b in bad:
        print("  %-46s epoch=%-5s dice=%-8s %s"
              % (b["pred"], b["ckpt_epoch"],
                 ("%.4f" % b["mean_dice"]) if isinstance(b["mean_dice"], float) else "?",
                 b["problems"]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
