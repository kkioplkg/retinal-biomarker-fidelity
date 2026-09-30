"""Audit -- and optionally repair -- the sigma scale behind every ``macro_mae_*``.

``macro_mae`` is ``mean_b |B_pred - B_GT| / sigma_b``. DECISIONS.md 2026-09-03
10:20 moved sigma from ``gateA_biomarker_scales.csv`` (fitted over **all**
reference masks, test images included -- a leak) to
``gateA_biomarker_scales_train.csv`` (training split only), and
``biomarker_eval.GATE_A_SCALES_CSV`` now points at the train file.

Runs scored before that switch still carry macro-MAE on the old scale, and
nothing on disk records which scale a run used. That is invisible in the CSV and
fatal for comparability: a method on the old scale is not comparable with one on
the new scale, yet Tab.2 puts them in the same column. Found 2026-09-16 --
``geometric`` on drive/chasedb1/fives was on the old scale while every other
method (and geometric/hrf) was on the new one, which made ``geometric`` look
like the best baseline on CHASE_DB1 and FIVES.

Every per_image.csv stores the raw ``bio_gt_* / bio_before_* / bio_after_*``
columns, so macro-MAE can be recomputed exactly rather than re-run: the
recomputation reproduces the stored value to <1e-9 for every correctly-scored
run, which is what makes the repair safe (and is checked here before anything is
written).

CLI
---
    python -m src.eval.rescale_macro_mae            # audit only
    python -m src.eval.rescale_macro_mae --apply    # rewrite mismatched files
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Dict, Optional

import numpy as np

from .biomarker_eval import (GATE_A_SCALES_ALL_CSV, GATE_A_SCALES_TRAIN_CSV,
                             PRIMARY)
from .tables import EXP_ROOT, RUNS_DIR, _live_glob, _read_csv

PIPES = ("pvbm", "skan")


def _scales(path: str) -> Dict[str, Dict[str, float]]:
    s = _read_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    if s is None:
        return out
    for _, r in s.iterrows():
        key = str(r["dataset"]).lower().replace("_", "")
        out.setdefault(key, {})[str(r["biomarker"])] = float(r["sigma"])
    return out


def _macro(df, sg: Dict[str, float], stage: str):
    acc = []
    for b in PRIMARY:
        for p in PIPES:
            col = f"{b}_{p}"
            s = sg.get(col, np.nan)
            a, g = df.get(f"bio_{stage}_{col}"), df.get(f"bio_gt_{col}")
            if a is None or g is None or not np.isfinite(s) or s <= 0:
                continue
            acc.append(np.abs(a - g) / s)
    if not acc:
        return None
    return np.nanmean(np.vstack(acc), axis=0)


def audit(apply: bool = False):
    import pandas as pd

    train, allsc = _scales(GATE_A_SCALES_TRAIN_CSV), _scales(GATE_A_SCALES_ALL_CSV)
    stamp = time.strftime("%m%d_%H%M")
    paths = (_live_glob(os.path.join(RUNS_DIR, "repair", "*", "*", "seed*", "per_image.csv"))
             + _live_glob(os.path.join(RUNS_DIR, "rigr", "*", "*", "seed*", "per_image.csv")))
    rows = []
    for p in paths:
        df = _read_csv(p)
        if df is None or "macro_mae_before" not in df.columns:
            continue
        ds = str(df["dataset"].iloc[0]).lower()
        if ds not in train:
            continue
        rec_t, rec_a = _macro(df, train[ds], "before"), _macro(df, allsc.get(ds, {}), "before")
        stored = df["macro_mae_before"].to_numpy(dtype=float)
        d_t = np.nanmax(np.abs(stored - rec_t)) if rec_t is not None else np.nan
        d_a = np.nanmax(np.abs(stored - rec_a)) if rec_a is not None else np.nan
        verdict = ("train" if d_t < 1e-6 else
                   "ALL(leaked)" if d_a < 1e-6 else "unknown")
        rows.append(dict(run=os.path.relpath(p, EXP_ROOT), dataset=ds,
                         scale=verdict, max_dev_train=d_t, max_dev_all=d_a))
        if apply and verdict != "train":
            after_t = _macro(df, train[ds], "after")
            if rec_t is None or after_t is None:
                print(f"[rescale][skip] {p}: cannot recompute"); continue
            shutil.copy2(p, f"{p}.bak_sigma_{verdict}_{stamp}")
            df["macro_mae_before"] = rec_t
            df["macro_mae_after"] = after_t
            df["macro_mae_delta"] = after_t - rec_t
            df.to_csv(p, index=False)
            sp = os.path.join(os.path.dirname(p), "summary.json")
            if os.path.exists(sp):
                try:
                    blob = json.load(open(sp))
                    shutil.copy2(sp, f"{sp}.bak_sigma_{verdict}_{stamp}")
                    m = blob.get("means", {})
                    m["macro_mae_before"] = float(np.nanmean(rec_t))
                    m["macro_mae_after"] = float(np.nanmean(after_t))
                    m["macro_mae_delta"] = float(np.nanmean(after_t - rec_t))
                    blob["macro_mae_sigma_rescaled"] = {
                        "from": verdict, "to": "train", "when": stamp,
                        "why": "gateA sigma switched to the training-split table "
                               "(DECISIONS.md 2026-09-03 10:20); this run predated it"}
                    json.dump(blob, open(sp, "w"), indent=2)
                except Exception as e:
                    print(f"[rescale][warn] {sp}: {e}")
            print(f"[rescale] {os.path.relpath(p, EXP_ROOT)}: {verdict} -> train")

    t = pd.DataFrame(rows)
    bad = t[t.scale != "train"]
    print("\n%d run(s) scored; scale breakdown:" % len(t))
    print(t.groupby("scale").size().to_string())
    if len(bad):
        print("\nNOT on the canonical training-split sigma:")
        print(bad[["run", "dataset", "scale"]].to_string(index=False))
    else:
        print("\nAll runs are on the canonical training-split sigma.")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite macro_mae_* of mismatched runs (keeps .bak_sigma_*)")
    audit(**vars(ap.parse_args()))
