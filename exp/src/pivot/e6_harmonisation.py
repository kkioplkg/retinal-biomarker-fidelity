"""E6 -- constant-offset harmonisation: what it fixes and what it cannot fix.

Part A (from P2, ``results/pivot/p2_calibration.csv``): in-domain vs LODO bias
removal for the per-dataset constant-offset calibrator.

Part B (new): the direct demonstration that a constant offset is **invisible to
any within-cohort classifier**.  We apply a single constant per
(dataset, biomarker) offset

    B_harm,i = B_pred,i - mean_j (B_pred,j - B_GT,j)

to every image of the cohort and re-run the P3 downstream protocol (logreg and
GBDT, 5-fold stratified CV x 3 repeats, macro one-vs-rest AUC of the pooled
out-of-fold probabilities, SEED = 0).  The AUC must be identical to the raw
predicted features to numerical precision -- logistic regression is fitted on
standardised features (a shift changes only the intercept) and a tree splits on
order statistics (a shift moves every threshold by the same amount).

This is the honest version of P2/P3's ``offset_per_ds`` row, whose small
residual differences came from the 5-fold cross-fitting of the offset, not from
the harmonisation itself.

CLI
---
    python -m src.pivot.e6_harmonisation
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS, PRIMARY_COLS_ZONEB, sigma_table

SEED = 0
N_REPEAT = 3


def macro_auc(y, proba, classes):
    from sklearn.metrics import roc_auc_score
    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() in (0, len(yy)):
            continue
        a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def cv_proba(X, y, kind, classes, n_splits=5):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    acc = np.zeros((len(y), len(classes)), float)
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            if kind == "logreg":
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
            else:
                m = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=150, learning_rate=0.08,
                    min_samples_leaf=10, l2_regularization=1.0,
                    early_stopping=False, random_state=SEED + rep)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            order = [list(m.classes_).index(c) for c in classes]
            acc[te] += p[:, order]
    return acc / N_REPEAT


# ------------------------------------------------------------- part A
def part_a(out_dir: str) -> pd.DataFrame:
    p2 = pd.read_csv(os.path.join(PIVOT_DIR, "p2_calibration.csv"))
    keep = ["biomarker", "dataset", "n",
            "bias_uncal", "bias_offset_per_ds", "bias_offset_lodo",
            "mae_uncal", "mae_offset_per_ds", "mae_offset_lodo",
            "red_offset_per_ds", "red_offset_lodo",
            "cov90_offset_per_ds", "cov90_offset_lodo",
            "width_offset_per_ds", "width_offset_lodo"]
    t = p2[[c for c in keep if c in p2.columns]].copy()
    t["abs_bias_uncal"] = t["bias_uncal"].abs()
    t["abs_bias_in"] = t["bias_offset_per_ds"].abs()
    t["abs_bias_lodo"] = t["bias_offset_lodo"].abs()
    t["bias_removed_in_frac"] = 1.0 - t["abs_bias_in"] / t["abs_bias_uncal"]
    t["bias_removed_lodo_frac"] = 1.0 - t["abs_bias_lodo"] / t["abs_bias_uncal"]
    t = t[t["biomarker"].isin(PRIMARY_COLS)]
    out = os.path.join(out_dir, "e6_offset_bias_removal.csv")
    t.to_csv(out, index=False)
    print("wrote", out, t.shape)
    return t


# ------------------------------------------------------------- part B
def part_b(out_dir: str) -> pd.DataFrame:
    # same input file and same row order as ``src/pivot/p3_downstream.py``, so
    # the ``auc_gt`` / ``auc_pred`` columns reproduce P3.1 exactly
    pred = pd.read_csv(os.path.join(PIVOT_DIR, "p2_predictions.csv"), low_memory=False)
    rows: List[dict] = []
    specs = [("fives", "all"), ("fives", "test"), ("hrf", "all")]

    for ds, split in specs:
        cols = list(PRIMARY_COLS)
        d = pred[pred["dataset"] == ds].copy()
        if split != "all":
            d = d[d["split"] == split]
        d = d[d["disease"].notna()].reset_index(drop=True)
        y = d["disease"].to_numpy()
        classes = sorted(pd.unique(y))

        Xp = d[[c + "__pred" for c in cols]].to_numpy(float)
        Xg = d[[c + "__gt" for c in cols]].to_numpy(float)
        Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
        Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0)
        # the single constant per (dataset, biomarker) offset
        offs = np.nanmean(Xp - Xg, axis=0)
        Xh = Xp - offs[None, :]

        for clf in ("logreg", "gbdt"):
            a_gt = macro_auc(y, cv_proba(Xg, y, clf, classes), classes)
            a_pred = macro_auc(y, cv_proba(Xp, y, clf, classes), classes)
            a_harm = macro_auc(y, cv_proba(Xh, y, clf, classes), classes)
            rows.append(dict(dataset=ds, split=split, clf=clf, n=len(y),
                             n_classes=len(classes),
                             auc_gt=a_gt, auc_pred=a_pred, auc_harmonised=a_harm,
                             abs_diff_harm_minus_pred=abs(a_harm - a_pred),
                             attenuation_gt_minus_pred=a_gt - a_pred,
                             offsets_sigma=";".join(
                                 f"{c}={o/sigma_table()[ds][c]:+.3f}"
                                 for c, o in zip(cols, offs))))
    t = pd.DataFrame(rows)
    out = os.path.join(out_dir, "e6_offset_invariance.csv")
    t.to_csv(out, index=False)
    print("wrote", out, t.shape)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    args = ap.parse_args()
    a = part_a(args.out_dir)
    b = part_b(args.out_dir)
    pd.set_option("display.width", 240)
    print(a.round(4).to_string(index=False))
    print(b[["dataset", "split", "clf", "n", "auc_gt", "auc_pred",
             "auc_harmonised", "abs_diff_harm_minus_pred"]].to_string(index=False))


if __name__ == "__main__":
    main()
