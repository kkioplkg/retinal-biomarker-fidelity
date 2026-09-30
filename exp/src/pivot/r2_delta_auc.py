"""R2 -- the reference-gap AUC difference, absolute and under both relative
denominators.

Reviewer point 15 (``review/user_cmig_review_20260918.md``)
----------------------------------------------------------
"Delete 'a fifth of the discrimination' and just write the Delta AUC."

The phrase is ambiguous because two different denominators are in circulation
and they disagree by roughly a factor of two:

    rel_of_auc     Delta AUC / AUC_reference
                   "how much of the reference AUC is given up"
    rel_of_excess  Delta AUC / (AUC_reference - 0.5)
                   "how much of the discrimination ABOVE CHANCE is given up"

This script computes the absolute difference and both ratios, each with the
same paired image-level bootstrap the manuscript already uses
(``src/pivot/p3_downstream.py::boot_ci``: the out-of-fold probability matrices
are fixed and whole images are resampled, so the two arms are always compared on
the same images), so the paper can quote the absolute number and, if it wants a
ratio at all, quote it with its denominator named.

Output: ``results/pivot/r2/r2_delta_auc.csv``.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_delta_auc
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS

SEED = 0
N_REPEAT = 3
N_SPLITS = 5
N_BOOT = 1000
PANELS = {"skan4": [c for c in PRIMARY_COLS if c.endswith("_skan")],
          "primary8": list(PRIMARY_COLS)}


def macro_auc(y, proba, classes) -> float:
    from sklearn.metrics import roc_auc_score
    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() in (0, len(yy)):
            continue
        a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def cv_proba(X, y, kind, classes):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    acc = np.zeros((len(y), len(classes)), float)
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(PIVOT_DIR,
                                                   "p2_predictions.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    ap.add_argument("--datasets", default="hrf,fives")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    pred = pd.read_csv(args.pred, low_memory=False)
    rows: List[dict] = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        d = pred[(pred["dataset"] == ds) & pred["disease"].notna()
                 ].reset_index(drop=True)
        if len(d) < 30:
            continue
        y = d["disease"].to_numpy()
        classes = sorted(pd.unique(y))
        for panel, cols in PANELS.items():
            Xg = np.nan_to_num(d[[c + "__gt" for c in cols]].to_numpy(float),
                               nan=0.0, posinf=0.0, neginf=0.0)
            Xp = np.nan_to_num(d[[c + "__pred" for c in cols]].to_numpy(float),
                               nan=0.0, posinf=0.0, neginf=0.0)
            for kind in ("logreg", "gbdt"):
                pg = cv_proba(Xg, y, kind, classes)
                pp = cv_proba(Xp, y, kind, classes)
                a_g, a_p = macro_auc(y, pg, classes), macro_auc(y, pp, classes)
                delta = a_g - a_p
                rel_auc = delta / a_g if a_g > 0 else float("nan")
                rel_exc = delta / (a_g - 0.5) if a_g > 0.5 else float("nan")

                rng = np.random.RandomState(SEED)
                n = len(y)
                dd, ra, re, gg, ppv = [], [], [], [], []
                for _ in range(N_BOOT):
                    idx = rng.randint(0, n, n)
                    if len(np.unique(y[idx])) < len(classes):
                        continue
                    bg = macro_auc(y[idx], pg[idx], classes)
                    bp = macro_auc(y[idx], pp[idx], classes)
                    if not (np.isfinite(bg) and np.isfinite(bp)):
                        continue
                    dd.append(bg - bp)
                    gg.append(bg)
                    ppv.append(bp)
                    if bg > 0:
                        ra.append((bg - bp) / bg)
                    if bg > 0.5:
                        re.append((bg - bp) / (bg - 0.5))

                def pc(v):
                    v = np.asarray(v, float)
                    return ((float(np.percentile(v, 2.5)),
                             float(np.percentile(v, 97.5))) if v.size
                            else (float("nan"), float("nan")))

                d_lo, d_hi = pc(dd)
                ra_lo, ra_hi = pc(ra)
                re_lo, re_hi = pc(re)
                g_lo, g_hi = pc(gg)
                p_lo, p_hi = pc(ppv)
                rows.append(dict(
                    dataset=ds, featureset=panel, clf=kind, n=n,
                    n_classes=len(classes),
                    auc_reference=a_g, auc_reference_lo=g_lo,
                    auc_reference_hi=g_hi,
                    auc_predicted=a_p, auc_predicted_lo=p_lo,
                    auc_predicted_hi=p_hi,
                    delta_auc_absolute=delta, delta_auc_lo=d_lo,
                    delta_auc_hi=d_hi,
                    rel_of_auc=rel_auc, rel_of_auc_lo=ra_lo,
                    rel_of_auc_hi=ra_hi,
                    rel_of_excess=rel_exc, rel_of_excess_lo=re_lo,
                    rel_of_excess_hi=re_hi,
                    denominator_note=("rel_of_auc divides by AUC_reference; "
                                      "rel_of_excess divides by "
                                      "(AUC_reference - 0.5), the "
                                      "discrimination above chance. The two "
                                      "differ by the factor "
                                      "AUC_reference/(AUC_reference-0.5) = "
                                      "%.2f here." % (a_g / (a_g - 0.5)
                                                      if a_g > 0.5 else np.nan)),
                    source=("results/pivot/p2_predictions.csv; 5-fold x 3 "
                            "repeats stratified CV, macro one-vs-rest AUC; "
                            "1000-draw paired image bootstrap of the fixed "
                            "out-of-fold probability matrices, identical to "
                            "src/pivot/p3_downstream.py::boot_ci")))
                print("[dauc] %-6s %-9s %-7s  dAUC=%.4f  /AUC=%.4f  "
                      "/(AUC-0.5)=%.4f" % (ds, panel, kind, delta, rel_auc,
                                           rel_exc), flush=True)

    out = pd.DataFrame(rows)
    p = os.path.join(args.out_dir, "r2_delta_auc.csv")
    out.to_csv(p, index=False)
    print("wrote", p, out.shape)
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(out[["dataset", "featureset", "clf", "n", "auc_reference",
                   "auc_predicted", "delta_auc_absolute", "delta_auc_lo",
                   "delta_auc_hi", "rel_of_auc", "rel_of_auc_lo",
                   "rel_of_auc_hi", "rel_of_excess", "rel_of_excess_lo",
                   "rel_of_excess_hi"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
