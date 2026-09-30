"""E1 -- the GT-oracle downstream arm and which error component explains it.

Table 1 (``e1_oracle_downstream.csv``): for FIVES (n = 400, 4 classes) and HRF
(n = 45, 3 classes), the macro one-vs-rest AUC from the reference-mask
biomarkers (the oracle ceiling) against the AUC from the segmentation-derived
biomarkers, with the paired-bootstrap attenuation -- read verbatim from
``results/pivot/p3_downstream.csv`` -- joined to the E1 decomposition summary
(mean |constant bias|, mean per-image residual SD, mean reliability r) for the
same dataset.

Table 2 (``e1_attenuation_link.csv``): the per-image test.  A within-cohort
classifier cannot see a constant shift, so only the per-image residual can
attenuate it.  For each dataset we score every image with the pooled
out-of-fold probability of its true class under the *predicted*-feature
classifier, and correlate that against the per-image macro absolute error
``mean_c |B_pred,c - B_GT,c| / sigma_c`` and against its bias-removed version
``mean_c |(B_pred,c - B_GT,c)/sigma_c - bias_c|``.

CLI
---
    python -m src.pivot.e1_oracle
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS, sigma_table

SEED = 0
N_REPEAT = 3
N_BOOT = 1000


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


def spearman_ci(a, b, n_boot=N_BOOT, seed=SEED):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 4:
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(a, b)[0])
    rng = np.random.RandomState(seed)
    v = []
    for _ in range(n_boot):
        i = rng.randint(0, a.size, a.size)
        r = spearmanr(a[i], b[i])[0]
        if np.isfinite(r):
            v.append(r)
    return rho, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    args = ap.parse_args()

    dec = pd.read_csv(os.path.join(PIVOT_DIR, "e1_decomposition.csv"))
    w = dec.pivot_table(index=["dataset", "biomarker"], columns="quantity",
                        values="value").reset_index()

    # The audit summary that travels with each downstream row must be computed
    # on the SAME measurement panel as the classifier features, or the table
    # silently pairs a four-column classifier with an eight-column fidelity
    # mean.  PANELS maps the p3 featureset name to the biomarker columns.
    PANELS = {
        "skan4": [c for c in PRIMARY_COLS if c.endswith("_skan")],
        "primary8": list(PRIMARY_COLS),
    }

    def panel_summary(cols: List[str]) -> pd.DataFrame:
        ww = w[w["biomarker"].isin(cols)]
        out = ww.groupby("dataset").agg(
            mean_abs_bias=("bias_const", lambda s: float(np.mean(np.abs(s)))),
            mean_resid_sd=("resid_sd", "mean"),
            mean_r_pearson=("r_pearson", "mean"),
            min_r_pearson=("r_pearson", "min"),
            mean_topology=("topology", "mean"),
        ).reset_index()
        nt = ww[~ww["biomarker"].str.startswith("tortuosity")]
        return out.merge(
            nt.groupby("dataset").agg(
                mean_r_pearson_no_tort=("r_pearson", "mean"),
                min_r_pearson_no_tort=("r_pearson", "min")).reset_index(),
            on="dataset", how="left")

    # ---- table 1: reference-mask arm, verbatim from p3_downstream.csv
    p3_all = pd.read_csv(os.path.join(PIVOT_DIR, "p3_downstream.csv"))
    rows: List[dict] = []
    for fs, cols in PANELS.items():
        p3 = p3_all[(p3_all["featureset"] == fs) &
                    (p3_all["source"].isin(["gt", "pred"]))]
        if len(p3) == 0:
            print(f"[warn] featureset {fs} absent from p3_downstream.csv")
            continue
        summ = panel_summary(cols)
        sub: List[dict] = []
        for (ds, split, clf), g in p3.groupby(["dataset", "split", "clf"]):
            gt = g[g["source"] == "gt"]
            pr = g[g["source"] == "pred"]
            if len(gt) == 0 or len(pr) == 0:
                continue
            sub.append(dict(
                featureset=fs, n_features=len(cols),
                dataset=ds, split=split, clf=clf, n=int(gt["n"].iloc[0]),
                n_classes=int(gt["n_classes"].iloc[0]),
                auc_gt=float(gt["macro_auc"].iloc[0]),
                auc_gt_lo=float(gt["ci_lo"].iloc[0]), auc_gt_hi=float(gt["ci_hi"].iloc[0]),
                auc_pred=float(pr["macro_auc"].iloc[0]),
                auc_pred_lo=float(pr["ci_lo"].iloc[0]), auc_pred_hi=float(pr["ci_hi"].iloc[0]),
                attenuation=float(pr["gt_minus_this"].iloc[0]),
                att_lo=float(pr["diff_ci_lo"].iloc[0]), att_hi=float(pr["diff_ci_hi"].iloc[0]),
                class_counts=str(gt["class_counts"].iloc[0]),
                source=f"results/pivot/p3_downstream.csv (featureset={fs})"))
        rows += pd.DataFrame(sub).merge(summ, on="dataset", how="left")                   .to_dict("records")
    t1 = pd.DataFrame(rows)
    o1 = os.path.join(args.out_dir, "e1_oracle_downstream.csv")
    t1.to_csv(o1, index=False)
    print("wrote", o1)

    # ---- table 2: per-image link
    sig = sigma_table()
    pred = pd.read_csv(os.path.join(PIVOT_DIR, "p2_predictions.csv"), low_memory=False)
    rows2: List[dict] = []
    for ds in ("fives", "hrf"):
        d = pred[pred["dataset"] == ds].copy()
        d = d[d["disease"].notna()].reset_index(drop=True)
        y = d["disease"].to_numpy()
        classes = sorted(pd.unique(y))
        Xp = np.nan_to_num(d[[c + "__pred" for c in PRIMARY_COLS]].to_numpy(float),
                           nan=0.0, posinf=0.0, neginf=0.0)
        Xg = np.nan_to_num(d[[c + "__gt" for c in PRIMARY_COLS]].to_numpy(float),
                           nan=0.0, posinf=0.0, neginf=0.0)
        s = np.array([sig[ds][c] for c in PRIMARY_COLS], float)
        z = (Xp - Xg) / s[None, :]                # signed error in sigma
        bias = np.nanmean(z, axis=0)              # the constant component
        err_total = np.nanmean(np.abs(z), axis=1)
        err_resid = np.nanmean(np.abs(z - bias[None, :]), axis=1)
        for clf in ("logreg", "gbdt"):
            P = cv_proba(Xp, y, clf, classes)
            Pg = cv_proba(Xg, y, clf, classes)
            idx = np.array([classes.index(v) for v in y])
            ptrue = P[np.arange(len(y)), idx]
            ptrue_gt = Pg[np.arange(len(y)), idx]
            for nm, e in (("abs_err_total", err_total), ("abs_err_resid", err_resid)):
                rho, lo, hi = spearman_ci(e, ptrue)
                rows2.append(dict(dataset=ds, clf=clf, n=len(y), error=nm,
                                  target="p_true(pred features)",
                                  rho=rho, lo=lo, hi=hi))
                rho, lo, hi = spearman_ci(e, ptrue_gt)
                rows2.append(dict(dataset=ds, clf=clf, n=len(y), error=nm,
                                  target="p_true(GT features)",
                                  rho=rho, lo=lo, hi=hi))
            rho, lo, hi = spearman_ci(err_total, ptrue - ptrue_gt)
            rows2.append(dict(dataset=ds, clf=clf, n=len(y), error="abs_err_total",
                              target="p_true(pred) - p_true(GT)",
                              rho=rho, lo=lo, hi=hi))
        rows2.append(dict(dataset=ds, clf="-", n=len(y), error="const_bias_spread",
                          target="SD over images of the constant component",
                          rho=0.0, lo=0.0, hi=0.0))
    t2 = pd.DataFrame(rows2)
    t2["source"] = ("results/pivot/p2_predictions.csv features + "
                    "src/pivot/p3_downstream.py CV protocol; sigma = "
                    "results/gateA_biomarker_scales_train.csv; 1000-resample "
                    "image bootstrap")
    o2 = os.path.join(args.out_dir, "e1_attenuation_link.csv")
    t2.to_csv(o2, index=False)
    print("wrote", o2)

    pd.set_option("display.width", 250)
    print(t1.round(4).drop(columns=["source", "class_counts"]).to_string(index=False))
    print(t2.round(4).drop(columns=["source"]).to_string(index=False))


if __name__ == "__main__":
    main()
