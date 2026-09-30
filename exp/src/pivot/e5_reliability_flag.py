"""E5 -- cross-view spread as a label-free per-image unreliability flag.

Two parts.

**E5.1 flag validity.**  Spearman between the across-view spread of a biomarker
(SD and CV over the K = 8 anatomy-preserving acquisition views of
``results/fig4a_stability_<ds>_none_per_view.csv``, aggregated in
``results/pivot/p4_per_image.csv``) and the per-image absolute error, with a
1000-resample image bootstrap CI.  Two error definitions:

    err_pred  |B_pred(unperturbed seed-0 mask) - B_GT|   <- deployment relevant
    err_mean  |B_view-mean                    - B_GT|   <- reproduces P4.2

**E5.2 gating.**  Rank images by a single label-free unreliability score
(mean over the eight primary columns of the rank-normalised across-view CV),
drop the top q% (q = 10/20/30), and ask what the retained set buys:

* reliability r(B_pred, B_GT) of the retained set, per biomarker and macro;
* downstream macro-AUC of the retained set (HRF only -- the 60 FIVES images
  that have views are Glaucoma 50 / AMD 5 / Normal 4 / DR 1 and cannot support
  a 4-class CV; regenerating views for a class-balanced FIVES subset needs GPU
  inference, which was not available for this analysis-only task).

Both are compared against **random drops of the same size** (the exact null):
1000 random retained subsets, reported as mean [2.5, 97.5] percentile with a
one-sided p = P(random >= gated).

CLI
---
    python -m src.pivot.e5_reliability_flag
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS

SEED = 0
N_BOOT = 1000
N_RAND = 1000
N_REPEAT = 3
QS = (10, 20, 30)


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


def pearson(a, b):
    from scipy.stats import pearsonr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pearsonr(a[ok], b[ok])[0])


def boot_pair(x, y, fn, n_boot=N_BOOT, seed=SEED):
    """Percentile CI **and** a two-sided bootstrap p-value.

    The p-value is the achieved significance level of the bootstrap
    distribution, reported with the conservative ``(b + 1) / (B + 1)``
    correction so that it can never be printed as ``0.000``: with B = 1000
    resamples the smallest reportable value is 1/1001, i.e. ``p < 0.001``.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.RandomState(seed)
    v = []
    for _ in range(n_boot):
        i = rng.randint(0, x.size, x.size)
        r = fn(x[i], y[i])
        if np.isfinite(r):
            v.append(r)
    if not v:
        return float("nan"), float("nan"), float("nan"), 0
    a = np.asarray(v, float)
    b = min(int((a <= 0).sum()), int((a >= 0).sum()))
    p = min(1.0, 2.0 * (b + 1) / (a.size + 1))
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
            float(p), int(a.size))


def macro_auc(y, proba, classes):
    from sklearn.metrics import roc_auc_score
    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() in (0, len(yy)):
            continue
        a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def cv_macro_auc(X, y, classes, n_splits=5, seed=SEED):
    """Same protocol as ``src/pivot/p3_downstream.py``: logreg, 5-fold x 3."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _, cnt = np.unique(y, return_counts=True)
    k = int(min(n_splits, cnt.min()))
    if k < 2:
        return float("nan")
    acc = np.zeros((len(y), len(classes)), float)
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(X, y):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=1.0))
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            order = [list(m.classes_).index(c) for c in classes]
            acc[te] += p[:, order]
    return macro_auc(y, acc / N_REPEAT, classes)


# ------------------------------------------------------------------ data
def load(dataset: str):
    per = pd.read_csv(os.path.join(PIVOT_DIR, "p4_per_image.csv"))
    per = per[per["dataset"] == dataset].copy()
    m = pd.read_csv(os.path.join(PIVOT_DIR, "bio_master.csv"), low_memory=False)
    m = m[m["dataset"] == dataset]
    gt = m[m["source"] == "gt"]
    gt = gt[gt["split"] == "test"][["image_id", "disease"] + list(PRIMARY_COLS)]
    gt = gt.rename(columns={c: c + "__gt" for c in PRIMARY_COLS})
    d = per.merge(gt, on="image_id", how="inner")
    return d


def unreliability_score(d: pd.DataFrame) -> np.ndarray:
    """Label-free per-image score: mean rank-normalised CV over the 8 columns."""
    ranks = []
    n = len(d)
    for c in PRIMARY_COLS:
        v = d[c + "__cv"].to_numpy(float)
        r = pd.Series(v).rank(method="average", na_option="keep").to_numpy(float)
        ranks.append((r - 1.0) / max(n - 1, 1))
    return np.nanmean(np.vstack(ranks), axis=0)


# ------------------------------------------------------------- E5.1 flag
def flag_validity(rows: List[dict]) -> None:
    for ds in ("hrf", "fives", "drive", "chasedb1"):
        d = load(ds)
        if len(d) == 0:
            continue
        for c in PRIMARY_COLS:
            g = d[c + "__gt"].to_numpy(float)
            for errname, pcol in (("err_pred", c + "__pred"),
                                  ("err_mean", c + "__mean")):
                if pcol not in d.columns:
                    continue
                err = np.abs(d[pcol].to_numpy(float) - g)
                for spread in ("std", "cv"):
                    s = d[c + "__" + spread].to_numpy(float)
                    rho = spearman(s, err)
                    lo, hi, pval, nb = boot_pair(s, err, spearman)
                    rows.append(dict(
                        dataset=ds, biomarker=c, err=errname, spread=spread,
                        n=int(len(d)), rho=rho, lo=lo, hi=hi,
                        p_boot=pval, n_boot=nb,
                        source="results/pivot/p4_per_image.csv (8 acquisition "
                               "views, seed-0 baseline segmenter) x "
                               "results/pivot/bio_master.csv source=gt; "
                               "1000-resample image bootstrap; p_boot is the "
                               "two-sided achieved significance level with the "
                               "(b+1)/(B+1) correction"))


# ---------------------------------------------------------- E5.2 gating
def gating(rows_rel: List[dict], rows_auc: List[dict]) -> None:
    rng_master = np.random.RandomState(SEED)
    for ds in ("hrf", "fives"):
        d = load(ds).reset_index(drop=True)
        n = len(d)
        score = unreliability_score(d)
        order = np.argsort(-score)            # most unreliable first
        y = d["disease"].astype(str).to_numpy()
        classes = sorted(set(y))
        Xp = np.nan_to_num(d[[c + "__pred" for c in PRIMARY_COLS]]
                           .to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        Xg = np.nan_to_num(d[[c + "__gt" for c in PRIMARY_COLS]]
                           .to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
        can_auc = (ds == "hrf")

        def macro_r(idx):
            rs = [pearson(d[c + "__pred"].to_numpy(float)[idx],
                          d[c + "__gt"].to_numpy(float)[idx])
                  for c in PRIMARY_COLS]
            return float(np.nanmean(rs)), rs

        keep_all = np.arange(n)
        r_all, rs_all = macro_r(keep_all)
        auc_all = cv_macro_auc(Xp, y, classes) if can_auc else float("nan")
        auc_gt = cv_macro_auc(Xg, y, classes) if can_auc else float("nan")
        rows_rel.append(dict(dataset=ds, q=0, n_kept=n, macro_r=r_all,
                             rand_mean=np.nan, rand_lo=np.nan, rand_hi=np.nan,
                             p_random_ge=np.nan,
                             **{f"r_{c}": rs_all[i] for i, c in enumerate(PRIMARY_COLS)}))
        if can_auc:
            rows_auc.append(dict(dataset=ds, q=0, n_kept=n, auc_pred=auc_all,
                                 auc_gt=auc_gt, rand_mean=np.nan, rand_lo=np.nan,
                                 rand_hi=np.nan, p_random_ge=np.nan))

        for q in QS:
            n_drop = int(round(n * q / 100.0))
            if n_drop < 1:
                continue
            keep = np.sort(order[n_drop:])
            n_kept = len(keep)
            r_gate, rs_gate = macro_r(keep)
            # random-drop null
            rng = np.random.RandomState(SEED + q)
            rr, ra = [], []
            for _ in range(N_RAND):
                k = np.sort(rng.choice(n, n_kept, replace=False))
                rr.append(macro_r(k)[0])
            rr = np.array(rr, float)
            rows_rel.append(dict(
                dataset=ds, q=q, n_kept=n_kept, macro_r=r_gate,
                rand_mean=float(np.nanmean(rr)),
                rand_lo=float(np.nanpercentile(rr, 2.5)),
                rand_hi=float(np.nanpercentile(rr, 97.5)),
                p_random_ge=float(np.mean(rr >= r_gate)),
                **{f"r_{c}": rs_gate[i] for i, c in enumerate(PRIMARY_COLS)}))

            if can_auc:
                a_gate = cv_macro_auc(Xp[keep], y[keep], classes)
                a_gt = cv_macro_auc(Xg[keep], y[keep], classes)
                rng2 = np.random.RandomState(SEED + 100 + q)
                n_rand_auc = 200        # 200 x 15 logreg fits per q
                for _ in range(n_rand_auc):
                    k = np.sort(rng2.choice(n, n_kept, replace=False))
                    ra.append(cv_macro_auc(Xp[k], y[k], classes))
                ra = np.array(ra, float)
                rows_auc.append(dict(
                    dataset=ds, q=q, n_kept=n_kept, auc_pred=a_gate, auc_gt=a_gt,
                    rand_mean=float(np.nanmean(ra)),
                    rand_lo=float(np.nanpercentile(ra, 2.5)),
                    rand_hi=float(np.nanpercentile(ra, 97.5)),
                    p_random_ge=float(np.nanmean(ra >= a_gate)),
                    n_random=n_rand_auc))
    _ = rng_master


def gating_per_biomarker(rows: List[dict]) -> None:
    """Gate each biomarker with ITS OWN across-view CV (the per-measurement flag)."""
    for ds in ("hrf", "fives", "drive"):
        d = load(ds).reset_index(drop=True)
        n = len(d)
        for c in PRIMARY_COLS:
            cv = d[c + "__cv"].to_numpy(float)
            p = d[c + "__pred"].to_numpy(float)
            g = d[c + "__gt"].to_numpy(float)
            order = np.argsort(-np.nan_to_num(cv, nan=-np.inf))
            rows.append(dict(dataset=ds, biomarker=c, q=0, n_kept=n,
                             r=pearson(p, g), rand_mean=np.nan, rand_lo=np.nan,
                             rand_hi=np.nan, p_random_ge=np.nan))
            for q in QS:
                n_drop = int(round(n * q / 100.0))
                if n_drop < 1:
                    continue
                keep = np.sort(order[n_drop:])
                r_g = pearson(p[keep], g[keep])
                rng = np.random.RandomState(SEED + q)
                rr = np.array([pearson(p[k], g[k]) for k in
                               (np.sort(rng.choice(n, len(keep), replace=False))
                                for _ in range(N_RAND))], float)
                rows.append(dict(
                    dataset=ds, biomarker=c, q=q, n_kept=len(keep), r=r_g,
                    rand_mean=float(np.nanmean(rr)),
                    rand_lo=float(np.nanpercentile(rr, 2.5)),
                    rand_hi=float(np.nanpercentile(rr, 97.5)),
                    p_random_ge=float(np.nanmean(rr >= r_g))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    ap.add_argument("--only", default="all", choices=("all", "flag"),
                    help="'flag' recomputes only the (cheap) flag-validity "
                         "table and leaves the cross-validated gating outputs "
                         "on disk untouched")
    args = ap.parse_args()

    flag: List[dict] = []
    flag_validity(flag)
    f = pd.DataFrame(flag)
    f.to_csv(os.path.join(args.out_dir, "e5_flag_validity.csv"), index=False)
    print("wrote e5_flag_validity.csv", f.shape)
    if args.only == "flag":
        return

    rel: List[dict] = []
    auc: List[dict] = []
    gating(rel, auc)
    pd.DataFrame(rel).to_csv(os.path.join(args.out_dir, "e5_gating_reliability.csv"),
                             index=False)
    pd.DataFrame(auc).to_csv(os.path.join(args.out_dir, "e5_gating_auc.csv"),
                             index=False)
    print("wrote e5_gating_reliability.csv / e5_gating_auc.csv")

    pb: List[dict] = []
    gating_per_biomarker(pb)
    pd.DataFrame(pb).to_csv(os.path.join(args.out_dir, "e5_gating_per_biomarker.csv"),
                            index=False)
    print("wrote e5_gating_per_biomarker.csv")


if __name__ == "__main__":
    main()
