"""Probe P4 -- does test-time aggregation over anatomy-preserving views help?

Reads the Fig.4A stability tables ``results/fig4a_stability_<ds>_none_per_view.csv``
(baseline seed-0 segmenter, K = 8 anatomy-preserving acquisition perturbations
per test image, ``src/robust/perturb_views.py``) and the reference biomarkers of
``results/pivot/bio_master.csv`` (``source == "gt"``), and asks three things:

P4.1  reliability ``r(B, B_GT)`` for the view-mean / view-median aggregate
      versus a single view, per dataset and biomarker (Spearman + Pearson,
      paired image bootstrap CI of the difference);
P4.2  does the per-image spread across views predict the per-image |error|
      (a ground-truth-free reliability proxy)?
P4.3  the P3 downstream macro-AUC with view-aggregated features versus single
      view, the unperturbed prediction, and the reference masks.

Provenance note: the per-view biomarkers were computed by
``src/robust/run_stability.py`` with ``compute_all(..., disc=None)`` and the
default ``fd_rotations = 25``; ``bio_master.csv`` used ``fd_rotations = 5`` and
a per-image optic disc.  The FD estimator is the same, only the number of
averaged rotations differs; every comparison below is between columns of the
*same* provenance except where a ``pred`` (unperturbed) row is shown for
context, which is flagged in the output.

CLI
---
    python -m src.pivot.p4_views
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS, RESULTS_DIR
from src.pivot.p3_downstream import boot_ci, cv_proba, macro_auc

SEED = 0
N_BOOT = 1000
AGGS = ("view0", "mean", "median")


# --------------------------------------------------------------------------
def load_views(dataset: str) -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, f"fig4a_stability_{dataset}_none_per_view.csv")
    df = pd.read_csv(p)
    df = df[df["repair"] == "none"] if "repair" in df.columns else df
    return df


def aggregate(views: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """One row per image: view0 / mean / median / std / cv for every biomarker."""
    out = []
    for iid, g in views.groupby("image_id"):
        g = g.sort_values("view_id")
        row = {"image_id": iid, "n_views": len(g)}
        for c in cols:
            v = g[c].to_numpy(dtype=float)
            v0 = float(g[g["view_id"] == g["view_id"].min()][c].iloc[0])
            row[c + "__view0"] = v0
            row[c + "__mean"] = float(np.nanmean(v))
            row[c + "__median"] = float(np.nanmedian(v))
            row[c + "__std"] = float(np.nanstd(v, ddof=1))
            row[c + "__cv"] = float(np.nanstd(v, ddof=1) / max(abs(np.nanmean(v)), 1e-12))
        for c in cols:
            for k in range(len(g)):
                row[f"{c}__v{int(g['view_id'].iloc[k])}"] = float(g[c].iloc[k])
        out.append(row)
    return pd.DataFrame(out).sort_values("image_id").reset_index(drop=True)


def _corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.allclose(a[ok], a[ok][0]) or np.allclose(b[ok], b[ok][0]):
        return float("nan")
    if kind == "spearman":
        return float(stats.spearmanr(a[ok], b[ok]).statistic)
    return float(stats.pearsonr(a[ok], b[ok])[0])


def reliability_table(per_image: pd.DataFrame, gt: pd.DataFrame, dataset: str,
                      n_views: int, cols: Sequence[str]) -> List[dict]:
    d = per_image.merge(gt, on="image_id", suffixes=("", "_gtrow"))
    rows: List[dict] = []
    rng = np.random.RandomState(SEED)
    n = len(d)
    boot_idx = [rng.randint(0, n, n) for _ in range(N_BOOT)]
    for c in cols:
        y = d[c + "__gt"].to_numpy(dtype=float)
        singles = np.column_stack([d[f"{c}__v{k}"].to_numpy(dtype=float)
                                   for k in range(n_views)])
        for kind in ("spearman", "pearson"):
            r_single = float(np.nanmean([_corr(singles[:, k], y, kind)
                                         for k in range(n_views)]))
            base = dict(dataset=dataset, biomarker=c, stat=kind, n_images=n,
                        n_views=n_views, r_single_mean_over_views=r_single,
                        r_view0=_corr(d[c + "__view0"].to_numpy(dtype=float), y, kind))
            if (c + "__pred") in d.columns:
                base["r_pred_unperturbed"] = _corr(
                    d[c + "__pred"].to_numpy(dtype=float), y, kind)
            for agg in ("mean", "median"):
                x = d[c + f"__{agg}"].to_numpy(dtype=float)
                r_agg = _corr(x, y, kind)
                diffs = []
                for idx in boot_idx:
                    ra = _corr(x[idx], y[idx], kind)
                    rs = np.nanmean([_corr(singles[idx, k], y[idx], kind)
                                     for k in range(n_views)])
                    if np.isfinite(ra) and np.isfinite(rs):
                        diffs.append(ra - rs)
                base[f"r_{agg}"] = r_agg
                base[f"d_{agg}"] = r_agg - r_single
                base[f"d_{agg}_lo"] = float(np.percentile(diffs, 2.5)) if diffs else np.nan
                base[f"d_{agg}_hi"] = float(np.percentile(diffs, 97.5)) if diffs else np.nan
            rows.append(base)
    return rows


def variance_proxy(per_image: pd.DataFrame, gt: pd.DataFrame, dataset: str,
                   cols: Sequence[str]) -> List[dict]:
    """Spearman(view spread, |view-mean - GT|) -- a GT-free reliability proxy."""
    d = per_image.merge(gt, on="image_id")
    rows = []
    for c in cols:
        y = d[c + "__gt"].to_numpy(dtype=float)
        err = np.abs(d[c + "__mean"].to_numpy(dtype=float) - y)
        rows.append(dict(dataset=dataset, biomarker=c, n_images=len(d),
                         rho_std_vs_abserr=_corr(d[c + "__std"].to_numpy(dtype=float),
                                                 err, "spearman"),
                         rho_cv_vs_abserr=_corr(d[c + "__cv"].to_numpy(dtype=float),
                                                err, "spearman")))
    return rows


def downstream(per_image: pd.DataFrame, gt: pd.DataFrame, dataset: str,
               cols: Sequence[str]) -> List[dict]:
    d = per_image.merge(gt, on="image_id")
    d = d[d["disease"].notna()].reset_index(drop=True)
    if len(d) < 24:
        return []
    y = d["disease"].to_numpy()
    classes = sorted(pd.unique(y))
    counts = {c: int((y == c).sum()) for c in classes}
    if min(counts.values()) < 8:
        print(f"[{dataset}] downstream skipped: class counts {counts} "
              "-- the Fig.4A view subset is not class-balanced", flush=True)
        return []
    sources = {"gt": [c + "__gt" for c in cols]}
    if all((c + "__pred") in d.columns for c in cols):
        sources["pred_unperturbed"] = [c + "__pred" for c in cols]
    for agg in AGGS:
        sources[agg] = [c + f"__{agg}" for c in cols]
    rows = []
    for kind in ("logreg", "gbdt"):
        probas = {}
        for s, names in sources.items():
            X = np.nan_to_num(d[names].to_numpy(dtype=float), nan=0.0,
                              posinf=0.0, neginf=0.0)
            probas[s] = cv_proba(X, y, kind, classes)
        ci, dci = boot_ci(y, probas, classes)
        for s, p in probas.items():
            rows.append(dict(dataset=dataset, clf=kind, source=s, n=len(d),
                             n_classes=len(classes),
                             macro_auc=macro_auc(y, p, classes),
                             ci_lo=ci.get(s, (np.nan,) * 2)[0],
                             ci_hi=ci.get(s, (np.nan,) * 2)[1],
                             gt_minus_this=macro_auc(y, probas["gt"], classes)
                             - macro_auc(y, p, classes),
                             diff_ci_lo=dci.get(s, (np.nan,) * 2)[0],
                             diff_ci_hi=dci.get(s, (np.nan,) * 2)[1],
                             class_counts=";".join("%s=%d" % (c, int((y == c).sum()))
                                                   for c in classes)))
    return rows


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hrf,fives,drive,chasedb1")
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    args = ap.parse_args(argv)

    master = pd.read_csv(os.path.join(PIVOT_DIR, "bio_master.csv"))
    cols = list(PRIMARY_COLS)

    rel, prox, down, per_img_all = [], [], [], []
    for ds in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        views = load_views(ds)
        n_views = int(views["view_id"].nunique())
        per_image = aggregate(views, cols)

        m = master[master["dataset"] == ds]
        gt = m[m["source"] == "gt"][["image_id", "disease", "split"] + cols].copy()
        gt = gt.rename(columns={c: c + "__gt" for c in cols})
        pr = m[m["source"] == "pred_test"][["image_id"] + cols].copy()
        pr = pr.rename(columns={c: c + "__pred" for c in cols})
        gt = gt.merge(pr, on="image_id", how="left")

        per_image = per_image.merge(gt[["image_id"] + [c + "__pred" for c in cols]],
                                    on="image_id", how="left")
        gt = gt.drop(columns=[c + "__pred" for c in cols])
        keep = gt[gt["image_id"].isin(per_image["image_id"])]
        print(f"[{ds}] {len(per_image)} images x {n_views} views, "
              f"{len(keep)} matched to GT", flush=True)

        rel += reliability_table(per_image, keep, ds, n_views, cols)
        prox += variance_proxy(per_image, keep, ds, cols)
        down += downstream(per_image, keep, ds, cols)
        pi = per_image.copy(); pi.insert(0, "dataset", ds)
        per_img_all.append(pi)

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(rel).to_csv(os.path.join(args.out_dir, "p4_reliability.csv"), index=False)
    pd.DataFrame(prox).to_csv(os.path.join(args.out_dir, "p4_variance_proxy.csv"), index=False)
    pd.DataFrame(down).to_csv(os.path.join(args.out_dir, "p4_downstream.csv"), index=False)
    pd.concat(per_img_all, ignore_index=True).to_csv(
        os.path.join(args.out_dir, "p4_per_image.csv"), index=False)

    with pd.option_context("display.width", 240, "display.max_rows", 300):
        print("\n=== P4.1 reliability (spearman) ===")
        r = pd.DataFrame(rel)
        print(r[r["stat"] == "spearman"][
            ["dataset", "biomarker", "r_single_mean_over_views", "r_view0",
             "r_pred_unperturbed", "r_mean", "d_mean", "d_mean_lo", "d_mean_hi",
             "r_median"]].round(3).to_string(index=False))
        print("\n=== P4.1 reliability (pearson) ===")
        print(r[r["stat"] == "pearson"][
            ["dataset", "biomarker", "r_single_mean_over_views", "r_view0",
             "r_pred_unperturbed", "r_mean", "d_mean", "d_mean_lo", "d_mean_hi",
             "r_median"]].round(3).to_string(index=False))
        print("\n=== P4.2 view-spread vs |error| ===")
        print(pd.DataFrame(prox).round(3).to_string(index=False))
        print("\n=== P4.3 downstream ===")
        print(pd.DataFrame(down).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
