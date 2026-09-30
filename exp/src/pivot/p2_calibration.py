"""Probe P2 -- post-hoc biomarker bias calibration with conformal intervals (D2).

Data
----
``results/pivot/bio_master.csv``.  Fitting uses only **out-of-fold training**
predictions (``source == pred_oof``) paired with the reference biomarkers of
the same image; scoring uses the held-out **test** predictions
(``source == pred_test``, seed 0).  No test image ever enters a fit.

Parameterisation
----------------
Length-like biomarkers differ by an order of magnitude between DRIVE (584 px)
and HRF (3504 px), so nothing is regressed in raw units.  For every dataset and
biomarker the robust scale

    s_ds = 1.4826 * MAD( B_pred over that dataset's fitting images )

is computed **from predictions only** -- it needs no ground truth and is
therefore available for a brand-new domain.  The model predicts the relative
correction

    y = (B_gt - B_pred) / s_ds        ->        B_hat = B_pred + s_ds * y_hat

Reported errors are divided by the pre-registered GT sigma of
``results/gateA_biomarker_scales_train.csv``, so they are comparable with the
signed-bias table of ``RESULTS_DIGEST.md`` section 4.1.

Models
------
(a0) ``offset_per_ds``  subtract the mean signed bias of that dataset -- the
                        trivial calibration every learned model must beat
(a)  ``linear_per_ds``  ridge on [B_pred_z, observables], fitted per dataset
(b)  ``gbdt_pooled``    one gradient-boosted tree over all datasets, with the
                        dataset identity as a feature
(c0) ``offset_lodo``    mean signed bias of the *other* datasets
(c)  ``gbdt_lodo``      leave-one-dataset-out GBDT, no dataset feature

Split-conformal 90 % intervals: the OOF rows are split 70/30 into fit/calibration
with a fixed seed; the half-width is the finite-sample-corrected 90th percentile
of the absolute calibration residual.

Outputs
-------
``p2_pairs.csv``        the assembled (B_pred, B_gt, features) table
``p2_predictions.csv``  per-image calibrated biomarkers for every model.  Rows
                        with ``role == fit`` are **cross-fitted** (5-fold within
                        the fitting set, or LODO for model (c)) so probe P3 can
                        use training images without an in-sample advantage.
``p2_calibration.csv``  the MAE / bias / coverage summary

CLI
---
    python -m src.pivot.p2_calibration
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import (PIVOT_DIR, PRIMARY_COLS, PRIMARY_COLS_ZONEB,
                              sigma_table)

#: calibrated targets: the 8 primary columns plus their zone-B variants
#: (secondary -- the zone-B sigma is not in the Gate A train table for every
#: dataset, so those rows can carry NaN errors)
BIO_COLS = list(PRIMARY_COLS) + list(PRIMARY_COLS_ZONEB)

SEED = 0
ALPHA = 0.10                      # 90 % intervals
MIN_CAL = 5                       # conformal_q needs >= 5 calibration points
N_FOLDS = 5
MODELS = ("offset_per_ds", "linear_per_ds", "gbdt_pooled",
          "offset_lodo", "gbdt_lodo")

#: dimensionless observables usable at inference time
FEATS_RAW: Tuple[str, ...] = (
    "feat_pred_density", "feat_prob_mean", "feat_prob_std", "feat_prob_entropy",
    "feat_frac_lowconf", "feat_frac_mid", "feat_soft_hard_ratio",
    "feat_prob_in_fg", "feat_cnr", "feat_vessel_bg_diff", "feat_green_mean",
    "feat_green_std", "feat_contrast", "feat_fov_frac", "disc_r_rel",
    "disc_confident",
)


def robust_scale(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - med)))
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    return s if (np.isfinite(s) and s > 0) else float("nan")


def build_frame(master: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, image_id, role) with B_pred, B_gt and observables."""
    gt = master[master["source"] == "gt"].set_index(["dataset", "image_id"])
    out = []
    for src, tag in (("pred_oof", "fit"), ("pred_test", "eval")):
        pr = master[master["source"] == src].copy()
        if not len(pr):
            continue
        idx = pd.MultiIndex.from_arrays([pr["dataset"], pr["image_id"]])
        keep = idx.isin(gt.index)
        pr, idx = pr[keep].copy(), idx[keep]
        for c in BIO_COLS:
            if c in gt.columns:
                pr["gt_" + c] = gt.reindex(idx)[c].to_numpy(dtype=float)
        pr["role"] = tag
        out.append(pr)
    df = pd.concat(out, ignore_index=True).reset_index(drop=True)

    for ds, g in df.groupby("dataset"):
        sel = df["dataset"] == ds
        fitv = g[g["role"] == "fit"]
        for c in BIO_COLS:
            if c not in df.columns:
                continue
            ref = fitv[c].to_numpy(dtype=float)
            if ref.size < 3:
                ref = g[c].to_numpy(dtype=float)
            s = robust_scale(ref)
            med = float(np.nanmedian(ref))
            df.loc[sel, "spred_" + c] = s
            df.loc[sel, "z_" + c] = (df.loc[sel, c].to_numpy(dtype=float) - med) / s
    return df


def feature_matrix(df: pd.DataFrame, with_dataset: bool,
                   datasets: List[str]) -> np.ndarray:
    cols = ["z_" + c for c in PRIMARY_COLS] + [c for c in FEATS_RAW if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    X = np.column_stack([X, np.log10(df["feat_longest"].to_numpy(dtype=float))])
    if with_dataset:
        oh = np.column_stack([(df["dataset"] == d).to_numpy(dtype=float)
                              for d in datasets])
        X = np.column_stack([X, oh])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _make(kind: str):
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 20)))
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=10,
        l2_regularization=1.0, random_state=SEED)


def conformal_q(resid: np.ndarray, alpha: float = ALPHA) -> float:
    r = np.abs(resid[np.isfinite(resid)])
    n = r.size
    if n < 5:
        return float("nan")
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(r)[k - 1])


# --------------------------------------------------------------------------
def _predict_one_biomarker(df: pd.DataFrame, bio: str, datasets: List[str],
                           cal_mask: np.ndarray, folds: np.ndarray
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """yhat and conformal half-width per model, for every row of ``df``."""
    role = df["role"].to_numpy()
    ds_arr = df["dataset"].to_numpy()
    y = ((df["gt_" + bio].to_numpy(dtype=float) - df[bio].to_numpy(dtype=float))
         / df["spred_" + bio].to_numpy(dtype=float))
    ok = np.isfinite(y)
    is_fit = role == "fit"
    X_nods = feature_matrix(df, False, datasets)
    X_ds = feature_matrix(df, True, datasets)

    yhat = {m: np.full(len(df), np.nan) for m in MODELS}
    half = {m: np.full(len(df), np.nan) for m in MODELS}

    # ---- (a0) per-dataset constant offset --------------------------------
    # the simplest possible calibration: subtract the mean signed bias.  Any
    # feature-based model has to beat THIS, not the uncalibrated value.
    for ds in datasets:
        d_tr = ok & is_fit & (ds_arr == ds) & (~cal_mask)
        d_ca = ok & is_fit & (ds_arr == ds) & cal_mask
        if d_tr.sum() < 3:
            continue
        mu = float(np.mean(y[d_tr]))
        yhat["offset_per_ds"][(~is_fit) & (ds_arr == ds)] = mu
        half["offset_per_ds"][ds_arr == ds] = (conformal_q(y[d_ca] - mu)
                                               if d_ca.sum() >= 5 else np.nan)
        for f in range(N_FOLDS):
            tr = ok & is_fit & (ds_arr == ds) & (folds != f) & (~cal_mask)
            te_f = is_fit & (ds_arr == ds) & (folds == f)
            if tr.sum() >= 3 and te_f.any():
                yhat["offset_per_ds"][te_f] = float(np.mean(y[tr]))

    # ---- (c0) pooled constant offset, leave-one-dataset-out ---------------
    for ds in datasets:
        tr = ok & is_fit & (ds_arr != ds) & (~cal_mask)
        ca = ok & is_fit & (ds_arr != ds) & cal_mask
        if tr.sum() < 10:
            continue
        mu = float(np.mean(y[tr]))
        yhat["offset_lodo"][ds_arr == ds] = mu
        half["offset_lodo"][ds_arr == ds] = (conformal_q(y[ca] - mu)
                                             if ca.sum() >= 5 else np.nan)

    # ---- (a) per-dataset ridge -------------------------------------------
    for ds in datasets:
        d_tr = ok & is_fit & (ds_arr == ds) & (~cal_mask)
        d_ca = ok & is_fit & (ds_arr == ds) & cal_mask
        if d_tr.sum() < 6:
            continue
        m = _make("ridge").fit(X_nods[d_tr], y[d_tr])
        te = (~is_fit) & (ds_arr == ds)
        if te.any():
            yhat["linear_per_ds"][te] = m.predict(X_nods[te])
        q = conformal_q(y[d_ca] - m.predict(X_nods[d_ca])) if d_ca.sum() >= 5 else np.nan
        half["linear_per_ds"][(ds_arr == ds)] = q
        # cross-fitted values for the fitting rows
        for f in range(N_FOLDS):
            tr = ok & is_fit & (ds_arr == ds) & (folds != f) & (~cal_mask)
            te_f = is_fit & (ds_arr == ds) & (folds == f)
            if tr.sum() < 6 or not te_f.any():
                continue
            yhat["linear_per_ds"][te_f] = _make("ridge").fit(
                X_nods[tr], y[tr]).predict(X_nods[te_f])

    # ---- (b) pooled GBDT with dataset feature ----------------------------
    tr = ok & is_fit & (~cal_mask)
    ca = ok & is_fit & cal_mask
    if tr.sum() >= 20:
        m = _make("gbdt").fit(X_ds[tr], y[tr])
        te = ~is_fit
        yhat["gbdt_pooled"][te] = m.predict(X_ds[te])
        half["gbdt_pooled"][:] = conformal_q(y[ca] - m.predict(X_ds[ca]))
        for f in range(N_FOLDS):
            tr_f = ok & is_fit & (folds != f) & (~cal_mask)
            te_f = is_fit & (folds == f)
            if tr_f.sum() < 20 or not te_f.any():
                continue
            yhat["gbdt_pooled"][te_f] = _make("gbdt").fit(
                X_ds[tr_f], y[tr_f]).predict(X_ds[te_f])

    # ---- (c) LODO GBDT, no dataset feature -------------------------------
    for ds in datasets:
        tr = ok & is_fit & (ds_arr != ds) & (~cal_mask)
        ca = ok & is_fit & (ds_arr != ds) & cal_mask
        if tr.sum() < 20:
            continue
        m = _make("gbdt").fit(X_nods[tr], y[tr])
        sel = ds_arr == ds                      # both roles: ds never in the fit
        yhat["gbdt_lodo"][sel] = m.predict(X_nods[sel])
        half["gbdt_lodo"][sel] = (conformal_q(y[ca] - m.predict(X_nods[ca]))
                                  if ca.sum() >= 5 else np.nan)
    return yhat, half


def run(df: pd.DataFrame, sigmas: Dict[str, Dict[str, float]]
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(SEED)
    datasets = sorted(df["dataset"].unique())
    is_fit = (df["role"] == "fit").to_numpy()

    cal_mask = np.zeros(len(df), dtype=bool)
    folds = np.full(len(df), -1, dtype=int)
    for ds in datasets:
        i = np.where(is_fit & (df["dataset"].to_numpy() == ds))[0]
        rng.shuffle(i)
        # at least MIN_CAL rows, or conformal_q returns NaN and the interval
        # silently degenerates (HRF: 15 fitting rows -> round(4.5) == 4 < 5)
        n_cal = min(max(MIN_CAL, int(np.ceil(0.3 * len(i)))), max(0, len(i) - 8))
        cal_mask[i[:n_cal]] = True
        rest = i[n_cal:]
        folds[rest] = np.arange(len(rest)) % N_FOLDS
        folds[i[:n_cal]] = np.arange(n_cal) % N_FOLDS

    wide = df[["dataset", "image_id", "role", "split", "disease"]].copy()
    rows: List[dict] = []
    for bio in BIO_COLS:
        if bio not in df.columns or ("gt_" + bio) not in df.columns:
            continue
        yhat, half = _predict_one_biomarker(df, bio, datasets, cal_mask, folds)
        B_pred = df[bio].to_numpy(dtype=float)
        B_gt = df["gt_" + bio].to_numpy(dtype=float)
        s_pred = df["spred_" + bio].to_numpy(dtype=float)
        sig = np.array([sigmas.get(d, {}).get(bio, np.nan) for d in df["dataset"]],
                       dtype=float)
        wide[bio + "__pred"] = B_pred
        wide[bio + "__gt"] = B_gt
        for m in MODELS:
            wide[bio + "__cal_" + m] = B_pred + s_pred * yhat[m]
            wide[bio + "__half_" + m] = s_pred * half[m]

        ev = (~is_fit) & np.isfinite(B_gt) & np.isfinite(B_pred)
        for ds in datasets + ["ALL"]:
            sel = ev if ds == "ALL" else (ev & (df["dataset"].to_numpy() == ds))
            if sel.sum() == 0:
                continue
            e0 = (B_pred - B_gt)[sel] / sig[sel]
            base = dict(biomarker=bio, dataset=ds, n=int(sel.sum()),
                        mae_uncal=float(np.nanmean(np.abs(e0))),
                        bias_uncal=float(np.nanmean(e0)))
            for m in MODELS:
                Bh = B_pred + s_pred * yhat[m]
                e = (Bh - B_gt)[sel] / sig[sel]
                base["mae_" + m] = float(np.nanmean(np.abs(e)))
                base["bias_" + m] = float(np.nanmean(e))
                base["red_" + m] = float(1.0 - base["mae_" + m] / base["mae_uncal"])
                hw = s_pred * half[m]
                cov = (B_gt >= Bh - hw) & (B_gt <= Bh + hw)
                base["cov90_" + m] = float(np.nanmean(cov[sel].astype(float)))
                base["width_" + m] = float(np.nanmean((2 * hw)[sel] / sig[sel]))
            rows.append(base)
    return pd.DataFrame(rows), wide


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    args = ap.parse_args(argv)

    master = pd.read_csv(args.master)
    df = build_frame(master)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "p2_pairs.csv"), index=False)
    print("[p2] pairs by dataset/role:")
    print(df.groupby(["dataset", "role"]).size())

    res, wide = run(df, sigma_table())
    res.to_csv(os.path.join(args.out_dir, "p2_calibration.csv"), index=False)
    wide.to_csv(os.path.join(args.out_dir, "p2_predictions.csv"), index=False)
    show = (["biomarker", "dataset", "n", "mae_uncal"]
            + ["mae_" + m for m in MODELS] + ["red_" + m for m in MODELS])
    prim = res[res["biomarker"].isin(PRIMARY_COLS)]
    with pd.option_context("display.width", 240, "display.max_rows", 400):
        print(prim[show].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
