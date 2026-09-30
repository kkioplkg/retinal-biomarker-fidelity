"""R2 -- method-comparison calibration of predicted against reference biomarkers,
plus a third "proportional distortion" arm for the injection dose-response.

Reviewer point 5 (``review/user_cmig_review_20260918.md``)
---------------------------------------------------------
"Report calibration slope / intercept, CCC and the Bland-Altman proportional
bias; add a proportional-distortion arm to the injection experiment."

The audit as shipped reports a *constant* offset and a residual SD.  That is a
location-and-scatter description and it is silent about a **slope**: a predicted
biomarker can be unbiased at the cohort mean and still compress or stretch the
between-image spread, which is exactly the failure mode that matters for a
biomarker meant to rank patients.  This script adds the standard
method-comparison battery.

Part 1 -- calibration (``r2_calibration.csv``, one row per dataset x biomarker
x regression method)

    OLS      B_pred = alpha + beta * B_ref + eps, ordinary least squares.
             Consistent only if B_ref is measured without error; it is not
             (the inter-observer discrepancy on DRIVE/CHASE_DB1 is of the same
             order as the machine's own error), so OLS beta is attenuated.
    Deming   the errors-in-variables fit with error-variance ratio
             ``lambda = var(err_pred) / var(err_ref)``.  ``lambda = 1``
             (orthogonal regression) is the primary setting and is reported as
             ``deming_l1``; ``deming_l4`` (lambda = 4, i.e. the prediction
             assumed four times noisier than the reference) is carried as a
             sensitivity so the reader can see how much the slope moves.

    Each row also carries: residual SD about the fitted line, Pearson r,
    Spearman rho, Lin's concordance correlation coefficient, the Bland-Altman
    mean difference and 95 % limits of agreement, and the proportional-bias
    test -- OLS of (pred - ref) on (pred + ref)/2, reporting slope, its
    bootstrap CI and a two-sided p-value.

    All CIs are 2000-draw image bootstraps (paired; the same resampled images
    enter every statistic in a row).

Part 2 -- figures (``figs/pivot/r2_calibration_<dataset>.png|pdf``)
    A 4 x 2 grid per dataset: for each primary-panel (skan) biomarker, the
    identity-line scatter with the OLS and Deming fits on the left and the
    Bland-Altman plot with mean difference, limits of agreement and the
    proportional-bias line on the right.

Part 3 -- the proportional-distortion injection arm
(``r2_dose_proportional.csv``, ``r2_dose_proportional_summary.csv``)

    ``B'_i = mean_j B_j + k * (B_i - mean_j B_j)`` for
    ``k in {0.5, 0.75, 1.25, 1.5, 2}`` applied to the **reference** biomarkers
    of HRF and FIVES, traced through the identical downstream protocol as
    ``src/pivot/e1_dose_response.py`` (5-fold x 3 repeats, logistic regression
    and HistGradientBoosting, macro one-vs-rest AUC).

    The expectation, stated before running: a per-biomarker affine map of the
    features is absorbed by a *retrained* classifier.  For logistic regression
    with a StandardScaler in front it is absorbed exactly (the scaler undoes it
    and the fit is invariant); for a tree ensemble it is absorbed up to the
    monotone-split invariance, so any movement is CV/tie noise rather than a
    real effect.  A *fixed* rule -- a threshold, a published cut-off, a frozen
    model shipped with coefficients -- would be moved, and that is the point:
    the invariance is a property of the retrained-classifier protocol, not of
    the biomarker.  The script therefore adds a fixed-rule arm
    (``r2_dose_proportional_fixedrule.csv``) that fits the classifier on the
    undistorted reference features once and applies it, frozen, to the
    distorted features -- the contrast that makes the explanation checkable.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_calibration
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import EXP_ROOT, PIVOT_DIR, PRIMARY_COLS, sigma_table

SEED = 0
N_BOOT = 2000
N_REPEAT = 3
N_SPLITS = 5
DATASETS = ("drive", "chasedb1", "hrf", "fives")
PANEL_SKAN = [c for c in PRIMARY_COLS if c.endswith("_skan")]
PANEL = {c: ("primary" if c.endswith("_skan") else "sensitivity")
         for c in PRIMARY_COLS}
K_GRID = (0.5, 0.75, 1.25, 1.5, 2.0)


# --------------------------------------------------------------- regressions
def ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """(alpha, beta) of y = alpha + beta x."""
    if x.size < 3 or np.std(x) == 0:
        return (float("nan"), float("nan"))
    beta, alpha = np.polyfit(x, y, 1)
    return (float(alpha), float(beta))


def deming(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> Tuple[float, float]:
    """Deming (errors-in-variables) slope/intercept with error-variance ratio
    ``lam = var(err_y) / var(err_x)``.

    beta = ( Syy - lam*Sxx + sqrt( (Syy - lam*Sxx)^2 + 4*lam*Sxy^2 ) )
           / ( 2*Sxy )
    alpha = mean(y) - beta * mean(x)
    """
    n = x.size
    if n < 3:
        return (float("nan"), float("nan"))
    mx, my = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - mx) ** 2) / (n - 1))
    syy = float(np.sum((y - my) ** 2) / (n - 1))
    sxy = float(np.sum((x - mx) * (y - my)) / (n - 1))
    if sxy == 0:
        return (float("nan"), float("nan"))
    disc = (syy - lam * sxx) ** 2 + 4.0 * lam * sxy ** 2
    beta = (syy - lam * sxx + np.sqrt(max(disc, 0.0))) / (2.0 * sxy)
    return (float(my - beta * mx), float(beta))


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    if x.size < 3:
        return float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx = float(np.var(x, ddof=0))
    vy = float(np.var(y, ddof=0))
    cov = float(np.mean((x - mx) * (y - my)))
    den = vx + vy + (mx - my) ** 2
    return float(2.0 * cov / den) if den > 0 else float("nan")


def stats_pack(x: np.ndarray, y: np.ndarray, method: str,
               with_spearman: bool = True) -> Dict[str, float]:
    """Every scalar a calibration row needs, for one resample.

    ``with_spearman=False`` is used inside the bootstrap: Spearman's rho needs a
    re-rank on every draw and is the only expensive term here, so it is computed
    on the point estimate only and reported without a bootstrap CI.
    """
    if method == "ols":
        a, b = ols(x, y)
    elif method == "deming_l1":
        a, b = deming(x, y, 1.0)
    elif method == "deming_l4":
        a, b = deming(x, y, 4.0)
    else:
        raise ValueError(method)
    resid = y - (a + b * x)
    d = y - x
    m = (y + x) / 2.0
    ba_a, ba_b = ols(m, d)
    out = {
        "alpha": a, "beta": b,
        "resid_sd": float(np.std(resid, ddof=1)) if x.size > 2 else float("nan"),
        "ccc": ccc(x, y),
        "ba_mean_diff": float(np.mean(d)),
        "ba_sd_diff": float(np.std(d, ddof=1)) if x.size > 2 else float("nan"),
        "prop_bias_slope": ba_b,
        "prop_bias_intercept": ba_a,
    }
    out["ba_loa_lo"] = out["ba_mean_diff"] - 1.96 * out["ba_sd_diff"]
    out["ba_loa_hi"] = out["ba_mean_diff"] + 1.96 * out["ba_sd_diff"]
    sx, sy = float(np.std(x)), float(np.std(y))
    out["r_pearson"] = (float(np.corrcoef(x, y)[0, 1]) if (sx > 0 and sy > 0)
                        else float("nan"))
    if with_spearman:
        from scipy.stats import spearmanr
        try:
            out["r_spearman"] = float(spearmanr(x, y)[0])
        except Exception:                                        # noqa: BLE001
            out["r_spearman"] = float("nan")
    return out


def prop_bias_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided p for the slope of (y-x) on (y+x)/2 (classical OLS t-test)."""
    from scipy import stats

    d = y - x
    m = (y + x) / 2.0
    n = d.size
    if n < 4 or np.std(m) == 0:
        return float("nan")
    a, b = ols(m, d)
    resid = d - (a + b * m)
    s2 = float(np.sum(resid ** 2) / (n - 2))
    sxx = float(np.sum((m - np.mean(m)) ** 2))
    if sxx <= 0 or s2 <= 0:
        return float("nan")
    t = b / np.sqrt(s2 / sxx)
    return float(2.0 * stats.t.sf(abs(t), df=n - 2))


def calibration_rows(master: pd.DataFrame, sig: Dict[str, Dict[str, float]]
                     ) -> pd.DataFrame:
    rows: List[dict] = []
    rng_master = np.random.RandomState(SEED)
    for ds in DATASETS:
        m = master[master["dataset"] == ds]
        gt = m[(m["source"] == "gt") & (m["split"] == "test")]
        pr = m[m["source"] == "pred_test"]
        cols = ["image_id"] + list(PRIMARY_COLS)
        d = gt[cols].merge(pr[cols], on="image_id", suffixes=("__gt", "__pred"))
        for col in PRIMARY_COLS:
            s = sig.get(ds, {}).get(col, float("nan"))
            xg = d[col + "__gt"].to_numpy(float)
            yp = d[col + "__pred"].to_numpy(float)
            ok = np.isfinite(xg) & np.isfinite(yp)
            x, y = xg[ok], yp[ok]
            if x.size < 4:
                continue
            # Everything is computed on the SIGMA axis, CENTRED on the
            # reference median of the same cohort, so that a biomarker whose
            # value is large relative to sigma (tortuosity sits at ~130 sigma)
            # does not produce a meaningless intercept.  Centring subtracts the
            # same constant from both variables, so beta, CCC, r, the residual
            # SD, the Bland-Altman limits and the proportional-bias slope are
            # all unchanged; only alpha is affected, and in the centred frame
            # alpha IS the interpretable quantity -- the predicted-minus-
            # reference offset an image at the cohort median would carry.
            med = float(np.median(x))
            xs, ys = (x - med) / s, (y - med) / s
            idx = rng_master.randint(0, x.size, size=(N_BOOT, x.size))
            for method in ("ols", "deming_l1", "deming_l4"):
                point = stats_pack(xs, ys, method)
                draws: Dict[str, List[float]] = {
                    k: [] for k in point if k != "r_spearman"}
                for i in idx:
                    if np.std(xs[i]) == 0:
                        continue
                    st = stats_pack(xs[i], ys[i], method, with_spearman=False)
                    for k, v in st.items():
                        if np.isfinite(v):
                            draws[k].append(v)
                ci = {}
                for k, v in draws.items():
                    a = np.asarray(v, float)
                    ci[k + "_lo"] = (float(np.percentile(a, 2.5)) if a.size
                                     else float("nan"))
                    ci[k + "_hi"] = (float(np.percentile(a, 97.5)) if a.size
                                     else float("nan"))
                rows.append(dict(
                    dataset=ds, biomarker=col, panel=PANEL[col], method=method,
                    n=int(x.size), sigma=s, ref_median_raw=med,
                    beta_excludes_1=bool(np.isfinite(ci["beta_lo"]) and
                                         (ci["beta_lo"] > 1.0 or
                                          ci["beta_hi"] < 1.0)),
                    alpha_excludes_0=bool(np.isfinite(ci["alpha_lo"]) and
                                          (ci["alpha_lo"] > 0.0 or
                                           ci["alpha_hi"] < 0.0)),
                    prop_bias_p=prop_bias_p(xs, ys),
                    **point, **ci,
                    units=("sigma (results/gateA_biomarker_scales_train.csv), "
                           "centred on the reference median of the same cohort "
                           "(ref_median_raw, native units): alpha is therefore "
                           "the predicted-minus-reference offset at a "
                           "median-valued image. beta, CCC, r, residual SD, "
                           "the Bland-Altman limits and the proportional-bias "
                           "slope are unaffected by the centring"),
                    source=("results/pivot/bio_master.csv "
                            "(source=gt[test] vs pred_test, seg seed 0); "
                            "2000-draw paired image bootstrap"),
                ))
    return pd.DataFrame(rows)


# ------------------------------------------------------- the main-text table
def main_table(cal: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """The main-text row set fixed by the methods consultation:

        mu (constant offset), alpha (calibration intercept), beta (slope),
        residual SD, CCC, r, and the downstream AUC gap.

    The consultation (``review/cmig_plan_reply.md``, point 24) also says CCC
    must not sit below r in prominence, so CCC is placed before r here.
    ``mu`` is the Bland-Altman mean difference, which is exactly the audit's
    constant offset; ``alpha`` is the Deming intercept in the median-centred
    frame, i.e. the offset a median-valued image carries once the slope is
    allowed to differ from 1.  The AUC gap is joined from
    ``r2_delta_auc.csv`` where it exists (HRF and FIVES only, per dataset --
    it is a dataset-level quantity, not a per-biomarker one).
    """
    d = cal[cal["method"] == "deming_l1"].copy()
    gap = {}
    p = os.path.join(out_dir, "r2_delta_auc.csv")
    if os.path.exists(p):
        g = pd.read_csv(p)
        g = g[(g["featureset"] == "skan4") & (g["clf"] == "logreg")]
        for r in g.itertuples(index=False):
            gap[r.dataset] = (r.delta_auc_absolute, r.delta_auc_lo,
                              r.delta_auc_hi)
    rows: List[dict] = []
    for r in d.itertuples(index=False):
        ga = gap.get(r.dataset, (float("nan"),) * 3)
        rows.append(dict(
            dataset=r.dataset, biomarker=r.biomarker, panel=r.panel, n=r.n,
            mu=r.ba_mean_diff, mu_lo=r.ba_mean_diff_lo, mu_hi=r.ba_mean_diff_hi,
            alpha=r.alpha, alpha_lo=r.alpha_lo, alpha_hi=r.alpha_hi,
            beta=r.beta, beta_lo=r.beta_lo, beta_hi=r.beta_hi,
            resid_sd=r.resid_sd, resid_sd_lo=r.resid_sd_lo,
            resid_sd_hi=r.resid_sd_hi,
            ccc=r.ccc, ccc_lo=r.ccc_lo, ccc_hi=r.ccc_hi,
            r_pearson=r.r_pearson, r_pearson_lo=r.r_pearson_lo,
            r_pearson_hi=r.r_pearson_hi,
            prop_bias_slope=r.prop_bias_slope,
            prop_bias_slope_lo=r.prop_bias_slope_lo,
            prop_bias_slope_hi=r.prop_bias_slope_hi,
            prop_bias_p=r.prop_bias_p,
            auc_gap_skan4_logreg=ga[0], auc_gap_lo=ga[1], auc_gap_hi=ga[2],
            units=r.units,
            source=("results/pivot/r2/r2_calibration.csv (method=deming_l1) "
                    "joined to results/pivot/r2/r2_delta_auc.csv "
                    "(featureset=skan4, clf=logreg) for the dataset-level "
                    "AUC gap")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def make_figures(master: pd.DataFrame, cal: pd.DataFrame,
                 sig: Dict[str, Dict[str, float]], fig_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.savefig_util import save_fig

    written: List[str] = []
    for ds in DATASETS:
        m = master[master["dataset"] == ds]
        gt = m[(m["source"] == "gt") & (m["split"] == "test")]
        pr = m[m["source"] == "pred_test"]
        cols = ["image_id"] + list(PRIMARY_COLS)
        d = gt[cols].merge(pr[cols], on="image_id", suffixes=("__gt", "__pred"))
        fig, axes = plt.subplots(len(PANEL_SKAN), 2,
                                 figsize=(9.0, 3.0 * len(PANEL_SKAN)))
        for i, col in enumerate(PANEL_SKAN):
            s = sig.get(ds, {}).get(col, float("nan"))
            x0 = d[col + "__gt"].to_numpy(float)
            y0 = d[col + "__pred"].to_numpy(float)
            ok = np.isfinite(x0) & np.isfinite(y0)
            med = float(np.median(x0[ok])) if ok.any() else 0.0
            x, y = (x0[ok] - med) / s, (y0[ok] - med) / s
            axL, axR = axes[i, 0], axes[i, 1]

            # ---- left: identity-line scatter with the two fits
            axL.scatter(x, y, s=14, alpha=0.65, color="#3b6ea5",
                        edgecolor="none")
            if x.size >= 3:
                xx = np.linspace(float(np.min(x)), float(np.max(x)), 50)
                axL.plot(xx, xx, color="0.35", lw=1.0, ls=":", label="identity")
                r = cal[(cal["dataset"] == ds) & (cal["biomarker"] == col)]
                for meth, colr, lab in (("ols", "#c0504d", "OLS"),
                                        ("deming_l1", "#9bbb59", "Deming")):
                    rr = r[r["method"] == meth]
                    if len(rr):
                        a = float(rr["alpha"].iloc[0])
                        b = float(rr["beta"].iloc[0])
                        axL.plot(xx, a + b * xx, color=colr, lw=1.2,
                                 label=f"{lab}  b={b:.2f}")
            axL.set_xlabel("reference - cohort median  [sigma]", fontsize=8)
            axL.set_ylabel("predicted - cohort median  [sigma]", fontsize=8)
            axL.set_title(f"{col}  (n={x.size})", fontsize=9)
            axL.legend(fontsize=7, frameon=False)
            axL.grid(alpha=0.22, lw=0.5)
            axL.tick_params(labelsize=7)

            # ---- right: Bland-Altman
            dd, mm = y - x, (x + y) / 2.0
            axR.scatter(mm, dd, s=14, alpha=0.65, color="#8064a2",
                        edgecolor="none")
            if dd.size >= 3:
                md = float(np.mean(dd))
                sdd = float(np.std(dd, ddof=1))
                axR.axhline(md, color="#c0504d", lw=1.1,
                            label=f"mean {md:+.2f}")
                for lv in (md - 1.96 * sdd, md + 1.96 * sdd):
                    axR.axhline(lv, color="#c0504d", lw=0.9, ls="--")
                axR.axhline(0.0, color="0.35", lw=0.8, ls=":")
                a, b = ols(mm, dd)
                xx = np.linspace(float(np.min(mm)), float(np.max(mm)), 50)
                axR.plot(xx, a + b * xx, color="#e8a33d", lw=1.2,
                         label=f"prop. bias {b:+.2f}")
                axR.set_ylim(md - 3.4 * sdd, md + 3.4 * sdd)
            axR.set_xlabel("mean of the two - cohort median  [sigma]", fontsize=8)
            axR.set_ylabel("predicted - reference  [sigma]", fontsize=8)
            axR.set_title("Bland-Altman", fontsize=9)
            axR.legend(fontsize=7, frameon=False)
            axR.grid(alpha=0.22, lw=0.5)
            axR.tick_params(labelsize=7)

        fig.suptitle(f"{ds.upper()} -- calibration of predicted against "
                     f"reference biomarkers (primary skan panel)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        written += save_fig(fig, os.path.join(fig_dir,
                                              f"r2_calibration_{ds}.png"),
                            dpi=200)
        plt.close(fig)
    return written


# ---------------------------------------- part 3: proportional distortion arm
def macro_auc(y: np.ndarray, proba: np.ndarray, classes: Sequence) -> float:
    from sklearn.metrics import roc_auc_score

    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() in (0, len(yy)):
            continue
        a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def _model(kind: str, rep: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=1.0))
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.08, min_samples_leaf=10,
        l2_regularization=1.0, early_stopping=False, random_state=SEED + rep)


def cv_proba(X: np.ndarray, y: np.ndarray, kind: str, classes: Sequence
             ) -> np.ndarray:
    """The identical retrained-classifier protocol used everywhere else."""
    from sklearn.model_selection import StratifiedKFold

    acc = np.zeros((len(y), len(classes)), float)
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            m = _model(kind, rep)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            order = [list(m.classes_).index(c) for c in classes]
            acc[te] += p[:, order]
    return acc / N_REPEAT


def cv_proba_fixed(X_fit: np.ndarray, X_apply: np.ndarray, y: np.ndarray,
                   kind: str, classes: Sequence) -> np.ndarray:
    """Fit on the UNDISTORTED features, score the DISTORTED ones.

    Same folds; inside each fold the model is trained on ``X_fit[tr]`` and then
    applied, frozen, to ``X_apply[te]``.  This is the "fixed rule" contrast:
    the coefficients (or split thresholds) never see the distorted scale.
    """
    from sklearn.model_selection import StratifiedKFold

    acc = np.zeros((len(y), len(classes)), float)
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=SEED + rep)
        for tr, te in skf.split(X_fit, y):
            m = _model(kind, rep)
            m.fit(X_fit[tr], y[tr])
            p = m.predict_proba(X_apply[te])
            order = [list(m.classes_).index(c) for c in classes]
            acc[te] += p[:, order]
    return acc / N_REPEAT


def locked_rule_metrics(y: np.ndarray, proba: np.ndarray, classes: Sequence
                        ) -> Dict[str, float]:
    """What a LOCKED decision rule actually does, beyond the AUC.

    The methods consultation is explicit that AUC is the wrong endpoint for a
    frozen-threshold arm, because AUC does not depend on the threshold at all.
    So the locked arm also reports what the decision rule does: the
    argmax-class decision, its macro sensitivity and specificity, the predicted
    positive rate per class (decision drift), and two calibration statistics --
    the mean probability assigned to the true class and the multi-class Brier
    score.
    """
    cls = list(classes)
    pred = np.array([cls[i] for i in np.argmax(proba, axis=1)])
    sens, spec, ppr = [], [], []
    for i, c in enumerate(cls):
        pos = (y == c)
        dec = (pred == c)
        if pos.sum():
            sens.append(float((dec & pos).sum() / pos.sum()))
        if (~pos).sum():
            spec.append(float(((~dec) & (~pos)).sum() / (~pos).sum()))
        ppr.append(float(dec.mean()))
    onehot = np.zeros_like(proba)
    for i, c in enumerate(cls):
        onehot[y == c, i] = 1.0
    return {
        "accuracy": float((pred == y).mean()),
        "macro_sensitivity": float(np.mean(sens)) if sens else float("nan"),
        "macro_specificity": float(np.mean(spec)) if spec else float("nan"),
        "ppr": ";".join("%s=%.4f" % (c, v) for c, v in zip(cls, ppr)),
        "mean_p_true_class": float(np.mean(np.sum(proba * onehot, axis=1))),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
    }


def fidelity(Xa: np.ndarray, Xb: np.ndarray) -> Dict[str, float]:
    from scipy.stats import pearsonr

    rs = []
    for j in range(Xa.shape[1]):
        a, b = Xa[:, j], Xb[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        rs.append(float(pearsonr(a[ok], b[ok])[0]))
    if not rs:
        return dict(r_mean=float("nan"), r_min=float("nan"))
    return dict(r_mean=float(np.mean(rs)), r_min=float(np.min(rs)))


def resid_sd_sigma(Xa: np.ndarray, Xb: np.ndarray, sig: np.ndarray) -> float:
    z = (Xa - Xb) / sig[None, :]
    return float(np.mean(np.std(z - np.nanmean(z, axis=0)[None, :], axis=0,
                                ddof=1)))


def proportional_arm(pred_csv: str, datasets: Sequence[str],
                     clfs: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(pred_csv, low_memory=False)
    sigtab = sigma_table()
    rows: List[dict] = []
    fixed: List[dict] = []
    src = ("results/pivot/p2_predictions.csv (skan4 panel, reference-mask "
           "biomarkers); B' = mean + k*(B - mean) per biomarker, applied to "
           "the REFERENCE features; same CV protocol as "
           "src/pivot/e1_dose_response.py (5-fold x 3 repeats, seeds 0..2)")
    for ds in datasets:
        d = pred[(pred["dataset"] == ds) & pred["disease"].notna()
                 ].reset_index(drop=True)
        if len(d) < 30:
            continue
        y = d["disease"].to_numpy()
        classes = sorted(pd.unique(y))
        Xg = np.nan_to_num(d[[c + "__gt" for c in PANEL_SKAN]].to_numpy(float),
                           nan=0.0, posinf=0.0, neginf=0.0)
        sig = np.array([sigtab[ds][c] for c in PANEL_SKAN], float)
        mu = Xg.mean(axis=0)
        for kind in clfs:
            pg = cv_proba(Xg, y, kind, classes)
            auc_ref = macro_auc(y, pg, classes)
            ref_locked = locked_rule_metrics(y, pg, classes)
            rows.append(dict(dataset=ds, clf=kind, injection="proportional",
                             arm="reference", gamma=1.0, k=1.0, n=len(y),
                             n_classes=len(classes), r_mean=1.0, r_min=1.0,
                             resid_sd=0.0, macro_auc=auc_ref,
                             delta_auc=0.0, source=src))
            fixed.append(dict(dataset=ds, clf=kind, arm="reference", gamma=1.0,
                              k=1.0, n=len(y), macro_auc=auc_ref,
                              delta_auc=0.0, **ref_locked,
                              d_accuracy=0.0, d_macro_sensitivity=0.0,
                              d_macro_specificity=0.0, d_mean_p_true_class=0.0,
                              d_brier=0.0,
                              source=src + "; locked-deployment arm: model, "
                                           "preprocessing and decision rule "
                                           "fitted on the UNDISTORTED "
                                           "reference features"))
            for k in K_GRID:
                X = mu[None, :] + k * (Xg - mu[None, :])
                f = fidelity(X, Xg)
                a = macro_auc(y, cv_proba(X, y, kind, classes), classes)
                rows.append(dict(
                    dataset=ds, clf=kind, injection="proportional",
                    arm="injected", gamma=float(k), k=float(k), n=len(y),
                    n_classes=len(classes), r_mean=f["r_mean"],
                    r_min=f["r_min"], resid_sd=resid_sd_sigma(X, Xg, sig),
                    macro_auc=a, delta_auc=float(a - auc_ref), source=src))
                pf = cv_proba_fixed(Xg, X, y, kind, classes)
                af = macro_auc(y, pf, classes)
                lk = locked_rule_metrics(y, pf, classes)
                fixed.append(dict(
                    dataset=ds, clf=kind, arm="injected", gamma=float(k),
                    k=float(k), n=len(y), macro_auc=af,
                    delta_auc=float(af - auc_ref), **lk,
                    d_accuracy=float(lk["accuracy"] - ref_locked["accuracy"]),
                    d_macro_sensitivity=float(lk["macro_sensitivity"]
                                              - ref_locked["macro_sensitivity"]),
                    d_macro_specificity=float(lk["macro_specificity"]
                                              - ref_locked["macro_specificity"]),
                    d_mean_p_true_class=float(lk["mean_p_true_class"]
                                              - ref_locked["mean_p_true_class"]),
                    d_brier=float(lk["brier"] - ref_locked["brier"]),
                    source=src + "; locked-deployment arm: model, "
                                 "preprocessing, calibration and decision rule "
                                 "fitted on the UNDISTORTED features and "
                                 "applied frozen to the distorted ones. AUC is "
                                 "threshold-free and is reported only for "
                                 "continuity; the decision-drift columns "
                                 "(accuracy, macro sensitivity/specificity, "
                                 "predicted positive rate, mean probability of "
                                 "the true class, Brier) are the endpoints "
                                 "that a locked rule actually moves."))
    return pd.DataFrame(rows), pd.DataFrame(fixed)


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--pred", default=os.path.join(PIVOT_DIR, "p2_predictions.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    ap.add_argument("--fig_dir", default=os.path.join(EXP_ROOT, "figs", "pivot"))
    ap.add_argument("--datasets", default="hrf,fives")
    ap.add_argument("--clfs", default="logreg,gbdt")
    ap.add_argument("--skip_calibration", action="store_true")
    ap.add_argument("--skip_dose", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    sig = sigma_table()

    if not args.skip_calibration:
        master = pd.read_csv(args.master, low_memory=False)
        cal = calibration_rows(master, sig)
        p = os.path.join(args.out_dir, "r2_calibration.csv")
        cal.to_csv(p, index=False)
        print("wrote", p, cal.shape)
        mt = main_table(cal, args.out_dir)
        p = os.path.join(args.out_dir, "r2_main_table.csv")
        mt.to_csv(p, index=False)
        print("wrote", p, mt.shape)
        make_figures(master, cal, sig, args.fig_dir)
        with pd.option_context("display.width", 230, "display.max_rows", 400):
            print("\n[calibration, primary skan panel, Deming lambda=1]")
            show = cal[(cal["panel"] == "primary") &
                       (cal["method"] == "deming_l1")]
            print(show[["dataset", "biomarker", "n", "alpha", "alpha_lo",
                        "alpha_hi", "beta", "beta_lo", "beta_hi", "resid_sd",
                        "r_pearson", "ccc", "ba_mean_diff", "ba_loa_lo",
                        "ba_loa_hi", "prop_bias_slope", "prop_bias_p"]]
                  .round(4).to_string(index=False))
            print("\n[calibration, primary skan panel, OLS]")
            show = cal[(cal["panel"] == "primary") & (cal["method"] == "ols")]
            print(show[["dataset", "biomarker", "alpha", "beta", "beta_lo",
                        "beta_hi", "ccc", "prop_bias_slope", "prop_bias_p"]]
                  .round(4).to_string(index=False))

    if not args.skip_dose:
        dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        clfs = [c.strip() for c in args.clfs.split(",") if c.strip()]
        rows, fixed = proportional_arm(args.pred, dsets, clfs)
        p = os.path.join(args.out_dir, "r2_dose_proportional.csv")
        rows.to_csv(p, index=False)
        print("wrote", p, rows.shape)
        q = os.path.join(args.out_dir, "r2_dose_proportional_fixedrule.csv")
        fixed.to_csv(q, index=False)
        print("wrote", q, fixed.shape)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print("\n[proportional distortion, retrained classifier]")
            print(rows[["dataset", "clf", "arm", "gamma", "r_mean",
                        "resid_sd", "macro_auc",
                        "delta_auc"]].round(6).to_string(index=False))
            print("\n[proportional distortion, FIXED rule "
                  "(fit undistorted, apply frozen)]")
            print(fixed[["dataset", "clf", "arm", "gamma", "macro_auc",
                         "delta_auc", "accuracy", "d_accuracy",
                         "macro_sensitivity", "d_macro_sensitivity",
                         "macro_specificity", "d_macro_specificity",
                         "mean_p_true_class", "brier", "d_brier", "ppr"]]
                  .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
