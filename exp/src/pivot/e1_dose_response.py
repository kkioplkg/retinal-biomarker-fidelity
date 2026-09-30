"""E1 -- within-dataset error-injection dose-response, two arms.

Why this script exists
----------------------
The manuscript's association between image-level measurement fidelity and the
downstream reference gap is an **ecological contrast between two labelled
cohorts** (HRF and FIVES), which differ in resolution, sample size, disease mix
and annotation protocol as well as in fidelity.  A reviewer proposed the cheap
experiment that turns the association into a within-dataset statement, and the
design consultation of ``review/design_e4_reply.md`` fixed its shape
(``exp/DECISIONS.md`` 2026-09-17 13:30, item 10).  This is it: start from the
**reference** biomarkers of one cohort, perturb them in one of two controlled
ways, and trace the downstream AUC.

``shift``     axis 1.  ``B_i <- B_i + c * sigma_b``, the same constant for every
              image.  The dataset mean moves and nothing else does.
``residual``  axis 2.  ``B_i <- B_i + sigma_b * delta_i`` with ``delta_i`` drawn
              image by image, **calibrated to hit a target image-level Pearson
              r with the reference** and drawn from a multivariate normal whose
              correlation matrix is the one the *observed* de-biased residuals
              have across the four panel biomarkers.  So the injected error has
              the cross-biomarker structure the real error has, and the x-axis
              of the curve is the same fidelity coefficient the audit reports.

Calibration.  If ``B`` has SD ``g_j`` in sigma_b units and an independent
residual of SD ``s_j`` is added, the Pearson correlation between injected and
reference is ``r_j = g_j / sqrt(g_j^2 + s_j^2)``, so a target ``r`` needs
``s_j = g_j * sqrt(1/r^2 - 1)``.  The achieved r is measured, not assumed, and
is what the CSV and the figure use.

Everything else -- labels, classifier, folds, CV repeats -- is held fixed, so
the only thing that varies along a curve is the measurement error.  No
segmenter is retrained; this is CPU-only.

What it can and cannot show.  It is a simulation of measurement error on top of
the reference masks, so it shows how this classifier on this cohort responds to
error of a given size and shape.  The injected residual is Gaussian and
homoscedastic across images (the real error is neither, though its
cross-biomarker correlation is preserved), and it is independent of the disease
label by construction, whereas real segmentation error need not be.  Both are
stated in the manuscript rather than hidden.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.e1_dose_response
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS, sigma_table

SEED = 0
N_REPEAT = 3
N_SPLITS = 5
PANEL = [c for c in PRIMARY_COLS if c.endswith("_skan")]

#: target image-level Pearson r for the residual arm.  1.0 is the untouched
#: reference arm; the grid brackets both cohorts' observed fidelity.
TARGET_R = (1.0, 0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40)
#: constant-shift doses, in units of the frozen sigma_b
SHIFT_DOSES = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)


def macro_auc(y: np.ndarray, proba: np.ndarray, classes: Sequence) -> float:
    from sklearn.metrics import roc_auc_score

    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() in (0, len(yy)):
            continue
        a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def cv_proba(X: np.ndarray, y: np.ndarray, kind: str, classes: Sequence
             ) -> np.ndarray:
    """The identical protocol used everywhere else in this project."""
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


def resid_sd(Xa: np.ndarray, Xb: np.ndarray, sig: np.ndarray) -> float:
    """Mean over columns of the de-biased residual SD in sigma_b units."""
    z = (Xa - Xb) / sig[None, :]
    return float(np.mean(np.std(z - np.nanmean(z, axis=0)[None, :], axis=0,
                                ddof=1)))


def residual_structure(Xp: np.ndarray, Xg: np.ndarray, sig: np.ndarray
                       ) -> np.ndarray:
    """Correlation matrix of the OBSERVED de-biased residuals across the panel.

    This is what makes the injected error look like the real one: segmentation
    error in length, density and fractal dimension moves together, and an
    injection of independent noise would understate how a multivariate
    classifier degrades.
    """
    z = (Xp - Xg) / sig[None, :]
    z = z - np.nanmean(z, axis=0)[None, :]
    R = np.corrcoef(z, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    # nearest-PSD nudge: clip tiny negative eigenvalues from sampling noise
    w, V = np.linalg.eigh(R)
    if w.min() < 1e-8:
        w = np.clip(w, 1e-8, None)
        R = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
    return R


def run_dataset(d: pd.DataFrame, ds: str, sig: np.ndarray, clfs: Sequence[str],
                n_seeds: int, n_seeds_gbdt: int = 0) -> List[dict]:
    y = d["disease"].to_numpy()
    classes = sorted(pd.unique(y))
    Xg = np.nan_to_num(d[[c + "__gt" for c in PANEL]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    Xp = np.nan_to_num(d[[c + "__pred" for c in PANEL]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)

    R = residual_structure(Xp, Xg, sig)
    L = np.linalg.cholesky(R)
    g = np.std(Xg / sig[None, :], axis=0, ddof=1)     # reference spread, sigma units

    rows: List[dict] = []
    src = ("results/pivot/p2_predictions.csv (skan4 panel, reference-mask "
           "biomarkers); sigma = results/gateA_biomarker_scales_train.csv; "
           "residual arm drawn MVN with the correlation matrix of the observed "
           "de-biased residuals of the same cohort, scaled per biomarker to hit "
           "the target r; same CV protocol as p3_downstream (5-fold x 3 "
           f"repeats, seeds {SEED}..{SEED + N_REPEAT - 1})")
    rstruct = ";".join(f"{PANEL[i]}~{PANEL[j]}={R[i, j]:.3f}"
                       for i in range(len(PANEL)) for j in range(i + 1, len(PANEL)))

    for kind in clfs:
        # ---- the two measured anchors
        rows.append(dict(dataset=ds, clf=kind, injection="observed",
                         arm="reference", target_r=1.0, dose=0.0, seed=-1,
                         n=len(y), n_classes=len(classes), r_mean=1.0,
                         r_min=1.0, resid_sd=0.0,
                         macro_auc=macro_auc(y, cv_proba(Xg, y, kind, classes),
                                             classes),
                         resid_corr=rstruct, source=src))
        f = fidelity(Xp, Xg)
        rows.append(dict(dataset=ds, clf=kind, injection="observed",
                         arm="predicted", target_r=float("nan"),
                         dose=float("nan"), seed=-1, n=len(y),
                         n_classes=len(classes), r_mean=f["r_mean"],
                         r_min=f["r_min"], resid_sd=resid_sd(Xp, Xg, sig),
                         macro_auc=macro_auc(y, cv_proba(Xp, y, kind, classes),
                                             classes),
                         resid_corr=rstruct, source=src))

        # ---- arm 2: image-specific correlated residual, calibrated to target r
        ns = n_seeds_gbdt if (kind == "gbdt" and n_seeds_gbdt) else n_seeds
        for tr_ in TARGET_R:
            reps = 1 if tr_ >= 1.0 else ns
            s = np.zeros_like(g) if tr_ >= 1.0 \
                else g * np.sqrt(1.0 / (tr_ ** 2) - 1.0)
            for k in range(reps):
                rng = np.random.RandomState(1000 * k + 7)
                delta = (rng.standard_normal(Xg.shape) @ L.T) * s[None, :]
                X = Xg + delta * sig[None, :]
                f = fidelity(X, Xg)
                rows.append(dict(
                    dataset=ds, clf=kind, injection="residual", arm="injected",
                    target_r=float(tr_), dose=float(np.mean(s)), seed=k,
                    n=len(y), n_classes=len(classes), r_mean=f["r_mean"],
                    r_min=f["r_min"], resid_sd=resid_sd(X, Xg, sig),
                    macro_auc=macro_auc(y, cv_proba(X, y, kind, classes),
                                        classes),
                    resid_corr=rstruct, source=src))

        # ---- arm 1: a constant shift, deterministic
        for c in SHIFT_DOSES:
            X = Xg + c * sig[None, :]
            f = fidelity(X, Xg)
            rows.append(dict(
                dataset=ds, clf=kind, injection="shift", arm="injected",
                target_r=1.0, dose=float(c), seed=0, n=len(y),
                n_classes=len(classes), r_mean=f["r_mean"], r_min=f["r_min"],
                resid_sd=resid_sd(X, Xg, sig),
                macro_auc=macro_auc(y, cv_proba(X, y, kind, classes), classes),
                resid_corr=rstruct, source=src))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(PIVOT_DIR, "p2_predictions.csv"))
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    ap.add_argument("--datasets", default="fives,hrf")
    ap.add_argument("--clfs", default="logreg,gbdt")
    ap.add_argument("--n_seeds", type=int, default=20,
                    help="independent residual draws per target r")
    ap.add_argument("--n_seeds_gbdt", type=int, default=0,
                    help="separate draw count for the slower GBDT sensitivity "
                         "arm; 0 means use --n_seeds")
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred, low_memory=False)
    sigtab = sigma_table()
    clfs = [c.strip() for c in args.clfs.split(",") if c.strip()]

    rows: List[dict] = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        d = pred[pred["dataset"] == ds]
        d = d[d["disease"].notna()].reset_index(drop=True)
        if len(d) < 30:
            continue
        sig = np.array([sigtab[ds][c] for c in PANEL], float)
        t0 = time.time()
        rows += run_dataset(d, ds, sig, clfs, args.n_seeds, args.n_seeds_gbdt)
        print(f"[dose] {ds}: {len(d)} images, {round(time.time() - t0, 1)}s")

    out = pd.DataFrame(rows)
    p = os.path.join(args.out_dir, "e1_dose_response.csv")
    out.to_csv(p, index=False)
    print("wrote", p, out.shape)

    g = (out[out["injection"] != "observed"]
         .groupby(["dataset", "clf", "injection", "target_r", "dose"],
                  as_index=False, dropna=False)
         .agg(n_draws=("macro_auc", "size"),
              r_mean=("r_mean", "mean"),
              resid_sd=("resid_sd", "mean"),
              auc=("macro_auc", "mean"),
              auc_lo=("macro_auc", lambda v: float(np.percentile(v, 2.5))),
              auc_hi=("macro_auc", lambda v: float(np.percentile(v, 97.5)))))
    g["source"] = ("mean over independent residual draws of "
                   "results/pivot/e1_dose_response.csv")
    q = os.path.join(args.out_dir, "e1_dose_response_summary.csv")
    g.to_csv(q, index=False)
    print("wrote", q, g.shape)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(g.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
