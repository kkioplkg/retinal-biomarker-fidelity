"""R2 -- robustness panel for the residual-injection dose-response.

Reviewer point 8 (``review/user_cmig_review_20260918.md``)
---------------------------------------------------------
"The injection model is too friendly: add an empirical residual bootstrap, a
heteroscedastic arm, a label-dependent arm and a proportional-distortion arm."

The shipped experiment (``src/pivot/e1_dose_response.py``) injects a *Gaussian,
homoscedastic, label-independent* residual whose only concession to realism is
that it carries the cross-biomarker correlation matrix of the observed
residuals.  Every one of those three properties is a place where the simulation
could be flattering the conclusion, so this script re-runs the same
dose-response under six injection mechanisms and asks whether the
AUC-versus-achieved-fidelity curve moves.

The mechanisms
--------------
All of them inject ``delta`` in sigma units and are scaled per biomarker to hit
a target image-level Pearson ``r`` with the reference, using the same
calibration as the shipped script: if the reference has SD ``g_j`` in sigma
units and independent noise of SD ``s_j`` is added, then
``r_j = g_j / sqrt(g_j^2 + s_j^2)``, so ``s_j = g_j * sqrt(1/r^2 - 1)``.  The
*achieved* r is measured on the injected features, never assumed, and is the
x-axis of every curve.

``gaussian_indep``    independent Gaussian per biomarker.  The naive model.
``gaussian_corr``     Gaussian with the correlation matrix of the observed
                      de-biased residuals.  **This is the shipped arm**, repeated
                      here so the panel is self-contained and the other five can
                      be read against it.
``resid_bootstrap``   the observed de-biased residual *vectors* are resampled
                      with replacement, whole rows at a time.  This keeps the
                      real marginal shape (skew, heavy tails, the near-ties that
                      tortuosity produces) and the real cross-biomarker
                      dependence, including whatever is non-Gaussian about it.
                      Each column is then standardised to unit SD and rescaled
                      to ``s_j`` so the arm sits on the same r axis.
``hetero_reference``  heteroscedastic in the reference value: the per-image
                      noise multiplier is proportional to the image's reference
                      biomarker level (normalised to unit mean square), so large
                      vessels carry proportionally larger error.
``hetero_quality``    heteroscedastic in an image-quality proxy computable at
                      inference time without any label -- ``feat_contrast`` from
                      ``bio_master.csv`` (robust-percentile Michelson contrast
                      inside the FOV).  The multiplier is larger for low-contrast
                      images.
``label_dependent``   the per-class residual mean and SD of the **observed**
                      pred-minus-reference residuals are estimated per disease
                      class and per biomarker, and the injected residual
                      reproduces that class structure (relative per-class means
                      and SD ratios held fixed, overall level scaled to the
                      target r).  This is the one arm in which the injected
                      error carries information about the label, so it is the
                      arm that can move the AUC in either direction.
``proportional``      ``B'_i = mean + k*(B_i - mean)``.  A pure affine map, so its
                      achieved r is 1 by construction; it is plotted as a
                      separate marker rather than on the r axis.

Outputs (``results/pivot/r2/``)
-------------------------------
``r2_injection_panel.csv``          one row per draw
``r2_injection_panel_summary.csv``  mean and 2.5/97.5 percentile over draws
``figs/pivot/r2_injection_panel.png|pdf``
                                    AUC vs achieved r, one panel per
                                    (dataset, classifier), one line per
                                    mechanism, with the real measurement marked.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_injection_panel
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import EXP_ROOT, PIVOT_DIR, PRIMARY_COLS, sigma_table

SEED = 0
N_REPEAT = 3
N_SPLITS = 5
PANEL = [c for c in PRIMARY_COLS if c.endswith("_skan")]
TARGET_R = (0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40)
K_GRID = (0.5, 0.75, 1.25, 1.5, 2.0)
MECHANISMS = ("gaussian_indep", "gaussian_corr", "resid_bootstrap",
              "hetero_reference", "hetero_quality", "label_dependent")


# --------------------------------------------------------------- downstream
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
    rs = []
    for j in range(Xa.shape[1]):
        a, b = Xa[:, j], Xb[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        rs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    if not rs:
        return dict(r_mean=float("nan"), r_min=float("nan"))
    return dict(r_mean=float(np.mean(rs)), r_min=float(np.min(rs)))


def resid_sd(Xa: np.ndarray, Xb: np.ndarray, sig: np.ndarray) -> float:
    z = (Xa - Xb) / sig[None, :]
    return float(np.mean(np.std(z - np.nanmean(z, axis=0)[None, :], axis=0,
                                ddof=1)))


def residual_corr(Z: np.ndarray) -> np.ndarray:
    R = np.corrcoef(Z, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh(R)
    if w.min() < 1e-8:
        w = np.clip(w, 1e-8, None)
        R = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
    return R


# --------------------------------------------------------- noise generators
def _unit_cols(a: np.ndarray) -> np.ndarray:
    """Centre and scale every column to unit SD (so the caller controls s_j)."""
    a = a - a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return a / sd


def draw_noise(mech: str, rng: np.random.RandomState, n: int, L: np.ndarray,
               Z: np.ndarray, ref_z: np.ndarray, quality: np.ndarray,
               y: np.ndarray, classes: Sequence) -> np.ndarray:
    """Unit-SD-per-column noise of shape (n, p) with the mechanism's structure."""
    p = L.shape[0]
    if mech == "gaussian_indep":
        e = rng.standard_normal((n, p))
    elif mech == "gaussian_corr":
        e = rng.standard_normal((n, p)) @ L.T
    elif mech == "resid_bootstrap":
        idx = rng.randint(0, Z.shape[0], n)
        e = Z[idx].copy()
    elif mech in ("hetero_reference", "hetero_quality"):
        base = rng.standard_normal((n, p)) @ L.T
        if mech == "hetero_reference":
            # multiplier proportional to the reference level, per biomarker
            w = ref_z - ref_z.min(axis=0, keepdims=True) + 0.1
        else:
            # low contrast -> larger error; map the proxy to [0.1, 1.1]
            q = quality.astype(float)
            ok = np.isfinite(q)
            if ok.sum() < 3 or np.nanstd(q) == 0:
                w = np.ones((n, 1))
            else:
                qq = np.where(ok, q, np.nanmedian(q[ok]))
                lo, hi = np.nanpercentile(qq, [5, 95])
                u = np.clip((qq - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
                w = (1.1 - u)[:, None]
            w = np.repeat(w, p, axis=1)
        w = w / np.sqrt(np.mean(w ** 2, axis=0, keepdims=True))
        e = base * w
    elif mech == "label_dependent":
        base = rng.standard_normal((n, p)) @ L.T
        e = np.zeros((n, p))
        for c in classes:
            m_obs = (y == c)
            m_new = (y == c)
            if m_obs.sum() < 3:
                e[m_new] = base[m_new]
                continue
            mu_c = Z[m_obs].mean(axis=0)
            sd_c = Z[m_obs].std(axis=0, ddof=1)
            sd_all = Z.std(axis=0, ddof=1)
            ratio = np.where(sd_all > 0, sd_c / sd_all, 1.0)
            e[m_new] = base[m_new] * ratio[None, :] + mu_c[None, :]
    else:
        raise ValueError(mech)
    return _unit_cols(e)


# --------------------------------------------------------------------------
def run_dataset(d: pd.DataFrame, ds: str, sig: np.ndarray, quality: np.ndarray,
                clfs: Sequence[str], n_seeds: int, n_seeds_gbdt: int
                ) -> List[dict]:
    y = d["disease"].to_numpy()
    classes = sorted(pd.unique(y))
    Xg = np.nan_to_num(d[[c + "__gt" for c in PANEL]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    Xp = np.nan_to_num(d[[c + "__pred" for c in PANEL]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y)

    Zraw = (Xp - Xg) / sig[None, :]
    Z = Zraw - Zraw.mean(axis=0, keepdims=True)      # de-biased residuals
    R = residual_corr(Z)
    L = np.linalg.cholesky(R)
    g = np.std(Xg / sig[None, :], axis=0, ddof=1)    # reference spread, sigma
    ref_z = Xg / sig[None, :]
    mu = Xg.mean(axis=0)

    src = ("results/pivot/p2_predictions.csv (skan4 panel, reference-mask "
           "biomarkers); sigma = results/gateA_biomarker_scales_train.csv; "
           "quality proxy feat_contrast from results/pivot/bio_master.csv "
           "(source=gt); CV protocol identical to src/pivot/e1_dose_response.py "
           f"(5-fold x {N_REPEAT} repeats, seeds {SEED}..{SEED + N_REPEAT - 1})")

    rows: List[dict] = []
    for kind in clfs:
        ns = n_seeds_gbdt if kind == "gbdt" else n_seeds
        auc_ref = macro_auc(y, cv_proba(Xg, y, kind, classes), classes)
        rows.append(dict(dataset=ds, clf=kind, mechanism="observed",
                         arm="reference", target_r=1.0, k=float("nan"), seed=-1,
                         n=n, n_classes=len(classes), r_mean=1.0, r_min=1.0,
                         resid_sd=0.0, macro_auc=auc_ref, delta_auc=0.0,
                         source=src))
        f = fidelity(Xp, Xg)
        rows.append(dict(dataset=ds, clf=kind, mechanism="observed",
                         arm="real_measurement", target_r=float("nan"),
                         k=float("nan"), seed=-1, n=n, n_classes=len(classes),
                         r_mean=f["r_mean"], r_min=f["r_min"],
                         resid_sd=resid_sd(Xp, Xg, sig),
                         macro_auc=macro_auc(y, cv_proba(Xp, y, kind, classes),
                                             classes),
                         delta_auc=float("nan"), source=src))
        rows[-1]["delta_auc"] = rows[-1]["macro_auc"] - auc_ref

        for mech in MECHANISMS:
            for tr_ in TARGET_R:
                s = g * np.sqrt(1.0 / (tr_ ** 2) - 1.0)
                for k in range(ns):
                    rng = np.random.RandomState(1000 * k + 7)
                    e = draw_noise(mech, rng, n, L, Z, ref_z, quality, y,
                                   classes)
                    X = Xg + (e * s[None, :]) * sig[None, :]
                    fi = fidelity(X, Xg)
                    a = macro_auc(y, cv_proba(X, y, kind, classes), classes)
                    rows.append(dict(
                        dataset=ds, clf=kind, mechanism=mech, arm="injected",
                        target_r=float(tr_), k=float("nan"), seed=k, n=n,
                        n_classes=len(classes), r_mean=fi["r_mean"],
                        r_min=fi["r_min"], resid_sd=resid_sd(X, Xg, sig),
                        macro_auc=a, delta_auc=float(a - auc_ref), source=src))

        for kk in K_GRID:
            X = mu[None, :] + kk * (Xg - mu[None, :])
            fi = fidelity(X, Xg)
            a = macro_auc(y, cv_proba(X, y, kind, classes), classes)
            rows.append(dict(
                dataset=ds, clf=kind, mechanism="proportional", arm="injected",
                target_r=1.0, k=float(kk), seed=0, n=n,
                n_classes=len(classes), r_mean=fi["r_mean"], r_min=fi["r_min"],
                resid_sd=resid_sd(X, Xg, sig), macro_auc=a,
                delta_auc=float(a - auc_ref), source=src))
    return rows


# ------------------------------------------------------------------ figure
def make_figure(summ: pd.DataFrame, raw: pd.DataFrame, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.savefig_util import save_fig

    colours = {"gaussian_indep": "#7f7f7f", "gaussian_corr": "#3b6ea5",
               "resid_bootstrap": "#c0504d", "hetero_reference": "#9bbb59",
               "hetero_quality": "#e8a33d", "label_dependent": "#8064a2"}
    dsets = [d for d in ("hrf", "fives") if d in set(summ["dataset"])]
    clfs = [c for c in ("logreg", "gbdt") if c in set(summ["clf"])]
    fig, axes = plt.subplots(len(dsets), len(clfs),
                             figsize=(5.2 * len(clfs), 4.0 * len(dsets)),
                             squeeze=False)
    for i, ds in enumerate(dsets):
        for j, kind in enumerate(clfs):
            ax = axes[i][j]
            for mech, colr in colours.items():
                g = summ[(summ["dataset"] == ds) & (summ["clf"] == kind) &
                         (summ["mechanism"] == mech)].sort_values("r_mean")
                if not len(g):
                    continue
                ax.plot(g["r_mean"], g["auc"], "-o", ms=3.2, lw=1.2,
                        color=colr, label=mech)
                ax.fill_between(g["r_mean"], g["auc_lo"], g["auc_hi"],
                                color=colr, alpha=0.13, lw=0)
            obs = raw[(raw["dataset"] == ds) & (raw["clf"] == kind) &
                      (raw["arm"] == "real_measurement")]
            ref = raw[(raw["dataset"] == ds) & (raw["clf"] == kind) &
                      (raw["arm"] == "reference")]
            if len(ref):
                ax.axhline(float(ref["macro_auc"].iloc[0]), color="0.35",
                           lw=0.9, ls=":", label="reference masks")
            if len(obs):
                ax.plot([float(obs["r_mean"].iloc[0])],
                        [float(obs["macro_auc"].iloc[0])], marker="*",
                        ms=15, color="k", ls="none",
                        label="real measurement", zorder=5)
            prop = summ[(summ["dataset"] == ds) & (summ["clf"] == kind) &
                        (summ["mechanism"] == "proportional")]
            if len(prop):
                ax.plot(prop["r_mean"], prop["auc"], marker="s", ms=5,
                        color="#2b8c6a", ls="none",
                        label="proportional (r=1)", zorder=4)
            ax.set_xlim(1.02, 0.34)
            ax.set_xlabel("achieved image-level fidelity  r", fontsize=9)
            ax.set_ylabel("macro one-vs-rest AUC", fontsize=9)
            ax.set_title(f"{ds.upper()}  --  {kind}", fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
            ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("R2  injection-mechanism robustness panel: the AUC-vs-fidelity "
                 "curve under six error models", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_fig(fig, out_png, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(PIVOT_DIR, "p2_predictions.csv"))
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    ap.add_argument("--fig_dir", default=os.path.join(EXP_ROOT, "figs", "pivot"))
    ap.add_argument("--datasets", default="hrf,fives")
    ap.add_argument("--clfs", default="logreg,gbdt")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n_seeds_gbdt", type=int, default=5)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    pred = pd.read_csv(args.pred, low_memory=False)
    master = pd.read_csv(args.master, low_memory=False)
    qmap = (master[master["source"] == "gt"]
            .set_index(["dataset", "image_id"])["feat_contrast"].to_dict())
    sigtab = sigma_table()
    clfs = [c.strip() for c in args.clfs.split(",") if c.strip()]

    rows: List[dict] = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        d = pred[(pred["dataset"] == ds) & pred["disease"].notna()
                 ].reset_index(drop=True)
        if len(d) < 30:
            continue
        sig = np.array([sigtab[ds][c] for c in PANEL], float)
        q = np.array([qmap.get((ds, i), np.nan)
                      for i in d["image_id"].astype(str)], float)
        t0 = time.time()
        rows += run_dataset(d, ds, sig, q, clfs, args.n_seeds,
                            args.n_seeds_gbdt)
        print("[panel] %s: %d images, quality proxy on %d/%d, %.1fs"
              % (ds, len(d), int(np.isfinite(q).sum()), len(d),
                 time.time() - t0), flush=True)

    out = pd.DataFrame(rows)
    p = os.path.join(args.out_dir, "r2_injection_panel.csv")
    out.to_csv(p, index=False)
    print("wrote", p, out.shape)

    inj = out[out["arm"] == "injected"]
    summ = (inj.groupby(["dataset", "clf", "mechanism", "target_r", "k"],
                        as_index=False, dropna=False)
            .agg(n_draws=("macro_auc", "size"),
                 r_mean=("r_mean", "mean"),
                 resid_sd=("resid_sd", "mean"),
                 auc=("macro_auc", "mean"),
                 auc_lo=("macro_auc", lambda v: float(np.percentile(v, 2.5))),
                 auc_hi=("macro_auc", lambda v: float(np.percentile(v, 97.5))),
                 delta_auc=("delta_auc", "mean")))
    summ["source"] = ("mean over independent draws of "
                      "results/pivot/r2/r2_injection_panel.csv")
    q = os.path.join(args.out_dir, "r2_injection_panel_summary.csv")
    summ.to_csv(q, index=False)
    print("wrote", q, summ.shape)

    make_figure(summ, out, os.path.join(args.fig_dir, "r2_injection_panel.png"))

    with pd.option_context("display.width", 200, "display.max_rows", 900):
        print("\n[summary]")
        print(summ.round(4).to_string(index=False))
        print("\n[where the real measurement lands]")
        print(out[out["arm"].isin(["reference", "real_measurement"])]
              [["dataset", "clf", "arm", "r_mean", "resid_sd", "macro_auc",
                "delta_auc"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
