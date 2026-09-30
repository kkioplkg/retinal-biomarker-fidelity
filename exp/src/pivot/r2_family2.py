"""R2 -- the C1 measurement audit repeated for the SECOND segmenter families.

Reviewer point 9 / the second-family arm
----------------------------------------
The audit as shipped rests on one U-Net lineage.  The question a methods
reviewer actually asks is whether the measurement-error structure -- a
dataset-level offset of order 1 sigma, a slope below 1, a per-image residual of
the same order, and a large downstream reference gap only where image-level
fidelity is low -- is a property of *measuring biomarkers from an automatic
segmentation* or a quirk of that one architecture.  This script recomputes the
whole audit for the second families on exactly the frozen conventions of
family 1, so the two can be read side by side.

Families it handles (auto-discovered under ``runs/pivot/r2``)
-------------------------------------------------------------
``lwnet``         a public 68 k-parameter W-Net checkpoint, applied zero-shot.
                  **DRIVE is this checkpoint's own training set**, so only
                  CHASE_DB1 / HRF / FIVES are genuinely zero-shot; every row
                  carries a ``zero_shot`` flag.
``segformer_b0``  3.7 M-parameter transformer trained here on the same splits,
                  resolution and early-stopping rule as the U-Net, 2 seeds.

Any directory matching ``runs/pivot/r2/<family>/<dataset>[/seed<k>]/pred/bio.csv``
is picked up, so the script does not need editing when the SegFormer seeds land.

Pre-declared mask-failure rules (DECISIONS.md 2026-09-18 11:03, locked before
any family-2 audit number was looked at)
-------------------------------------------------------------------------------
1. ``fg_px == 0``  -> the prediction is a **segmentation failure**, not a
   measurement-calibration datum.  Excluded from every calibration and fidelity
   estimate and **counted separately**.
2. ``pred_fg_frac < 0.02`` (and not empty) -> **degenerate** mask.  **Kept in
   the main analysis**, excluded in a sensitivity analysis.  Both are reported.
3. The two analysis sets are therefore
   ``main``            = all images except the empty ones
   ``sens_no_degen``   = ``main`` minus the degenerate ones
   and a third, ``all_incl_empty``, is carried for the downstream arm only, so
   the cost of the failures can be priced rather than assumed away.

Everything else is frozen from family 1
---------------------------------------
Same primary panel (the four skan biomarkers, PVBM as sensitivity), same frozen
per-cohort sigma (``results/gateA_biomarker_scales_train.csv``), the same common
scale ``s_k`` (``src.pivot.r2_scale.common_scale``, the definition of
``review/cmig_plan_reply.md`` 2A), the same median-centred frame for alpha, the
same Deming lambda = 1 primary fit, the same 2000-draw paired image bootstrap,
and the same downstream protocol as ``p3_downstream.py`` (5-fold x 3 repeats,
macro one-vs-rest AUC, 1000-draw paired image bootstrap).

Outputs (``results/pivot/r2/``)
-------------------------------
``family2_mask_quality.csv``         per family x dataset: n, empty, degenerate,
                                     and the per-disease-class failure counts.
``family2_audit.csv``                mu (raw / sigma_own / s_k common), alpha,
                                     beta, residual SD, CCC, r, Bland-Altman
                                     proportional-bias slope, all with CIs, per
                                     family x seed x dataset x biomarker x
                                     analysis set.
``family2_delta_auc.csv``            reference-gap Delta AUC on HRF and FIVES,
                                     logistic + GBDT, per analysis set.
``family2_failure_attribution.csv``  how much of a family's FIVES gap is the
                                     mask failures: Delta AUC with vs without
                                     the failed images.
``family2_replication.csv``          does each family-1 conclusion replicate.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_family2
    OMP_NUM_THREADS=1 python -m src.pivot.r2_family2 --families lwnet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS, RUNS_DIR, sigma_table
from src.pivot.r2_calibration import ccc, deming, ols, prop_bias_p
from src.pivot.r2_scale import LENGTH_LIKE, common_scale

SEED = 0
N_BOOT = 2000
N_BOOT_AUC = 1000
N_REPEAT = 3
N_SPLITS = 5
DATASETS = ("drive", "chasedb1", "hrf", "fives")
PANEL_SKAN = [c for c in PRIMARY_COLS if c.endswith("_skan")]
PANEL = {c: ("primary" if c.endswith("_skan") else "sensitivity")
         for c in PRIMARY_COLS}

#: pre-declared thresholds (DECISIONS.md 2026-09-18 11:03)
EMPTY_FG_PX = 0
DEGENERATE_FG_FRAC = 0.02

#: which (family, dataset) pairs are genuinely zero-shot.  LWNet's public
#: checkpoint was trained on DRIVE, so its DRIVE column is in-domain and must
#: not be presented as zero-shot.
LWNET_TRAIN_DOMAIN = ("drive",)

#: Families trained here, in-domain, on each dataset's own training split.
#: These are NEVER zero-shot -- SegFormer-B0 is trained per dataset on the same
#: split, resolution and early-stopping rule as the U-Net, which is the whole
#: point of it as a controlled architecture comparison.
IN_DOMAIN_FAMILIES = ("segformer_b0", "unet")


def is_zero_shot(family: str, dataset: str) -> bool:
    if family in IN_DOMAIN_FAMILIES:
        return False
    if family == "lwnet":
        return dataset not in LWNET_TRAIN_DOMAIN
    # unknown family: be explicit rather than guess
    return False

ANALYSIS_SETS = ("main", "sens_no_degen")

#: A Deming slope whose bootstrap CI is wider than this is treated as
#: unidentified and excluded from the slope-direction verdict.
IDENT_CI_WIDTH = 2.0


# ------------------------------------------------------------- discovery
def discover(root: str, families: Optional[Sequence[str]] = None
             ) -> List[dict]:
    """Every ``pred/bio.csv`` under ``runs/pivot/r2``, with (family, ds, seed)."""
    out: List[dict] = []
    for p in sorted(glob.glob(os.path.join(root, "*", "**", "pred", "bio.csv"),
                              recursive=True)):
        rel = os.path.relpath(p, root).replace("\\", "/").split("/")
        if len(rel) < 3:
            continue
        family = rel[0]
        if family.startswith("_") or family == "logs":
            continue
        if families and family not in families:
            continue
        mid = rel[1:-2]                      # between family and pred/bio.csv
        ds, seed = None, None
        for part in mid:
            m = re.fullmatch(r"s(?:eed)?(\d+)", part)
            if m:
                seed = int(m.group(1))
                continue
            # <dataset>_seed<k>, <dataset>_s<k> (the layout the GPU agent
            # actually used) and <dataset>-seed<k> all mean the same thing
            m = re.fullmatch(r"(.+?)[-_]s(?:eed)?(\d+)", part)
            if m and m.group(1) in DATASETS:
                ds, seed = m.group(1), int(m.group(2))
                continue
            if part in DATASETS:
                ds = part
        if ds is None:
            continue
        if seed is None:
            sm = os.path.join(os.path.dirname(os.path.dirname(p)),
                              "summary.json")
            if os.path.exists(sm):
                try:
                    with open(sm, encoding="utf-8") as fh:
                        seed = int(json.load(fh).get("seed", 0))
                except Exception:                            # noqa: BLE001
                    seed = 0
            else:
                seed = 0
        out.append(dict(family=family, dataset=ds, seed=int(seed), path=p))
    return out


# ------------------------------------------------------------------ data
def load_reference(master_path: str) -> pd.DataFrame:
    m = pd.read_csv(master_path, low_memory=False)
    gt = m[m["source"] == "gt"]
    cols = ["dataset", "image_id", "split", "disease", "disc_fov_diameter"] \
        + list(PRIMARY_COLS)
    return gt[cols].reset_index(drop=True)


def fg_fraction(pred: pd.DataFrame) -> np.ndarray:
    """Predicted foreground fraction inside the FOV.

    ``density_skan`` is by definition ``vessel_area / region_area`` inside the
    FOV, which is the same quantity ``lwnet_mask_quality.csv`` records as
    ``pred_fg_frac``; ``fg_px / skan_fov_area_px`` is used when the density
    column is missing or NaN (an empty mask makes some columns NaN).
    """
    frac = pd.to_numeric(pred.get("density_skan"), errors="coerce").to_numpy(float)
    fg = pd.to_numeric(pred.get("fg_px"), errors="coerce").to_numpy(float)
    area = pd.to_numeric(pred.get("skan_fov_area_px"), errors="coerce"
                         ).to_numpy(float)
    alt = np.where((area > 0) & np.isfinite(fg), fg / np.where(area > 0, area, 1),
                   np.nan)
    return np.where(np.isfinite(frac), frac, alt)


def flag_masks(pred: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-declared failure rules; never look at the audit first."""
    d = pred.copy()
    d["pred_fg_frac"] = fg_fraction(d)
    fg = pd.to_numeric(d.get("fg_px"), errors="coerce").to_numpy(float)
    d["empty_mask"] = (np.isfinite(fg) & (fg <= EMPTY_FG_PX))
    d["degenerate_fg"] = ((~d["empty_mask"]) &
                          np.isfinite(d["pred_fg_frac"]) &
                          (d["pred_fg_frac"] < DEGENERATE_FG_FRAC))
    d["primary_nan"] = d[PANEL_SKAN].isna().any(axis=1)
    return d


def pairs(ref: pd.DataFrame, pred: pd.DataFrame, ds: str) -> pd.DataFrame:
    r = ref[ref["dataset"] == ds]
    keep = ["image_id", "split", "disease", "disc_fov_diameter"] \
        + list(PRIMARY_COLS)
    pcols = ["image_id", "empty_mask", "degenerate_fg", "primary_nan",
             "pred_fg_frac"] + list(PRIMARY_COLS)
    d = r[keep].merge(pred[pcols], on="image_id", suffixes=("__gt", "__pred"))
    return d.reset_index(drop=True)


def subset(d: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "all_incl_empty":
        return d
    if which == "main":
        return d[~d["empty_mask"]].reset_index(drop=True)
    if which == "sens_no_degen":
        return d[(~d["empty_mask"]) & (~d["degenerate_fg"])].reset_index(drop=True)
    raise ValueError(which)


# -------------------------------------------------------------- statistics
def boot_ci(x: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, x.size, size=(n_boot, x.size))
    v = np.array([fn(x[i]) for i in idx], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def calib_pack(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Deming lambda=1 calibration plus the method-comparison battery.

    ``x``/``y`` arrive already on the median-centred sigma axis, exactly the
    frame family 1 uses, so alpha is the offset a median-valued image carries.
    """
    a, b = deming(x, y, 1.0)
    resid = y - (a + b * x)
    diff = y - x
    mean = (y + x) / 2.0
    _pa, pb = ols(mean, diff)
    sx, sy = float(np.std(x)), float(np.std(y))
    return {
        "alpha": a, "beta": b,
        "resid_sd": float(np.std(resid, ddof=1)) if x.size > 2 else float("nan"),
        "ccc": ccc(x, y),
        "mu": float(np.mean(diff)),
        "ba_sd_diff": float(np.std(diff, ddof=1)) if x.size > 2 else float("nan"),
        "prop_bias_slope": pb,
        "r_pearson": (float(np.corrcoef(x, y)[0, 1]) if (sx > 0 and sy > 0)
                      else float("nan")),
    }


def audit(recs: List[dict], ref: pd.DataFrame, sig: Dict[str, Dict[str, float]],
          skcommon: Dict[str, float], qual: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows: List[dict] = []
    rng_master = np.random.RandomState(SEED)
    for rec in recs:
        ds, fam, sd_ = rec["dataset"], rec["family"], rec["seed"]
        pred = flag_masks(pd.read_csv(rec["path"], low_memory=False))
        d0 = pairs(ref, pred, ds)
        if not len(d0):
            print("[audit] %s/%s seed%d: no reference overlap -- skipped"
                  % (fam, ds, sd_))
            continue
        zs = is_zero_shot(fam, ds)
        for aset in ANALYSIS_SETS:
            d = subset(d0, aset)
            for col in PRIMARY_COLS:
                s = sig.get(ds, {}).get(col, float("nan"))
                sk = skcommon.get(col, float("nan"))
                fam_key = col.rsplit("_", 1)[0]
                xg = d[col + "__gt"].to_numpy(float)
                yp = d[col + "__pred"].to_numpy(float)
                fovd = d["disc_fov_diameter"].to_numpy(float)
                ok = np.isfinite(xg) & np.isfinite(yp)
                if ok.sum() < 4:
                    continue
                xr, yr = xg[ok], yp[ok]
                # --- mu in the three units the paper reports
                mu_raw = float(np.mean(yr - xr))
                if fam_key in LENGTH_LIKE:
                    fd = fovd[ok]
                    good = np.isfinite(fd) & (fd > 0)
                    mu_fov = (float(np.mean((yr[good] - xr[good]) / fd[good]))
                              if good.any() else float("nan"))
                    xc, yc = xr / np.where(good, fd, np.nan), \
                        yr / np.where(good, fd, np.nan)
                    mu_common = (float(np.nanmean(yc - xc)) / sk
                                 if np.isfinite(sk) and sk > 0 else float("nan"))
                else:
                    mu_fov = float("nan")
                    mu_common = (mu_raw / sk if np.isfinite(sk) and sk > 0
                                 else float("nan"))
                mu_sigma = mu_raw / s if np.isfinite(s) and s > 0 else float("nan")

                # --- calibration battery on the median-centred sigma axis
                med = float(np.median(xr))
                xs, ys = (xr - med) / s, (yr - med) / s
                point = calib_pack(xs, ys)
                idx = rng_master.randint(0, xs.size, size=(N_BOOT, xs.size))
                draws: Dict[str, List[float]] = {k: [] for k in point}
                draws["mu_raw"] = []
                for i in idx:
                    if np.std(xs[i]) == 0:
                        continue
                    st = calib_pack(xs[i], ys[i])
                    for k, v in st.items():
                        if np.isfinite(v):
                            draws[k].append(v)
                    draws["mu_raw"].append(float(np.mean(yr[i] - xr[i])))
                ci = {}
                for k, v in draws.items():
                    a = np.asarray(v, float)
                    ci[k + "_lo"] = (float(np.percentile(a, 2.5)) if a.size
                                     else float("nan"))
                    ci[k + "_hi"] = (float(np.percentile(a, 97.5)) if a.size
                                     else float("nan"))
                rows.append(dict(
                    family=fam, seed=sd_, dataset=ds, zero_shot=zs,
                    analysis_set=aset, biomarker=col, panel=PANEL[col],
                    n=int(ok.sum()), n_all=int(len(d0)),
                    n_empty_excluded=int(d0["empty_mask"].sum()),
                    n_degenerate=int(d0["degenerate_fg"].sum()),
                    sigma_own=s, sigma_common=sk,
                    mu_raw=mu_raw, mu_raw_lo=ci["mu_raw_lo"],
                    mu_raw_hi=ci["mu_raw_hi"],
                    mu_fovnorm=mu_fov, mu_sigma_own=mu_sigma,
                    mu_sigma_common=mu_common,
                    mu_centred_sigma=point["mu"], mu_lo=ci["mu_lo"],
                    mu_hi=ci["mu_hi"],
                    alpha=point["alpha"], alpha_lo=ci["alpha_lo"],
                    alpha_hi=ci["alpha_hi"],
                    beta=point["beta"], beta_lo=ci["beta_lo"],
                    beta_hi=ci["beta_hi"],
                    beta_below_1=bool(np.isfinite(ci["beta_hi"]) and
                                      ci["beta_hi"] < 1.0),
                    beta_excludes_1=bool(np.isfinite(ci["beta_lo"]) and
                                         (ci["beta_lo"] > 1.0 or
                                          ci["beta_hi"] < 1.0)),
                    resid_sd=point["resid_sd"],
                    resid_sd_lo=ci["resid_sd_lo"],
                    resid_sd_hi=ci["resid_sd_hi"],
                    ccc=point["ccc"], ccc_lo=ci["ccc_lo"], ccc_hi=ci["ccc_hi"],
                    r_pearson=point["r_pearson"],
                    r_pearson_lo=ci["r_pearson_lo"],
                    r_pearson_hi=ci["r_pearson_hi"],
                    r_spearman=float(spearmanr(xr, yr)[0]) if ok.sum() > 3
                    else float("nan"),
                    ba_loa_lo=point["mu"] - 1.96 * point["ba_sd_diff"],
                    ba_loa_hi=point["mu"] + 1.96 * point["ba_sd_diff"],
                    prop_bias_slope=point["prop_bias_slope"],
                    prop_bias_slope_lo=ci["prop_bias_slope_lo"],
                    prop_bias_slope_hi=ci["prop_bias_slope_hi"],
                    prop_bias_p=prop_bias_p(xs, ys),
                    units=("mu_raw in native units; mu_fovnorm in FOV "
                           "diameters (length-like only); mu_sigma_own uses the "
                           "frozen per-cohort Gate A sigma; mu_sigma_common "
                           "uses the common scale s_k. alpha/beta/CCC/r/"
                           "residual SD/LoA/prop-bias are on the "
                           "median-centred sigma_own axis, identical to "
                           "family 1"),
                    source=("%s vs reference-mask biomarkers of %s; frozen "  # noqa: E501
                            "sigma = results/gateA_biomarker_scales_train.csv; "
                            "common scale s_k per "
                            "results/pivot/r2/r2_common_scale_formula.txt; "
                            "mask-failure rules pre-declared in DECISIONS.md "
                            "2026-09-18 11:03; 2000-draw paired image bootstrap"
                            % (os.path.relpath(rec["path"], RUNS_DIR
                                               ).replace("\\", "/"),
                               "results/pivot/r2/bio_master_full.csv")),
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------------- mask quality
def mask_quality(recs: List[dict], ref: pd.DataFrame,
                 shipped: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for rec in recs:
        ds, fam, sd_ = rec["dataset"], rec["family"], rec["seed"]
        pred = flag_masks(pd.read_csv(rec["path"], low_memory=False))
        d = pairs(ref, pred, ds)
        if not len(d):
            continue
        dis = d["disease"].astype(str).fillna("NA")
        per_class = []
        for c in sorted(set(dis)):
            m = (dis == c)
            per_class.append("%s=%d/%d" % (c, int(d.loc[m, "empty_mask"].sum() +
                                                  d.loc[m, "degenerate_fg"].sum()),
                                           int(m.sum())))
        chk = ""
        if shipped is not None and fam == "lwnet":
            s = shipped[shipped["dataset"] == ds]
            if len(s):
                chk = ("shipped lwnet_mask_quality.csv: empty=%d degenerate=%d"
                       % (int(s["empty_mask"].sum()),
                          int(s["degenerate_fg"].sum())))
        rows.append(dict(
            family=fam, seed=sd_, dataset=ds,
            zero_shot=is_zero_shot(fam, ds),
            n=int(len(d)),
            n_empty=int(d["empty_mask"].sum()),
            n_degenerate=int(d["degenerate_fg"].sum()),
            n_failed_total=int(d["empty_mask"].sum() + d["degenerate_fg"].sum()),
            frac_failed=float((d["empty_mask"] | d["degenerate_fg"]).mean()),
            n_primary_nan=int(d["primary_nan"].sum()),
            fg_frac_median=float(np.nanmedian(d["pred_fg_frac"])),
            fg_frac_min=float(np.nanmin(d["pred_fg_frac"])),
            failed_per_class=";".join(per_class),
            failed_image_ids=";".join(
                sorted(d.loc[d["empty_mask"] | d["degenerate_fg"],
                             "image_id"].astype(str))[:40]),
            crosscheck=chk,
            rules=("empty: fg_px == 0 (excluded, counted); degenerate: "
                   "pred_fg_frac < 0.02 (kept in main, excluded in "
                   "sensitivity) -- DECISIONS.md 2026-09-18 11:03"),
            source=os.path.relpath(rec["path"], RUNS_DIR).replace("\\", "/")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- downstream
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


def gap_rows(d: pd.DataFrame, cols: Sequence[str], label: dict) -> List[dict]:
    y = d["disease"].astype(str).to_numpy()
    classes = sorted(set(y))
    if len(classes) < 2 or len(y) < 30:
        return []
    Xg = np.nan_to_num(d[[c + "__gt" for c in cols]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    Xp = np.nan_to_num(d[[c + "__pred" for c in cols]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    out: List[dict] = []
    for kind in ("logreg", "gbdt"):
        pg, pp = cv_proba(Xg, y, kind, classes), cv_proba(Xp, y, kind, classes)
        a_g, a_p = macro_auc(y, pg, classes), macro_auc(y, pp, classes)
        rng = np.random.RandomState(SEED)
        n = len(y)
        dd, gg, pv = [], [], []
        for _ in range(N_BOOT_AUC):
            i = rng.randint(0, n, n)
            if len(set(y[i])) < len(classes):
                continue
            bg, bp = macro_auc(y[i], pg[i], classes), macro_auc(y[i], pp[i], classes)
            if np.isfinite(bg) and np.isfinite(bp):
                dd.append(bg - bp); gg.append(bg); pv.append(bp)

        def pc(v):
            v = np.asarray(v, float)
            return ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                    if v.size else (float("nan"), float("nan")))
        d_lo, d_hi = pc(dd)
        g_lo, g_hi = pc(gg)
        p_lo, p_hi = pc(pv)
        out.append(dict(
            **label, clf=kind, n=n, n_classes=len(classes),
            auc_reference=a_g, auc_reference_lo=g_lo, auc_reference_hi=g_hi,
            auc_predicted=a_p, auc_predicted_lo=p_lo, auc_predicted_hi=p_hi,
            delta_auc=a_g - a_p, delta_auc_lo=d_lo, delta_auc_hi=d_hi,
            class_counts=";".join("%s=%d" % (c, int((y == c).sum()))
                                  for c in classes),
            source=("family-2 predictions vs reference-mask biomarkers; "
                    "5-fold x 3 repeats stratified CV, macro one-vs-rest AUC, "
                    "1000-draw paired image bootstrap -- identical protocol to "
                    "src/pivot/p3_downstream.py; NaN features from failed "
                    "masks are zero-filled exactly as a naive pipeline would")))
    return out


def downstream(recs: List[dict], ref: pd.DataFrame,
               datasets: Sequence[str] = ("hrf", "fives")
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    attrib: List[dict] = []
    for rec in recs:
        ds, fam, sd_ = rec["dataset"], rec["family"], rec["seed"]
        if ds not in datasets:
            continue
        pred = flag_masks(pd.read_csv(rec["path"], low_memory=False))
        d0 = pairs(ref, pred, ds)
        if not len(d0) or d0["disease"].isna().all():
            continue
        splits = sorted(set(d0["split"].astype(str)))
        cover = ("test-only (no out-of-fold predictions exist for this "
                 "family on the training split)" if splits == ["test"]
                 else "splits: " + ",".join(splits))
        zs = is_zero_shot(fam, ds)
        per_set = {}
        # When a family has no mask failures on this dataset the three analysis
        # sets are the SAME rows, so the fit is computed once and copied.  This
        # is what keeps the whole re-run to minutes once the remaining
        # SegFormer seeds land: only LWNet/FIVES actually needs three fits.
        done_by_n: Dict[Tuple[int, str, str], dict] = {}
        for aset in ("all_incl_empty",) + ANALYSIS_SETS:
            d = subset(d0, aset)
            for pname, cols in (("skan4", PANEL_SKAN),
                                ("primary8", list(PRIMARY_COLS))):
                label = dict(
                    family=fam, seed=sd_, dataset=ds, zero_shot=zs,
                    analysis_set=aset, featureset=pname,
                    n_empty=int(d0["empty_mask"].sum()),
                    n_degenerate=int(d0["degenerate_fg"].sum()),
                    split_coverage=cover)
                cached = [g for (n_, pn, _c), g in done_by_n.items()
                          if n_ == len(d) and pn == pname]
                if cached:
                    got = []
                    for g in cached:
                        gg = dict(g)
                        gg.update(label)
                        gg["source"] = (g["source"] + "; identical row set to "
                                        "analysis_set='%s' (no mask failures "
                                        "distinguish them), fit reused"
                                        % g["analysis_set"])
                        got.append(gg)
                else:
                    got = gap_rows(d, cols, label)
                    for g in got:
                        done_by_n[(len(d), pname, g["clf"])] = g
                rows += got
                for g in got:
                    per_set[(aset, pname, g["clf"])] = g
        # --- how much of the gap is the mask failures?
        for (aset, pname, clf), g in sorted(per_set.items()):
            if aset == "all_incl_empty":
                continue
            base = per_set.get(("all_incl_empty", pname, clf))
            nodeg = per_set.get(("sens_no_degen", pname, clf))
            if base is None:
                continue
            attrib.append(dict(
                family=fam, seed=sd_, dataset=ds, featureset=pname, clf=clf,
                analysis_set=aset,
                n_all=base["n"], n_this=g["n"],
                n_empty=int(d0["empty_mask"].sum()),
                n_degenerate=int(d0["degenerate_fg"].sum()),
                delta_auc_all=base["delta_auc"],
                delta_auc_all_lo=base["delta_auc_lo"],
                delta_auc_all_hi=base["delta_auc_hi"],
                delta_auc_this=g["delta_auc"],
                delta_auc_this_lo=g["delta_auc_lo"],
                delta_auc_this_hi=g["delta_auc_hi"],
                delta_auc_no_degenerate=(nodeg["delta_auc"] if nodeg else
                                         float("nan")),
                gap_removed_by_dropping=base["delta_auc"] - g["delta_auc"],
                # A ratio to a near-zero baseline is meaningless (it printed
                # 526 % in one cell), so the fraction is only defined when the
                # all-images gap is at least 0.02 AUC in magnitude; otherwise
                # quote gap_removed_by_dropping in absolute AUC.
                frac_of_gap_from_failures=(
                    (base["delta_auc"] - g["delta_auc"]) / base["delta_auc"]
                    if (np.isfinite(base["delta_auc"]) and
                        abs(base["delta_auc"]) >= 0.02) else float("nan")),
                frac_defined=bool(np.isfinite(base["delta_auc"]) and
                                  abs(base["delta_auc"]) >= 0.02),
                split_coverage=cover,
                source=("results/pivot/r2/family2_delta_auc.csv; "
                        "'all_incl_empty' keeps the failed masks with their "
                        "NaN biomarkers zero-filled, which is what a naive "
                        "pipeline would do; the difference is the share of the "
                        "reference gap attributable to segmentation failure "
                        "rather than to measurement calibration")))
    return pd.DataFrame(rows), pd.DataFrame(attrib)


# ---------------------------------------------------------- replication
def replication(au: pd.DataFrame, dn: pd.DataFrame,
                aset: str = "main") -> pd.DataFrame:
    """Does each family-1 conclusion reproduce in each family?

    Family-1 reference values (U-Net, primary skan panel, frozen sigma):
      mean |mu| HRF 1.540 > FIVES 0.249  (ratio 6.2)
      mean residual SD HRF 1.480 > FIVES 0.779
      all four HRF Deming slopes < 1 (0.37-0.81)
      Delta AUC HRF +0.217 (logreg) vs FIVES -0.003
    """
    F1 = dict(mu_hrf=1.5397, mu_fives=0.2485, sd_hrf=1.4795, sd_fives=0.7787,
              hrf_beta_below_1=4, dauc_hrf=0.2170, dauc_fives=-0.0027)
    rows: List[dict] = []
    a = au[(au["panel"] == "primary") & (au["analysis_set"] == aset)]
    for (fam, sd_), g in a.groupby(["family", "seed"]):
        per_ds = g.groupby("dataset").agg(
            mu=("mu_centred_sigma", lambda s: float(np.mean(np.abs(s)))),
            sd=("resid_sd", "mean"), r=("r_pearson", "mean"))
        has = set(per_ds.index)
        # --- offset ordering
        if {"hrf", "fives"} <= has:
            h, f = float(per_ds.loc["hrf", "mu"]), float(per_ds.loc["fives", "mu"])
            rows.append(dict(
                family=fam, seed=sd_, analysis_set=aset,
                conclusion="offset ordering: HRF > FIVES",
                family1_value="1.540 vs 0.249 (ratio 6.2)",
                family2_value="%.3f vs %.3f (ratio %.2f)"
                % (h, f, h / f if f else float("nan")),
                replicates=bool(h > f)))
            hr, fr = float(per_ds.loc["hrf", "r"]), float(per_ds.loc["fives", "r"])
            rows.append(dict(
                family=fam, seed=sd_, analysis_set=aset,
                conclusion="image-level fidelity ordering: FIVES > HRF",
                family1_value="r 0.824 vs 0.487",
                family2_value="r %.3f vs %.3f" % (fr, hr),
                replicates=bool(fr > hr)))
            h2, f2 = float(per_ds.loc["hrf", "sd"]), float(per_ds.loc["fives", "sd"])
            rows.append(dict(
                family=fam, seed=sd_, analysis_set=aset,
                conclusion="residual-scatter ordering: HRF > FIVES",
                family1_value="1.480 vs 0.779 (ratio 1.9)",
                family2_value="%.3f vs %.3f (ratio %.2f)"
                % (h2, f2, h2 / f2 if f2 else float("nan")),
                replicates=bool(h2 > f2)))
        # --- HRF slope < 1
        hb = g[g["dataset"] == "hrf"]
        if len(hb):
            n_lt = int((hb["beta"] < 1.0).sum())
            n_ci = int(hb["beta_below_1"].sum())
            w = (pd.to_numeric(hb["beta_hi"], errors="coerce")
                 - pd.to_numeric(hb["beta_lo"], errors="coerce"))
            ident = np.isfinite(w) & (w <= IDENT_CI_WIDTH)
            n_id = int(ident.sum())
            n_id_lt = int(((hb["beta"] < 1.0) & ident).sum())
            rows.append(dict(
                family=fam, seed=sd_, analysis_set=aset,
                conclusion="HRF attenuation: Deming slope < 1",
                family1_value="4/4 below 1 (0.37-0.81)",
                family2_value=("%d/%d below 1 (%s), %d with CI entirely below "
                               "1; identified: %d/%d, of which %d below 1"
                               % (n_lt, len(hb),
                                  ", ".join("%.2f" % v for v in hb["beta"]),
                                  n_ci, n_id, len(hb), n_id_lt)),
                n_identified=n_id, n_identified_below_1=n_id_lt,
                replicates=(None if n_id < 0.5 * len(hb)
                            else bool(n_id_lt >= max(1, n_id - 1)))))
        # --- slope < 1 on EVERY dataset this family covers (family 1 gives
        # 4/4 on DRIVE and HRF, 2/4 on CHASE_DB1 and FIVES), so the SegFormer
        # DRIVE-only run is still checkable before its other datasets land
        F1_BLT1 = {"drive": 4, "chasedb1": 2, "hrf": 4, "fives": 2}
        for ds in sorted(has):
            gd = g[g["dataset"] == ds]
            if not len(gd):
                continue
            n_lt = int((gd["beta"] < 1.0).sum())
            # A Deming slope is only interpretable when it is IDENTIFIED.  On a
            # low-fidelity cohort the correlation is near zero and the slope's
            # bootstrap interval spans tens of units (SegFormer HRF FD:
            # beta = 2.40, CI [-16.1, 21.0], r = 0.20), so a bare "how many
            # slopes are below 1" count is noise rather than a replication
            # criterion.  Slopes whose CI is wider than IDENT_CI_WIDTH are
            # reported but excluded from the verdict, and the verdict is None
            # (undecidable) when fewer than half the columns are identified.
            w = (pd.to_numeric(gd["beta_hi"], errors="coerce")
                 - pd.to_numeric(gd["beta_lo"], errors="coerce"))
            ident = np.isfinite(w) & (w <= IDENT_CI_WIDTH)
            n_id = int(ident.sum())
            n_id_lt = int(((gd["beta"] < 1.0) & ident).sum())
            if n_id == 0 or n_id < 0.5 * len(gd):
                verdict = None
            else:
                verdict = bool(n_id_lt >= max(1, n_id - 1))
            rows.append(dict(
                family=fam, seed=sd_, analysis_set=aset,
                conclusion="%s calibration slope < 1 (attenuation)" % ds.upper(),
                family1_value="%d/4 below 1" % F1_BLT1.get(ds, -1),
                family2_value=("%d/%d below 1 (%s); identified slopes "
                               "(bootstrap CI width <= %.0f): %d/%d, of which "
                               "%d below 1"
                               % (n_lt, len(gd),
                                  ", ".join("%.2f" % v for v in gd["beta"]),
                                  IDENT_CI_WIDTH, n_id, len(gd), n_id_lt)),
                n_identified=n_id, n_identified_below_1=n_id_lt,
                replicates=verdict))

        # --- Delta AUC gap magnitude
        dd = dn[(dn["family"] == fam) & (dn["seed"] == sd_) &
                (dn["analysis_set"] == aset) &
                (dn["featureset"] == "skan4") & (dn["clf"] == "logreg")]
        gh = dd[dd["dataset"] == "hrf"]["delta_auc"]
        gf = dd[dd["dataset"] == "fives"]["delta_auc"]
        if len(gh) or len(gf):
            rows.append(dict(
                family=fam, seed=sd_, analysis_set=aset,
                conclusion="reference gap large on HRF, small on FIVES",
                family1_value="+0.217 vs -0.003",
                family2_value="%s vs %s"
                % ("%.4f" % gh.iloc[0] if len(gh) else "n/a",
                   "%.4f" % gf.iloc[0] if len(gf) else "n/a"),
                replicates=(bool(gh.iloc[0] > gf.iloc[0])
                            if (len(gh) and len(gf)) else None)))
    for r in rows:
        r["source"] = ("results/pivot/r2/family2_audit.csv + "
                       "family2_delta_auc.csv; family-1 values from "
                       "results/pivot/r2/r2_main_table.csv and r2_delta_auc.csv")
    return pd.DataFrame(rows)


# ------------------------------------------------------------ common scale
def load_common_scale(out_dir: str, ref: pd.DataFrame
                      ) -> Tuple[Dict[str, float], str]:
    """The frozen family-1 ``s_k``, loaded rather than recomputed.

    Family 1 wrote it into ``r2_scale_sensitivity.csv`` (``cohort == "common"``)
    from the reference masks available at the time.  Reusing that exact number
    is what makes the two families comparable; recomputing would drift with the
    reference sample (the FIVES training split has grown from 200 to 600 images
    since, which moves ``s_k`` by 2-4 %).
    """
    p = os.path.join(out_dir, "r2_scale_sensitivity.csv")
    if os.path.exists(p):
        s = pd.read_csv(p)
        s = s[s["cohort"] == "common"][["biomarker", "sigma"]].drop_duplicates()
        lut = {str(r.biomarker): float(r.sigma) for r in s.itertuples(index=False)}
        if all(c in lut for c in PRIMARY_COLS):
            return lut, "frozen, loaded from r2_scale_sensitivity.csv"
    # fallback: recompute with the identical formula, and say so loudly
    print("[family2] WARNING: frozen common scale not found in %s -- "
          "recomputing; this may not match family 1" % p)
    lut = {}
    for col in PRIMARY_COLS:
        vals = {}
        for ds in DATASETS:
            r = ref[(ref["dataset"] == ds) & (ref["split"] == "train")]
            x = r[col].to_numpy(float)
            if col.rsplit("_", 1)[0] in LENGTH_LIKE:
                fd = r["disc_fov_diameter"].to_numpy(float)
                x = x / np.where(fd > 0, fd, np.nan)
            vals[ds] = x
        lut[col] = common_scale(vals)
    return lut, "RECOMPUTED (frozen table missing)"


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(RUNS_DIR, "pivot", "r2"))
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "r2",
                                                     "bio_master_full.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    ap.add_argument("--families", default="",
                    help="comma-separated subset; default = all discovered")
    ap.add_argument("--skip_downstream", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    fams = [f.strip() for f in args.families.split(",") if f.strip()] or None
    recs = discover(args.root, fams)
    if not recs:
        print("[family2] nothing found under %s" % args.root)
        return 1
    print("[family2] %d prediction sets:" % len(recs))
    for r in recs:
        print("   %-14s %-9s seed%d  %s" % (r["family"], r["dataset"],
                                            r["seed"], r["path"]))

    master = args.master
    if not os.path.exists(master):
        master = os.path.join(PIVOT_DIR, "bio_master.csv")
        print("[family2] full master missing, falling back to", master)
    ref = load_reference(master)
    sig = sigma_table()

    # The common scale s_k must be the SAME NUMBER family 1 used, or the two
    # families are not on one denominator and the whole comparison is void.
    # It is therefore LOADED from the frozen family-1 sensitivity table and
    # only recomputed if that file is missing.  (Recomputing it here would
    # silently differ: family 1 was run against bio_master.csv, whose FIVES
    # training split had 200 images, while bio_master_full.csv now has 600.)
    skcommon, sk_src = load_common_scale(args.out_dir, ref)
    print("[family2] common scale s_k (%s): " % sk_src
          + ", ".join("%s=%.6g" % (c, skcommon[c]) for c in PANEL_SKAN))

    shipped = None
    p = os.path.join(args.out_dir, "lwnet_mask_quality.csv")
    if os.path.exists(p):
        shipped = pd.read_csv(p)

    mq = mask_quality(recs, ref, shipped)
    q = os.path.join(args.out_dir, "family2_mask_quality.csv")
    mq.to_csv(q, index=False)
    print("wrote", q, mq.shape)

    au = audit(recs, ref, sig, skcommon, mq)
    q = os.path.join(args.out_dir, "family2_audit.csv")
    au.to_csv(q, index=False)
    print("wrote", q, au.shape)

    dn, attrib = (pd.DataFrame(), pd.DataFrame())
    if not args.skip_downstream:
        dn, attrib = downstream(recs, ref)
        q = os.path.join(args.out_dir, "family2_delta_auc.csv")
        dn.to_csv(q, index=False)
        print("wrote", q, dn.shape)
        q = os.path.join(args.out_dir, "family2_failure_attribution.csv")
        attrib.to_csv(q, index=False)
        print("wrote", q, attrib.shape)

    empty_dn = pd.DataFrame(columns=["family", "seed", "analysis_set",
                                     "featureset", "clf", "dataset",
                                     "delta_auc"])
    # both analysis sets, so a conclusion that only fails because of the
    # degenerate masks is visible as such rather than as a non-replication
    rep = pd.concat([replication(au, dn if len(dn) else empty_dn, aset)
                     for aset in ANALYSIS_SETS], ignore_index=True)
    q = os.path.join(args.out_dir, "family2_replication.csv")
    rep.to_csv(q, index=False)
    print("wrote", q, rep.shape)

    with pd.option_context("display.width", 230, "display.max_rows", 400):
        print("\n[mask quality -- pre-declared rules]")
        print(mq[["family", "dataset", "zero_shot", "n", "n_empty",
                  "n_degenerate", "frac_failed", "fg_frac_median",
                  "fg_frac_min", "failed_per_class"]]
              .round(4).to_string(index=False))
        print("\n[audit, primary skan panel, analysis_set=main]")
        print(au[(au["panel"] == "primary") & (au["analysis_set"] == "main")]
              [["family", "dataset", "biomarker", "n", "mu_raw",
                "mu_sigma_own", "mu_sigma_common", "alpha", "beta", "beta_lo",
                "beta_hi", "resid_sd", "ccc", "r_pearson", "prop_bias_slope",
                "prop_bias_p"]].round(4).to_string(index=False))
        if len(dn):
            print("\n[reference-gap Delta AUC]")
            print(dn[["family", "dataset", "analysis_set", "featureset", "clf",
                      "n", "auc_reference", "auc_predicted", "delta_auc",
                      "delta_auc_lo", "delta_auc_hi", "split_coverage"]]
                  .round(4).to_string(index=False))
        if len(attrib):
            print("\n[how much of the gap is mask failure]")
            print(attrib[["family", "dataset", "featureset", "clf",
                          "analysis_set", "n_all", "n_this", "delta_auc_all",
                          "delta_auc_this", "gap_removed_by_dropping",
                          "frac_of_gap_from_failures"]]
                  .round(4).to_string(index=False))
        print("\n[replication of the family-1 conclusions]")
        print(rep.drop(columns=["source"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
