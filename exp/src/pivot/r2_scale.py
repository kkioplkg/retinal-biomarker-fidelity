"""R2 (revision-2) -- the sigma axis audited in RAW units, and the scale made
explicit and falsifiable.

Reviewer points 3 and 4 (``review/user_cmig_review_20260918.md``)
----------------------------------------------------------------
3.  ``sigma`` is not a common unit across datasets.  It is a *within-dataset*
    effect-size scale, so a number in sigma units on HRF and a number in sigma
    units on FIVES are not the same physical quantity.  Report the audit in
    raw units as well, say what sigma is, and show what happens under a common
    reference scale.
4.  Write the robust-scale estimator down as a formula, and show the headline
    numbers under MAD / SD / IQR.

What this script produces (all under ``results/pivot/r2/``)
----------------------------------------------------------
``r2_sigma_estimators.csv``
    Per (dataset, biomarker, estimator): n, the estimator's value, a 2000-draw
    image bootstrap CI **of sigma itself**, the exact formula as a string, and
    the ratio to the frozen Gate A sigma (a reproduction check).

``r2_audit_raw_units.csv``
    Per (dataset, biomarker): the three audit quantities -- constant offset,
    residual SD, Pearson/Spearman r -- in
      * ``raw`` native units (density fraction; total length in px;
        FD and tortuosity dimensionless),
      * ``fovnorm`` units where a length is involved (px / FOV diameter px,
        i.e. "FOV diameters"; the FOV diameter is the one already measured per
        image in ``bio_master.csv::disc_fov_diameter``),
      * ``sigma`` units (the frozen Gate A train scale),
    each with a 2000-draw paired image bootstrap CI.  ``r`` is scale-free and is
    therefore identical in all three blocks -- it is reported once, which is
    itself part of the answer to point 3.

``r2_scale_sensitivity.csv``
    The headline sigma-scaled offsets and residual SDs under nine scale
    variants: three estimators x three reference cohorts
      * ``own``    -- the dataset's own training-split reference masks (status quo)
      * ``pooled`` -- one scale per biomarker, pooled over the four datasets'
                      training-split reference masks (a COMMON reference scale)
      * ``fives``  -- FIVES' training-split reference masks used as a fixed
                      calibration cohort for every dataset
    Pooling and transporting a scale is only meaningful for a *dimensionless*
    or *FOV-normalised* biomarker, so the pooled/FIVES variants are computed on
    the FOV-normalised representation (density, FD, tortuosity are already
    dimensionless; total length is divided by the FOV diameter).  The ``own``
    variant is reported on both representations so the two can be compared.

``r2_scale_claims.csv``
    The two headline claims, evaluated under every variant:
      * ``hrf_gt_fives`` -- does HRF still show a larger offset / residual
        scatter than FIVES?
      * ``axis_ratio``   -- is the offset/residual axis still one to two orders
        of magnitude above the topology axis?

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_scale
"""
from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import (EXP_ROOT, GATEA_DS, PIVOT_DIR, PRIMARY_COLS,
                              RESULTS_DIR, sigma_table)

SEED = 0
N_BOOT = 2000
DATASETS = ("drive", "chasedb1", "hrf", "fives")

#: primary = the four skan columns (DECISIONS.md 2026-09-17 13:30);
#: the PVBM duplicates are carried as a sensitivity panel.
PANEL = {c: ("primary" if c.endswith("_skan") else "sensitivity")
         for c in PRIMARY_COLS}

#: raw physical unit of each biomarker family
RAW_UNIT = {
    "FD": "dimensionless (box-counting slope)",
    "tortuosity": "dimensionless (arc/chord ratio)",
    "density": "fraction of FOV area (dimensionless)",
    "total_length": "skeleton pixels",
}
#: biomarkers that carry a length dimension and therefore need FOV normalisation
LENGTH_LIKE = ("total_length",)

# --------------------------------------------------------------------------
# the estimators, written out exactly as they are computed
# --------------------------------------------------------------------------
ESTIMATOR_FORMULA = {
    "mad1.4826": ("sigma = 1.4826 * median_i | x_i - median_j x_j |   "
                  "(normal-consistent median absolute deviation)"),
    "sd": "sigma = sqrt( (1/(n-1)) * sum_i (x_i - mean_j x_j)^2 )   (ordinary SD)",
    "iqr1.349": ("sigma = ( Q75(x) - Q25(x) ) / 1.349   "
                 "(normal-consistent interquartile range; 1.349 = 2 * Phi^-1(0.75))"),
}
#: the estimator frozen in results/gateA_biomarker_scales_train.csv
GATEA_ESTIMATOR = "mad1.4826"


def est_mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def est_sd(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def est_iqr(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    q75, q25 = np.percentile(x, [75.0, 25.0])
    return float((q75 - q25) / 1.349)


ESTIMATORS: Dict[str, Callable[[np.ndarray], float]] = {
    "mad1.4826": est_mad, "sd": est_sd, "iqr1.349": est_iqr,
}

#: The COMMON reference scale, in the single form the methods consultation
#: (``review/cmig_plan_reply.md`` section 2A) said it would accept.  Written out
#: here because it is the only common-scale definition the manuscript may use.
COMMON_SCALE_FORMULA = (
    "For biomarker k, using REFERENCE-MASK measurements x_ick^ref only:\n"
    "  (1) per-cohort reference centre   m_ck   = median_i x_ick^ref\n"
    "  (2) cohort-centred deviations     d_ick  = x_ick^ref - m_ck\n"
    "  (3) common scale                  s_k    = 1.4826 * weighted_median_ick |d_ick|,\n"
    "      with weights w_ick = 1/n_c so every cohort carries equal total weight\n"
    "  (4) every cohort and every segmenter then uses the SAME denominator:\n"
    "      e_ick = ( x_ick^pred - x_ick^ref ) / s_k\n"
    "Properties: the denominator never depends on model performance; a large "
    "cohort (FIVES n=800) cannot dominate it; between-cohort location shifts "
    "cannot inflate it. Raw units remain the primary result; this is a "
    "cross-cohort SENSITIVITY scale and must not be renamed a 'natural unit' "
    "or a 'population SD'. Length-like biomarkers are normalised by the FOV "
    "diameter before step (1), because a pixel count is not comparable across "
    "cohorts of different resolution."
)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted median: the smallest value whose cumulative weight reaches half."""
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = np.asarray(x, float)[ok], np.asarray(w, float)[ok]
    if x.size == 0:
        return float("nan")
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[int(np.searchsorted(c, 0.5 * c[-1]))])


def common_scale(values_by_cohort: Dict[str, np.ndarray]) -> float:
    """``s_k`` of ``COMMON_SCALE_FORMULA`` from per-cohort reference values."""
    d, w = [], []
    for _c, x in values_by_cohort.items():
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            continue
        d.append(np.abs(x - np.median(x)))
        w.append(np.full(x.size, 1.0 / x.size))
    if not d:
        return float("nan")
    return float(1.4826 * weighted_median(np.concatenate(d),
                                          np.concatenate(w)))


# --------------------------------------------------------------------------
def boot_ci(x: np.ndarray, fn: Callable[[np.ndarray], float],
            n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, x.size, size=(n_boot, x.size))
    v = np.array([fn(x[i]) for i in idx], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def boot_ci_paired(a: np.ndarray, b: np.ndarray,
                   fn: Callable[[np.ndarray, np.ndarray], float],
                   n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        i = rng.randint(0, a.size, a.size)
        v = fn(a[i], b[i])
        if np.isfinite(v):
            out.append(v)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import pearsonr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pearsonr(a[ok], b[ok])[0])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_master(path: str) -> pd.DataFrame:
    m = pd.read_csv(path, low_memory=False)
    need = {"dataset", "image_id", "split", "source", "disc_fov_diameter"}
    missing = need - set(m.columns)
    if missing:
        raise SystemExit("bio_master is missing %s" % sorted(missing))
    return m


def fov_factor(df: pd.DataFrame, col: str) -> np.ndarray:
    """Divisor that turns the raw column into FOV-normalised units.

    A length-like biomarker is divided by the per-image FOV diameter in pixels;
    everything else is already dimensionless and is divided by 1.
    """
    fam = col.rsplit("_", 1)[0]
    if fam in LENGTH_LIKE:
        d = df["disc_fov_diameter"].to_numpy(float)
        return np.where(np.isfinite(d) & (d > 0), d, np.nan)
    return np.ones(len(df), float)


def train_reference(master: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Training-split reference (observer-1 ground-truth) masks of one dataset."""
    m = master[(master["dataset"] == ds) & (master["source"] == "gt") &
               (master["split"] == "train")]
    return m.reset_index(drop=True)


def test_pairs(master: pd.DataFrame, ds: str) -> pd.DataFrame:
    """Test-split (reference, seed-0 prediction) pairs of one dataset."""
    m = master[master["dataset"] == ds]
    gt = m[(m["source"] == "gt") & (m["split"] == "test")]
    pr = m[m["source"] == "pred_test"]
    cols = ["image_id", "disc_fov_diameter"] + list(PRIMARY_COLS)
    d = gt[cols].merge(pr[cols], on="image_id", suffixes=("__gt", "__pred"))
    # the FOV diameter is detected from the fundus image alone, so the two
    # copies are identical; keep one under a plain name.
    d["disc_fov_diameter"] = d["disc_fov_diameter__gt"]
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------
# part 1 -- the sigma estimators, with a CI of sigma itself
# --------------------------------------------------------------------------
def sigma_estimators(master: pd.DataFrame, gatea: Dict[str, Dict[str, float]]
                     ) -> pd.DataFrame:
    rows: List[dict] = []
    for ds in DATASETS:
        ref = train_reference(master, ds)
        for col in PRIMARY_COLS:
            if col not in ref.columns:
                continue
            raw = ref[col].to_numpy(float)
            fov = fov_factor(ref, col)
            for rep, x in (("raw", raw), ("fovnorm", raw / fov)):
                for name, fn in ESTIMATORS.items():
                    v = fn(x)
                    lo, hi = boot_ci(x, fn)
                    g = gatea.get(ds, {}).get(col, float("nan"))
                    rows.append(dict(
                        dataset=ds, biomarker=col, panel=PANEL[col],
                        representation=rep, estimator=name,
                        formula=ESTIMATOR_FORMULA[name],
                        n=int(np.isfinite(x).sum()),
                        median=float(np.nanmedian(x)),
                        mean=float(np.nanmean(x)),
                        mad_raw=float(np.nanmedian(np.abs(
                            x[np.isfinite(x)] - np.nanmedian(x)))),
                        sigma=v, sigma_lo=lo, sigma_hi=hi,
                        sigma_cv=(float((hi - lo) / (2 * 1.96 * v))
                                  if np.isfinite(v) and v > 0 else float("nan")),
                        unit=(RAW_UNIT[col.rsplit("_", 1)[0]] if rep == "raw"
                              else ("FOV diameters"
                                    if col.rsplit("_", 1)[0] in LENGTH_LIKE
                                    else RAW_UNIT[col.rsplit("_", 1)[0]])),
                        gateA_sigma=g,
                        ratio_to_gateA=(v / g if (rep == "raw" and
                                                  np.isfinite(g) and g > 0)
                                        else float("nan")),
                        source=("results/pivot/bio_master.csv "
                                "(source=gt, split=train) -- training-split "
                                "reference masks only; 2000-draw image "
                                "bootstrap CI of sigma; frozen scale = "
                                "results/gateA_biomarker_scales_train.csv"),
                    ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# part 2 -- the audit in raw / FOV-normalised / sigma units
# --------------------------------------------------------------------------
def audit_units(master: pd.DataFrame, gatea: Dict[str, Dict[str, float]]
                ) -> pd.DataFrame:
    rows: List[dict] = []
    for ds in DATASETS:
        d = test_pairs(master, ds)
        fovd = d["disc_fov_diameter"].to_numpy(float)
        for col in PRIMARY_COLS:
            g = d[col + "__gt"].to_numpy(float)
            p = d[col + "__pred"].to_numpy(float)
            fam = col.rsplit("_", 1)[0]
            s = gatea.get(ds, {}).get(col, float("nan"))
            reps = [("raw", g, p, RAW_UNIT[fam], 1.0)]
            if fam in LENGTH_LIKE:
                reps.append(("fovnorm", g / fovd, p / fovd, "FOV diameters",
                             float(np.nanmedian(fovd))))
            reps.append(("sigma", g / s, p / s,
                         "sigma (train-split 1.4826*MAD)", s))
            for rep, gg, pp, unit, divisor in reps:
                e = pp - gg
                ok = np.isfinite(e)
                if ok.sum() < 3:
                    continue
                off = float(np.mean(e[ok]))
                sd = float(np.std(e[ok], ddof=1))
                olo, ohi = boot_ci(e[ok], np.mean)
                slo, shi = boot_ci(e[ok], lambda v: np.std(v, ddof=1))
                rp = pearson(p, g)
                rplo, rphi = boot_ci_paired(p, g, pearson)
                rows.append(dict(
                    dataset=ds, biomarker=col, panel=PANEL[col],
                    representation=rep, unit=unit, divisor=divisor,
                    n=int(ok.sum()),
                    ref_median=float(np.nanmedian(gg)),
                    ref_iqr=float(np.nanpercentile(gg, 75)
                                  - np.nanpercentile(gg, 25)),
                    offset=off, offset_lo=olo, offset_hi=ohi,
                    resid_sd=sd, resid_sd_lo=slo, resid_sd_hi=shi,
                    r_pearson=rp, r_pearson_lo=rplo, r_pearson_hi=rphi,
                    r_spearman=spearman(p, g),
                    source=("results/pivot/bio_master.csv "
                            "(source=gt[test] vs pred_test, seg seed 0); "
                            "FOV diameter = bio_master::disc_fov_diameter "
                            "(detected from the fundus image alone); "
                            "sigma = results/gateA_biomarker_scales_train.csv; "
                            "2000-draw image bootstrap"),
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# part 3 -- scale sensitivity: 3 estimators x 3 reference cohorts
# --------------------------------------------------------------------------
def scale_variants(master: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """``{(cohort, estimator, representation): {(dataset,biomarker) -> sigma}}``.

    ``cohort`` is one of

      own     the dataset's own training-split reference masks
      pooled  all four datasets' training-split reference masks, concatenated
              (one scale per biomarker, shared by every dataset)
      fives   FIVES' training-split reference masks, used for every dataset
    """
    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    refs = {ds: train_reference(master, ds) for ds in DATASETS}
    for rep in ("raw", "fovnorm"):
        vals: Dict[str, Dict[str, np.ndarray]] = {}
        for ds in DATASETS:
            r = refs[ds]
            vals[ds] = {}
            for col in PRIMARY_COLS:
                x = r[col].to_numpy(float)
                if rep == "fovnorm":
                    x = x / fov_factor(r, col)
                vals[ds][col] = x
        for est, fn in ESTIMATORS.items():
            own, pooled, fiv = {}, {}, {}
            for col in PRIMARY_COLS:
                cat = np.concatenate([vals[ds][col] for ds in DATASETS])
                sp = fn(cat)
                sf = fn(vals["fives"][col])
                for ds in DATASETS:
                    own[f"{ds}|{col}"] = fn(vals[ds][col])
                    pooled[f"{ds}|{col}"] = sp
                    fiv[f"{ds}|{col}"] = sf
            out[("own", est, rep)] = own
            out[("pooled_naive", est, rep)] = pooled
            out[("fives", est, rep)] = fiv

        # --- the ACCEPTED common scale (COMMON_SCALE_FORMULA).  It has one
        # estimator by definition -- 1.4826 x weighted-median MAD of the
        # cohort-centred reference deviations -- so it is not crossed with the
        # estimator grid.
        com = {}
        for col in PRIMARY_COLS:
            s = common_scale({ds: vals[ds][col] for ds in DATASETS})
            for ds in DATASETS:
                com[f"{ds}|{col}"] = s
        out[("common", "mad1.4826_weighted_median", rep)] = com
    return out


def sensitivity(master: pd.DataFrame) -> pd.DataFrame:
    var = scale_variants(master)
    rows: List[dict] = []
    for ds in DATASETS:
        d = test_pairs(master, ds)
        fovd = d["disc_fov_diameter"].to_numpy(float)
        for col in PRIMARY_COLS:
            g0 = d[col + "__gt"].to_numpy(float)
            p0 = d[col + "__pred"].to_numpy(float)
            fam = col.rsplit("_", 1)[0]
            for (cohort, est, rep), lut in var.items():
                # pooling / transporting a scale is only defined on a
                # dimensionless (FOV-normalised) representation
                if cohort in ("pooled_naive", "fives", "common") \
                        and rep != "fovnorm":
                    continue
                s = lut.get(f"{ds}|{col}", float("nan"))
                if not np.isfinite(s) or s <= 0:
                    continue
                if rep == "fovnorm" and fam in LENGTH_LIKE:
                    g, p = g0 / fovd, p0 / fovd
                else:
                    g, p = g0, p0
                e = (p - g) / s
                ok = np.isfinite(e)
                if ok.sum() < 3:
                    continue
                rows.append(dict(
                    dataset=ds, biomarker=col, panel=PANEL[col],
                    cohort=cohort, estimator=est, representation=rep,
                    sigma=s, n=int(ok.sum()),
                    offset=float(np.mean(e[ok])),
                    abs_offset=float(abs(np.mean(e[ok]))),
                    resid_sd=float(np.std(e[ok], ddof=1)),
                    r_pearson=pearson(p, g),
                    source=("results/pivot/bio_master.csv; scale cohort "
                            "'%s' x estimator '%s' on representation '%s'; "
                            "scales from training-split reference masks only"
                            % (cohort, est, rep)),
                ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# part 3b -- the headline table, RAW UNITS PRIMARY
# --------------------------------------------------------------------------
def headline(au: pd.DataFrame, sens: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, biomarker) with the RAW-unit audit as the primary
    columns and the two standardised scales as sensitivity columns.

    The methods consultation (``review/cmig_plan_reply.md`` 2A) is explicit that
    raw units are the main result and that the common scale is a cross-cohort
    sensitivity, never a renamed "natural unit"; this table is laid out that way
    so the paper agent cannot accidentally invert the emphasis.
    """
    rows: List[dict] = []
    for (ds, col), g in au.groupby(["dataset", "biomarker"]):
        fam = col.rsplit("_", 1)[0]
        prim_rep = "fovnorm" if fam in LENGTH_LIKE else "raw"
        r = g[g["representation"] == prim_rep]
        raw = g[g["representation"] == "raw"]
        sg = g[g["representation"] == "sigma"]
        if not len(r) or not len(sg):
            continue
        r, raw, sg = r.iloc[0], raw.iloc[0], sg.iloc[0]
        com = sens[(sens["dataset"] == ds) & (sens["biomarker"] == col) &
                   (sens["cohort"] == "common")]
        rows.append(dict(
            dataset=ds, biomarker=col, panel=PANEL[col], n=int(r["n"]),
            # ---- PRIMARY: raw / anatomically normalised units
            primary_unit=r["unit"], reference_median=r["ref_median"],
            offset=r["offset"], offset_lo=r["offset_lo"],
            offset_hi=r["offset_hi"], resid_sd=r["resid_sd"],
            resid_sd_lo=r["resid_sd_lo"], resid_sd_hi=r["resid_sd_hi"],
            r_pearson=r["r_pearson"], r_pearson_lo=r["r_pearson_lo"],
            r_pearson_hi=r["r_pearson_hi"], r_spearman=r["r_spearman"],
            # ---- native pixel counts kept alongside for length-like columns
            offset_native_px=(raw["offset"] if fam in LENGTH_LIKE
                              else float("nan")),
            resid_sd_native_px=(raw["resid_sd"] if fam in LENGTH_LIKE
                                else float("nan")),
            # ---- SENSITIVITY 1: the dataset's own sigma (status quo axis)
            sens_sigma_own=sg["divisor"], offset_sigma_own=sg["offset"],
            resid_sd_sigma_own=sg["resid_sd"],
            # ---- SENSITIVITY 2: the accepted COMMON reference scale
            sens_sigma_common=(float(com["sigma"].iloc[0]) if len(com)
                               else float("nan")),
            offset_sigma_common=(float(com["offset"].iloc[0]) if len(com)
                                 else float("nan")),
            resid_sd_sigma_common=(float(com["resid_sd"].iloc[0]) if len(com)
                                   else float("nan")),
            common_scale_formula=COMMON_SCALE_FORMULA,
            source=("PRIMARY columns: results/pivot/r2/r2_audit_raw_units.csv "
                    "(raw native units; length-like biomarkers normalised by "
                    "the per-image FOV diameter). SENSITIVITY columns: the "
                    "per-cohort Gate A sigma and the common reference scale of "
                    "results/pivot/r2/r2_scale_sensitivity.csv (cohort=common)."
                    )))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# part 4 -- do the headline claims survive?
# --------------------------------------------------------------------------
def claims(sens: pd.DataFrame, topo_path: str) -> pd.DataFrame:
    rows: List[dict] = []
    prim = sens[sens["panel"] == "primary"]

    # --- topology axis: largest primary-panel paired |dB_topo| - |dB_ctrl|
    topo_max = float("nan")
    topo_src = topo_path
    if os.path.exists(topo_path):
        t = pd.read_csv(topo_path)
        t = t[(t["scope"] == "sever") & (t["panel"] == "primary")]
        if len(t):
            topo_max = float(t["h_net"].abs().max())

    for (cohort, est, rep), g in prim.groupby(
            ["cohort", "estimator", "representation"]):
        per_ds = g.groupby("dataset").agg(
            mean_abs_offset=("abs_offset", "mean"),
            mean_resid_sd=("resid_sd", "mean"),
            median_abs_offset=("abs_offset", "median"),
            median_resid_sd=("resid_sd", "median"))
        if not {"hrf", "fives"} <= set(per_ds.index):
            continue
        h, f = per_ds.loc["hrf"], per_ds.loc["fives"]
        big = float(max(per_ds["mean_abs_offset"].max(),
                        per_ds["mean_resid_sd"].max()))
        rows.append(dict(
            claim="hrf_offset_gt_fives", cohort=cohort, estimator=est,
            representation=rep,
            what_a="HRF mean |offset| (sigma)",
            what_b="FIVES mean |offset| (sigma)",
            hrf=float(h["mean_abs_offset"]), fives=float(f["mean_abs_offset"]),
            ratio=float(h["mean_abs_offset"] / f["mean_abs_offset"])
            if f["mean_abs_offset"] > 0 else float("nan"),
            survives=bool(h["mean_abs_offset"] > f["mean_abs_offset"]),
            source="results/pivot/r2/r2_scale_sensitivity.csv (primary panel)"))
        rows.append(dict(
            claim="hrf_residsd_gt_fives", cohort=cohort, estimator=est,
            representation=rep,
            what_a="HRF mean residual SD (sigma)",
            what_b="FIVES mean residual SD (sigma)",
            hrf=float(h["mean_resid_sd"]), fives=float(f["mean_resid_sd"]),
            ratio=float(h["mean_resid_sd"] / f["mean_resid_sd"])
            if f["mean_resid_sd"] > 0 else float("nan"),
            survives=bool(h["mean_resid_sd"] > f["mean_resid_sd"]),
            source="results/pivot/r2/r2_scale_sensitivity.csv (primary panel)"))
        rows.append(dict(
            claim="offset_resid_axis_over_topology", cohort=cohort,
            estimator=est, representation=rep,
            what_a=("largest primary-panel mean |offset| or residual SD over "
                    "the four datasets (sigma) -- NOT an HRF-specific value"),
            what_b=("largest primary-panel paired topology harm |h_net| "
                    "(sigma) -- NOT a FIVES-specific value"),
            hrf=big, fives=topo_max,
            ratio=float(big / topo_max) if (np.isfinite(topo_max) and
                                            topo_max > 0) else float("nan"),
            survives=bool(np.isfinite(topo_max) and topo_max > 0 and
                          big / topo_max >= 10.0),
            source=("largest primary-panel mean |offset| or residual SD under "
                    "this scale variant vs largest primary-panel paired "
                    "topology harm in %s" % topo_src)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    master = load_master(args.master)
    gatea = sigma_table()

    se = sigma_estimators(master, gatea)
    p = os.path.join(args.out_dir, "r2_sigma_estimators.csv")
    se.to_csv(p, index=False)
    print("wrote", p, se.shape)

    au = audit_units(master, gatea)
    p = os.path.join(args.out_dir, "r2_audit_raw_units.csv")
    au.to_csv(p, index=False)
    print("wrote", p, au.shape)

    sens = sensitivity(master)
    p = os.path.join(args.out_dir, "r2_scale_sensitivity.csv")
    sens.to_csv(p, index=False)
    print("wrote", p, sens.shape)

    hl = headline(au, sens)
    p = os.path.join(args.out_dir, "r2_headline_raw_primary.csv")
    hl.to_csv(p, index=False)
    print("wrote", p, hl.shape)

    with open(os.path.join(args.out_dir, "r2_common_scale_formula.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(COMMON_SCALE_FORMULA + "\n\nPer-biomarker values (FOV-"
                 "normalised representation, training-split reference masks of "
                 "DRIVE + CHASE_DB1 + HRF + FIVES, equal cohort weight):\n")
        cs = sens[sens["cohort"] == "common"][["biomarker", "sigma"]
                                             ].drop_duplicates()
        for rr in cs.itertuples(index=False):
            fh.write("  %-20s %.6g\n" % (rr.biomarker, rr.sigma))
    print("wrote", os.path.join(args.out_dir, "r2_common_scale_formula.txt"))

    cl = claims(sens, os.path.join(PIVOT_DIR, "e1_topology_matched.csv"))
    p = os.path.join(args.out_dir, "r2_scale_claims.csv")
    cl.to_csv(p, index=False)
    print("wrote", p, cl.shape)

    # ---- console summary -------------------------------------------------
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        chk = se[(se["representation"] == "raw") &
                 (se["estimator"] == GATEA_ESTIMATOR)]
        print("\n[sigma reproduction check: our 1.4826*MAD vs frozen Gate A]")
        print(chk[["dataset", "biomarker", "n", "sigma", "gateA_sigma",
                   "ratio_to_gateA"]].round(6).to_string(index=False))

        print("\n[raw-unit audit, primary skan panel]")
        pr = au[(au["panel"] == "primary")]
        print(pr[["dataset", "biomarker", "representation", "unit", "n",
                  "ref_median", "offset", "offset_lo", "offset_hi",
                  "resid_sd", "r_pearson"]].round(5).to_string(index=False))

        print("\n[scale sensitivity: per-dataset mean over the primary panel]")
        agg = (sens[sens["panel"] == "primary"]
               .groupby(["cohort", "estimator", "representation", "dataset"],
                        as_index=False)
               .agg(mean_abs_offset=("abs_offset", "mean"),
                    mean_resid_sd=("resid_sd", "mean")))
        print(agg.round(4).to_string(index=False))

        print("\n[claims]")
        print(cl.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
