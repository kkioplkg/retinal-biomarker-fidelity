"""E2 analysis -- measurement fidelity of the ReliSeg fine-tuning grid.

Consumes exactly what ``src.pivot.e2_run`` produced and writes seven artefacts
under ``results/pivot/``:

    e2_fidelity.csv    per (dataset, config, seed, checkpoint, biomarker,
                       pipeline): n, sigma, r_pearson, r_spearman,
                       bias_const_sigma, resid_sd_sigma, mae_sigma
    e2_delta.csv       paired image bootstrap (2000 resamples, 95 % CI) of
                       delta-r against the *same-seed* reference -- BOTH the
                       untouched baseline and the same-budget `continued`
                       control -- per seed and pooled over seeds (mean of the
                       per-seed delta-r, with a paired-bootstrap CI of that
                       mean, a seed-level t CI over the three seeds, and the
                       seed-level sign agreement)
    e2_pixel.csv       Dice / clDice and their paired deltas vs the same-seed
                       baseline (from pred*/pixel_metrics.csv; the run-level
                       means are cross-checked against infer_meta.json)
    e2_downstream.csv  the P3 downstream protocol (src/pivot/p3_downstream.py:
                       standardised multinomial logreg + small GBDT, 5-fold
                       stratified CV repeated 3x, pooled out-of-fold macro
                       one-vs-rest AUC) on the FIVES *test* biomarkers for the
                       GT / baseline / ReliSeg / CF-Loss arms, with a paired
                       image bootstrap of delta-AUC; HRF is reported the same
                       way as development-set evidence
    e2_zeroshot.csv    CHASE_DB1 / STARE fidelity of every main-grid last.pt
    e2_safety.csv      the pre-registered safety endpoints
    E2_REPORT.md       the narrative, every number carrying a source path

Definitions are the ones already used by ``src/pivot/e1_decompose.py``, on the
same sigma axis (``results/gateA_biomarker_scales_train.csv``, the robust
train-split scale 1.4826 * MAD):

    bias_const  mean_i (B_pred,i - B_GT,i) / sigma
    resid_sd    SD_i[(B_pred,i - B_GT,i) / sigma]        (ddof = 1)
    r_pearson   pearson  r(B_pred, B_GT)   over the test images
    r_spearman  spearman rho(B_pred, B_GT) over the test images

Nothing here selects a configuration on a test metric.  The pre-specified
primary endpoint is FIVES (confirmatory set), ``last.pt``, the **skan**
pipeline; PVBM is a sensitivity analysis.  The validation-fidelity checkpoint
(``fid``) is reported alongside and always labelled as such.  Safety margins
(DECISIONS 2026-09-17 13:30): Dice/clDice may not drop by more than 0.01
absolute, tortuosity/density delta-r >= -0.05, |delta bias| <= 0.25 sigma.

CLI
---
    python -m src.pivot.e2_analysis                 # everything that is ready
    python -m src.pivot.e2_analysis --skip-downstream
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import (EXP_ROOT, PIVOT_DIR, PRIMARY, PRIMARY_COLS,
                              PIPES, sigma_table)
from src.pivot import e2_run as E2

N_BOOT = 2000
SEED = 0
GT_MASTER = os.path.join(PIVOT_DIR, "bio_master.csv")
SIGMA_SRC = "results/gateA_biomarker_scales_train.csv"

#: the three pre-registered primary biomarkers (proposal v4, "预注册终点")
PRIMARY_ENDPOINT = ("density", "total_length", "FD")
#: primary panel = the skan pipeline (DECISIONS 2026-09-17 13:30); PVBM is
#: reported as a sensitivity analysis, never as the headline.
PRIMARY_PIPELINE = "skan"
SENSITIVITY_PIPELINE = "pvbm"
#: the four markers of the primary panel, in report order
PANEL = ("density", "total_length", "FD", "tortuosity")
#: pre-registered safety endpoints (proposal v4 "安全终点")
SAFETY_MARKERS = ("tortuosity", "density")

# ---- pre-registered safety margins (DECISIONS 2026-09-17 13:30) ----
#: clDice (and Dice) non-inferiority: the drop may not exceed this, absolute
PIXEL_MARGIN = 0.01
#: tortuosity / density fidelity may not fall by more than this in r
DELTA_R_MARGIN = -0.05
#: ... nor may their constant bias move by more than this, in sigma
DELTA_BIAS_MARGIN = 0.25

#: reference arms for delta-r, in order: the untouched baseline first, then
#: the same-budget `continued` control (DECISIONS 2026-09-17 13:30).
REFERENCES = ("baseline", "continued")

MAIN_CONFIGS = E2.MAIN_CONFIGS

#: column order of results/pivot/e2_delta.csv (also used to hand the report an
#: empty-but-well-formed frame before the biomarker stage has produced rows)
DELTA_COLS = [
    "dataset", "target", "config", "reference", "seed", "checkpoint",
    "biomarker", "pipeline", "n", "n_boot", "d_r", "lo", "hi",
    "d_bias_sigma", "seed_lo", "seed_hi", "n_seeds", "n_seeds_positive",
    "direction_consistent", "pred_path", "gt_path", "sigma_source",
]


def rel(p: str) -> str:
    """Path relative to the exp root, forward-slashed, for the report."""
    try:
        return os.path.relpath(p, EXP_ROOT).replace("\\", "/")
    except ValueError:
        return p.replace("\\", "/")


def split_col(c: str) -> Tuple[str, str]:
    for p in PIPES:
        if c.endswith("_" + p):
            return c[: -len(p) - 1], p
    return c, ""


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def pred_root(ds: str, seed: int, cfg: str, which: str,
              target: Optional[str] = None) -> str:
    suff = "" if which == "last" else "_fid"
    name = "pred" + suff + (("_" + target) if target else "")
    return os.path.join(EXP_ROOT, E2.run_dir(ds, seed, cfg), name)


def _read_bio(root: str) -> Optional[pd.DataFrame]:
    f = os.path.join(root, "bio.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d["__src"] = rel(f)
    return d


def gt_table(ds: str) -> pd.DataFrame:
    """GT biomarkers of the test split.

    In-domain (FIVES / HRF) they come from ``bio_master.csv`` (source=gt,
    split=test); the zero-shot targets have their own ``gtbio`` product
    ``results/pivot/e2_gt_<ds>.csv``, written by the same estimator path.
    """
    ext = os.path.join(PIVOT_DIR, f"e2_gt_{ds}.csv")
    if ds in ("chasedb1", "stare") and os.path.exists(ext):
        d = pd.read_csv(ext)
        d["__src"] = rel(ext)
        return d
    m = pd.read_csv(GT_MASTER)
    d = m[(m["dataset"] == ds) & (m["source"] == "gt") &
          (m["split"] == "test")].copy()
    d["__src"] = rel(GT_MASTER) + " (source=gt, split=test)"
    return d


def _key(s: pd.Series) -> pd.Series:
    """Join key tolerant of the FIVES split prefix (``test_100_D`` vs ``100_D``)."""
    return s.astype(str).str.replace(r"^(train|val|test)_", "", regex=True)


def sigma_for(ds: str, sig: Dict[str, Dict[str, float]],
              gt: pd.DataFrame) -> Tuple[Dict[str, float], str]:
    """sigma per biomarker column, and a human-readable provenance string."""
    if ds in sig:
        return sig[ds], SIGMA_SRC
    # STARE has no Gate A train scale: fall back to the robust scale of this
    # dataset's own GT test biomarkers, and say so everywhere it is used.
    out = {}
    for c in PRIMARY_COLS:
        v = pd.to_numeric(gt[c], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        out[c] = float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else np.nan
    return out, ("1.4826*MAD of this dataset's own GT test biomarkers "
                 "(no Gate A train scale for %s)" % ds)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def boot_idx(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """One fixed (n_boot, n) resample matrix -- shared by every comparison so
    all deltas on a dataset are paired on the same resampled images."""
    return np.random.RandomState(seed).randint(0, n, size=(n_boot, n))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    a, b = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    from scipy import stats
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def boot_r(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Pearson r of every bootstrap resample, vectorised: (n_boot,)."""
    a, b = x[idx], y[idx]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.where(den > 0, den, np.nan)


def ci(v: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size < 10:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def t_ci(v: Sequence[float]) -> Tuple[float, float]:
    """Two-sided 95 % t interval of the mean of a handful of seed estimates."""
    from scipy import stats
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return float("nan"), float("nan")
    m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))
    h = float(stats.t.ppf(0.975, v.size - 1)) * se
    return m - h, m + h


# --------------------------------------------------------------------------
# 1 -- fidelity, 2 -- delta-r, 5 -- zero-shot
# --------------------------------------------------------------------------
def _arm_frames(ds: str, which: str, target: Optional[str],
                pairs: Sequence[Tuple[str, int]]
                ) -> Tuple[Dict[Tuple[str, int], pd.DataFrame], List[str]]:
    arms: Dict[Tuple[str, int], pd.DataFrame] = {}
    missing = []
    for cfg, k in sorted(set(pairs)):
        # the baseline has no fine-tuning and therefore only one checkpoint;
        # it is the comparator for both `last` and `fid`.
        w = "last" if cfg == "baseline" else which
        root = pred_root(ds, k, cfg, w, target)
        d = _read_bio(root)
        if d is None:
            missing.append(rel(os.path.join(root, "bio.csv")))
            continue
        arms[(cfg, k)] = d
    return arms, missing


def items_to_pairs(items: Sequence[Tuple[int, str]]) -> List[Tuple[str, int]]:
    return [(c, k) for k, c in items]


def fidelity_and_delta(units: Sequence[Tuple[str, int, str, str]],
                       sig: Dict[str, Dict[str, float]],
                       target: Optional[str] = None
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Fidelity rows + paired-bootstrap delta-r rows for one evaluation family.

    ``units`` are (dataset, seed, config, which) tuples that share a test set;
    ``target`` None = in-domain, else the zero-shot dataset.
    """
    frows: List[dict] = []
    drows: List[dict] = []
    missing: List[str] = []

    fam: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for ds, k, cfg, which in units:
        fam.setdefault((ds, which), []).append((k, cfg))

    for (ds, which), items in sorted(fam.items()):
        # the baseline is the comparator for every family, including `fid`
        # (it has no fine-tuning, so its single checkpoint serves both); only
        # the seeds that actually have a fine-tuned arm need a baseline.
        pairs = set(items_to_pairs(items))
        seeds = sorted({k for k, _ in items})
        pairs |= {("baseline", k) for k in seeds}
        arms, miss = _arm_frames(ds, which, target, sorted(pairs))
        missing += miss
        if not arms:
            continue
        gt = gt_table(target or ds)
        gt_src = str(gt["__src"].iloc[0])
        sigs, sig_src = sigma_for(target or ds, sig, gt)
        gt = gt.assign(__k=_key(gt["image_id"]))

        # ---- common image set across every arm present (paired throughout) ----
        common = set(gt["__k"])
        for d in arms.values():
            common &= set(_key(d["image_id"]))
        common = sorted(common)
        if len(common) < 5:
            continue
        g = gt.set_index("__k").loc[common]
        A = {ab: d.assign(__k=_key(d["image_id"])).set_index("__k").loc[common]
             for ab, d in arms.items()}
        idx = boot_idx(len(common))

        for c in PRIMARY_COLS:
            bm, pipe = split_col(c)
            s = float(sigs.get(c, np.nan))
            y = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
            xs = {ab: pd.to_numeric(d[c], errors="coerce").to_numpy(float)
                  for ab, d in A.items()}
            ok = np.isfinite(y)
            for x in xs.values():
                ok &= np.isfinite(x)
            if ok.sum() < 5:
                continue
            yy = y[ok]
            xx = {ab: x[ok] for ab, x in xs.items()}
            # bootstrap indices restricted to the surviving rows
            bidx = boot_idx(int(ok.sum()))

            for (cfg, k), x in xx.items():
                if cfg == "baseline" and which != "last":
                    continue      # one baseline row per (ds, seed), under `last`
                e = (x - yy) / s if np.isfinite(s) and s > 0 else np.full_like(x, np.nan)
                frows.append(dict(
                    dataset=ds, target=target or ds, config=cfg, seed=k,
                    checkpoint=which, biomarker=bm, pipeline=pipe,
                    n=int(ok.sum()), sigma=s,
                    r_pearson=pearson(x, yy), r_spearman=spearman(x, yy),
                    bias_const_sigma=float(np.mean(e)),
                    resid_sd_sigma=float(np.std(e, ddof=1)),
                    mae_sigma=float(np.mean(np.abs(e))),
                    pred_path=rel(os.path.join(
                        pred_root(ds, k, cfg,
                                  "last" if cfg == "baseline" else which,
                                  target), "bio.csv")),
                    gt_path=gt_src, sigma_source=sig_src))

            # ---- delta-r vs each reference arm, same seed ----
            # REFERENCES: the untouched baseline AND the same-budget
            # `continued` control (DECISIONS 2026-09-17 13:30) -- a gain that
            # survives the second reference is not just "more training".
            for ref in REFERENCES:
                ref_boot = {k: boot_r(xx[(ref, k)], yy, bidx)
                            for k in seeds if (ref, k) in xx}
                if not ref_boot:
                    continue
                per_cfg_boot: Dict[str, Dict[int, np.ndarray]] = {}
                per_cfg_pt: Dict[str, Dict[int, float]] = {}
                per_cfg_bias: Dict[str, Dict[int, float]] = {}
                for (cfg, k), x in xx.items():
                    if cfg == ref or k not in ref_boot:
                        continue
                    if cfg in REFERENCES and REFERENCES.index(cfg) <                             REFERENCES.index(ref):
                        continue          # baseline-vs-continued once only
                    dd = boot_r(x, yy, bidx) - ref_boot[k]
                    lo, hi = ci(dd)
                    pt = pearson(x, yy) - pearson(xx[(ref, k)], yy)
                    db = (float(np.mean((x - yy) / s) - np.mean((xx[(ref, k)] - yy) / s))
                          if np.isfinite(s) and s > 0 else np.nan)
                    drows.append(dict(
                        dataset=ds, target=target or ds, config=cfg,
                        reference=ref, seed=str(k),
                        checkpoint=which, biomarker=bm, pipeline=pipe,
                        n=int(ok.sum()), n_boot=N_BOOT,
                        d_r=pt, lo=lo, hi=hi, d_bias_sigma=db,
                        seed_lo=np.nan, seed_hi=np.nan,
                        n_seeds=1, n_seeds_positive=int(pt > 0),
                        direction_consistent="",
                        pred_path=rel(os.path.join(
                            pred_root(ds, k, cfg, which, target), "bio.csv")),
                        gt_path=gt_src, sigma_source=sig_src))
                    per_cfg_boot.setdefault(cfg, {})[k] = dd
                    per_cfg_pt.setdefault(cfg, {})[k] = pt
                    per_cfg_bias.setdefault(cfg, {})[k] = db

                # ---- pooled over seeds: mean of the per-seed delta-r ----
                for cfg, per_seed in per_cfg_boot.items():
                    ks = sorted(per_seed)
                    if len(ks) < 2:
                        continue
                    pooled = np.vstack([per_seed[k] for k in ks]).mean(0)
                    pts = [per_cfg_pt[cfg][k] for k in ks]
                    lo, hi = ci(pooled)
                    slo, shi = t_ci(pts)
                    npos = int(sum(1 for v in pts if v > 0))
                    drows.append(dict(
                        dataset=ds, target=target or ds, config=cfg,
                        reference=ref,
                        seed="pooled(%s)" % ",".join(str(k) for k in ks),
                        checkpoint=which, biomarker=bm, pipeline=pipe,
                        n=int(ok.sum()), n_boot=N_BOOT,
                        d_r=float(np.mean(pts)), lo=lo, hi=hi,
                        d_bias_sigma=float(np.nanmean(
                            [per_cfg_bias[cfg][k] for k in ks])),
                        seed_lo=slo, seed_hi=shi,
                        n_seeds=len(ks), n_seeds_positive=npos,
                        direction_consistent=("yes" if npos in (0, len(ks))
                                              else "no"),
                        pred_path="runs/pivot/%s/seed{%s}/%s/%s%s/bio.csv"
                        % (ds, ",".join(str(k) for k in ks), cfg,
                           "pred" if which == "last" else "pred_fid",
                           ("_" + target) if target else ""),
                        gt_path=gt_src, sigma_source=sig_src))
    return pd.DataFrame(frows), pd.DataFrame(drows), missing


# --------------------------------------------------------------------------
# 3 -- pixel metrics
# --------------------------------------------------------------------------
def pixel_table(units: Sequence[Tuple[str, int, str, str]]) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[dict] = []
    missing: List[str] = []
    per: Dict[Tuple[str, str], Dict[Tuple[str, int], pd.DataFrame]] = {}
    for ds, k, cfg, which in units:
        root = pred_root(ds, k, cfg, which)
        f = os.path.join(root, "pixel_metrics.csv")
        mf = os.path.join(root, "infer_meta.json")
        if not os.path.exists(f):
            missing.append(rel(f))
            continue
        px = pd.read_csv(f)
        meta = json.load(open(mf, encoding="utf-8")) if os.path.exists(mf) else {}
        rows.append(dict(
            dataset=ds, config=cfg, seed=k, checkpoint=which, n=len(px),
            mean_dice=float(px["dice"].mean()),
            mean_cldice=float(px["cldice"].mean()),
            meta_mean_dice=meta.get("mean_dice"),
            meta_mean_cldice=meta.get("mean_cldice"),
            ckpt_epoch=meta.get("ckpt_epoch"), ckpt=meta.get("ckpt"),
            d_dice=np.nan, d_dice_lo=np.nan, d_dice_hi=np.nan,
            d_cldice=np.nan, d_cldice_lo=np.nan, d_cldice_hi=np.nan,
            px_path=rel(f), meta_path=rel(mf)))
        per.setdefault((ds, which), {})[(cfg, k)] = px

    base_by = {}
    for (ds, which), arms in per.items():
        for (cfg, k), px in arms.items():
            if cfg == "baseline":
                base_by[(ds, k)] = px
    for r in rows:
        if r["config"] == "baseline":
            continue
        b = base_by.get((r["dataset"], r["seed"]))
        if b is None:
            continue
        a = per[(r["dataset"], r["checkpoint"])][(r["config"], r["seed"])]
        m = a[["image", "dice", "cldice"]].merge(
            b[["image", "dice", "cldice"]], on="image", suffixes=("_a", "_b"))
        if len(m) < 5:
            continue
        idx = boot_idx(len(m))
        for name in ("dice", "cldice"):
            d = (m[name + "_a"] - m[name + "_b"]).to_numpy(float)
            bs = d[idx].mean(1)
            lo, hi = ci(bs)
            r["d_" + name] = float(d.mean())
            r["d_%s_lo" % name], r["d_%s_hi" % name] = lo, hi
    return pd.DataFrame(rows), missing


# --------------------------------------------------------------------------
# 4 -- downstream (the P3 protocol)
# --------------------------------------------------------------------------
def downstream_table(datasets=("fives", "hrf"),
                     configs=MAIN_CONFIGS, which: str = "last"
                     ) -> Tuple[pd.DataFrame, List[str]]:
    """The P3 downstream protocol applied to the E2 arms.

    Frozen primary panel (DECISIONS 2026-09-17 13:30): the four skan
    biomarkers.  The eight-column panel duplicates density exactly (both
    pipelines read the same foreground fraction) and carries the PVBM
    tortuosity column that Gate A disqualified, so it is kept as a
    sensitivity analysis rather than as the headline.  delta-AUC is reported
    against both reference arms, exactly as delta-r is.
    """
    from src.pivot.p3_downstream import cv_proba, macro_auc

    panels = {"skan4": [c for c in PRIMARY_COLS if c.endswith("_skan")],
              "primary8": list(PRIMARY_COLS)}
    rows: List[dict] = []
    missing: List[str] = []
    for ds in datasets:
        gt = gt_table(ds)
        gt_src = str(gt["__src"].iloc[0])
        gt = gt.assign(__k=_key(gt["image_id"]))
        arms, miss = _arm_frames(ds, which, None,
                                 [(c, k) for c in configs for k in E2.SEEDS])
        missing += miss
        if not arms:
            continue
        common = set(gt["__k"])
        for d in arms.values():
            common &= set(_key(d["image_id"]))
        common = sorted(common)
        g = gt.set_index("__k").loc[common]
        g = g[g["disease"].notna()]
        common = list(g.index)
        if len(common) < 24:
            continue
        y = g["disease"].to_numpy()
        classes = sorted(pd.unique(y))
        if len(classes) < 2 or min(int((y == c).sum()) for c in classes) < 8:
            continue
        idx = boot_idx(len(y))

        def X_of(d: pd.DataFrame, cols) -> np.ndarray:
            d = d.assign(__k=_key(d["image_id"])).set_index("__k").loc[common]
            return np.nan_to_num(d[list(cols)].to_numpy(float),
                                 nan=0.0, posinf=0.0, neginf=0.0)

        for fs, cols in panels.items():
            feats = {("gt", -1): np.nan_to_num(g[list(cols)].to_numpy(float),
                                               nan=0.0, posinf=0.0, neginf=0.0)}
            for ab, d in arms.items():
                feats[ab] = X_of(d, cols)
            for kind in ("logreg", "gbdt"):
                probas = {ab: cv_proba(X, y, kind, classes)
                          for ab, X in feats.items()}
                aucs = {ab: macro_auc(y, p, classes) for ab, p in probas.items()}
                # one resample serves every arm, so all deltas are paired
                boots: Dict[Tuple[str, int], List[float]] = {ab: [] for ab in probas}
                for ix in idx:
                    if len(np.unique(y[ix])) < len(classes):
                        continue
                    for ab, p in probas.items():
                        boots[ab].append(macro_auc(y[ix], p[ix], classes))
                for ab in probas:
                    cfg, k = ab
                    lo, hi = ci(boots[ab])
                    gt_gap = np.asarray(boots[("gt", -1)]) - np.asarray(boots[ab])
                    glo, ghi = ci(gt_gap)
                    for ref_cfg in REFERENCES:
                        ref = (ref_cfg, k)
                        d_auc = d_lo = d_hi = np.nan
                        if cfg not in (ref_cfg, "gt") and ref in boots:
                            dv = np.asarray(boots[ab]) - np.asarray(boots[ref])
                            d_auc = aucs[ab] - aucs[ref]
                            d_lo, d_hi = ci(dv)
                        elif ref_cfg != REFERENCES[0]:
                            continue     # gt / the reference itself: one row
                        rows.append(dict(
                            dataset=ds, checkpoint=which, clf=kind,
                            featureset=fs,
                            is_primary=bool(fs == "skan4" and kind == "logreg"
                                            and ds == "fives"
                                            and which == "last"),
                            config=cfg, reference=ref_cfg,
                            seed=("-" if cfg == "gt" else k),
                            n=len(y), n_classes=len(classes), n_boot=N_BOOT,
                            macro_auc=aucs[ab], ci_lo=lo, ci_hi=hi,
                            d_auc_vs_ref=d_auc, d_auc_lo=d_lo, d_auc_hi=d_hi,
                            gt_minus_this=aucs[("gt", -1)] - aucs[ab],
                            gt_gap_lo=glo, gt_gap_hi=ghi,
                            class_counts=";".join(
                                "%s=%d" % (c, int((y == c).sum()))
                                for c in classes),
                            features="+".join(cols),
                            protocol="src/pivot/p3_downstream.py cv_proba "
                                     "(5-fold stratified x3 repeats, "
                                     "logreg/GBDT)",
                            pred_path=(gt_src if cfg == "gt" else rel(
                                os.path.join(pred_root(ds, k, cfg, which),
                                             "bio.csv")))))
    return pd.DataFrame(rows), missing


# --------------------------------------------------------------------------
# 6 -- safety
# --------------------------------------------------------------------------
def safety_table(delta: pd.DataFrame, pixel: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    if len(delta):
        d = delta[delta["biomarker"].isin(SAFETY_MARKERS)].copy()
        for r in d.itertuples(index=False):
            # pre-registered margin: delta-r >= -0.05.  The point estimate
            # breaching it is a BREACH; the CI reaching past it while the
            # point estimate does not is flagged separately as AT-RISK.
            breach = np.isfinite(r.d_r) and r.d_r < DELTA_R_MARGIN
            at_risk = (not breach) and np.isfinite(r.lo) and r.lo < DELTA_R_MARGIN
            rows.append(dict(
                endpoint="delta_r", dataset=r.dataset, target=r.target,
                config=r.config, reference=r.reference, seed=r.seed,
                checkpoint=r.checkpoint,
                biomarker=r.biomarker, pipeline=r.pipeline,
                value=r.d_r, lo=r.lo, hi=r.hi, margin_value=DELTA_R_MARGIN,
                margin="pre-registered (DECISIONS 2026-09-17 13:30): "
                       "tortuosity/density delta-r >= -0.05",
                flag="BREACH" if breach else ("AT-RISK" if at_risk else "ok"),
                source=r.pred_path))
            rows.append(dict(
                endpoint="delta_bias_sigma", dataset=r.dataset, target=r.target,
                config=r.config, reference=r.reference, seed=r.seed,
                checkpoint=r.checkpoint,
                biomarker=r.biomarker, pipeline=r.pipeline,
                value=r.d_bias_sigma, lo=np.nan, hi=np.nan,
                margin_value=DELTA_BIAS_MARGIN,
                margin="pre-registered (DECISIONS 2026-09-17 13:30): "
                       "|delta bias| <= 0.25 sigma",
                flag=("BREACH" if (np.isfinite(r.d_bias_sigma)
                                   and abs(r.d_bias_sigma) > DELTA_BIAS_MARGIN)
                      else "ok"),
                source=r.pred_path))
    if len(pixel):
        p = pixel[pixel["config"] != "baseline"]
        for r in p.itertuples(index=False):
            for name, v, lo, hi in (("delta_dice", r.d_dice, r.d_dice_lo, r.d_dice_hi),
                                    ("delta_cldice", r.d_cldice, r.d_cldice_lo,
                                     r.d_cldice_hi)):
                rows.append(dict(
                    endpoint=name, dataset=r.dataset, target=r.dataset,
                    config=r.config, reference="baseline", seed=r.seed,
                    checkpoint=r.checkpoint,
                    biomarker="pixel", pipeline="-",
                    value=v, lo=lo, hi=hi, margin_value=-PIXEL_MARGIN,
                    margin="pre-registered non-inferiority: the drop may not "
                           "exceed %.2f absolute (proposal v4 安全终点; "
                           "DECISIONS 2026-09-17 13:30)" % PIXEL_MARGIN,
                    flag=("BREACH" if (np.isfinite(v) and v < -PIXEL_MARGIN)
                          else ("ok" if np.isfinite(v) else "na")),
                    source=r.px_path))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt(v, nd=3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"


def srt(df, by: Sequence[str]):
    """sort_values that tolerates an empty frame or absent columns."""
    if df is None or not len(df):
        return df if df is not None else pd.DataFrame()
    have = [c for c in by if c in df.columns]
    return df.sort_values(have) if have else df


def _tbl(df: pd.DataFrame, cols: Sequence[str], nd: int = 3) -> str:
    if df is None or not len(df):
        return "_(no rows yet)_\n"
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return "_(no rows yet)_\n"
    d = df[list(cols)].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.{nd}f}")
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    body = "".join("| " + " | ".join(str(x) for x in r) + " |\n"
                   for r in d.itertuples(index=False))
    return head + body


def coverage_table() -> pd.DataFrame:
    """Which (dataset, config, checkpoint) cells already have a bio.csv.

    The grid is run incrementally, so a report can be produced before every
    cell exists.  This table is what tells the reader which rows of the
    report are final and which are still pending -- it is derived from the
    files on disk, never asserted.
    """
    rows = []
    fam = {}
    for ds, k, cfg, which in E2.eval_units():
        fam.setdefault((ds, cfg, which), []).append(k)
    for (ds, cfg, which), seeds in sorted(fam.items()):
        have, want = [], sorted(seeds)
        for k in want:
            w = "last" if cfg == "baseline" else which
            if os.path.exists(os.path.join(pred_root(ds, k, cfg, w), "bio.csv")):
                have.append(k)
        # The zero-shot sweep is deliberately frozen at E2.SEEDS (0-2): the
        # review-16 extra seeds were added for the in-domain null only, and
        # ``main`` builds ``zunits`` from E2.SEEDS explicitly.  Counting the
        # denominator over ``want`` instead made complete cells read "6/12".
        zs_seeds = [k for k in want if k in E2.SEEDS]
        zs = 0
        if cfg in MAIN_CONFIGS and which == "last":
            for t in E2.EXTERNAL:
                for k in zs_seeds:
                    if os.path.exists(os.path.join(
                            pred_root(ds, k, cfg, "last", t), "bio.csv")):
                        zs += 1
        rows.append(dict(
            dataset=ds, config=cfg, checkpoint=which,
            seeds_done="%d/%d" % (len(have), len(want)),
            seeds=",".join(str(k) for k in have) or "-",
            zeroshot_done="%d/%d" % (zs, 2 * len(zs_seeds))
            if (cfg in MAIN_CONFIGS and which == "last") else "n/a",
            status=("complete" if len(have) == len(want)
                    else ("partial" if have else "PENDING"))))
    return pd.DataFrame(rows)


def write_report(fid: pd.DataFrame, delta: pd.DataFrame, pixel: pd.DataFrame,
                 down: pd.DataFrame, zero: pd.DataFrame, safety: pd.DataFrame,
                 missing: Sequence[str], out: str,
                 out_dir: Optional[str] = None) -> None:
    L: List[str] = []
    A = L.append
    SK = PRIMARY_PIPELINE

    def dsel(ds, which="last", ref="baseline", pooled=True, pipe=SK,
             markers=PANEL, abl=None):
        """Slice of e2_delta.csv, with the ablation rows in or out."""
        if not len(delta):
            return pd.DataFrame(columns=DELTA_COLS)
        d = delta[(delta["dataset"] == ds) & (delta["checkpoint"] == which)
                  & (delta["reference"] == ref) & (delta["pipeline"] == pipe)
                  & delta["biomarker"].isin(markers)]
        ispool = d["seed"].astype(str).str.startswith("pooled")
        d = d[ispool if pooled else ~ispool]
        isabl = d["config"].astype(str).str.startswith("abl_")
        if abl is True:
            d = d[isabl]
        elif abl is False:
            d = d[~isabl]
        return d

    A("# E2 -- measurement-fidelity fine-tuning grid (ReliSeg vs faithful CF-Loss)\n")
    A("Generated %s by `src/pivot/e2_analysis.py`.\n"
      % time.strftime("%Y-%m-%d %H:%M:%S"))

    A("\n## Pre-registration this report is read against\n")
    A("- Design: `DECISIONS.md` 2026-09-17 00:30 / 02:00 / 13:30 and "
      "`proposal/07_pivot_proposal_v4.md`.\n"
      "- **Primary endpoint**: FIVES (confirmatory set), `last.pt`, "
      "density / total_length / FD, **skan pipeline**. PVBM is a sensitivity "
      "analysis (section 8), never the headline.\n"
      "- **Two reference arms**: the untouched `baseline` "
      "(`runs/seg/fives/seed<k>/best.pt`, re-inferred through the identical "
      "E2 path) and `continued` -- the same-budget control: same starting "
      "checkpoint, same 50 epochs, same LR schedule and batch size, base loss "
      "only (CE-Dice + 0.5*clDice, lambda = 0). A gain that survives the "
      "second reference is not simply more training.\n"
      "- HRF is the development/mechanism set (its test split was touched by "
      "probes P1-P5); FIVES is confirmatory.\n"
      "- `fid` = the checkpoint with the best *validation* measurement "
      "fidelity. Reported alongside, always labelled. Nothing is selected on "
      "a test metric.\n")
    A("- Safety margins: Dice/clDice drop <= %.2f absolute; tortuosity and "
      "density delta-r >= %+.2f; |delta bias| <= %.2f sigma.\n"
      % (PIXEL_MARGIN, DELTA_R_MARGIN, DELTA_BIAS_MARGIN))
    A("- sigma axis: `%s` (robust train-split scale, 1.4826 * MAD); "
      "bias_const / resid_sd / r defined exactly as in "
      "`src/pivot/e1_decompose.py`.\n" % SIGMA_SRC)

    A("\n## Artefacts\n")
    for f in ("e2_fidelity.csv", "e2_delta.csv", "e2_pixel.csv",
              "e2_downstream.csv", "e2_zeroshot.csv", "e2_safety.csv"):
        q = os.path.join(out_dir or PIVOT_DIR, f)
        try:                     # a stage with nothing to write yet is empty
            n = len(pd.read_csv(q)) if os.path.exists(q) else 0
        except pd.errors.EmptyDataError:
            n = 0
        A("- `%s` -- %d rows%s\n" % (rel(q), n, "" if n else "  *(pending)*"))

    # ---- 0 coverage ----
    cov = coverage_table()
    npend = int((cov["status"] != "complete").sum())
    A("\n## 0. Coverage -- which cells this version of the report covers\n")
    if npend:
        A("**This is a partial report: %d of %d (dataset x config x "
          "checkpoint) cells are not complete.** Rows for them are absent "
          "below, not zero. Re-run `python -m src.pivot.e2_analysis` when the "
          "CPU biomarker stage finishes.\n\n" % (npend, len(cov)))
    else:
        A("All %d (dataset x config x checkpoint) cells are complete.\n\n"
          % len(cov))
    A(_tbl(srt(cov[cov["status"] != "complete"],
               ["dataset", "config", "checkpoint"]),
           ["dataset", "config", "checkpoint", "seeds_done", "seeds",
            "zeroshot_done", "status"]))
    A("\nComplete cells:\n\n")
    A(_tbl(srt(cov[cov["status"] == "complete"],
               ["dataset", "config", "checkpoint"]),
           ["dataset", "config", "checkpoint", "seeds_done", "zeroshot_done"]))

    # ---- 1 primary endpoint ----
    A("\n## 1. Primary endpoint -- FIVES confirmatory set, `last.pt`, skan\n")
    A("Paired image bootstrap over the FIVES test images, %d resamples, 95%% "
      "percentile CI, against the **same-seed** reference. "
      "Source: `results/pivot/e2_delta.csv`.\n" % N_BOOT)
    for ref in REFERENCES:
        # The seed set is no longer uniform across configs (CMIG review 16
        # added FIVES seeds 3-5 for baseline/continued/reliseg only), so the
        # heading must not hard-code a range -- the per-row ``n_seeds`` column
        # is the authority.  Nothing about the estimator changed.
        A("\n### vs `%s`, pooled over seeds (mean of the per-seed delta-r; "
          "`n_seeds` gives how many seeds each row pools)\n\n" % ref)
        A(_tbl(dsel("fives", "last", ref, abl=False)
               .sort_values(["config", "biomarker"]),
               ["config", "biomarker", "d_r", "lo", "hi", "seed_lo", "seed_hi",
                "n_seeds_positive", "direction_consistent", "d_bias_sigma"]))
    A("\n`lo`/`hi` = paired image-bootstrap CI of the seed-averaged delta-r; "
      "`seed_lo`/`seed_hi` = 95% t interval over the `n_seeds` per-seed point "
      "estimates; `n_seeds_positive` out of `n_seeds` with "
      "`direction_consistent` gives the seed-level sign agreement.\n")
    for ref in REFERENCES:
        A("\n### Seed by seed (vs `%s`)\n\n" % ref)
        A(_tbl(dsel("fives", "last", ref, pooled=False, abl=False)
               .sort_values(["config", "biomarker", "seed"]),
               ["config", "seed", "biomarker", "d_r", "lo", "hi"]))
    A("\n### `fid` checkpoint (validation-fidelity selection; secondary)\n\n")
    A(_tbl(dsel("fives", "fid", "baseline", abl=False)
           .sort_values(["config", "biomarker"]),
           ["config", "biomarker", "d_r", "lo", "hi", "direction_consistent"]))

    # ---- 2 absolute fidelity ----
    A("\n## 2. Absolute fidelity (FIVES, `last.pt`, skan, seed-averaged)\n")
    A("Source: `results/pivot/e2_fidelity.csv`.\n\n")
    if len(fid):
        f = fid[(fid["dataset"] == "fives") & (fid["checkpoint"] == "last")
                & (fid["pipeline"] == SK) & fid["biomarker"].isin(PANEL)]
        if len(f):
            agg = (f.groupby(["config", "biomarker"])
                   [["r_pearson", "r_spearman", "bias_const_sigma",
                     "resid_sd_sigma"]].mean().reset_index())
            A(_tbl(agg.sort_values(["biomarker", "config"]),
                   ["config", "biomarker", "r_pearson", "r_spearman",
                    "bias_const_sigma", "resid_sd_sigma"]))

    # ---- 3 pixel ----
    A("\n## 3. Pixel metrics (non-inferiority margin %.2f absolute)\n"
      % PIXEL_MARGIN)
    A("Source: `results/pivot/e2_pixel.csv` (per-image "
      "`runs/pivot/<ds>/seed<k>/<cfg>/pred*/pixel_metrics.csv`, cross-checked "
      "against `infer_meta.json`). Deltas are vs the same-seed baseline.\n\n")
    if len(pixel):
        px = pixel[(pixel["checkpoint"] == "last") & (pixel["config"] != "baseline")]
        agg = (px.groupby(["dataset", "config"])[["mean_dice", "mean_cldice",
                                                  "d_dice", "d_cldice"]]
               .mean().reset_index())
        A(_tbl(agg, ["dataset", "config", "mean_dice", "mean_cldice",
                     "d_dice", "d_cldice"], nd=4))

    # ---- 4 downstream ----
    A("\n## 4. Downstream utility (P3 protocol)\n")
    A("Source: `results/pivot/e2_downstream.csv`; protocol "
      "`src/pivot/p3_downstream.py` (standardised multinomial logistic "
      "regression and a small GBDT, 5-fold stratified CV repeated 3x, pooled "
      "out-of-fold macro one-vs-rest AUC, paired image bootstrap of delta-AUC "
      "vs the same-seed baseline).\n\n")
    if len(down):
        cols = ["clf", "config", "reference", "seed", "macro_auc", "ci_lo",
                "ci_hi", "d_auc_vs_ref", "d_auc_lo", "d_auc_hi"]
        prim = down[(down["featureset"] == "skan4")
                    & (down["checkpoint"] == "last")]
        A("Primary panel (skan4), FIVES (confirmatory):\n\n")
        A(_tbl(srt(prim[prim["dataset"] == "fives"],
                   ["clf", "config", "reference", "seed"]), cols, nd=4))
        A("\nPrimary panel (skan4), HRF (development set):\n\n")
        A(_tbl(srt(prim[prim["dataset"] == "hrf"],
                   ["clf", "config", "reference", "seed"]), cols, nd=4))
        A("\nSensitivity -- the eight-column panel (both pipelines), FIVES:\n\n")
        A(_tbl(srt(down[(down["featureset"] == "primary8")
                        & (down["dataset"] == "fives")
                        & (down["checkpoint"] == "last")],
                   ["clf", "config", "reference", "seed"]), cols, nd=4))

    # ---- 5 zero-shot ----
    A("\n## 5. Zero-shot cross-dataset fidelity (CHASE_DB1 / STARE)\n")
    A("**These two sets are the only masked evidence outside the development "
      "set**: neither model ever saw them, and no probe touched them. They are "
      "therefore reported seed by seed rather than pooled -- with n this "
      "small, a seed mean hides more than it summarises.\n\n"
      "Sources: `results/pivot/e2_zeroshot.csv` (fidelity + delta-r against "
      "both references), `results/pivot/e2_zeroshot_delta.csv` (every delta "
      "row incl. the seed-pooled ones), `results/pivot/e2_zeroshot_delta_"
      "pooled.csv`. GT biomarkers: `results/pivot/e2_gt_chasedb1.csv` (n=8) "
      "and `results/pivot/e2_gt_stare.csv` (n=10), written by `p5_eval gtbio` "
      "on the same estimator path. Inference runs at the *source* model's "
      "scale (longest side 1536, patch 768).\n\n"
      "**Caveat that governs every number here: n = 8 (CHASE_DB1) and n = 10 "
      "(STARE) test images.** A Pearson r on 8 points has a 95 %% interval "
      "roughly +-0.5 wide, and STARE has no Gate A train scale, so its sigma "
      "is the robust scale of its own GT biomarkers (stated in the "
      "`sigma_source` column). Read these as directional evidence, not as "
      "estimates.\n")
    if len(zero):
        z = zero[(zero["pipeline"] == SK) & zero["biomarker"].isin(PANEL)]
        for target in sorted(z["target"].unique()):
            zt = z[z["target"] == target]
            A("\n### %s -- absolute fidelity, seed by seed\n\n" % target)
            A(_tbl(srt(zt, ["dataset", "biomarker", "config", "seed"]),
                   ["dataset", "config", "seed", "biomarker", "n",
                    "r_pearson", "r_spearman", "bias_const_sigma",
                    "resid_sd_sigma"]))
            for ref in REFERENCES:
                cols = [c for c in ("d_r_vs_%s" % ref, "d_r_vs_%s_lo" % ref,
                                    "d_r_vs_%s_hi" % ref) if c in zt.columns]
                if not cols or not zt[cols[0]].notna().any():
                    A("\n#### delta-r vs `%s`: _(pending -- the `%s` arm has "
                      "no zero-shot biomarkers yet)_\n" % (ref, ref))
                    continue
                A("\n#### %s -- delta-r vs `%s` (paired bootstrap over the "
                  "same images, %d resamples)\n\n" % (target, ref, N_BOOT))
                A(_tbl(srt(zt[zt[cols[0]].notna()],
                           ["dataset", "biomarker", "config", "seed"]),
                       ["dataset", "config", "seed", "biomarker"] + cols))

    # ---- 6 safety ----
    A("\n## 6. Safety endpoints\n")
    A("Source: `results/pivot/e2_safety.csv`. Margins are the pre-registered "
      "ones listed at the top. `BREACH` = the point estimate is past the "
      "margin; `AT-RISK` = the point estimate holds but the 95% CI reaches "
      "past it.\n\n")
    if len(safety):
        s6 = safety[(safety["checkpoint"] == "last")
                    & (safety["flag"].isin(("BREACH", "AT-RISK")))]
        A("Flagged rows: %d of %d.\n\n" % (len(s6), len(safety)))
        A(_tbl(s6.sort_values(["dataset", "config", "endpoint", "biomarker"])
               .head(80),
               ["dataset", "config", "reference", "seed", "endpoint",
                "biomarker", "pipeline", "value", "lo", "hi", "flag"]))

    # ---- 6b HRF, the development set ----
    A("\n## 6b. HRF -- development set, all seeds, `last.pt`, skan\n")
    A("HRF's test split was touched by probes P1-P5, so nothing here is "
      "confirmatory. It is reported in full because it is where the headroom "
      "is: the HRF baseline sits at r = 0.29-0.36 for FD and total_length "
      "against 0.90-0.98 on FIVES, so an effect that exists at all should be "
      "visible here first.\n\n")
    for ref in REFERENCES:
        A("\n### Pooled over seeds 0-2, vs `%s`\n\n" % ref)
        A(_tbl(srt(dsel("hrf", "last", ref, abl=False),
                   ["config", "biomarker"]),
               ["config", "biomarker", "d_r", "lo", "hi", "seed_lo", "seed_hi",
                "n_seeds_positive", "direction_consistent", "d_bias_sigma"]))
        A("\n### Seed by seed, vs `%s`\n\n" % ref)
        A(_tbl(srt(dsel("hrf", "last", ref, pooled=False, abl=False),
                   ["config", "biomarker", "seed"]),
               ["config", "seed", "biomarker", "d_r", "lo", "hi"]))
    A("\n### HRF safety flags (clDice non-inferiority and the unconstrained "
      "markers)\n\n")
    if len(safety):
        hs = safety[(safety["dataset"] == "hrf")
                    & (safety["checkpoint"] == "last")
                    & (safety["endpoint"].isin(("delta_cldice", "delta_dice",
                                                "delta_r", "delta_bias_sigma")))
                    & (safety["flag"] != "ok")]
        A("Flagged HRF rows: %d.\n\n" % len(hs))
        A(_tbl(srt(hs, ["endpoint", "config", "biomarker", "seed"]),
               ["config", "reference", "seed", "endpoint", "biomarker",
                "pipeline", "value", "lo", "hi", "margin_value", "flag"], nd=4))
    A("\nThe clDice rows are complete for the whole grid already: they come "
      "from `pred*/pixel_metrics.csv`, written by the inference stage, and do "
      "not wait on the CPU biomarker stage.\n")

    # ---- 7 ablations ----
    A("\n## 7. Ablations (HRF seed 0 only -- development set, skan)\n")
    A("Leave-one-out (`abl_no_*`) says which term is dispensable; single-term "
      "(`abl_*_only`) says which term carries the effect. Reference: the "
      "same-seed baseline.\n\n")
    A(_tbl(dsel("hrf", "last", "baseline", pooled=False, abl=True)
           .sort_values(["config", "biomarker"]),
           ["config", "biomarker", "d_r", "lo", "hi"]))

    # ---- 8 PVBM sensitivity ----
    A("\n## 8. Sensitivity -- the PVBM pipeline (FIVES, `last.pt`)\n")
    A("The same primary contrast measured with the second biomarker pipeline. "
      "Agreement with section 1 is the check that the effect is a property of "
      "the segmentation, not of one estimator.\n\n")
    A(_tbl(dsel("fives", "last", "baseline", pipe=SENSITIVITY_PIPELINE,
                abl=False).sort_values(["config", "biomarker"]),
           ["config", "biomarker", "d_r", "lo", "hi", "direction_consistent"]))

    if missing:
        A("\n## Not yet available\n")
        A("These products were missing when the analysis ran (a CPU biomarker "
          "job or an inference job had not finished):\n\n")
        for m in sorted(set(missing)):
            A("- `%s`\n" % m)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(L))
    print("-> %s" % rel(out))

# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-downstream", action="store_true")
    ap.add_argument("--out-dir", default=PIVOT_DIR)
    a = ap.parse_args(argv)
    os.chdir(EXP_ROOT)
    os.makedirs(a.out_dir, exist_ok=True)

    sig = sigma_table()
    units = E2.eval_units()
    missing: List[str] = []

    print("[e2a] in-domain fidelity / delta-r ...", flush=True)
    fid, delta, m = fidelity_and_delta(units, sig)
    missing += m

    print("[e2a] pixel metrics ...", flush=True)
    pixel, m = pixel_table(units)
    missing += m

    print("[e2a] zero-shot ...", flush=True)
    zrows, zdelta = [], []
    zunits = [(ds, k, cfg, "last") for ds in ("fives", "hrf")
              for cfg in MAIN_CONFIGS for k in E2.SEEDS]
    for target in E2.EXTERNAL:
        if not os.path.exists(os.path.join(PIVOT_DIR, f"e2_gt_{target}.csv")):
            missing.append(rel(os.path.join(PIVOT_DIR, f"e2_gt_{target}.csv")))
            continue
        zf, zd, m = fidelity_and_delta(zunits, sig, target=target)
        zrows.append(zf)
        zdelta.append(zd)
        missing += m
    zero = pd.concat([z for z in zrows if len(z)], ignore_index=True) \
        if any(len(z) for z in zrows) else pd.DataFrame()
    zerod = pd.concat([z for z in zdelta if len(z)], ignore_index=True) \
        if any(len(z) for z in zdelta) else pd.DataFrame()
    if len(zero) and len(zerod):
        # One row per (target, source model, config, seed, biomarker,
        # pipeline), carrying delta-r against BOTH references side by side --
        # merging on the delta table as-is would duplicate every row once per
        # reference.
        keys = ["target", "dataset", "config", "seed", "biomarker", "pipeline"]
        zero = zero.assign(seed=zero["seed"].astype(str))
        per_seed = zerod[~zerod["seed"].astype(str).str.startswith("pooled")]
        for ref in REFERENCES:
            zd = per_seed[per_seed["reference"] == ref][
                keys + ["d_r", "lo", "hi"]].rename(columns={
                    "d_r": f"d_r_vs_{ref}", "lo": f"d_r_vs_{ref}_lo",
                    "hi": f"d_r_vs_{ref}_hi"})
            if len(zd):
                zero = zero.merge(zd, on=keys, how="left")
        zpool = zerod[zerod["seed"].astype(str).str.startswith("pooled")]
        if len(zpool):
            zpool.to_csv(os.path.join(a.out_dir, "e2_zeroshot_delta_pooled.csv"),
                         index=False)
        zerod.to_csv(os.path.join(a.out_dir, "e2_zeroshot_delta.csv"),
                     index=False)

    down = pd.DataFrame()
    if not a.skip_downstream:
        print("[e2a] downstream (this takes a few minutes) ...", flush=True)
        down, m = downstream_table()
        missing += m
    else:
        # --skip-downstream must not destroy a downstream table an earlier run
        # already paid for: reload it so the report still carries those rows.
        q = os.path.join(a.out_dir, "e2_downstream.csv")
        if os.path.exists(q):
            try:
                down = pd.read_csv(q)
                print("[e2a] downstream skipped; kept %d existing rows from %s"
                      % (len(down), rel(q)), flush=True)
            except pd.errors.EmptyDataError:
                pass

    safety = safety_table(delta, pixel)

    for name, df in (("e2_fidelity.csv", fid), ("e2_delta.csv", delta),
                     ("e2_pixel.csv", pixel), ("e2_downstream.csv", down),
                     ("e2_zeroshot.csv", zero), ("e2_safety.csv", safety)):
        p = os.path.join(a.out_dir, name)
        df.to_csv(p, index=False)
        print("-> %s (%d rows)" % (rel(p), len(df)))
    write_report(fid, delta, pixel, down, zero, safety, missing,
                 os.path.join(a.out_dir, "E2_REPORT.md"), out_dir=a.out_dir)
    if missing:
        print("[e2a] %d missing products (see the report tail)" % len(set(missing)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
