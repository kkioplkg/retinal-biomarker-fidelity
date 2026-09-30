"""Post-hoc matched-FCR comparison, inside the range the sweeps actually reach.

The pre-registered Fig.3 operating points are FCR = 1 / 2.5 / 5 %. No arm gets
near them: sweeping lambda in {0.5,1,2,4} x tau in {0.3..0.9}, the mean matched
FCR bottoms out around 0.51-0.71 on DRIVE/HRF/FIVES (the only sub-0.5 points are
degenerate ones where nothing is accepted, so FCR is 0 by the |A| = 0
convention). ``_interp_at_fcr`` refuses to extrapolate, so
``results/fig3_matched_fcr.csv`` comes out empty -- which reads like missing runs
but is really "the question was asked outside the achievable range".

This module asks the same question at FCR targets that the sweeps do reach
(default 0.55 / 0.60 / 0.65) for the three RiGR modes and for EVAPORE-e2e, whose
acceptance threshold tau gives it its own operating curve
(``runs/repair/evapore_e2e_sweep/<ds>/seed0/tau*``).

Output: ``results/fig3_matched_fcr_posthoc.csv``, one row per
(dataset, method, fcr_target) with TRR_recall and macro-MAE benefit
(= macro_mae_before - macro_mae_after, so **positive = improvement**),
plus an image-level bootstrap CI.

CLI
---
    cd exp
    python -m src.eval.matched_fcr_posthoc
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from .tables import (EXP_ROOT, RESULTS_DIR, RUNS_DIR, _live_glob, _read_csv,
                     add_matched_fcr, df_to_markdown)

FCR_TARGETS_POSTHOC = (0.55, 0.60, 0.65)
PRESPECIFIED = (0.01, 0.025, 0.05)
MAIN_DATASETS = ("drive", "chasedb1", "hrf", "fives")
METRICS = ("TRR_recall", "macro_mae_benefit")


def _interp(fcr: np.ndarray, val: np.ndarray, target: float) -> float:
    """Linear interpolation at *target*, never extrapolating."""
    ok = np.isfinite(fcr) & np.isfinite(val)
    x, y = fcr[ok], val[ok]
    if x.size < 2:
        return float("nan")
    o = np.argsort(x)
    x, y = x[o], y[o]
    if target < x.min() or target > x.max():
        return float("nan")
    return float(np.interp(target, x, y))


def _rigr_points() -> Dict[tuple, "object"]:
    """(dataset, mode) -> per-image sweep frame with one row per (op, image).

    Prefers ``sweep2_per_image.csv`` where it exists. The risk and uniform arms
    have no threshold knob, so the original ``--sweep`` gives them only four
    operating points in a narrow FCR band -- too few to interpolate a
    matched-FCR comparison against ``prob``. ``sweep2`` adds the U-threshold
    axis (7 lambda x 4 quantiles = 28 cells) and widens the reachable range;
    its ``(lam=1, u0q=0)`` cell is the pre-registered operating point, so the
    two sweeps agree where they overlap. ``prob`` keeps the original sweep,
    whose tau axis already gives it a curve.
    """
    import pandas as pd

    out: Dict[tuple, list] = {}
    frames: Dict[tuple, str] = {}
    for fname, opcols in (("sweep2_per_image.csv", ("lam", "u0q")),
                          ("sweep_per_image.csv", ("lam", "tau"))):
        for p in _live_glob(os.path.join(RUNS_DIR, "rigr", "*", "*", "seed*",
                                         fname)):
            d = _read_csv(p)
            if d is None or not set(opcols) | {"image", "FCR"} <= set(d.columns):
                continue
            ds = str(d["dataset"].iloc[0]).lower()
            mode = "rigr_" + str(d["mode"].iloc[0])
            if ds not in MAIN_DATASETS:
                continue
            # first file wins per (dataset, mode): sweep2 before sweep
            if frames.get((ds, mode), fname) != fname:
                continue
            frames[(ds, mode)] = fname
            d = add_matched_fcr(d.copy())
            if "macro_mae_benefit" not in d.columns:
                if {"macro_mae_before", "macro_mae_after"} <= set(d.columns):
                    d["macro_mae_benefit"] = d.macro_mae_before - d.macro_mae_after
                else:
                    d["macro_mae_benefit"] = np.nan
            d["op"] = (d[opcols[0]].astype(str) + "|"
                       + d[opcols[1]].astype(str))
            if "n_accepted" not in d.columns:
                d["n_accepted"] = np.nan
            out.setdefault((ds, mode), []).append(
                d[["op", "image", "seed", "FCR", "n_accepted"] + list(METRICS)])
    for k, v in sorted(frames.items()):
        print("[fcr_posthoc] %s/%s <- %s" % (k[0], k[1], v))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def _evapore_points() -> Dict[tuple, "object"]:
    """(dataset, 'evapore_e2e') -> per-image frame, one row per (tau, image)."""
    import pandas as pd

    out: Dict[tuple, list] = {}
    # canonical layout only: <ds>/seed<k>/tau<value>/.  This deliberately skips
    # siblings such as `hrf_first10`, a 10-image partial run.
    for p in _live_glob(os.path.join(RUNS_DIR, "repair", "evapore_e2e_sweep",
                                     "*", "seed*", "tau*", "per_image.csv")):
        parts = p.replace("\\", "/").split("/")
        ds, tau = parts[-4], parts[-2]
        if ds not in MAIN_DATASETS:
            continue
        d = _read_csv(p)
        if d is None or "FCR" not in d.columns:
            continue
        d = add_matched_fcr(d.copy())
        d["macro_mae_benefit"] = d.macro_mae_before - d.macro_mae_after
        d["op"] = tau
        d["image"] = d["image"] if "image" in d.columns else d["image_id"]
        d["seed"] = d["seed"] if "seed" in d.columns else 0
        if "n_accepted" not in d.columns:
            d["n_accepted"] = np.nan
        out.setdefault((ds, "evapore_e2e"), []).append(
            d[["op", "image", "seed", "FCR", "n_accepted"] + list(METRICS)])
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def _as_matrices(df):
    """(op x image) matrices for FCR, n_accepted and each metric.

    The bootstrap resamples *images*, so laying the data out as op-by-image
    matrices turns one resample into a column gather plus a row mean -- pure
    numpy. The previous implementation rebuilt the frame with a per-image
    ``concat`` on every resample, i.e. n_images filtering passes x n_boot per
    cell (60 x 1000 on FIVES), which did not finish in ten minutes.
    """
    import pandas as pd

    cols = ["FCR", "n_accepted"] + list(METRICS)
    # Average over seeds FIRST. Several (dataset, mode) frames pool 3 seeds, so
    # (op, image) is not unique in the raw frame and a scatter-assign would keep
    # only the last seed instead of the mean of the three.
    agg = df.groupby(["op", "image"], as_index=False)[cols].mean()
    ops = sorted(agg["op"].unique())
    imgs = sorted(agg["image"].unique())
    oi = {o: i for i, o in enumerate(ops)}
    ii = {m: i for i, m in enumerate(imgs)}
    M = {c: np.full((len(ops), len(imgs)), np.nan) for c in cols}
    r = agg["op"].map(oi).to_numpy()
    k = agg["image"].map(ii).to_numpy()
    for c in cols:
        v = pd.to_numeric(agg[c], errors="coerce").to_numpy(dtype=float)
        M[c][r, k] = v
    return ops, np.array(imgs, dtype=object), M


def _curve_from(M, cols_idx: Optional[np.ndarray] = None):
    """Mean over images (optionally a resample) -> {col: per-op vector}, with
    zero-acceptance operating points masked out.

    With |A| = 0 the matched FCR is 0 by convention (DECISIONS.md 2026-09-03
    10:20), so a cell that repairs nothing looks like a perfect-precision point
    at FCR = 0 and anchors the curve there -- which would let interpolation
    report a "reachable" FCR of 0 and invent a matched comparison between two
    do-nothing arms.
    """
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c, m in M.items():
            sub = m if cols_idx is None else m[:, cols_idx]
            out[c] = np.nanmean(sub, axis=1)
    na = out.get("n_accepted")
    keep = np.ones_like(out["FCR"], dtype=bool)
    if na is not None and np.isfinite(na).any():
        keep &= ~(na < 0.5)
    out["_keep"] = keep
    return out


def build(results_dir: str = RESULTS_DIR, n_boot: int = 1000,
          targets: Sequence[float] = FCR_TARGETS_POSTHOC, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    pts = {}
    pts.update(_rigr_points())
    pts.update(_evapore_points())
    if not pts:
        print("[fcr_posthoc] no sweep data found")
        return None

    # Per dataset, the FCR band reachable by ALL THREE RiGR modes at once: only
    # inside it is a matched-FCR comparison between them possible at all.  With
    # sweep2 widening risk/uniform this band is much larger than it was, so add
    # three targets inside it (25/50/75 %) on top of the fixed ones.
    common: Dict[str, tuple] = {}
    for ds in MAIN_DATASETS:
        los, his = [], []
        for mode in ("rigr_prob", "rigr_uniform", "rigr_risk"):
            d = pts.get((ds, mode))
            if d is None:
                continue
            _, _, M = _as_matrices(d)
            c = _curve_from(M)
            f = c["FCR"][c["_keep"]]
            f = f[np.isfinite(f)]
            if f.size == 0:
                continue
            los.append(float(f.min())); his.append(float(f.max()))
        if len(los) == 3:
            lo, hi = max(los), min(his)
            if hi > lo:
                common[ds] = (lo, hi)
                print("[fcr_posthoc] %s common reachable FCR band "
                      "[%.3f, %.3f]" % (ds, lo, hi))

    rows: List[dict] = []
    for (ds, method), df in sorted(pts.items()):
        ops, imgs, M = _as_matrices(df)
        cur = _curve_from(M)
        keep = cur["_keep"]
        fcr_all = cur["FCR"][keep]
        fcr_all = fcr_all[np.isfinite(fcr_all)]
        if fcr_all.size == 0:
            continue
        fcr_lo, fcr_hi = float(fcr_all.min()), float(fcr_all.max())
        extra = []
        if ds in common:
            lo, hi = common[ds]
            extra = [round(lo + f * (hi - lo), 4) for f in (0.25, 0.5, 0.75)]
        for t in list(targets) + list(PRESPECIFIED) + extra:
            rec = dict(dataset=ds, method=method, fcr_target=t,
                       n_ops=int(keep.sum()), fcr_min=fcr_lo, fcr_max=fcr_hi,
                       in_range=bool(fcr_lo <= t <= fcr_hi),
                       n_images=len(imgs), n_boot=0)
            for m in METRICS:
                rec[m] = _interp(cur["FCR"][keep], cur[m][keep], t)
                rec[m + "_ci_lo"] = np.nan
                rec[m + "_ci_hi"] = np.nan
            if rec["in_range"] and len(imgs) >= 4:
                boots = {m: [] for m in METRICS}
                n_img = len(imgs)
                for _ in range(n_boot):
                    idx = rng.integers(0, n_img, size=n_img)
                    c = _curve_from(M, idx)
                    k2 = c["_keep"]
                    for m in METRICS:
                        boots[m].append(_interp(c["FCR"][k2], c[m][k2], t))
                for m in METRICS:
                    b = np.array(boots[m], dtype=float)
                    b = b[np.isfinite(b)]
                    if b.size >= 20:
                        rec[m + "_ci_lo"] = float(np.percentile(b, 2.5))
                        rec[m + "_ci_hi"] = float(np.percentile(b, 97.5))
                rec["n_boot"] = n_boot
            rec["common_band"] = ds in common and (
                common[ds][0] <= t <= common[ds][1])
            rec["src"] = "runs/rigr/<mode>/%s/seed*/sweep{2,}_per_image.csv" % ds \
                if method.startswith("rigr_") \
                else "runs/repair/evapore_e2e_sweep/%s/seed0/tau*/per_image.csv" % ds
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(
        ["dataset", "method", "fcr_target"]).reset_index(drop=True)
    csv = os.path.join(results_dir, "fig3_matched_fcr_posthoc.csv")
    out.to_csv(csv, index=False)
    print(f"[fcr_posthoc] wrote {csv}")

    md = ["# Post-hoc matched-FCR comparison (achievable range)\n",
          "`macro_mae_benefit` = macro_mae_before - macro_mae_after, so "
          "**positive = improvement**. CIs are image-level bootstrap "
          f"({n_boot} resamples). Rows with `in_range=False` are the "
          "pre-specified 1/2.5/5% targets, which **no arm reaches** -- they are "
          "listed so the gap is explicit rather than an empty file.\n"]
    show = out[out.in_range][["dataset", "method", "fcr_target", "TRR_recall",
                              "TRR_recall_ci_lo", "TRR_recall_ci_hi",
                              "macro_mae_benefit", "macro_mae_benefit_ci_lo",
                              "macro_mae_benefit_ci_hi", "n_images"]]
    md.append("\n## In-range operating points\n")
    md.append(df_to_markdown(show))
    md.append("\n## Achievable FCR range per arm\n")
    rng_tbl = out.groupby(["dataset", "method"])[["fcr_min", "fcr_max", "n_ops"]] \
                 .first().reset_index()
    md.append(df_to_markdown(rng_tbl))
    md_path = os.path.join(results_dir, "fig3_matched_fcr_posthoc.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"[fcr_posthoc] wrote {md_path}")
    return out


if __name__ == "__main__":
    build()
