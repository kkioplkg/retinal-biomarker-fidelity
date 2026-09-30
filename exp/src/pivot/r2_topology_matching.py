"""R2 -- covariate-matched controls for the topology counterfactual.

Reviewer point 20 (``review/user_cmig_review_20260918.md``)
----------------------------------------------------------
"The topology control needs to be matched on local calibre, branch order,
position and edit length, or else stratified."

What the shipped analysis does.  Every verified structural edit in the C1
corpus is paired with a *control* edit placed elsewhere in the same image that
changes the **same number of foreground pixels** without changing connectivity.
The topology-specific harm is the paired difference
``h_net = |dB_topo|/sigma - |dB_ctrl|/sigma`` (``e1_topology_matched.csv``).
Matching on the pixel budget alone leaves the control free to sit on a vessel
of a different calibre, at a different eccentricity, in a different local
density.

What this script does
---------------------
1.  ``r2_topology_covariates.csv`` -- the inventory the reviewer asked for:
    which covariates the corpus actually records per event and per control,
    what each one is, and its coverage on the audit subset.

2.  ``r2_topology_balance.csv`` -- covariate balance between the topological
    edit and its own control, on the covariates that exist for **both** sides
    (local radius, edit length in pixels, radial distance from the optic disc
    centre in disc radii), as a standardised mean difference, before and after
    re-matching.

3.  ``r2_topology_matched_cem.csv`` -- coarsened exact matching.  Both sides are
    binned on radius x edit-length x radial-distance (and, as a further
    restriction, the event's local-density bin); only pairs whose control falls
    in the **same cell on every coordinate** are kept, and the surviving cells
    are reweighted to the full topological-event distribution so the estimand
    stays "the mean topology-specific harm over the events we generated" rather
    than "over the events that happened to match".

4.  ``r2_topology_matched_nn.csv`` -- nearest-neighbour re-matching across
    events.  Each topological edit is re-paired with the control edit of a
    *different* event in the same image whose donor covariates are closest in a
    standardised (log radius, log edit length, radial distance) metric, with
    a caliper.  This is the stronger design: it does not rely on the original
    within-event control being balanced at all.

5.  ``r2_topology_strata.csv`` -- stratified sensitivity: ``h_net`` inside
    tertiles of vessel calibre, of radial distance, and of local vessel
    density.

6.  ``r2_topology_claims.csv`` -- does "the topology-specific effect is one to
    two orders of magnitude below the offset and the residual scatter" survive
    every one of the above?

A limitation stated rather than hidden: the corpus records the donor's
**position and radius** and the matched pixel budget, but not the donor's local
vessel density or local contrast.  Local density therefore enters this analysis
as a *stratification* of the events (design 5) and as an extra exact-matching
coordinate on the event side (design 3), not as a two-sided matching covariate.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_topology_matching
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import (EXP_ROOT, GATEA_DS, PIVOT_DIR, PRIMARY_COLS,
                              RESULTS_DIR)

SEED = 0
N_BOOT = 2000
DATASETS = ("drive", "chasedb1", "hrf", "fives")
PANEL = {c: ("primary" if c.endswith("_skan") else "sensitivity")
         for c in PRIMARY_COLS}
PRIMARY_PANEL = [c for c in PRIMARY_COLS if c.endswith("_skan")]

#: what the corpus records, and on which side of the comparison it exists.
COVARIATE_DOC: List[Tuple[str, str, str]] = [
    # (column, side, description)
    ("gt_r_loc", "event", "local vessel radius at the edit locus, px at work scale"),
    ("gt_d_loc", "event", "local vessel diameter at the edit locus, px"),
    ("gt_radius_bin", "event", "coarsened calibre bin of the edited vessel (0..3)"),
    ("gt_branch_order", "event", "branch order of the edited branch in the reference skeleton"),
    ("gt_branch_length", "event", "length of the edited branch, px"),
    ("gt_branch_type", "event", "branch type label (terminal / internal / junction-to-junction)"),
    ("gt_dist_junction", "event", "distance from the locus to the nearest skeleton junction, px"),
    ("gt_dist_endpoint", "event", "distance from the locus to the nearest skeleton endpoint, px"),
    ("gt_zone", "event", "disc-relative zone (A/B/C) of the locus"),
    ("gt_density", "event", "local vessel density in a window around the locus"),
    ("gt_density_bin", "event", "coarsened local-density bin"),
    ("gt_contrast", "event", "local image contrast at the locus"),
    ("gt_contrast_bin", "event", "coarsened local-contrast bin"),
    ("phi_r_a", "event", "radius of the first endpoint of the affected candidate, px"),
    ("phi_r_b", "event", "radius of the second endpoint (bridge-type events only)"),
    ("phi_r_mean", "event", "mean radius of the two endpoints, px"),
    ("phi_d", "event", "endpoint separation, px (bridge-type events)"),
    ("phi_gap_len", "event", "gap length along the reconnection path, px"),
    ("phi_zone", "event", "disc-relative zone from the phi feature builder"),
    ("phi_fov_dist", "event", "distance from the FOV centre, px"),
    ("phi_fov_dist_over_r", "event", "the same, divided by the local radius"),
    ("phi_density", "event", "predicted local vessel density near the locus"),
    ("phi_density_far", "event", "predicted vessel density in a wider ring"),
    ("phi_density_ratio", "event", "local / far density ratio"),
    ("phi_contrast", "event", "local vessel-to-background contrast"),
    ("phi_n_endpoints_local", "event", "skeleton endpoints in the local window"),
    ("phi_n_junctions_local", "event", "skeleton junctions in the local window"),
    ("phi_skel_density_local", "event", "skeleton pixel density in the local window"),
    ("phi_curvature_a", "event", "local curvature at the first endpoint"),
    ("phi_curvature_b", "event", "local curvature at the second endpoint"),
    ("n_changed", "both", "EDIT LENGTH: foreground pixels changed by the edit"),
    ("locus_row", "event", "edit locus row, work-scale px"),
    ("locus_col", "event", "edit locus column, work-scale px"),
    ("disc_cx", "both", "optic-disc centre column, work-scale px"),
    ("disc_cy", "both", "optic-disc centre row, work-scale px"),
    ("disc_r", "both", "optic-disc radius, work-scale px (the radial-distance unit)"),
    ("control_donor_row", "control", "control edit donor row, work-scale px"),
    ("control_donor_col", "control", "control edit donor column, work-scale px"),
    ("control_donor_r", "control", "local vessel radius at the control donor site, px"),
    ("control_n_changed", "control", "foreground pixels changed by the control edit"),
    ("control_match_tier", "control", "which matching tier the control was found at"),
    ("control_addition", "control", "whether the control edit added rather than removed pixels"),
    ("severity", "event", "severity label of the intervention"),
    ("severity_value", "event", "numeric severity of the intervention"),
    ("type", "event", "intervention type: sever / bridge / truncate / caliber"),
]

LOAD_COLS = (["dataset", "image_id", "subject_id", "split", "observer", "type",
              "severity", "severity_value", "control_ok", "control_match_tier",
              "control_addition", "n_changed", "control_n_changed",
              "control_donor_row", "control_donor_col", "control_donor_r",
              "locus_row", "locus_col", "disc_cx", "disc_cy", "disc_r",
              "img_h", "img_w", "work_scale", "event_id",
              "gt_r_loc", "gt_d_loc", "gt_radius_bin", "gt_branch_order",
              "gt_branch_length", "gt_density", "gt_density_bin",
              "gt_contrast", "gt_contrast_bin", "gt_zone", "gt_dist_junction",
              "gt_dist_endpoint", "gt_branch_type", "phi_r_a", "phi_r_b",
              "phi_r_mean", "phi_d", "phi_d_over_r", "phi_gap_len", "phi_zone",
              "phi_density", "phi_density_far", "phi_density_ratio",
              "phi_contrast", "phi_fov_dist", "phi_fov_dist_over_r",
              "phi_n_endpoints_local", "phi_n_junctions_local",
              "phi_skel_density_local", "phi_curvature_a", "phi_curvature_b"]
             + [f"absdB_{c}" for c in PRIMARY_COLS]
             + [f"cabsdB_{c}" for c in PRIMARY_COLS]
             + [f"nativef_{c}" for c in PRIMARY_COLS]
             + [f"sigmatr_{c}" for c in PRIMARY_COLS])


# --------------------------------------------------------------- bootstrap
def boot_cluster_mean(values: Dict[str, np.ndarray], cluster: np.ndarray,
                      n_boot: int = N_BOOT, seed: int = SEED
                      ) -> Dict[str, Tuple[float, float]]:
    """Image-clustered bootstrap of the mean of each array in ``values``.

    Events inside one image share the mask and are not independent, so whole
    images are resampled, never rows.
    """
    keys = list(values)
    out = {k: (float("nan"), float("nan")) for k in keys}
    uniq, inv = np.unique(cluster, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    n = len(groups)
    if n < 3:
        return out
    rng = np.random.RandomState(seed)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.randint(0, n, n)
        sel = np.concatenate([groups[j] for j in pick])
        for k in keys:
            with np.errstate(invalid="ignore"):
                m = np.nanmean(values[k][sel])
            if np.isfinite(m):
                draws[k].append(m)
    for k in keys:
        v = np.asarray(draws[k], float)
        if v.size:
            out[k] = (float(np.percentile(v, 2.5)),
                      float(np.percentile(v, 97.5)))
    return out


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (b - a) / pooled SD."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((b.mean() - a.mean()) / s) if s > 0 else float("nan")


# --------------------------------------------------------------------- data
def load_events(path: str, dataset_filter: Sequence[str]) -> pd.DataFrame:
    ev = pd.read_parquet(path, columns=LOAD_COLS)
    ev = ev[(ev["observer"] == "obs1") & (ev["split"] == "test") &
            (ev["control_ok"] == 1.0)].copy()
    inv = {v: k for k, v in GATEA_DS.items()}
    ev["ds"] = ev["dataset"].map(lambda d: inv.get(str(d), str(d).lower()))
    ev = ev[ev["ds"].isin(dataset_filter)].reset_index(drop=True)

    # radial distance from the optic-disc centre, in disc radii, for BOTH sides
    dr = pd.to_numeric(ev["disc_r"], errors="coerce").replace(0.0, np.nan)
    ev["rad_topo"] = np.sqrt(
        (ev["locus_row"] - ev["disc_cy"]) ** 2 +
        (ev["locus_col"] - ev["disc_cx"]) ** 2) / dr
    ev["rad_ctrl"] = np.sqrt(
        (ev["control_donor_row"] - ev["disc_cy"]) ** 2 +
        (ev["control_donor_col"] - ev["disc_cx"]) ** 2) / dr
    ev["r_topo"] = pd.to_numeric(ev["gt_r_loc"], errors="coerce")
    ev["r_ctrl"] = pd.to_numeric(ev["control_donor_r"], errors="coerce")
    ev["len_topo"] = pd.to_numeric(ev["n_changed"], errors="coerce")
    ev["len_ctrl"] = pd.to_numeric(ev["control_n_changed"], errors="coerce")
    return ev


def harms(ev: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    """(h_topo, h_ctrl) in training-split sigma units, native-unit converted."""
    conv = ev[f"nativef_{col}"].to_numpy(float)
    sig = ev[f"sigmatr_{col}"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ht = (ev[f"absdB_{col}"].to_numpy(float) * conv) / sig
        hc = (ev[f"cabsdB_{col}"].to_numpy(float) * conv) / sig
    return ht, hc


# ----------------------------------------------------------- 1. inventory
def inventory(ev: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for col, side, desc in COVARIATE_DOC:
        if col not in ev.columns:
            rows.append(dict(covariate=col, side=side, description=desc,
                             present=False, coverage=float("nan"),
                             n=0, median=float("nan"), q25=float("nan"),
                             q75=float("nan"),
                             source="results/c1_events_harm.parquet"))
            continue
        raw = ev[col]
        v = pd.to_numeric(raw, errors="coerce")
        if v.notna().sum() == 0 and raw.notna().sum() > 0:
            # categorical / string covariate: report coverage and levels
            lv = sorted(set(raw.dropna().astype(str)))[:8]
            rows.append(dict(
                covariate=col, side=side,
                description=desc + "  [categorical; levels: %s]" % ", ".join(lv),
                present=True, coverage=float(raw.notna().mean()),
                n=int(raw.notna().sum()), median=float("nan"),
                q25=float("nan"), q75=float("nan"),
                source=("results/c1_events_harm.parquet "
                        "(observer=obs1, split=test, control_ok=1)")))
            continue
        rows.append(dict(
            covariate=col, side=side, description=desc, present=True,
            coverage=float(v.notna().mean()), n=int(v.notna().sum()),
            median=float(v.median()) if v.notna().any() else float("nan"),
            q25=float(v.quantile(0.25)) if v.notna().any() else float("nan"),
            q75=float(v.quantile(0.75)) if v.notna().any() else float("nan"),
            source=("results/c1_events_harm.parquet "
                    "(observer=obs1, split=test, control_ok=1)")))
    rows.append(dict(
        covariate="rad_topo / rad_ctrl", side="both",
        description=("DERIVED radial distance from the optic-disc centre in "
                     "disc radii, computed here for the edit locus and for the "
                     "control donor site"),
        present=True, coverage=float(ev["rad_topo"].notna().mean()),
        n=int(ev["rad_topo"].notna().sum()),
        median=float(ev["rad_topo"].median()),
        q25=float(ev["rad_topo"].quantile(0.25)),
        q75=float(ev["rad_topo"].quantile(0.75)),
        source="derived in src/pivot/r2_topology_matching.py"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 2. balance
BALANCE_PAIRS = (("radius_px", "r_topo", "r_ctrl"),
                 ("edit_length_px", "len_topo", "len_ctrl"),
                 ("radial_dist_disc_radii", "rad_topo", "rad_ctrl"))


def balance(ev: pd.DataFrame, tag: str) -> List[dict]:
    rows: List[dict] = []
    for ds, g in ev.groupby("ds"):
        for name, a, b in BALANCE_PAIRS:
            x = pd.to_numeric(g[a], errors="coerce").to_numpy(float)
            y = pd.to_numeric(g[b], errors="coerce").to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            rows.append(dict(
                dataset=ds, design=tag, covariate=name, n=int(ok.sum()),
                mean_topo=float(np.nanmean(x[ok])) if ok.any() else np.nan,
                mean_ctrl=float(np.nanmean(y[ok])) if ok.any() else np.nan,
                smd=smd(x[ok], y[ok]),
                frac_exact_equal=float(np.mean(x[ok] == y[ok])) if ok.any() else np.nan,
                source="results/c1_events_harm.parquet"))
    return rows


# --------------------------------------------------- nested matching levels
#: Covariate priority fixed by the methods consultation
#: (``review/cmig_plan_reply.md`` 2D):
#:   edit burden > local calibre > branch order / topological context
#:   > radial position;  local vessel density is STRATIFIED, not hard-matched.
#: Each level adds one covariate to the previous one, so the harm estimate can
#: be read as a function of how much matching has been imposed.
MATCH_LEVELS: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [
    # (name, two-sided pair-matched covariates, event-side blocking covariates)
    ("L0_edit_burden", ("edit_length_px",), ()),
    ("L1_plus_calibre", ("edit_length_px", "radius_px"), ()),
    ("L2_plus_branch_order", ("edit_length_px", "radius_px"),
     ("gt_branch_order",)),
    ("L3_plus_radial", ("edit_length_px", "radius_px",
                        "radial_dist_disc_radii"), ("gt_branch_order",)),
]
#: relative weights used by the nearest-neighbour matcher, same priority order
NN_WEIGHTS = {"edit_length_px": 3.0, "radius_px": 2.0,
              "radial_dist_disc_radii": 1.0}


# --------------------------------------------------------------- binning
def qbin(x: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Bin ``x`` by fixed quantile edges computed on the topological side."""
    return np.digitize(x, edges, right=False)


def make_bins(ev: pd.DataFrame, n_bins: int = 4
              ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per coordinate: (topo bin, control bin, edges).

    Quantile edges are taken from the topological side, so the coarsening is
    defined by the estimand's own distribution and not by the control pool.

    **Local radius is the exception and is binned at a fixed 0.5 px
    resolution instead.**  ``gt_r_loc`` is a near-discrete quantity on the small
    cohorts (DRIVE's 25th, 50th and 75th percentiles are all 1 px), so quantile
    edges collapse to a single break and "matching" on them does nothing: the
    first version of this script left DRIVE's calibre SMD at +0.76 after
    supposedly matching on it.  A fixed 0.5 px grid makes the calibre coordinate
    an effectively exact match, which is what the covariate-priority list asks
    for.
    """
    out = {}
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    for name, a, b in BALANCE_PAIRS:
        x = pd.to_numeric(ev[a], errors="coerce").to_numpy(float)
        y = pd.to_numeric(ev[b], errors="coerce").to_numpy(float)
        if name == "radius_px":
            bt = np.where(np.isfinite(x), np.round(x * 2.0), -999)
            bc = np.where(np.isfinite(y), np.round(y * 2.0), -998)
            out[name] = (bt.astype(int), bc.astype(int),
                         np.array([0.5], float))
            continue
        fin = x[np.isfinite(x)]
        edges = np.unique(np.percentile(fin, qs)) if fin.size else np.array([])
        out[name] = (qbin(x, edges), qbin(y, edges), edges)
    return out


# ------------------------------------------------------------------ 3. CEM
def cem_level(ev: pd.DataFrame, n_bins: int, level: str,
              pair_cov: Sequence[str], block_cov: Sequence[str],
              density_block: bool = False
              ) -> Tuple[List[dict], List[dict]]:
    """Coarsened exact matching at one level of the covariate priority list.

    ``pair_cov``   covariates available on BOTH sides: the event and its control
                   must land in the same quantile bin of each.
    ``block_cov``  covariates recorded only on the event side (branch order):
                   these cannot be pair-matched, so they enter as exact blocking
                   -- the estimand is conditioned on them and the cells are
                   reweighted back to the full event distribution afterwards.
    Returns (harm rows, balance rows).
    """
    rows: List[dict] = []
    bal: List[dict] = []
    for ds, g in ev.groupby("ds"):
        g = g.reset_index(drop=True)
        bins = make_bins(g, n_bins)
        keep = np.ones(len(g), bool)
        for name, a, b in BALANCE_PAIRS:
            if name not in pair_cov:
                continue
            bt, bc, _ = bins[name]
            fa = np.isfinite(pd.to_numeric(g[a], errors="coerce").to_numpy(float))
            fb = np.isfinite(pd.to_numeric(g[b], errors="coerce").to_numpy(float))
            keep &= (bt == bc) & fa & fb

        parts = [np.array([str(int(bins[n][0][i])) for i in range(len(g))])
                 for n, _a, _b in BALANCE_PAIRS if n in pair_cov]
        for bc_name in block_cov:
            v = pd.to_numeric(g[bc_name], errors="coerce").to_numpy(float)
            fin = v[np.isfinite(v)]
            edges = (np.unique(np.percentile(
                fin, np.linspace(0, 100, n_bins + 1)[1:-1])) if fin.size
                else np.array([]))
            parts.append(np.array([f"{bc_name[:2]}{int(k)}"
                                   for k in qbin(v, edges)]))
        if density_block:
            db = pd.to_numeric(g["gt_density_bin"], errors="coerce"
                               ).fillna(-1).astype(int).to_numpy()
            parts.append(np.array([f"d{k}" for k in db]))
        cell = (np.array(["|".join(p[i] for p in parts) for i in range(len(g))])
                if parts else np.full(len(g), "all"))

        # CEM L1-style reweighting: bring the matched cells back to the share
        # each cell has among ALL the events we generated.
        tgt = pd.Series(cell).value_counts(normalize=True)
        src = pd.Series(cell[keep]).value_counts(normalize=True)
        w = np.array([(tgt.get(c, 0.0) / src[c]) if (keep[i] and c in src.index)
                      else 0.0 for i, c in enumerate(cell)], float)
        sel = keep & (w > 0)

        # ---- balance after this level, on the weighted matched sample
        for name, a, b in BALANCE_PAIRS:
            x = pd.to_numeric(g[a], errors="coerce").to_numpy(float)[sel]
            y = pd.to_numeric(g[b], errors="coerce").to_numpy(float)[sel]
            ok = np.isfinite(x) & np.isfinite(y)
            bal.append(dict(
                dataset=ds, design="cem_" + level, covariate=name,
                n=int(ok.sum()),
                mean_topo=float(np.mean(x[ok])) if ok.any() else np.nan,
                mean_ctrl=float(np.mean(y[ok])) if ok.any() else np.nan,
                smd=smd(x[ok], y[ok]),
                frac_exact_equal=(float(np.mean(x[ok] == y[ok])) if ok.any()
                                  else np.nan),
                source="results/c1_events_harm.parquet"))

        for col in PRIMARY_COLS:
            ht, hc = harms(g, col)
            ok = sel & np.isfinite(ht) & np.isfinite(hc)
            if ok.sum() < 10:
                continue
            ww = w[ok] / w[ok].mean()
            vals = {"h_topo": ht[ok] * ww, "h_ctrl": hc[ok] * ww,
                    "h_net": (ht[ok] - hc[ok]) * ww}
            ci = boot_cluster_mean(vals, g["image_id"].to_numpy(str)[ok])
            rows.append(dict(
                dataset=ds, biomarker=col, panel=PANEL[col],
                design="cem_" + level, level=level,
                pair_matched=",".join(pair_cov),
                blocked=",".join(list(block_cov) +
                                 (["gt_density_bin"] if density_block else [])),
                n_bins=n_bins, n_events=int(ok.sum()),
                n_events_before=int(np.isfinite(ht).sum()),
                match_rate=float(ok.sum()) / max(int(np.isfinite(ht).sum()), 1),
                n_images=int(pd.unique(g["image_id"].to_numpy(str)[ok]).size),
                n_cells=int(pd.unique(cell[ok]).size),
                h_topo=float(np.mean(vals["h_topo"])),
                h_topo_lo=ci["h_topo"][0], h_topo_hi=ci["h_topo"][1],
                h_ctrl=float(np.mean(vals["h_ctrl"])),
                h_ctrl_lo=ci["h_ctrl"][0], h_ctrl_hi=ci["h_ctrl"][1],
                h_net=float(np.mean(vals["h_net"])),
                h_net_lo=ci["h_net"][0], h_net_hi=ci["h_net"][1],
                source=("results/c1_events_harm.parquet; coarsened exact "
                        "matching, level '%s': pair-matched on quantile bins of "
                        "%s, blocked on %s; cells reweighted to the full event "
                        "distribution; 2000-draw image-clustered bootstrap"
                        % (level, ", ".join(pair_cov),
                           ", ".join(list(block_cov) +
                                     (["gt_density_bin"] if density_block
                                      else [])) or "nothing"))))
    return rows, bal


# -------------------------------------------------------------------- 4. NN
def nn_match(ev: pd.DataFrame, caliper: float) -> Tuple[List[dict], pd.DataFrame]:
    """Re-pair each topological edit with the control edit of a DIFFERENT event
    in the same image, nearest in standardised (log r, log len, radial dist)."""
    rows: List[dict] = []
    pair_frames: List[pd.DataFrame] = []
    for ds, g in ev.groupby("ds"):
        g = g.reset_index(drop=True)
        feats_t = np.column_stack([
            np.log1p(pd.to_numeric(g["r_topo"], errors="coerce").to_numpy(float)),
            np.log1p(pd.to_numeric(g["len_topo"], errors="coerce").to_numpy(float)),
            pd.to_numeric(g["rad_topo"], errors="coerce").to_numpy(float)])
        feats_c = np.column_stack([
            np.log1p(pd.to_numeric(g["r_ctrl"], errors="coerce").to_numpy(float)),
            np.log1p(pd.to_numeric(g["len_ctrl"], errors="coerce").to_numpy(float)),
            pd.to_numeric(g["rad_ctrl"], errors="coerce").to_numpy(float)])
        sd = np.nanstd(np.vstack([feats_t, feats_c]), axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        # priority weights (methods consultation 2D): edit burden > calibre >
        # radial position.  Columns are ordered (radius, edit length, radial).
        wt = np.array([NN_WEIGHTS["radius_px"],
                       NN_WEIGHTS["edit_length_px"],
                       NN_WEIGHTS["radial_dist_disc_radii"]], float)
        ft, fc = (feats_t / sd) * wt, (feats_c / sd) * wt

        partner = np.full(len(g), -1, int)
        dist = np.full(len(g), np.nan)
        for img, idx in g.groupby("image_id").indices.items():
            idx = np.asarray(idx)
            if idx.size < 2:
                continue
            A = ft[idx]
            B = fc[idx]
            okA = np.isfinite(A).all(axis=1)
            okB = np.isfinite(B).all(axis=1)
            if okA.sum() == 0 or okB.sum() < 2:
                continue
            D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
            D[:, ~okB] = np.inf
            D[~okA, :] = np.inf
            np.fill_diagonal(D, np.inf)      # never its own control
            j = np.argmin(D, axis=1)
            dmin = D[np.arange(len(idx)), j]
            good = np.isfinite(dmin) & (dmin <= caliper)
            partner[idx[good]] = idx[j[good]]
            dist[idx[good]] = dmin[good]
        sel = partner >= 0
        pair_frames.append(pd.DataFrame(dict(
            ds=ds, image_id=g["image_id"].to_numpy(str),
            matched=sel, nn_dist=dist,
            r_topo=g["r_topo"].to_numpy(float),
            r_ctrl_nn=np.where(sel, g["r_ctrl"].to_numpy(float)[
                np.clip(partner, 0, None)], np.nan),
            rad_topo=g["rad_topo"].to_numpy(float),
            rad_ctrl_nn=np.where(sel, g["rad_ctrl"].to_numpy(float)[
                np.clip(partner, 0, None)], np.nan),
            len_topo=g["len_topo"].to_numpy(float),
            len_ctrl_nn=np.where(sel, g["len_ctrl"].to_numpy(float)[
                np.clip(partner, 0, None)], np.nan))))

        for col in PRIMARY_COLS:
            ht, hc = harms(g, col)
            hc_nn = np.where(sel, hc[np.clip(partner, 0, None)], np.nan)
            ok = sel & np.isfinite(ht) & np.isfinite(hc_nn)
            if ok.sum() < 10:
                continue
            vals = {"h_topo": ht[ok], "h_ctrl": hc_nn[ok],
                    "h_net": ht[ok] - hc_nn[ok]}
            ci = boot_cluster_mean(vals, g["image_id"].to_numpy(str)[ok])
            rows.append(dict(
                dataset=ds, biomarker=col, panel=PANEL[col],
                design="nn_within_image", caliper=caliper,
                n_events=int(ok.sum()),
                n_events_before=int(np.isfinite(ht).sum()),
                match_rate=float(ok.sum()) / max(int(np.isfinite(ht).sum()), 1),
                n_images=int(pd.unique(g["image_id"].to_numpy(str)[ok]).size),
                mean_nn_dist=float(np.nanmean(dist[ok])),
                h_topo=float(np.mean(vals["h_topo"])),
                h_topo_lo=ci["h_topo"][0], h_topo_hi=ci["h_topo"][1],
                h_ctrl=float(np.mean(vals["h_ctrl"])),
                h_ctrl_lo=ci["h_ctrl"][0], h_ctrl_hi=ci["h_ctrl"][1],
                h_net=float(np.mean(vals["h_net"])),
                h_net_lo=ci["h_net"][0], h_net_hi=ci["h_net"][1],
                source=("results/c1_events_harm.parquet; each topological edit "
                        "re-paired with the control edit of a different event "
                        "in the same image, nearest in standardised "
                        "(log radius, log edit length, radial distance) with a "
                        "caliper of %.2f SD; 2000-draw image-clustered "
                        "bootstrap" % caliper)))
    return rows, pd.concat(pair_frames, ignore_index=True) if pair_frames \
        else pd.DataFrame()


# --------------------------------------------------------------- 5. strata
STRATA = (("calibre", "r_topo"), ("radial_distance", "rad_topo"),
          ("local_density", "gt_density"), ("edit_length", "len_topo"))


def strata(ev: pd.DataFrame, n_strata: int = 3) -> List[dict]:
    rows: List[dict] = []
    labels = ["low", "mid", "high"] if n_strata == 3 else \
        [f"q{i+1}" for i in range(n_strata)]
    for ds, g in ev.groupby("ds"):
        g = g.reset_index(drop=True)
        for sname, scol in STRATA:
            v = pd.to_numeric(g[scol], errors="coerce").to_numpy(float)
            fin = v[np.isfinite(v)]
            if fin.size < 30:
                continue
            edges = np.unique(np.percentile(
                fin, np.linspace(0, 100, n_strata + 1)[1:-1]))
            b = np.digitize(v, edges, right=False)
            for k in range(n_strata):
                m = (b == k) & np.isfinite(v)
                if m.sum() < 10:
                    continue
                for col in PRIMARY_COLS:
                    ht, hc = harms(g, col)
                    ok = m & np.isfinite(ht) & np.isfinite(hc)
                    if ok.sum() < 10:
                        continue
                    vals = {"h_topo": ht[ok], "h_ctrl": hc[ok],
                            "h_net": ht[ok] - hc[ok]}
                    ci = boot_cluster_mean(vals,
                                           g["image_id"].to_numpy(str)[ok])
                    rows.append(dict(
                        dataset=ds, biomarker=col, panel=PANEL[col],
                        stratum_by=sname,
                        stratum=labels[k] if k < len(labels) else f"q{k+1}",
                        stratum_lo=float(np.nanmin(v[ok])),
                        stratum_hi=float(np.nanmax(v[ok])),
                        n_events=int(ok.sum()),
                        n_images=int(pd.unique(
                            g["image_id"].to_numpy(str)[ok]).size),
                        h_topo=float(np.mean(vals["h_topo"])),
                        h_ctrl=float(np.mean(vals["h_ctrl"])),
                        h_net=float(np.mean(vals["h_net"])),
                        h_net_lo=ci["h_net"][0], h_net_hi=ci["h_net"][1],
                        source=("results/c1_events_harm.parquet; tertiles of "
                                "%s; 2000-draw image-clustered bootstrap"
                                % sname)))
    return rows


# ---------------------------------------------------------------- 6. claims
def claims(all_designs: pd.DataFrame, strat: pd.DataFrame,
           audit_path: str) -> pd.DataFrame:
    rows: List[dict] = []
    ref = {}
    if os.path.exists(audit_path):
        a = pd.read_csv(audit_path)
        a = a[(a["panel"] == "primary") & (a["representation"] == "sigma")]
        if len(a):
            ref["max_abs_offset"] = float(a["offset"].abs().max())
            ref["max_resid_sd"] = float(a["resid_sd"].max())
            ref["mean_abs_offset"] = float(a["offset"].abs().mean())
            ref["mean_resid_sd"] = float(a["resid_sd"].mean())
    for design, g in all_designs.groupby("design"):
        p = g[g["panel"] == "primary"]
        if not len(p):
            continue
        mx = float(p["h_net"].abs().max())
        for k, v in ref.items():
            rows.append(dict(
                claim="topology_below_%s" % k, design=design,
                topology_max_abs_h_net=mx, comparator=v,
                ratio=float(v / mx) if mx > 0 else float("inf"),
                orders_of_magnitude=(float(np.log10(v / mx)) if mx > 0 and v > 0
                                     else float("nan")),
                survives=bool(mx > 0 and v / mx >= 10.0),
                source=("max |h_net| over the primary skan panel and the four "
                        "datasets under design '%s' vs %s of "
                        "results/pivot/r2/r2_audit_raw_units.csv"
                        % (design, k))))
    if len(strat):
        p = strat[strat["panel"] == "primary"]
        mx = float(p["h_net"].abs().max())
        for k, v in ref.items():
            rows.append(dict(
                claim="topology_below_%s" % k, design="worst_stratum",
                topology_max_abs_h_net=mx, comparator=v,
                ratio=float(v / mx) if mx > 0 else float("inf"),
                orders_of_magnitude=(float(np.log10(v / mx)) if mx > 0 and v > 0
                                     else float("nan")),
                survives=bool(mx > 0 and v / mx >= 10.0),
                source=("max |h_net| over every stratum, primary skan panel "
                        "and four datasets, vs %s of "
                        "results/pivot/r2/r2_audit_raw_units.csv" % k)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(RESULTS_DIR,
                                                     "c1_events_harm.parquet"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    ap.add_argument("--scope", default="sever",
                    help="sever | all_types")
    ap.add_argument("--n_bins", type=int, default=4)
    ap.add_argument("--caliper", type=float, default=0.25)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    ev_all = load_events(args.events, DATASETS)
    scopes = {"sever": ev_all[ev_all["type"] == "sever"],
              "all_types": ev_all}
    ev = scopes[args.scope].reset_index(drop=True)
    print("[events] %d rows on scope '%s' (%d images, %d datasets)"
          % (len(ev), args.scope, ev["image_id"].nunique(), ev["ds"].nunique()))

    inv = inventory(ev)
    p = os.path.join(args.out_dir, "r2_topology_covariates.csv")
    inv.to_csv(p, index=False)
    print("wrote", p, inv.shape)

    bal = pd.DataFrame(balance(ev, "unmatched_pixelcount_only"))

    designs: List[dict] = []
    for level, pair_cov, block_cov in MATCH_LEVELS:
        r, b = cem_level(ev, args.n_bins, level, pair_cov, block_cov)
        designs += r
        bal = pd.concat([bal, pd.DataFrame(b)], ignore_index=True)
        print("[cem] %s: %d harm rows" % (level, len(r)), flush=True)
    # the full level plus local density as a blocking (not hard-matching)
    # variable -- the consultation's "stratify, do not hard-match" instruction
    r, b = cem_level(ev, args.n_bins, "L4_density_stratified",
                     MATCH_LEVELS[-1][1], MATCH_LEVELS[-1][2],
                     density_block=True)
    designs += r
    bal = pd.concat([bal, pd.DataFrame(b)], ignore_index=True)
    nn_rows, nn_pairs = nn_match(ev, args.caliper)
    designs += nn_rows
    dz = pd.DataFrame(designs)
    p = os.path.join(args.out_dir, "r2_topology_matched.csv")
    dz.to_csv(p, index=False)
    print("wrote", p, dz.shape)

    # balance after NN re-matching
    if len(nn_pairs):
        nn_ok = nn_pairs[nn_pairs["matched"]].rename(columns={
            "r_ctrl_nn": "r_ctrl", "rad_ctrl_nn": "rad_ctrl",
            "len_ctrl_nn": "len_ctrl"})
        nn_ok["ds"] = nn_ok["ds"]
        bal = pd.concat([bal, pd.DataFrame(balance(nn_ok, "nn_within_image"))],
                        ignore_index=True)
    p = os.path.join(args.out_dir, "r2_topology_balance.csv")
    bal.to_csv(p, index=False)
    print("wrote", p, bal.shape)

    st = pd.DataFrame(strata(ev))
    p = os.path.join(args.out_dir, "r2_topology_strata.csv")
    st.to_csv(p, index=False)
    print("wrote", p, st.shape)

    cl = claims(dz, st, os.path.join(args.out_dir, "r2_audit_raw_units.csv"))
    p = os.path.join(args.out_dir, "r2_topology_claims.csv")
    cl.to_csv(p, index=False)
    print("wrote", p, cl.shape)

    with pd.option_context("display.width", 230, "display.max_rows", 600):
        print("\n[covariate balance]")
        print(bal.round(4).to_string(index=False))
        print("\n[re-matched topology harm, primary skan panel]")
        print(dz[dz["panel"] == "primary"][
            ["dataset", "biomarker", "design", "pair_matched", "blocked",
             "n_events", "match_rate", "h_topo", "h_ctrl", "h_net",
             "h_net_lo", "h_net_hi"]]
              .round(4).to_string(index=False))
        print("\n[stratified sensitivity, primary skan panel]")
        print(st[st["panel"] == "primary"][
            ["dataset", "biomarker", "stratum_by", "stratum", "n_events",
             "h_topo", "h_ctrl", "h_net", "h_net_lo", "h_net_hi"]]
              .round(4).to_string(index=False))
        print("\n[claims]")
        print(cl.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
