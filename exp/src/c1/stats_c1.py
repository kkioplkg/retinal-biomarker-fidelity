"""C1 statistics: robust scales, net / deployment harm, mixed models, Fig. 2.

Plan reference: exp/EXPERIMENT_PLAN.md S3.3 and proposal section 3.1.2 /
section 3.1.4.

Four jobs
---------
(a) **Per-dataset robust scale** ``sigma_B``.  Pre-registered estimator
    ``sigma = 1.4826 * MAD`` over the **ground-truth** biomarker values of that
    dataset, taken from ``exp/results/gateA_biomarkers_gt.csv`` when Gate A has
    run, otherwise from the ``B0_*`` columns of ``c1_images.csv`` (the per-image
    ``B(M)`` cache written by ``run_c1``).  Computed through
    :func:`src.bio.biomarkers.standardise` so the two stages cannot drift.

(b) **Net and deployment harm** (proposal section 3.1.2, steps 1-2)

    ``H_net = (|B(M') - B(M)| - |B(M_C) - B(M)|) / sigma_B``   (signed)
    ``H_dep = max(0, H_net)``                                   (non-negative)

    ``H_net`` is the scientific quantity; ``H_dep`` is the deployment target of
    the BTR heads.  Both are written back into the event table.

(c) **Nested mixed-effects model** per primary biomarker (section 3.1.4):
    image random intercept with the locus nested inside it as a variance
    component, fixed effects for perturbation type, radius bin, zone, branch
    order, pipeline, a restricted cubic spline in severity (3 knots), and the
    ``type x radius`` / ``type x pipeline`` interactions.  Coefficient tables go
    to ``exp/results/c1_mixedlm_<biomarker>.csv``.

(d) **Fig. 2 data**: dose-response curves -- mean +- 95 % CI of the *signed*
    ``dB / sigma_B`` against severity, stratified by radius bin and zone, for
    the perturbation and its matched control -- to
    ``exp/results/fig2_dose_response.csv`` and ``exp/figs/fig2_dose_response.png``
    (2x2 panels: FD, tortuosity, density, total length).

Usage
-----
    cd exp
    python -m src.c1.stats_c1                      # everything
    python -m src.c1.stats_c1 --no-mixedlm         # scales + harms + Fig. 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from src.eval.savefig_util import save_fig

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
FIGS_DIR = os.path.join(EXP_ROOT, "figs")

PRIMARY = ("FD", "tortuosity", "density", "total_length")
PIPELINES = ("pvbm", "skan")

#: Which pipelines count as *primary* evidence for each biomarker.
#:
#: Gate A (``results/gateA_pipeline_agreement.csv``) found FD, density and
#: total_length to agree across the two pipelines (Spearman >= 0.92), so those
#: three are aggregated with the pre-registered **median over pipelines**.
#: **Tortuosity failed the gate** (rho 0.49-0.75): PVBM's median arc/chord
#: tortuosity and our length-weighted arc/chord tortuosity are not measuring the
#: same quantity, so tortuosity is carried by its single pre-registered pipeline
#: (skan, length-weighted) alone.  The ``tortuosity_pvbm`` columns stay in the
#: event table but are flagged non-primary and never enter a scalarisation.
PRIMARY_PIPELINES = {
    "FD": ("pvbm", "skan"),
    "tortuosity": ("skan",),
    "density": ("pvbm", "skan"),
    "total_length": ("pvbm", "skan"),
}

#: columns kept for completeness but excluded from every scalarisation
NON_PRIMARY_COLUMNS = tuple(
    "%s_%s" % (b, p) for b in PRIMARY for p in PIPELINES
    if p not in PRIMARY_PIPELINES[b]
)

#: pretty names for the Fig. 2 panels
PANEL_TITLE = {
    "FD": "Fractal dimension",
    "tortuosity": "Tortuosity",
    "density": "Vessel density",
    "total_length": "Total vessel length",
}


# --------------------------------------------------------------------------- #
# (a) per-dataset robust scale
# --------------------------------------------------------------------------- #
def robust_scales(events_df=None, images_csv: Optional[str] = None,
                  gate_a_csv: Optional[str] = None):
    """Per-dataset ``sigma_B`` of every primary biomarker column.

    Source priority: the Gate A ground-truth biomarker table, else the ``B0_*``
    ``B(M)`` cache in ``c1_images.csv``.  Returns the tidy table produced by
    :func:`src.bio.biomarkers.standardise` plus the source that was used.
    """
    import pandas as pd

    from ..bio.biomarkers import standardise

    cand = [f"{b}_{p}" for b in PRIMARY for p in PIPELINES]
    images_csv = images_csv or os.path.join(RESULTS_DIR, "c1_images.csv")

    # ---- Gate A scales (measured at native resolution) --------------------
    gate_scales = None
    base_dir = os.path.dirname(gate_a_csv) if gate_a_csv else RESULTS_DIR
    scale_csv = os.path.join(base_dir or RESULTS_DIR, "gateA_biomarker_scales.csv")
    if os.path.exists(scale_csv):
        g = pd.read_csv(scale_csv)
        if {"dataset", "biomarker", "sigma"} <= set(g.columns):
            gate_scales = g.copy()
            gate_scales["source"] = "gateA:" + os.path.basename(scale_csv)
    if gate_scales is None and gate_a_csv and os.path.exists(gate_a_csv):
        gt = pd.read_csv(gate_a_csv)
        if "dataset" in gt.columns:
            cols = [c for c in cand if c in gt.columns]
            gate_scales, _ = standardise(gt, dataset_col="dataset", columns=cols, add_z=False)
            gate_scales["source"] = "gateA:" + os.path.basename(gate_a_csv)

    # ---- C1 B(M) cache (measured at the C1 *working* resolution) ----------
    c1_scales = None
    work_scale = {}
    if os.path.exists(images_csv):
        im = pd.read_csv(images_csv)
        cols = [c for c in im.columns if c.startswith("B0_")]
        if cols:
            b0 = im[["dataset"] + cols].rename(columns={c: c[3:] for c in cols})
            keep = [c for c in cand if c in b0.columns]
            c1_scales, _ = standardise(b0, dataset_col="dataset", columns=keep, add_z=False)
            c1_scales["source"] = "c1_images_B0:" + os.path.basename(images_csv)
        if "work_scale" in im.columns:
            work_scale = im.groupby("dataset")["work_scale"].median().to_dict()

    if gate_scales is None and c1_scales is None:
        raise FileNotFoundError(
            "neither Gate A scales nor %s exist: cannot estimate sigma_B" % images_csv)

    # ---- pick a source per dataset ----------------------------------------
    # Gate A measured B(M) at **native** resolution.  Length-like biomarkers
    # (total_length above all) scale with resolution, so a dataset that C1
    # processes downscaled (HRF / FIVES at 1536 px) must take sigma_B from the
    # C1 B(M) cache -- otherwise |dB| and sigma_B live on different scales and
    # H_net is silently wrong by the scale factor.  Datasets processed at native
    # resolution use the pre-registered Gate A scales.
    frames = []
    datasets = set()
    for src in (gate_scales, c1_scales):
        if src is not None:
            datasets |= set(src["dataset"].astype(str))
    for ds in sorted(datasets):
        ws = float(work_scale.get(ds, 1.0))
        native = abs(ws - 1.0) < 1e-6
        in_gate = gate_scales is not None and (gate_scales["dataset"].astype(str) == ds).any()
        in_c1 = c1_scales is not None and (c1_scales["dataset"].astype(str) == ds).any()
        if native and in_gate:
            pick = gate_scales[gate_scales["dataset"].astype(str) == ds].copy()
        elif in_c1:
            pick = c1_scales[c1_scales["dataset"].astype(str) == ds].copy()
        elif in_gate:
            pick = gate_scales[gate_scales["dataset"].astype(str) == ds].copy()
            pick["source"] = pick["source"].astype(str) + "(SCALE-MISMATCH)"
        else:
            continue
        pick["work_scale"] = ws
        frames.append(pick)
    scales = pd.concat(frames, ignore_index=True)
    scales["primary"] = [
        int(str(b).rsplit("_", 1)[-1] in PRIMARY_PIPELINES.get(str(b).rsplit("_", 1)[0], ()))
        for b in scales["biomarker"].astype(str)
    ]
    return scales


def scale_lookup(scales) -> Dict[Tuple[str, str], float]:
    """``(dataset, biomarker_column) -> sigma`` (NaN and zero become NaN)."""
    out: Dict[Tuple[str, str], float] = {}
    for _, r in scales.iterrows():
        s = float(r["sigma"])
        out[(str(r["dataset"]), str(r["biomarker"]))] = s if np.isfinite(s) and s > 0 else np.nan
    return out


# --------------------------------------------------------------------------- #
# (b) net and deployment harm
# --------------------------------------------------------------------------- #
def train_scale_lookup(results_dir: Optional[str] = None):
    """``(dataset, biomarker_col) -> sigma`` from the TRAINING-split scales.

    The deployment target ``H_abs`` must be standardised by a scale that no test
    image informed, otherwise the decision scale leaks the held-out data.  Falls
    back to ``{}`` when the file has not been built (run
    ``python -m src.c1.train_scales``), in which case the caller keeps the
    all-mask scale and says so.
    """
    import pandas as pd

    rd = results_dir or RESULTS_DIR
    path = os.path.join(rd, "gateA_biomarker_scales_train.csv")
    if not os.path.exists(path):
        return {}, None
    g = pd.read_csv(path)
    lut = {}
    for _, r in g.iterrows():
        try:
            sig = float(r["sigma"])
        except (TypeError, ValueError):
            continue
        if sig and sig == sig and sig > 0:
            lut[(str(r["dataset"]), str(r["biomarker"]))] = sig
    return lut, os.path.basename(path)


#: Unit conversion from the C1 *working* resolution to NATIVE resolution.
#:
#: C1 perturbs HRF/FIVES at a 1536 px long side (work_scale s < 1), but the
#: deployed evaluator (``src/eval/biomarker_eval``) scores predictions mapped
#: back to native resolution, so the deployed harm must be expressed in native
#: sigma units.  Only *length-like* biomarkers carry a length dimension:
#:
#:   ``total_length``  pixels of centreline  ->  multiply by 1/s
#:   ``density``       vessel px / FOV px, a ratio                -> unchanged
#:   ``FD``            a box-counting dimension                   -> unchanged
#:   ``tortuosity``    arc/chord, a ratio                         -> unchanged
#:
#: CAVEAT, stated explicitly because it is an assumption and not an identity:
#: FD and tortuosity are scale-*invariant* only in the continuum.  Both are
#: estimated from a finite raster: PVBM's multifractal D0 sweeps a box range
#: fixed relative to the image, and skan's arc/chord is measured on a skeleton
#: whose thin branches can merge at 1536 px.  Their downscaled values are
#: therefore approximately, not exactly, the native ones, and no correction is
#: applied.  ``density`` is a ratio of two areas that rescale together, so it is
#: the most robust of the three.
LENGTH_LIKE = ("total_length",)


def native_factor(biomarker_col: str, work_scale: float) -> float:
    """Multiplier taking ``|dB|`` from the working resolution to native units."""
    try:
        s = float(work_scale)
    except (TypeError, ValueError):
        return 1.0
    if not (s > 0) or abs(s - 1.0) < 1e-9:
        return 1.0
    base = str(biomarker_col).rsplit("_", 1)[0]
    return (1.0 / s) if base in LENGTH_LIKE else 1.0


def add_harms(df, scales, train_lut=None, work_scale=None):
    """Add ``sigma_``, ``Hnet_`` and ``Hdep_`` columns for every primary column.

    ``H_net = (|dB| - |dB_ctrl|) / sigma_B``  (signed, proposal section 3.1.2)
    ``H_dep = max(0, H_net)``                 (the non-negative BTR target)
    ``H_abs = |dB| / sigma_B^train``          (the DEPLOYED target)

    ``H_net``/``H_dep`` keep the pre-registered all-mask ``sigma_B`` so the
    scientific dose-response is unchanged.  ``H_abs`` -- the quantity the
    deployed heads regress and a live system divides by -- uses the
    training-split-only scale instead, because a decision scale must not be
    informed by the images the system is judged on.  The all-mask version is
    retained alongside as ``Habs_all_*`` so the two can be compared.
    """
    import pandas as pd

    lut = scale_lookup(scales)
    train_lut = {} if train_lut is None else train_lut
    work_scale = {} if work_scale is None else work_scale
    ds = df["dataset"].astype(str)
    for b in PRIMARY:
        for p in PIPELINES:
            col = f"{b}_{p}"
            a_col, c_col = f"absdB_{col}", f"cabsdB_{col}"
            if a_col not in df.columns:
                continue
            sig = ds.map(lambda d, c=col: lut.get((d, c), np.nan)).astype(float)
            df[f"sigma_{col}"] = sig
            ctrl = (pd.to_numeric(df[c_col], errors="coerce") if c_col in df.columns
                    else pd.Series(np.nan, index=df.index))
            net = (pd.to_numeric(df[a_col], errors="coerce") - ctrl) / sig
            df[f"Hnet_{col}"] = net
            df[f"Hdep_{col}"] = net.clip(lower=0.0)
            # deployment target: standardised by the TRAIN-only scale when we
            # have one, with the all-mask version kept as a supplementary column
            sig_tr = ds.map(lambda d, c=col: train_lut.get((d, c), np.nan)).astype(float)
            df[f"sigmatr_{col}"] = sig_tr
            absdb = pd.to_numeric(df[a_col], errors="coerce")
            # supplementary: working-resolution units over the all-mask scale
            df[f"Habs_ws_{col}"] = (absdb / sig).clip(lower=0.0)
            # deployed: convert |dB| to NATIVE units, then divide by the
            # native training-split scale
            conv = ds.map(lambda d, c=col: native_factor(c, work_scale.get(d, 1.0)))
            df[f"nativef_{col}"] = conv.astype(float)
            df[f"Habs_{col}"] = ((absdb * conv.astype(float))
                                 / sig_tr.where(sig_tr > 0)).clip(lower=0.0)
            df[f"dBz_{col}"] = pd.to_numeric(df[f"dB_{col}"], errors="coerce") / sig
            if c_col in df.columns:
                df[f"cdBz_{col}"] = pd.to_numeric(df[f"cdB_{col}"], errors="coerce") / sig
    # macro-standardised scalarisation used as the Exp1 regression target:
    # equal weights over the four biomarkers, median over the two pipelines.
    for prefix, name in (("Hdep", "H_macro"), ("Hnet", "Hnet_macro"),
                         ("Habs", "Habs_macro")):
        per_b = []
        for b in PRIMARY:
            cols = [f"{prefix}_{b}_{p}" for p in PRIMARY_PIPELINES[b]
                    if f"{prefix}_{b}_{p}" in df.columns]
            if cols:
                per_b.append(df[cols].median(axis=1))
        if per_b:
            df[name] = pd.concat(per_b, axis=1).mean(axis=1)
    for c in NON_PRIMARY_COLUMNS:
        df[f"nonprimary_{c}"] = 1
    return df


# --------------------------------------------------------------------------- #
# severity handling
# --------------------------------------------------------------------------- #
def rcs_basis(x: np.ndarray, n_knots: int = 3) -> np.ndarray:
    """Harrell restricted (natural) cubic spline basis, ``n_knots - 1`` columns.

    Column 0 is ``x`` itself; the remaining columns are the truncated-power
    terms constrained to be linear beyond the boundary knots.  With the
    pre-registered ``n_knots = 3`` this yields one linear and one non-linear
    term, i.e. the "restricted cubic spline, 3 knots" of section 3.1.4.

    Implemented here rather than through ``patsy.cr`` because patsy's cubic
    spline builder raises on numpy >= 2 in this environment.  Knots sit at the
    10th / 50th / 90th percentiles (Harrell's recommendation for 3 knots).
    """
    x = np.asarray(x, dtype=float)
    fin = x[np.isfinite(x)]
    k = int(n_knots)
    if fin.size < k + 2 or np.ptp(fin) <= 0:
        return x.reshape(-1, 1)
    qs = np.linspace(10.0, 90.0, k)
    t = np.unique(np.percentile(fin, qs))
    if t.size < 3:
        return x.reshape(-1, 1)
    k = t.size
    cols = [x]
    denom = (t[-1] - t[0]) ** 2
    for j in range(k - 2):
        cube = lambda v: np.clip(v, 0.0, None) ** 3            # noqa: E731
        term = (cube(x - t[j])
                - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / denom)
    return np.column_stack(cols)


def add_severity_std(df):
    """Per-type severity rescaled to ``[0, 1]``.

    The four families have incommensurable severity units (capsule length in
    ``D``, ``rho``, bridge span in px, ``+-1`` px caliber).  The spline in the
    pooled model therefore runs on a per-type min-max rescaling of
    ``severity_value``; the type main effect and the ``type x ...``
    interactions absorb the level differences.
    """
    import pandas as pd

    v = pd.to_numeric(df["severity_value"], errors="coerce").abs()
    out = np.full(len(df), np.nan)
    for t, idx in df.groupby("type").groups.items():
        x = v.loc[idx].to_numpy(dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        out[df.index.get_indexer(idx)] = (x - lo) / (hi - lo) if hi > lo else 0.5
    df["severity_std"] = out
    B = rcs_basis(out, n_knots=3)
    for i in range(B.shape[1]):
        df[f"sev_rcs{i + 1}"] = B[:, i]
    df["order_bin"] = np.clip(pd.to_numeric(df.get("gt_branch_order"),
                                            errors="coerce").fillna(-1), -1, 4).astype(int)
    df["image_uid"] = df["dataset"].astype(str) + ":" + df["image_id"].astype(str) + ":" + \
        df["observer"].astype(str)
    return df


def to_long(df, biomarker: str, value: str = "Hnet"):
    """One row per (event, pipeline) for the pipeline factor of section 3.1.4."""
    import pandas as pd

    keep = ["image_uid", "dataset", "locus_id", "event_id", "type", "severity",
            "severity_std", "gt_radius_bin", "gt_zone", "order_bin", "subject_id"]
    keep += [c for c in df.columns if c.startswith("sev_rcs")]
    keep = [c for c in keep if c in df.columns]
    frames = []
    for p in PIPELINES:
        col = f"{value}_{biomarker}_{p}"
        if col not in df.columns:
            continue
        sub = df[keep].copy()
        sub["pipeline"] = p
        sub["y"] = pd.to_numeric(df[col], errors="coerce")
        frames.append(sub)
    if not frames:
        return None
    long = pd.concat(frames, ignore_index=True)
    return long.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "severity_std"])


# --------------------------------------------------------------------------- #
# (c) nested mixed-effects model
# --------------------------------------------------------------------------- #
def fit_mixedlm(long, out_csv: str, maxiter: int = 200):
    """MixedLM: image random intercept + locus variance component.

    ``y ~ sev_rcs1 + sev_rcs2 + C(type) + C(gt_radius_bin) + C(gt_zone)
          + C(order_bin) + C(pipeline) + C(type):C(gt_radius_bin)
          + C(type):C(pipeline)``
    with ``groups = image_uid`` and ``vc_formula = {"locus": "0 + C(locus_id)"}``
    -- the locus nested inside the image, as section 3.1.4 requires (the same
    anatomical site carries several severities and its observations are
    strongly correlated).
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    if long is None or len(long) < 30:
        return None, "too_few_rows"
    n_type = long["type"].nunique()
    spl = [c for c in long.columns if c.startswith("sev_rcs")
           and long[c].nunique() > 1]
    terms = list(spl) if spl else ["severity_std"]
    for c, name in (("type", "C(type)"), ("gt_radius_bin", "C(gt_radius_bin)"),
                    ("gt_zone", "C(gt_zone)"), ("order_bin", "C(order_bin)"),
                    ("pipeline", "C(pipeline)")):
        if c in long.columns and long[c].nunique() > 1:
            terms.append(name)
    if n_type > 1 and long["gt_radius_bin"].nunique() > 1:
        terms.append("C(type):C(gt_radius_bin)")
    if n_type > 1 and long["pipeline"].nunique() > 1:
        terms.append("C(type):C(pipeline)")
    # Fit the richest specification that is estimable.  With a full event table
    # every cell of type x radius and type x pipeline is populated and the first
    # formula fits; on small or unbalanced subsets the interaction (then the
    # higher-order main effect) cells can be empty, which makes the design
    # singular.  Rather than return nothing we drop terms from the end of the
    # ladder and record what was dropped.
    ladder = [list(terms)]
    for drop in ("C(type):C(pipeline)", "C(type):C(gt_radius_bin)",
                 "C(order_bin)", "C(gt_zone)", "C(gt_radius_bin)"):
        if drop in ladder[-1]:
            nxt = [t for t in ladder[-1] if t != drop]
            ladder.append(nxt)

    kw_base = dict(groups=long["image_uid"], re_formula="1")
    use_vc = long["locus_id"].nunique() > 1 and long["image_uid"].nunique() > 1

    res = None
    formula = ""
    dropped: List[str] = []
    errors: List[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cand in ladder:
            formula = "y ~ " + " + ".join(cand)
            dropped = [t for t in terms if t not in cand]
            for vc in ((True, False) if use_vc else (False,)):
                kw = dict(kw_base)
                if vc:
                    kw["vc_formula"] = {"locus": "0 + C(locus_id)"}
                try:
                    res = smf.mixedlm(formula, long, **kw).fit(
                        method="lbfgs", maxiter=maxiter, reml=True)
                    break
                except Exception as exc:  # noqa: BLE001 - singular design is common
                    errors.append(f"{formula[:40]}|vc={vc}: {type(exc).__name__}")
                    res = None
            if res is not None:
                break
    if res is None:
        return None, "all specifications singular :: " + "; ".join(errors[:3])

    tab = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.to_numpy(),
        "se": res.bse.reindex(res.params.index).to_numpy(),
        "z": res.tvalues.reindex(res.params.index).to_numpy(),
        "p": res.pvalues.reindex(res.params.index).to_numpy(),
    })
    ci = res.conf_int()
    tab["ci_lo"] = ci[0].reindex(res.params.index).to_numpy()
    tab["ci_hi"] = ci[1].reindex(res.params.index).to_numpy()
    tab["formula"] = formula
    tab["dropped_terms"] = ";".join(dropped)
    tab["n_obs"] = int(len(long))
    tab["n_groups"] = int(long["image_uid"].nunique())
    tab["converged"] = bool(getattr(res, "converged", True))
    tab.to_csv(out_csv, index=False)
    return res, formula


# --------------------------------------------------------------------------- #
# (d) Fig. 2 -- dose-response
# --------------------------------------------------------------------------- #
def _mean_ci(x: np.ndarray) -> Tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, n
    se = float(x.std(ddof=1) / np.sqrt(n))
    return m, m - 1.96 * se, m + 1.96 * se, n


def dose_response(df, out_csv: str):
    """Signed ``dB / sigma_B`` vs severity, by radius bin and zone, arm-wise."""
    import pandas as pd

    rows: List[Dict[str, object]] = []
    for b in PRIMARY:
        for p in PIPELINES:
            zcol, czcol = f"dBz_{b}_{p}", f"cdBz_{b}_{p}"
            if zcol not in df.columns:
                continue
            # group on the *categorical* severity level, not on the raw value:
            # the bridge family's severity_value is the continuous span in px and
            # would otherwise produce one singleton group per event.
            grp = ["dataset", "type", "severity", "gt_radius_bin", "gt_zone"]
            grp = [g for g in grp if g in df.columns]
            for keys, g in df.groupby(grp, dropna=False):
                base = dict(zip(grp, keys if isinstance(keys, tuple) else (keys,)))
                base["severity_value_mean"] = float(
                    pd.to_numeric(g["severity_value"], errors="coerce").mean())
                for arm, col in (("perturbation", zcol), ("matched_control", czcol)):
                    if col not in g.columns:
                        continue
                    m, lo, hi, n = _mean_ci(g[col].to_numpy(dtype=float))
                    rows.append(dict(base, biomarker=b, pipeline=p, arm=arm,
                                     mean=m, ci_lo=lo, ci_hi=hi, n=n))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def plot_fig2(dose, df, out_png: str):
    """2x2 panels (FD / tortuosity / density / total length), pooled over datasets.

    Each panel shows the signed standardised response against severity for the
    perturbation arm and its matched control, one line per radius bin, with the
    95 % CI shaded.  Severing events (the primary dose ladder
    ``L in {0.5, 1, 2, 4} D``) drive the x axis; the other families appear in
    the csv but not in the main figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    sev = df[df["type"] == "sever"].copy()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, b in zip(axes.ravel(), PRIMARY):
        zc = [f"dBz_{b}_{p}" for p in PIPELINES if f"dBz_{b}_{p}" in sev.columns]
        cc = [f"cdBz_{b}_{p}" for p in PIPELINES if f"cdBz_{b}_{p}" in sev.columns]
        if not zc:
            ax.set_axis_off()
            continue
        sev["_y"] = sev[zc].median(axis=1)
        sev["_c"] = sev[cc].median(axis=1) if cc else np.nan
        bins = sorted(pd.to_numeric(sev["gt_radius_bin"], errors="coerce").dropna().unique())
        for i, rb in enumerate(bins):
            g = sev[sev["gt_radius_bin"] == rb]
            xs, ms, los, his = [], [], [], []
            cms, clos, chis = [], [], []
            for L, gg in g.groupby("severity_value"):
                m, lo, hi, n = _mean_ci(gg["_y"].to_numpy(dtype=float))
                cm, clo, chi, _ = _mean_ci(gg["_c"].to_numpy(dtype=float))
                if n == 0:
                    continue
                xs.append(float(L)); ms.append(m); los.append(lo); his.append(hi)
                cms.append(cm); clos.append(clo); chis.append(chi)
            if not xs:
                continue
            o = np.argsort(xs)
            xs = np.array(xs)[o]
            col = colors[i % len(colors)]
            ax.plot(xs, np.array(ms)[o], "-o", color=col, label=f"radius bin {int(rb)}")
            ax.fill_between(xs, np.array(los)[o], np.array(his)[o], color=col, alpha=0.18)
            ax.plot(xs, np.array(cms)[o], "--s", color=col, alpha=0.65, markersize=4)
            ax.fill_between(xs, np.array(clos)[o], np.array(chis)[o], color=col, alpha=0.08)
        ax.axhline(0.0, color="0.4", lw=0.8)
        ax.set_title(PANEL_TITLE[b])
        ax.set_xlabel("capsule severance length  L / D")
        ax.set_ylabel(r"signed $\Delta B / \sigma_B$")
        ax.legend(fontsize=7, title="solid = perturbation\ndashed = matched control",
                  title_fontsize=7)
    # No suptitle -- the LaTeX float numbers and titles this figure, and an
    # embedded "Fig. 2" contradicts whatever number the document assigns.  What
    # the title said (capsule severance, median over the two pipelines, mean
    # +- 95 % CI, pooled over datasets) now lives in the LaTeX caption, in
    # paper/sections_supp/s14_moved_floats.tex.  Panel titles stay.
    save_fig(fig, out_png, dpi=200)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="C1 statistics: scales, harms, mixed models, Fig. 2")
    ap.add_argument("--events", default=os.path.join(RESULTS_DIR, "c1_events.parquet"),
                    help="merged event table written by run_c1 (which merges its "
                         "per-image shards into this file)")
    ap.add_argument("--images", default=os.path.join(RESULTS_DIR, "c1_images.csv"))
    ap.add_argument("--gate-a", default=os.path.join(RESULTS_DIR, "gateA_biomarkers_gt.csv"))
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default=RESULTS_DIR)
    ap.add_argument("--figs-dir", default=FIGS_DIR)
    ap.add_argument("--no-mixedlm", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args(argv)

    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    path = args.events
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv"
        if not os.path.exists(alt):
            raise FileNotFoundError(f"no event table at {path} or {alt}")
        path = alt
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"[in] {path}  rows={len(df)}  cols={len(df.columns)}")

    # (a) --------------------------------------------------------------------
    scales = robust_scales(df, images_csv=args.images, gate_a_csv=args.gate_a)
    sc_csv = os.path.join(args.out_dir, "c1_sigma_B.csv")
    scales.to_csv(sc_csv, index=False)
    print(f"[a] sigma_B from {scales['source'].iloc[0]} -> {sc_csv}")
    print(scales[["dataset", "biomarker", "n", "median", "sigma"]].to_string(index=False))

    # (b) --------------------------------------------------------------------
    train_lut, train_src = train_scale_lookup(args.out_dir)
    if train_src:
        miss = sorted({(d, b) for d, b in
                       zip(scales["dataset"].astype(str), scales["biomarker"].astype(str))
                       if (d, b) not in train_lut and
                       int(scales.loc[(scales["dataset"].astype(str) == d) &
                                      (scales["biomarker"].astype(str) == b),
                                      "primary"].max() or 0) == 1})
        print(f"[b] H_abs standardised by TRAIN-only sigma from {train_src} "
              f"({len(train_lut)} entries)")
        if miss:
            print(f"[b] WARNING no train sigma for {miss} -> H_abs is NaN there")
    else:
        print("[b] WARNING no gateA_biomarker_scales_train.csv; H_abs falls back "
              "to the all-mask sigma (run `python -m src.c1.train_scales`)")
    ws_map = {}
    if "work_scale" in scales.columns:
        ws_map = (scales.groupby(scales["dataset"].astype(str))["work_scale"]
                  .median().to_dict())
    print(f"[b] work_scale per dataset: "
          f"{ {k: round(float(v), 4) for k, v in ws_map.items()} }; "
          "length-like |dB| converted to NATIVE units before standardising")
    df = add_harms(df, scales, train_lut=train_lut, work_scale=ws_map)
    df = add_severity_std(df)
    ev_out = os.path.join(args.out_dir, "c1_events_harm.parquet")
    try:
        df.to_parquet(ev_out, index=False)
    except Exception:  # noqa: BLE001
        ev_out = os.path.join(args.out_dir, "c1_events_harm.csv")
        df.to_csv(ev_out, index=False)
    print(f"[b] H_net / H_dep added -> {ev_out}")
    hsum = {}
    for b in PRIMARY:
        for p in PIPELINES:
            c = f"Hnet_{b}_{p}"
            if c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce")
                hsum[c] = dict(mean=float(x.mean()), median=float(x.median()),
                               frac_positive=float((x > 0).mean()), n=int(x.notna().sum()))
    print(json.dumps(hsum, indent=2)[:2000])

    # (c) --------------------------------------------------------------------
    models: Dict[str, str] = {}
    if not args.no_mixedlm:
        for b in PRIMARY:
            long = to_long(df, b, value="Hnet")
            out_csv = os.path.join(args.out_dir, f"c1_mixedlm_{b}.csv")
            res, info = fit_mixedlm(long, out_csv, maxiter=args.maxiter)
            models[b] = info if res is None else f"ok n={len(long)} :: {info}"
            print(f"[c] {b:14s} {models[b]}")

    # (d) --------------------------------------------------------------------
    dr_csv = os.path.join(args.out_dir, "fig2_dose_response.csv")
    dose = dose_response(df, dr_csv)
    print(f"[d] dose-response rows={len(dose)} -> {dr_csv}")
    png = ""
    if not args.no_fig:
        png = plot_fig2(dose, df, os.path.join(args.figs_dir, "fig2_dose_response.png"))
        print(f"[d] {png}")

    with open(os.path.join(args.out_dir, "c1_stats_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(events=path, n_events=int(len(df)), sigma_source=str(scales["source"].iloc[0]),
                       habs_sigma_source=(train_src or "FALLBACK:all-mask sigma"),
                       habs_units="native (length-like |dB| x 1/work_scale)",
                       harm_summary=hsum, mixedlm=models, fig2=png), fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
