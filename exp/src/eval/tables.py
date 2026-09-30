"""Main-text tables (Tab.1 / Tab.2 / Tab.3), proposal v3 section 4.3.

Every ``build_tab*`` function is independently callable and independently
robust: a missing input file is reported with a one-line ``[skip]`` message
and the function returns ``None`` (or a partial table) instead of raising, so
``make_all.py`` can run this against whatever subset of the pipeline has
actually been executed.

Schema, as unified across producer modules (was a set of gaps; fixed by the
schema-reconciliation pass -- kept here as the contract, with the old-run
fallbacks the readers below still honour):

* ``src.rigr.run_rigr`` and ``src.baselines.run_baseline`` both write
  ``per_image.csv`` with an ``image`` column, ``before_*`` / ``after_*`` /
  ``delta_*`` metric prefixes, ``dataset, seed, method``, and (when biomarkers
  were requested) ``macro_mae_before`` / ``macro_mae_after`` /
  ``macro_mae_delta`` / ``bio_*`` columns computed identically via
  ``src.eval.biomarker_eval`` (same primary biomarkers, same single-source
  MAD scale, ``results/gateA_biomarker_scales.csv``). Older baseline runs
  predating this may still carry the previous ``image_id`` column and
  ``base_*`` / ``rep_*`` / ``d_*`` prefixes (with no ``macro_mae_*`` / ``bio_*``
  at all); the loaders below fall back to that schema when the new one is
  absent, so both kinds of run remain readable.
* ``TRR_recall`` / ``TRR_precision`` / ``FCR`` / ``n_should`` / ``n_accepted``
  / ``n_true`` / ``n_matched`` are named identically in both producers, so
  Tab.2's TRR/FCR columns merge cleanly.  Since DECISIONS.md 2026-09-03 10:20
  the **reported** ``TRR_precision`` / ``FCR`` are the *matched* pair
  ``m / |A|`` and ``1 - m / |A|`` (see :func:`add_matched_fcr`); the edge-level
  pair survives as ``TRR_precision_edge`` / ``FCR_edge``.
* ``sweep_curve.csv`` and ``sweep_per_image.csv`` (written by
  ``run_rigr --sweep``) both carry ``dataset, seed, mode`` (plus ``lam, tau``),
  so either can be told apart from another run's sweep once copied out of its
  run directory. Fig.3 (in ``figures.py``) reads ``sweep_per_image.csv`` for
  the per-image identity it needs to bootstrap a CI, falling back to
  ``sweep_curve.csv`` (point estimates only) when that is all that is
  available.

Tab.1 Exp2 schema (candidate-level natural intervention benchmark)
--------------------------------------------------------------------
``results/tab1_exp2.csv`` does not exist yet; ``src.rigr.run_rigr`` (S4 item
8) is expected to produce it with one row per **actually-applied** candidate
edge, columns:

    dataset     str    canonical dataset name (drive/chasedb1/hrf/fives)
    image       str    image key, matching the per_image.csv ``image`` column
    cand_id     int    candidate id, matching ``cand_<image>.csv``
    p_e         float  calibrated acceptance probability of the pair scorer
    dDice       float  Dice(after) - Dice(before) from actually adding e
    dclDice     float  clDice(after) - clDice(before)
    dBCS        float  BCS(after) - BCS(before)
    dBetti0     float  beta0_err(after) - beta0_err(before) (signed; more
                        negative = better topology)
    U_btr       float  the BTR-utility score assigned to e before it was
                        added (this is the "BTR" predictor of Tab.1's lower
                        half)
    dH_actual   float  the realised macro biomarker harm/benefit of adding e
                        (ground truth being predicted; sign convention:
                        positive = harm, i.e. macro-MAE increased)

``build_tab1`` computes, per predictor in
``{p_e, dDice, dclDice, dBCS, dBetti0, U_btr}``, the Spearman correlation with
``dH_actual`` and an equal-capacity GBDT-mapped held-out MAE/R^2 exactly as
``src.c1.btr.run_exp1`` does for Exp1, so the two halves of Tab.1 use the same
statistic family.

CLI
---
    cd exp
    python -m src.eval.tables
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
RUNS_DIR = os.path.join(EXP_ROOT, "runs")

PRIMARY_BIOMARKERS = ("FD", "tortuosity", "density", "total_length")
# S6 baselines: "evapore" is split into its two inference variants,
# which share one checkpoint and differ only in candidate generation and
# acceptance -- evapore_e2e is upstream end-to-end, evapore_scorer is their
# classifier inside our harness.  run_baseline writes these names into the
# `method` column, so the loaders must ask for them by name.
TAB2_METHODS = ["no_repair", "geometric", "rnca", "evapore_e2e",
                "evapore_scorer", "rigr_uniform", "rigr_risk"]
TAB2_DATASETS = ["drive", "chasedb1", "hrf", "fives"]
TAB2_METRICS = ["cldice", "bcs", "TRR_recall", "FCR", "macro_mae", "f1"]
TAB3_VARIANTS = ["appearance_only", "+orientation", "+curvature",
                 "+failcond", "+btr_utility", "frangi_vs_learned"]


# --------------------------------------------------------------------------- #
# generic io / formatting helpers
# --------------------------------------------------------------------------- #
def _read_csv(path: str):
    import pandas as pd

    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[tables] failed to read {path}: {e}")
        return None


def _read_json(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[tables] failed to read {path}: {e}")
        return None


def _glob(*patterns: str) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out))


#: Substrings that mark a run directory as **withdrawn** -- a superseded or
#: half-finished run kept on disk for provenance, or a development smoke run.
#: They live as siblings of the live run (``runs/rigr/risk/drive/seed0`` next to
#: ``runs/rigr/risk/drive/seed0_etascale_pre0906``), so a plain ``**`` glob
#: reads both and silently averages a withdrawn result into the headline number.
#:
#: * ``_etascale_pre0906`` / ``_pre`` -- the pre-C_geom-fix risk runs isolated by
#:   DECISIONS.md 2026-09-06 12:40(5); their utility is the one that "accepts
#:   almost no edge".
#: * ``_partial`` / ``_moved`` -- runs stopped mid-way (DECISIONS.md 2026-09-05
#:   23:29, 2026-09-06 16:12).
#: * ``smoke`` and any leading ``_`` -- development output (``runs/rigr/_smoke2``,
#:   ``runs/repair/smoke3``, ``runs/repair/smoke4``).
_WITHDRAWN_MARKERS = ("_etascale_pre", "_pre0906", "_partial", "_moved", "smoke")


def _is_live_run(path: str) -> bool:
    """False when any directory component of *path* (below ``EXP_ROOT``) marks a
    withdrawn or smoke run.  See :data:`_WITHDRAWN_MARKERS`."""
    try:
        rel = os.path.relpath(os.path.abspath(path), EXP_ROOT)
    except ValueError:                       # different drive; use it as given
        rel = path
    for seg in rel.replace("\\", "/").split("/"):
        if seg.startswith("_"):
            return False
        low = seg.lower()
        if any(mark in low for mark in _WITHDRAWN_MARKERS):
            return False
    return True


def _live_glob(*patterns: str) -> List[str]:
    """:func:`_glob` with the withdrawn/smoke run directories filtered out."""
    hits = _glob(*patterns)
    live, dropped = [], []
    for p in hits:
        (live if _is_live_run(p) else dropped).append(p)
    if dropped:
        shown = [os.path.relpath(d, EXP_ROOT) for d in dropped[:6]]
        print("[tables] ignoring %d withdrawn/smoke run file(s): %s%s"
              % (len(dropped), ", ".join(shown),
                 " ..." if len(dropped) > 6 else ""))
    return live


def df_to_markdown(df, floatfmt: str = "{:.4f}") -> str:
    """A dependency-free (no ``tabulate``) GitHub-flavoured markdown table."""
    import pandas as pd

    if df is None or len(df) == 0:
        return "_(no data)_\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return floatfmt.format(v)
        return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def df_to_latex_booktabs(df, caption: str = "", label: str = "",
                          floatfmt: str = "{:.4f}") -> str:
    import pandas as pd

    if df is None or len(df) == 0:
        return f"% {label}: no data\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            return "--" if np.isnan(v) else floatfmt.format(v)
        return "--" if v is None else str(v).replace("_", "\\_")

    colspec = "l" * len(cols)
    # Two presentation fixes, both needed by the Elsevier CAS classes the paper
    # uses.  (1) No "[t]" placement: CAS redefines the table environment to take
    # a KEY-VALUE optional argument, so a bare [t] empties fps@table and emits
    # "No positions in optional float specifier".  (2) Wide tables (the Exp1
    # benchmark is 11 columns) overflow the single-column text block, so they
    # are wrapped in a \\resizebox; narrow tables keep document-size type.
    wide = len(cols) > 6
    lines = [
        "\\begin{table}", "\\centering",
        f"\\caption{{{caption}}}" if caption else "\\caption{}",
        f"\\label{{{label}}}" if label else "\\label{}",
    ]
    if wide:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines += [
        "\\begin{tabular}{%s}" % colspec, "\\toprule",
        " & ".join(str(c).replace("_", "\\_") for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt(row[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if wide:
        lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    """Paired Wilcoxon signed-rank on aligned arrays, NaN-safe."""
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    d = d[d != 0]
    if d.size < 4:
        return float("nan"), float("nan"), int(ok.sum())
    try:
        stat, p = wilcoxon(d)
        return float(stat), float(p), int(ok.sum())
    except Exception:
        return float("nan"), float("nan"), int(ok.sum())


# --------------------------------------------------------------------------- #
# Tab.1 -- BTR predictive benchmark
# --------------------------------------------------------------------------- #
def _tab1_exp2(path: str, n_boot: int = 1000, seed: int = 0):
    """Candidate-level natural-intervention benchmark, see module docstring
    for the expected ``tab1_exp2.csv`` schema."""
    import pandas as pd

    df = _read_csv(path)
    if df is None:
        print(f"[tab1][skip] Exp2: {path} not found "
              "(src.rigr.run_rigr S4 item 8 has not produced it yet)")
        return None
    need = {"dataset", "image", "cand_id", "p_e", "dDice", "dclDice", "dBCS",
            "dBetti0", "U_btr", "dH_actual"}
    missing = need - set(df.columns)
    if missing:
        print(f"[tab1][skip] Exp2: {path} is missing columns {sorted(missing)}; "
              "see the tab1_exp2.csv schema in tables.py's module docstring")
        return None

    from scipy.stats import spearmanr

    y = pd.to_numeric(df["dH_actual"], errors="coerce").to_numpy(dtype=float)
    predictors = ["p_e", "dDice", "dclDice", "dBCS", "dBetti0", "U_btr"]
    rows = []
    for name in predictors:
        x = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            rows.append(dict(predictor=name, n=int(ok.sum()), spearman=np.nan,
                             spearman_p=np.nan, auc=np.nan))
            continue
        rho, pval = spearmanr(x[ok], y[ok])
        # classification AUC: does the predictor separate harmful (dH>0) from
        # beneficial/neutral (dH<=0) candidates? (proposal Tab.1 caption)
        auc = np.nan
        yb = (y[ok] > 0).astype(int)
        if 0 < yb.sum() < yb.size:
            try:
                from sklearn.metrics import roc_auc_score

                auc = float(roc_auc_score(yb, x[ok]))
            except Exception:
                pass
        rows.append(dict(predictor=name, n=int(ok.sum()), spearman=float(rho),
                         spearman_p=float(pval), auc=auc))
    tab = pd.DataFrame(rows)
    tab.insert(0, "exp", "exp2_candidate_level")
    return tab


def build_tab1(results_dir: str = RESULTS_DIR, out_dir: str = RESULTS_DIR,
               n_boot: int = 1000) -> Optional[Dict]:
    """Tab.1: BTR predictive benchmark (Exp1 controlled + Exp2 candidate-level).

    Reads ``results/tab1_exp1.csv`` + ``results/tab1_exp1_bootstrap.json``
    (written by ``src.c1.btr``) and ``results/tab1_exp2.csv`` (not yet
    produced by any module -- see the docstring above).  Writes
    ``results/tab1.md`` and ``results/tab1.tex``.  Returns a dict of the
    assembled frames, or ``None`` if nothing at all was available.
    """
    import pandas as pd

    # Primary Exp1 = the DEPLOYED target under the proper protocol.
    # ``tab1_exp1.csv`` is byte-identical to ``tab1_exp1_hdep.csv``, i.e. the
    # H_dep target that DECISIONS.md 2026-09-03 08:30 (gate B) retired -- the one
    # that is "mostly zero by construction" and that the deployed heads no longer
    # use.  ``tab1_exp1_indomain.csv`` is H_abs scored train-split-fit /
    # test-split-scored, which is what deployment actually does.  Fall back to
    # the legacy file only if the in-domain one is absent.
    exp1_indomain = _read_csv(os.path.join(results_dir, "tab1_exp1_indomain.csv"))
    exp1_legacy = _read_csv(os.path.join(results_dir, "tab1_exp1.csv"))
    if exp1_indomain is not None:
        exp1 = exp1_indomain[exp1_indomain["scope"].astype(str) == "ALL(test)"] \
            if "scope" in exp1_indomain.columns else exp1_indomain
        exp1_boot = _read_json(os.path.join(results_dir,
                                            "tab1_exp1_indomain_bootstrap.json"))
        exp1_target = "habs (deployed), fit train split / score test split"
    else:
        exp1, exp1_target = exp1_legacy, "hdep (LEGACY -- retired by gate B)"
        exp1_boot = _read_json(os.path.join(results_dir,
                                            "tab1_exp1_bootstrap.json"))
    if exp1 is None:
        print(f"[tab1][skip] Exp1: neither tab1_exp1_indomain.csv nor "
              f"tab1_exp1.csv found in {results_dir} "
              "(run `python -m src.c1.btr` first)")
    exp2 = _tab1_exp2(os.path.join(results_dir, "tab1_exp2.csv"), n_boot=n_boot)

    if exp1 is None and exp2 is None:
        print("[tab1] nothing to build")
        return None

    md_parts = ["## Tab.1 -- BTR predictive benchmark\n"]
    tex_parts = []

    if exp1 is not None:
        e1 = exp1.copy()
        e1.insert(0, "exp", "exp1_controlled")
        keep = [c for c in ("exp", "head", "block", "predictor", "n_features",
                            "n", "spearman", "spearman_p", "mae", "r2",
                            "best_conventional", "n_events", "n_images")
                if c in e1.columns]
        e1 = e1[keep]
        md_parts.append("### Exp1: controlled perturbation (raw association + "
                        "equal-capacity calibrated prediction)\n")
        md_parts.append(f"_Target: {exp1_target}._\n")
        md_parts.append(
            "_`block` separates **GT-informed conventional** predictors "
            "(dDice, dclDice, dBetti*, BCSdrop -- each needs the ground-truth "
            "mask, so none is available at inference time) from **no-GT "
            "observable** ones (obs_d, obs_radius, obs_density, ... computed "
            "from the image and the predicted mask alone). Only the second "
            "group is a deployable rival._\n")
        md_parts.append(df_to_markdown(e1))
        tex_parts.append(df_to_latex_booktabs(
            e1, caption="BTR Exp1: controlled-perturbation predictive benchmark.",
            label="tab:tab1_exp1"))
        if exp1_boot and isinstance(exp1_boot.get("bootstrap"), dict):
            md_parts.append(
                "\n**Image-level bootstrap CI on MAE(BTR) - MAE(rival)** "
                "(negative favours BTR), reported separately against the "
                "GT-informed and the inference-time-observable rival:\n")
            for kind, b in exp1_boot["bootstrap"].items():
                if not isinstance(b, dict):
                    continue
                for cmp_key, cmp_label in (("vs_conventional", "GT-informed"),
                                           ("vs_observable", "no-GT observable")):
                    c = b.get(cmp_key)
                    if not isinstance(c, dict):
                        continue
                    # newer files nest the stats under d_mae; older ones are flat
                    st = c.get("d_mae") if isinstance(c.get("d_mae"), dict) else c
                    sig = ("excludes zero" if st.get("excludes_zero")
                           else "crosses zero")
                    fav = st.get("favours_btr")
                    if fav is None:
                        fav = st.get("mean", 0.0) < 0
                    md_parts.append(
                        f"- head={kind} vs **{cmp_label}** rival "
                        f"`{c.get('rival')}`: dMAE={st.get('mean', float('nan')):+.6f} "
                        f"95% CI=[{st.get('ci_lo', float('nan')):+.6f}, "
                        f"{st.get('ci_hi', float('nan')):+.6f}] ({sig}); "
                        f"MAE BTR={c.get('mae_btr', float('nan')):.6f} vs "
                        f"rival={c.get('mae_rival', c.get('mae_best_conventional', float('nan'))):.6f} "
                        f"-> **{'BTR wins' if fav else 'rival wins'}**\n")
    else:
        md_parts.append("### Exp1: controlled perturbation\n_(missing -- see above)_\n")

    if exp2 is not None:
        md_parts.append("\n### Exp2: candidate-level natural intervention\n")
        md_parts.append(df_to_markdown(exp2))
        tex_parts.append(df_to_latex_booktabs(
            exp2, caption="BTR Exp2: candidate-level natural-intervention benchmark.",
            label="tab:tab1_exp2"))
    else:
        md_parts.append("\n### Exp2: candidate-level natural intervention\n"
                        "_(missing -- results/tab1_exp2.csv not produced yet)_\n")

    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "tab1.md")
    tex_path = os.path.join(out_dir, "tab1.tex")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_parts))
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(tex_parts) if tex_parts else "% no data\n")
    print(f"[tab1] wrote {md_path}")
    print(f"[tab1] wrote {tex_path}")
    return dict(exp1=exp1, exp1_boot=exp1_boot, exp2=exp2)


# --------------------------------------------------------------------------- #
# Tab.2 -- RiGR main results
# --------------------------------------------------------------------------- #
def _std_metric_frame(df, image_col: str, prefix_map: Dict[str, str],
                      dataset_col: str = "dataset", seed_col: str = "seed"):
    """Rename one producer's per_image.csv onto a common column set:
    dataset, seed, image, cldice, bcs, dice/f1, TRR_recall, FCR, macro_mae."""
    import pandas as pd

    out = pd.DataFrame()
    out["dataset"] = df[dataset_col].astype(str).str.lower()
    out["seed"] = df[seed_col] if seed_col in df.columns else 0
    out["image"] = df[image_col].astype(str)
    for std_name, src_name in prefix_map.items():
        if src_name in df.columns:
            out[std_name] = df[src_name]
        else:
            out[std_name] = np.nan
    for c in ("TRR_recall", "TRR_precision", "FCR", "n_should", "n_accepted",
              "n_true", "n_matched", "TRR_precision_matched", "FCR_matched"):
        out[c] = df[c] if c in df.columns else np.nan
    return add_matched_fcr(out)


def add_matched_fcr(out):
    """Make the **matched** precision / FCR the primary columns.

    DECISIONS.md 2026-09-03 10:20: the reported ``TRR_precision`` / ``FCR`` are
    ``m / |A|`` and ``1 - m / |A|`` -- the exact complement of ``TRR_recall =
    m / N_should``, so precision and recall share their numerator instead of
    being computed by two different criteria.  The edge-level pair
    (``|A_true| / |A|``) is kept as ``TRR_precision_edge`` / ``FCR_edge``.

    Runs produced after the ``src.eval.trr_fcr`` change already carry
    ``TRR_precision_matched`` / ``FCR_matched``; older ones are recomputed here
    from ``n_matched`` / ``n_accepted``, which every producer has always
    written.  The ``|A| = 0`` convention (precision 1, FCR 0) comes from
    :func:`src.eval.trr_fcr.matched_precision_fcr`, and those rows are flagged
    in ``no_accepted_edges``.  Since 2026-09-16 an image that accepted nothing
    has **NaN** precision/FCR rather than 1.0/0.0, so it drops out of any mean
    automatically; report ``n_fcr_defined`` next to an FCR mean.
    """
    import pandas as pd

    from src.eval.trr_fcr import matched_precision_fcr

    na = pd.to_numeric(out.get("n_accepted"), errors="coerce")
    nm = pd.to_numeric(out.get("n_matched"), errors="coerce")
    have = pd.to_numeric(out.get("FCR_matched"), errors="coerce")

    prec, fcr, empty = [], [], []
    for i in range(len(out)):
        a = na.iloc[i] if na is not None else np.nan
        m = nm.iloc[i] if nm is not None else np.nan
        if np.isfinite(a) and np.isfinite(m):
            pv, fv, ev = matched_precision_fcr(m, a)
        elif have is not None and np.isfinite(have.iloc[i]):
            fv = float(have.iloc[i])
            pv, ev = 1.0 - fv, False
        else:
            pv, fv, ev = np.nan, np.nan, False
        prec.append(pv)
        fcr.append(fv)
        empty.append(int(ev))

    out["TRR_precision_edge"] = out["TRR_precision"]
    out["FCR_edge"] = out["FCR"]
    out["TRR_precision_matched"] = prec
    out["FCR_matched"] = fcr
    out["no_accepted_edges"] = empty
    # the primary columns every table and figure reads
    # Fall back to the edge-level pair only where the matched pair could not be
    # computed at all (missing n_matched / n_accepted).  Rows that accepted
    # NOTHING keep NaN: |A| = 0 makes the rate 0/0, and filling it from the
    # edge-level column would reintroduce exactly the "connects nothing =
    # FCR 0" artefact the NaN convention exists to remove.
    # DECISIONS.md 2026-09-16 15:00 (review r3).
    _empty = out["no_accepted_edges"].astype(bool)
    out["TRR_precision"] = out["TRR_precision_matched"].where(
        _empty, out["TRR_precision_matched"].fillna(out["TRR_precision_edge"]))
    out["FCR"] = out["FCR_matched"].where(
        _empty, out["FCR_matched"].fillna(out["FCR_edge"]))
    return out


#: run-directory prefixes under ``runs/rigr/`` that are NOT main-table runs:
#: the Tab.3 ablation variants and the Tab.2 LODO block have their own readers
#: (``build_tab3`` / the LODO section of ``build_tab2``), and letting them fall
#: into ``_load_rigr_mode`` would silently average an ablation into the headline
#: number for its mode.
#: ``eta_sens`` joins them: ``runs/rigr/eta_sens/<eta>/<ds>/seed0`` is the
#: supplementary eta in {0.01, 0.1, 1} sweep of DECISIONS.md 2026-09-06 12:40(6),
#: all of it ``mode=risk``.  Its eta=0.1 arm re-runs the pre-registered setting,
#: so leaving it in would double-count DRIVE/CHASE seed 0 in the headline
#: ``rigr_risk`` row and mix in two off-spec eta values besides.
_NON_MAIN_RIGR_PREFIXES = ("ablation_", "lodo_", "eta_sens")


def _is_main_rigr_run(path: str) -> bool:
    parts = os.path.relpath(path, os.path.join(RUNS_DIR, "rigr")).replace(
        "\\", "/").split("/")
    return not any(seg.startswith(_NON_MAIN_RIGR_PREFIXES) for seg in parts)


def _check_unique(df, what: str):
    """Warn loudly if *df* has more than one row per (dataset, image, seed).

    Every Tab.2 loader is supposed to return exactly one row per image per
    seed; a duplicate means two run directories were merged into one method,
    which over-weights whichever seed was duplicated and inflates the ``n`` of
    every paired test downstream.  That is how the ``evapore_e2e_sweep`` tau
    sweep got averaged into the headline ``evapore_e2e`` row -- silently, and
    visible only as an impossible Wilcoxon n (200 for DRIVE's 60 image-seeds).
    """
    key = [c for c in ("dataset", "image", "seed") if c in df.columns]
    if len(key) < 3:
        return df
    dup = df.duplicated(subset=key).sum()
    if dup:
        srcs = sorted(set(df[df.duplicated(subset=key, keep=False)].get("_src", [])))
        print("[tables][WARNING] %s: %d duplicate (dataset,image,seed) row(s) "
              "from %d source file(s) -- the headline row is over-weighted. "
              "Sources: %s" % (what, dup, len(srcs),
                               ", ".join(os.path.relpath(s, EXP_ROOT)
                                         for s in srcs[:8])))
    return df


def _load_rigr_mode(mode: str) -> Optional["pd.DataFrame"]:
    """All ``per_image.csv`` under ``runs/rigr/**`` whose ``mode`` column
    equals ``mode`` (uniform / risk / prob), standardised.

    Ablation and LODO run directories are excluded (see
    :data:`_NON_MAIN_RIGR_PREFIXES`) -- they are separate experiments that
    happen to write the same ``mode`` value."""
    import pandas as pd

    paths = [q for q in _live_glob(os.path.join(RUNS_DIR, "rigr", "**", "per_image.csv"))
             if _is_main_rigr_run(q)]
    frames = []
    for p in paths:
        df = _read_csv(p)
        if df is None or "mode" not in df.columns:
            continue
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        std = _std_metric_frame(
            sub, "image",
            {"cldice": "after_cldice", "bcs": "after_bcs", "f1": "after_f1",
             "macro_mae": "macro_mae_after"})
        std["_src"] = p
        frames.append(std)
    if not frames:
        return None
    return _check_unique(pd.concat(frames, ignore_index=True),
                        "rigr_%s" % mode)


def _load_baseline_method(method: str) -> Optional["pd.DataFrame"]:
    """The headline ``per_image.csv`` of one baseline, standardised.

    Reads **only** the canonical run layout
    ``runs/repair/<method>/<dataset>/seed<k>/per_image.csv`` -- one row per
    (dataset, image, seed).

    It used to glob ``runs/repair/**`` and keep whatever rows carried the right
    ``method`` value, which silently absorbed any *sibling experiment* that
    reuses the method name.  ``runs/repair/evapore_e2e_sweep/<ds>/seed0/tau*/``
    is exactly that: an acceptance-threshold sweep whose rows are all stamped
    ``method=evapore_e2e``.  Pulling it in both mixed seven off-spec tau
    operating points into the headline row and counted **seed 0 eight times**
    (DRIVE: 200 rows where 20 images x 3 seeds = 60), which then skewed the
    "best baseline" choice the Wilcoxon block tests against.

    Reads the unified schema (``image`` column, ``after_*`` metric prefix)
    ``run_baseline.py`` now writes, including its ``macro_mae_after`` /
    ``bio_*`` biomarker columns when ``--bio`` was not ``none`` (schema items
    1-2); falls back to the pre-unification ``image_id`` / ``rep_*`` schema
    (no biomarker columns at all, so ``macro_mae`` reads as NaN) for runs
    that predate it.
    """
    import pandas as pd

    paths = _live_glob(os.path.join(RUNS_DIR, "repair", method, "*", "seed*",
                                    "per_image.csv"))
    frames = []
    for p in paths:
        df = _read_csv(p)
        if df is None or "method" not in df.columns:
            continue
        sub = df[df["method"].astype(str).str.startswith(method)]
        if sub.empty:
            continue
        image_col = "image" if "image" in sub.columns else "image_id"
        std = _std_metric_frame(
            sub, image_col,
            {"cldice": "after_cldice" if "after_cldice" in sub.columns else "rep_cldice",
             "bcs": "after_bcs" if "after_bcs" in sub.columns else "rep_bcs",
             "f1": "after_f1" if "after_f1" in sub.columns else "rep_f1",
             "macro_mae": "macro_mae_after"})  # absent (pre-unification runs) -> NaN
        std["_src"] = p
        frames.append(std)
    if not frames:
        return None
    return _check_unique(pd.concat(frames, ignore_index=True),
                        "baseline %s" % method)


#: where the unrepaired state is read from, in preference order.  Every repair
#: run stores the pre-repair metrics of the images IT actually processed, so
#: these are the only sources guaranteed to be on the same image set.
_NO_REPAIR_SOURCES = (
    ("rigr_uniform",   (RUNS_DIR, "rigr", "uniform", "*", "seed*")),
    ("rigr_risk",      (RUNS_DIR, "rigr", "risk", "*", "seed*")),
    ("geometric",      (RUNS_DIR, "repair", "geometric", "*", "seed*")),
    ("evapore_scorer", (RUNS_DIR, "repair", "evapore_scorer", "*", "seed*")),
    ("evapore_e2e",    (RUNS_DIR, "repair", "evapore_e2e", "*", "seed*")),
    ("rnca",           (RUNS_DIR, "repair", "rnca", "*", "seed*")),
)


def _before_frame(parts):
    """One row per (dataset, image, seed) of a method's pre-repair metrics."""
    import pandas as pd

    frames = []
    for p in _live_glob(os.path.join(*parts, "per_image.csv")):
        df = _read_csv(p)
        if df is None or "before_cldice" not in df.columns:
            continue
        std = _std_metric_frame(
            df, "image" if "image" in df.columns else "image_id",
            {"cldice": "before_cldice", "bcs": "before_bcs",
             "f1": "before_f1", "macro_mae": "macro_mae_before"})
        # _std_metric_frame copies the TRR / FCR / count columns straight from
        # the source, but those describe the REPAIR this run performed, not the
        # unrepaired state.  no_repair accepts nothing by definition, so every
        # one of them is undefined here; leaving them would have reported
        # rigr_uniform's own TRR/FCR on the no_repair row.
        for c in ("TRR_recall", "TRR_precision", "FCR", "n_should",
                  "n_accepted", "n_true", "n_matched",
                  "TRR_precision_matched", "FCR_matched",
                  "TRR_precision_edge", "FCR_edge"):
            if c in std.columns:
                std[c] = np.nan
        if "no_accepted_edges" in std.columns:
            std["no_accepted_edges"] = True
        std["_src"] = p
        frames.append(std)
    return pd.concat(frames, ignore_index=True) if frames else None


def _load_no_repair() -> Optional["pd.DataFrame"]:
    """The unrepaired state, taken from the repair runs' own ``before_*``.

    It used to come from ``results/seg_per_image.csv``, the full S2 evaluation.
    That file covers **every** test image, while the repair runs are capped --
    FIVES is evaluated on 60 of its 200 test images (``image_caps``). So the
    no_repair row was computed on a different, larger image set than every row
    it was being compared against: FIVES no_repair clDice read 0.9066 while
    every repair method, including ``geometric`` which barely alters a mask,
    read ~0.8694.  Restricting ``seg_per_image.csv`` to the 60 repair images
    gives 0.8694 -- the gap was entirely an image-set artefact, and it made the
    headline "repair costs 3.7 clDice points on FIVES" spurious.

    Every repair run stores the pre-repair metrics of the images it actually
    processed, and all six methods agree on those to < 1e-6 per (image, seed)
    on all four datasets (checked below, and loudly reported if ever false).
    So this reads the union over methods, preferring ``rigr_uniform``, and is
    by construction on exactly the image set the other Tab.2 rows use.
    """
    import pandas as pd

    ref = None
    ref_name = None
    for name, parts in _NO_REPAIR_SOURCES:
        f = _before_frame(parts)
        if f is None:
            continue
        if ref is None:
            ref, ref_name = f, name
            continue
        key = ["dataset", "image", "seed"]
        m = ref.merge(f, on=key, suffixes=("_a", "_b"))
        for c in ("cldice", "bcs", "f1", "macro_mae"):
            a, b = c + "_a", c + "_b"
            if a in m.columns and b in m.columns:
                d = float((m[a] - m[b]).abs().max(skipna=True) or 0.0)
                if np.isfinite(d) and d > 1e-6:
                    print("[tab2][WARNING] no_repair: %s and %s disagree on "
                          "before-repair %s by up to %.3e over %d shared rows "
                          "-- they are not the same unrepaired state"
                          % (ref_name, name, c, d, len(m)))
        # fill datasets/images the reference method never ran
        missing = f.merge(ref[key], on=key, how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
        if len(missing):
            ref = pd.concat([ref, missing], ignore_index=True)
    if ref is None:
        print("[tab2][skip] no_repair: no per_image.csv with before_* columns")
        return None
    return _check_unique(ref, "no_repair")


_TAB2_LOADERS = {
    "no_repair": lambda: _load_no_repair(),
    "geometric": lambda: _load_baseline_method("geometric"),
    "rnca": lambda: _load_baseline_method("rnca"),
    "evapore_e2e": lambda: _load_baseline_method("evapore_e2e"),
    "evapore_scorer": lambda: _load_baseline_method("evapore_scorer"),
    "rigr_uniform": lambda: _load_rigr_mode("uniform"),
    "rigr_risk": lambda: _load_rigr_mode("risk"),
}


def _aggregate_mean_sd(df, group_cols: Sequence[str], value_cols: Sequence[str]):
    import pandas as pd

    # mean over images within (dataset, seed), then mean +/- sd over seeds.
    # pandas .mean() skips NaN, which is what carries the |A| = 0 convention
    # through: an image that accepted nothing has NaN FCR and simply does not
    # enter the FCR mean.  n_fcr_defined records how many (image, seed) rows
    # actually had a defined FCR, so a mean over few images cannot masquerade
    # as a mean over all of them.
    per_seed = df.groupby(list(group_cols) + ["seed"])[list(value_cols)].mean().reset_index()
    agg = per_seed.groupby(list(group_cols))[list(value_cols)].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()
    if "FCR" in df.columns:
        n_def = (df.assign(_d=df["FCR"].notna())
                   .groupby(list(group_cols))["_d"].sum().reset_index()
                   .rename(columns={"_d": "n_fcr_defined"}))
        n_tot = (df.groupby(list(group_cols)).size().reset_index(name="n_rows"))
        agg = agg.merge(n_def, on=list(group_cols), how="left")                  .merge(n_tot, on=list(group_cols), how="left")
    return agg


def build_tab2(results_dir: str = RESULTS_DIR, out_dir: str = RESULTS_DIR,
              n_boot: int = 1000, smoke_label: bool = False) -> Optional[Dict]:
    """Tab.2: RiGR main results (methods x datasets) + LODO block +
    paired per-image Wilcoxon (risk vs uniform, risk vs best baseline)."""
    import pandas as pd

    per_method: Dict[str, "pd.DataFrame"] = {}
    for m in TAB2_METHODS:
        df = _TAB2_LOADERS[m]()
        if df is None:
            print(f"[tab2][skip] method={m}: no matching per_image.csv found")
            continue
        df = df.copy()
        df["method"] = m
        per_method[m] = df

    if not per_method:
        print("[tab2] nothing to build")
        return None

    all_df = pd.concat(per_method.values(), ignore_index=True)
    # Tab.2 is the pre-registered 4-dataset main table.  STARE is a
    # supplementary robustness domain (DECISIONS.md 2026-09-03 15:10) whose
    # risk arm was withdrawn with the C_geom fix and never re-run, so leaving
    # it in would add a row with no ``rigr_risk`` cell; it is reported
    # separately instead.  Say so rather than dropping it silently -- the
    # ``pd.Categorical`` below would otherwise turn it into a NaN dataset.
    extra_ds = sorted(set(all_df["dataset"].dropna().astype(str))
                      - set(TAB2_DATASETS))
    if extra_ds:
        n_extra = int(all_df["dataset"].astype(str).isin(extra_ds).sum())
        print("[tab2] excluding %d row(s) from non-main dataset(s) %s "
              "(Tab.2 is the pre-registered %s table)"
              % (n_extra, ", ".join(extra_ds), "/".join(TAB2_DATASETS)))
        all_df = all_df[~all_df["dataset"].astype(str).isin(extra_ds)]
    # n_accepted: edges actually applied per image.  Reported alongside FCR so a
    # low FCR cannot be read as "precise" when it is really "accepted almost
    # nothing", and so the over-connection question of the bias diagnostics is
    # answerable straight from Tab.2.
    value_cols = ["cldice", "bcs", "TRR_recall", "FCR", "n_accepted",
                  "macro_mae", "f1"]
    agg = _aggregate_mean_sd(all_df, ["method", "dataset"], value_cols)

    # order rows/cols per the pre-registered layout
    agg["method"] = pd.Categorical(agg["method"], categories=TAB2_METHODS, ordered=True)
    agg["dataset"] = pd.Categorical(agg["dataset"], categories=TAB2_DATASETS, ordered=True)
    agg = agg.sort_values(["dataset", "method"]).reset_index(drop=True)

    disp_cols = ["dataset", "method"] + [f"{c}_mean" for c in value_cols]
    disp = agg[disp_cols].copy()
    for c in value_cols:
        disp[f"{c}_sd"] = agg[f"{c}_std"]
    # how many (image, seed) rows the FCR mean is actually over -- an image that
    # accepted nothing has undefined FCR and is skipped (|A| = 0 convention)
    for c in ("n_fcr_defined", "n_rows"):
        if c in agg.columns:
            disp[c] = agg[c]

    # ---- paired per-image Wilcoxon: risk vs uniform, risk vs best baseline
    wil_rows = []
    if "rigr_risk" in per_method and "rigr_uniform" in per_method:
        for ds in TAB2_DATASETS:
            a = per_method["rigr_risk"]
            b = per_method["rigr_uniform"]
            a = a[a["dataset"] == ds]
            b = b[b["dataset"] == ds]
            m = a.merge(b, on=["dataset", "image", "seed"], suffixes=("_risk", "_uniform"))
            if m.empty:
                wil_rows.append(dict(dataset=ds, comparison="risk_vs_uniform",
                                     n=0, note="no overlapping (image, seed)"))
                continue
            for metric in ("macro_mae", "FCR", "cldice"):
                stat, p, n = _wilcoxon(m[f"{metric}_risk"], m[f"{metric}_uniform"])
                wil_rows.append(dict(dataset=ds, comparison="risk_vs_uniform",
                                     metric=metric, n=n, stat=stat, p=p))
    if "rigr_risk" in per_method:
        baseline_candidates = [m for m in ("geometric", "evapore_e2e",
                                           "evapore_scorer", "rnca")
                               if m in per_method]
        if baseline_candidates:
            # "best baseline" per dataset = lowest macro_mae if available,
            # else highest cldice, computed on the aggregated table above
            for ds in TAB2_DATASETS:
                sub = disp[(disp["dataset"] == ds)
                          & (disp["method"].astype(str).isin(baseline_candidates))]
                if sub.empty:
                    continue
                if sub["macro_mae_mean"].notna().any():
                    best = sub.loc[sub["macro_mae_mean"].idxmin(), "method"]
                else:
                    best = sub.loc[sub["cldice_mean"].idxmax(), "method"]
                a = per_method["rigr_risk"]
                b = per_method[str(best)]
                a = a[a["dataset"] == ds]
                b = b[b["dataset"] == ds]
                m = a.merge(b, on=["dataset", "image", "seed"],
                           suffixes=("_risk", f"_{best}"))
                if m.empty:
                    wil_rows.append(dict(dataset=ds,
                                         comparison=f"risk_vs_best_baseline({best})",
                                         n=0, note="no overlapping (image, seed)"))
                    continue
                # macro_mae is the biomarker-fidelity endpoint the claim is
                # actually about (``macro_mae`` here is ``macro_mae_after``,
                # see ``_std_metric_frame``); clDice is the topology endpoint.
                for metric in ("macro_mae", "cldice"):
                    col_b = f"{metric}_{best}" if f"{metric}_{best}" in m.columns else f"{metric}_y"
                    col_a = f"{metric}_risk" if f"{metric}_risk" in m.columns else f"{metric}_x"
                    stat, p, n = _wilcoxon(m[col_a], m[col_b])
                    wil_rows.append(dict(dataset=ds,
                                         comparison=f"risk_vs_best_baseline({best})",
                                         metric=metric, n=n, stat=stat, p=p))
    wil = pd.DataFrame(wil_rows) if wil_rows else None

    # ---- LODO blocks --------------------------------------------------
    # Three separate tables, because runs/rigr/lodo_<D>/ holds three different
    # experiments that all write mode=risk and were previously averaged into one
    # row:
    #   <D>/risk, <D>/uniform          seed 0, deployed R_false head  -> PRIMARY
    #   <D>/risk_c1bridge              seed 0, C1-bridge R_false head -> supp.
    #   <D>/{risk,uniform}_seed{1,2}   extra seeds                    -> supp.
    # The risk arms differ in which R_false head priced the false-repair term
    # (``btr_false_deployed.joblib`` vs ``btr_false_wo_<D>.joblib``), which is a
    # different method, not a repeat -- see summary.json ``risk_backend``.
    def _lodo_frames(arm_filter):
        frames = []
        for p in _live_glob(os.path.join(RUNS_DIR, "rigr", "lodo_*", "*",
                                         "per_image.csv")):
            parts = p.replace("\\", "/").split("/")
            held, arm = parts[-3], parts[-2]
            if not arm_filter(arm):
                continue
            df = _read_csv(p)
            if df is None:
                continue
            std = _std_metric_frame(
                df, "image",
                {"cldice": "after_cldice", "bcs": "after_bcs", "f1": "after_f1",
                 "macro_mae": "macro_mae_after"})
            std["mode"] = df["mode"] if "mode" in df.columns else "unknown"
            std["held_out"] = held
            std["arm"] = arm
            frames.append(std)
        return pd.concat(frames, ignore_index=True) if frames else None

    def _agg(frame, extra_cols=()):
        if frame is None:
            return None
        return _aggregate_mean_sd(frame, ["held_out", "mode", "dataset"] + list(extra_cols),
                                  value_cols)

    lodo = _agg(_lodo_frames(lambda a: a in ("risk", "uniform")))
    lodo_bridge = _agg(_lodo_frames(lambda a: a == "risk_c1bridge"))
    lodo_seeds = _agg(_lodo_frames(
        lambda a: a.startswith(("risk_seed", "uniform_seed"))))
    if lodo is None:
        print("[tab2][skip] LODO: no runs/rigr/lodo_*/{risk,uniform}/per_image.csv")

    # ---- STARE supplementary block ------------------------------------
    # STARE is not in the pre-registered 4-dataset main table (its rows were
    # dropped above), but the corrected risk run exists now, so report it
    # separately rather than discarding it.
    stare = None
    stare_frames = []
    stare_sources = dict(per_method)
    # probability-only is not one of the seven pre-registered Tab.2 methods, but
    # runs/rigr/prob/stare/seed0 exists and the STARE block is where the three
    # RiGR modes can be compared against each other.  No baseline was ever run
    # on STARE, so this block is RiGR-only.
    prob = _load_rigr_mode("prob")
    if prob is not None:
        stare_sources["rigr_prob"] = prob.assign(method="rigr_prob")
    for m, df_m in stare_sources.items():
        sub = df_m[df_m["dataset"].astype(str).str.lower() == "stare"]
        if not sub.empty:
            stare_frames.append(sub.assign(method=m))
    if stare_frames:
        stare = _aggregate_mean_sd(pd.concat(stare_frames, ignore_index=True),
                                   ["method", "dataset"], value_cols)

    # ---- write -----------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    label = " (SMOKE DATA)" if smoke_label else ""
    md = [f"## Tab.2 -- RiGR main results{label}\n", df_to_markdown(disp)]
    tex = [df_to_latex_booktabs(disp, caption="RiGR main results.", label="tab:tab2")]
    if wil is not None:
        md.append("\n### Paired per-image Wilcoxon\n")
        md.append(df_to_markdown(wil))
        tex.append(df_to_latex_booktabs(wil, caption="RiGR paired Wilcoxon tests.",
                                        label="tab:tab2_wilcoxon"))
    else:
        md.append("\n### Paired per-image Wilcoxon\n_(skipped: need both "
                  "rigr_risk and rigr_uniform / a baseline with overlapping images)_\n")
    if lodo is not None:
        md.append("\n### LODO block (primary: seed 0, deployed R_false head)\n")
        md.append(df_to_markdown(lodo))
        tex.append(df_to_latex_booktabs(lodo, caption="RiGR LODO block.",
                                        label="tab:tab2_lodo"))
    else:
        md.append("\n### LODO block\n_(skipped: no runs/rigr/lodo_*/ found)_\n")
    if lodo_bridge is not None:
        md.append("\n### Supplementary: LODO with the C1 bridge R_false head\n")
        md.append("_Same held-out miss head, but the false-repair term is priced "
                  "by `btr_false_wo_<D>.joblib` instead of the deployed "
                  "`btr_false_deployed.joblib` of the primary block._\n")
        md.append(df_to_markdown(lodo_bridge))
        tex.append(df_to_latex_booktabs(
            lodo_bridge, caption="LODO with the C1 bridge R_false head.",
            label="tab:tab2_lodo_c1bridge"))
    if lodo_seeds is not None:
        md.append("\n### Supplementary: LODO seeds 1-2\n")
        md.append(df_to_markdown(lodo_seeds))
        tex.append(df_to_latex_booktabs(lodo_seeds, caption="LODO, seeds 1-2.",
                                        label="tab:tab2_lodo_seeds"))
    if stare is not None:
        md.append("\n### Supplementary: STARE\n")
        md.append("_Not part of the pre-registered 4-dataset main table._\n")
        md.append(df_to_markdown(stare))
        tex.append(df_to_latex_booktabs(stare, caption="STARE (supplementary).",
                                        label="tab:tab2_stare"))

    md_path = os.path.join(out_dir, "tab2.md")
    tex_path = os.path.join(out_dir, "tab2.tex")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(tex))
    print(f"[tab2] wrote {md_path}")
    print(f"[tab2] wrote {tex_path}")
    return dict(disp=disp, wilcoxon=wil, lodo=lodo, per_method=per_method)


# --------------------------------------------------------------------------- #
# Tab.3 -- ablation
# --------------------------------------------------------------------------- #
def build_tab3(results_dir: str = RESULTS_DIR, out_dir: str = RESULTS_DIR
              ) -> Optional[Dict]:
    """Tab.3: ablation from
    ``runs/rigr/ablation_<variant>/<ds>/seed*/per_image.csv``."""
    import pandas as pd

    rows = []
    for variant in TAB3_VARIANTS:
        # '+' is filesystem-legal but let's tolerate both spellings
        stem = variant.lstrip("+")
        pats = [
            os.path.join(RUNS_DIR, "rigr", f"ablation_{variant}", "*", "seed*",
                        "per_image.csv"),
            os.path.join(RUNS_DIR, "rigr", f"ablation_{stem}", "*", "seed*",
                        "per_image.csv"),
        ]
        paths = _live_glob(*pats)
        if not paths:
            print(f"[tab3][skip] variant={variant}: no files matching "
                  f"runs/rigr/ablation_{variant}/<ds>/seed*/per_image.csv")
            continue
        for p in paths:
            df = _read_csv(p)
            if df is None:
                continue
            parts = p.replace("\\", "/").split("/")
            ds_guess = parts[-3] if len(parts) >= 3 else "unknown"
            seed_guess = parts[-2] if len(parts) >= 2 else "unknown"
            std = _std_metric_frame(
                df, "image" if "image" in df.columns else "image_id",
                {"cldice": "after_cldice" if "after_cldice" in df.columns else "rep_cldice",
                 "bcs": "after_bcs" if "after_bcs" in df.columns else "rep_bcs",
                 "f1": "after_f1" if "after_f1" in df.columns else "rep_f1",
                 "macro_mae": "macro_mae_after"})
            if "dataset" not in df.columns:
                std["dataset"] = ds_guess
            std["variant"] = variant
            rows.append(std)

    if not rows:
        print("[tab3] nothing to build")
        return None

    all_df = pd.concat(rows, ignore_index=True)
    value_cols = ["TRR_recall", "FCR", "macro_mae"]
    agg = _aggregate_mean_sd(all_df, ["variant", "dataset"], value_cols)
    agg["variant"] = pd.Categorical(agg["variant"], categories=TAB3_VARIANTS, ordered=True)
    agg = agg.sort_values(["variant", "dataset"]).reset_index(drop=True)

    disp_cols = ["variant", "dataset"] + [f"{c}_mean" for c in value_cols]
    disp = agg[disp_cols].copy()
    for c in value_cols:
        disp[f"{c}_sd"] = agg[f"{c}_std"]

    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "tab3.md")
    tex_path = os.path.join(out_dir, "tab3.tex")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("## Tab.3 -- ablation\n" + df_to_markdown(disp))
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(df_to_latex_booktabs(disp, caption="RiGR ablation.", label="tab:tab3"))
    print(f"[tab3] wrote {md_path}")
    print(f"[tab3] wrote {tex_path}")
    return dict(disp=disp)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    print("=== Tab.1 ===")
    build_tab1()
    print("\n=== Tab.2 ===")
    build_tab2(smoke_label=True)
    print("\n=== Tab.3 ===")
    build_tab3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
