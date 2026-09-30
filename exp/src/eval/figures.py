"""Main-text figures (Fig.2 / Fig.3 / Fig.4 / Fig.5), proposal v3 section 4.3.

Every ``build_fig*`` function is independently callable and robust to
missing inputs, printing a one-line ``[skip]`` message and returning
``None`` instead of raising, mirroring ``tables.py``.

CLI
---
    cd exp
    python -m src.eval.figures
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from src.eval.savefig_util import save_fig

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
RUNS_DIR = os.path.join(EXP_ROOT, "runs")
FIGS_DIR = os.path.join(EXP_ROOT, "figs")

FCR_TARGETS = (0.01, 0.025, 0.05)

# Fig.5 panels, pinned so a regeneration cannot silently change which cases the
# caption describes.  (dataset, image_id); see main() for the per-case counts.
FIG5_CASES = (
    ("drive", "04_test"),
    ("chasedb1", "Image_14L"),
    ("fives", "134_G"),
    ("hrf", "12_h"),
)
MODE_COLORS = {"prob": "#8c8c8c", "uniform": "#5b8def", "risk": "#d1495b"}
MODE_LABELS = {"prob": "probability-only", "uniform": "uniform-event-cost",
              "risk": "risk-guided"}


def _read_csv(path: str):
    import pandas as pd

    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:  # pragma: no cover
        print(f"[figures] failed to read {path}: {e}")
        return None


def _read_json(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _glob(*patterns: str) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(set(out))


def _live_glob(*patterns: str) -> List[str]:
    """:func:`_glob` minus the withdrawn / smoke run directories.

    Same filter the tables use (``src.eval.tables._is_live_run``): the
    pre-C_geom-fix ``*_etascale_pre0906`` runs, the ``*_partial_*`` /
    ``*_moved_*`` stopped runs and the ``_smoke*`` / ``smoke*`` development
    output all sit as siblings of the live run, so a ``runs/rigr/**`` glob
    otherwise plots a withdrawn result next to (or on top of) the real one."""
    from src.eval.tables import _is_live_run

    hits = _glob(*patterns)
    live, dropped = [], []
    for p in hits:
        (live if _is_live_run(p) else dropped).append(p)
    if dropped:
        print("[figures] ignoring %d withdrawn/smoke run file(s): %s%s"
              % (len(dropped),
                 ", ".join(os.path.relpath(d, EXP_ROOT) for d in dropped[:6]),
                 " ..." if len(dropped) > 6 else ""))
    return live


# --------------------------------------------------------------------------- #
# Fig.2 -- BTR dose-response (produced by src.c1.stats_c1, reused as-is)
# --------------------------------------------------------------------------- #
def build_fig2(figs_dir: str = FIGS_DIR) -> Optional[str]:
    """Fig.2 is produced by ``src.c1.stats_c1``; this only checks it exists
    (per the task instructions -- stats_c1.py is the owner of this figure)."""
    path = os.path.join(figs_dir, "fig2_dose_response.png")
    if os.path.exists(path):
        print(f"[fig2] found {path}")
        return path
    print(f"[fig2][skip] {path} not found (run `python -m src.c1.stats_c1` first)")
    return None


# --------------------------------------------------------------------------- #
# Fig.3 -- FCR-benefit tradeoff + matched-FCR statistics
# --------------------------------------------------------------------------- #
def _use_matched_fcr(df, src: str = ""):
    """Replace ``FCR`` by the matched FCR ``1 - m / |A|`` in a sweep frame.

    Fig.3 is an FCR-indexed curve, so it has to use the same FCR definition
    as Tab.2 or the pre-registered operating points would be read off a
    different axis.  DECISIONS.md 2026-09-03 10:20 makes the matched pair
    primary; ``src.eval.tables.add_matched_fcr`` holds the same logic for
    the tables, including the ``|A| = 0`` convention.
    """
    from src.eval.tables import add_matched_fcr

    if "FCR" not in df.columns:
        return df
    if "TRR_precision" not in df.columns:
        df = df.copy()
        df["TRR_precision"] = 1.0 - df["FCR"]
    n_before = df["FCR"].to_numpy(dtype=float, copy=True)
    df = add_matched_fcr(df.copy())
    changed = int(np.sum(~np.isclose(n_before, df["FCR"].to_numpy(dtype=float),
                                     equal_nan=True)))
    if changed and src:
        print("[fig3] %s: %d/%d rows switched to the matched FCR"
              % (os.path.basename(os.path.dirname(src)), changed, len(df)))
    return df


def _sweep_per_image_frames() -> List["pd.DataFrame"]:
    import pandas as pd

    # sweep2-preferred, per (dataset, mode).  The risk and uniform arms have no
    # tau knob, so the original --sweep gives them only four operating points;
    # sweep2 adds the U-threshold axis and widens their reachable FCR range.
    # src/eval/matched_fcr_posthoc.py applies exactly this precedence, and until
    # 2026-09-16 Fig.3 did not -- so the figure was drawn from sweep1 while the
    # matched-FCR table came from sweep2 and the two disagreed about where each
    # arm can operate.  The second knob is carried in the "tau" column whatever
    # it is called on disk, because Fig.3 uses (lam, tau) only as an operating
    # point identifier.
    frames = []
    chosen = {}
    for fname, knob in (("sweep2_per_image.csv", "u0q"),
                        ("sweep_per_image.csv", "tau")):
        for p in _live_glob(os.path.join(RUNS_DIR, "rigr", "**", fname)):
            df = _read_csv(p)
            if df is None:
                continue
            need = {"dataset", "seed", "mode", "image", "lam", knob,
                    "TRR_recall", "FCR"}
            if not need.issubset(df.columns):
                continue
            key = (str(df["dataset"].iloc[0]).lower(),
                   str(df["mode"].iloc[0]).lower())
            if chosen.get(key, fname) != fname:
                continue
            chosen[key] = fname
            df = df.copy()
            if knob != "tau":
                df["tau"] = df[knob]
            if "macro_mae_benefit" not in df.columns:
                if {"macro_mae_before", "macro_mae_after"}.issubset(df.columns):
                    df["macro_mae_benefit"] = (df["macro_mae_before"]
                                               - df["macro_mae_after"])
                else:
                    df["macro_mae_benefit"] = np.nan
            df = _use_matched_fcr(df, p)
            df["_src"] = p
            frames.append(df)
    for k, v in sorted(chosen.items()):
        print("[fig3] %s/%s <- %s" % (k[0], k[1], v))
    return frames


def _sweep_curve_fallback_frames() -> List["pd.DataFrame"]:
    """``sweep_curve.csv`` has no dataset/seed/mode columns of its own
    (see tables.py's module docstring); recover them from the sibling
    ``summary.json`` written into the same run directory, and mark the
    result as curve-level-only (no per-image data -> no image bootstrap)."""
    import pandas as pd

    paths = _live_glob(os.path.join(RUNS_DIR, "rigr", "**", "sweep_curve.csv"))
    frames = []
    for p in paths:
        run_dir = os.path.dirname(p)
        meta = _read_json(os.path.join(run_dir, "summary.json"))
        if meta is None:
            print(f"[fig3][warn] {p} has no sibling summary.json; "
                  "cannot recover dataset/mode, skipped")
            continue
        df = _read_csv(p)
        if df is None:
            continue
        df = df.copy()
        df["dataset"] = str(meta.get("dataset", "unknown"))
        df["mode"] = str(meta.get("mode", "unknown"))
        df["seed"] = meta.get("seed", 0)
        df = _use_matched_fcr(df, p)
        df["_src"] = p
        df["_curve_only"] = True
        frames.append(df)
    return frames


def _all_fcr(df, dataset: Optional[str], modes) -> np.ndarray:
    """Every mean-FCR operating point plotted in one panel, for the
    achievable-range shading of Fig.3."""
    vals = []
    for mode in modes:
        c = _curve_for(df, dataset, mode, "TRR_recall")
        if c is not None and not c.empty:
            vals.append(c["FCR"].to_numpy(dtype=float))
    return np.concatenate(vals) if vals else np.array([np.nan])


def _common_band(df, dataset: Optional[str], modes) -> Optional[tuple]:
    """FCR interval reachable by *every* arm in the panel.

    The union shaded by ``_all_fcr`` says where some arm can be placed; only
    the intersection says where the three arms can be compared at the same
    precision, which is what Conclusion 2 asks for.  Returns ``None`` when the
    arms occupy disjoint ranges (CHASE_DB1, whose uniform arm sits above 0.73).
    """
    lo, hi = -np.inf, np.inf
    n = 0
    for mode in ("prob", "uniform", "risk"):
        if mode not in list(modes):
            continue
        c = _curve_for(df, dataset, mode, "TRR_recall")
        if c is None or c.empty:
            continue
        f = c["FCR"].to_numpy(dtype=float)
        f = f[np.isfinite(f)]
        if f.size == 0:
            continue
        lo, hi, n = max(lo, float(f.min())), min(hi, float(f.max())), n + 1
    if n < 2 or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return (lo, hi)


def _interp_at_fcr(fcr: np.ndarray, val: np.ndarray, target: float) -> float:
    order = np.argsort(fcr)
    x, y = fcr[order], val[order]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")
    if target < x.min() or target > x.max():
        return float("nan")  # outside the achievable range -> no extrapolation
    return float(np.interp(target, x, y))


def _curve_for(df, dataset: Optional[str], mode: str, metric: str,
              agg_over: Sequence[str] = ("lam", "tau")):
    """Mean FCR / metric per operating point, optionally restricted to one
    dataset; aggregated over seeds and images (and datasets, if dataset is
    None)."""
    import pandas as pd

    sub = df[df["mode"] == mode]
    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]
    if sub.empty:
        return None
    cols = ["FCR", metric]
    if "n_accepted" in sub.columns:
        cols.append("n_accepted")
    g = sub.groupby(list(agg_over))[cols].mean().reset_index()
    if "n_accepted" in g.columns:
        # Drop do-nothing operating points.  With |A| = 0 the matched FCR is 0
        # by convention (DECISIONS.md 2026-09-03 10:20), so a cell that repairs
        # nothing plots as a perfect-precision point at FCR = 0 and anchors the
        # curve there -- which makes the pre-specified 1/2.5/5% targets look
        # reachable when nothing reaches them.  matched_fcr_posthoc.py masks
        # these the same way; before 2026-09-16 Fig.3 did not, and its
        # CHASE_DB1 panel announced "unreachable (min 0.00)".
        g = g[~(g["n_accepted"] < 0.5)]
        g = g.drop(columns=["n_accepted"])
    if g.empty:
        return None
    return g.sort_values("FCR")


def _bootstrap_matched_fcr(df, dataset: Optional[str], metric: str,
                           n_boot: int = 1000, seed: int = 0) -> Dict:
    """Image-level bootstrap CI of Delta(benefit) = risk - uniform at each
    pre-registered FCR target, using per-image sweep rows.  Falls back to a
    point estimate (no CI) when ``df`` carries no per-image identity
    (``_curve_only`` rows from sweep_curve.csv)."""
    import pandas as pd

    out = []
    curve_only = bool(df.get("_curve_only", pd.Series([False])).any())
    sub = df if dataset is None else df[df["dataset"] == dataset]
    has_uniform = "uniform" in sub["mode"].unique()
    has_risk = "risk" in sub["mode"].unique()
    if not (has_uniform and has_risk):
        for tgt in FCR_TARGETS:
            out.append(dict(dataset=dataset or "all", metric=metric,
                            fcr_target=tgt, risk_value=np.nan,
                            uniform_value=np.nan, delta=np.nan, ci_lo=np.nan,
                            ci_hi=np.nan, n_boot=0, bootstrap_level="none",
                            note="missing uniform and/or risk mode"))
        return out

    def point_estimate(d):
        c_r = _curve_for(d, dataset, "risk", metric)
        c_u = _curve_for(d, dataset, "uniform", metric)
        vals = {}
        for tgt in FCR_TARGETS:
            r = _interp_at_fcr(c_r["FCR"].to_numpy(), c_r[metric].to_numpy(), tgt) \
                if c_r is not None else float("nan")
            u = _interp_at_fcr(c_u["FCR"].to_numpy(), c_u[metric].to_numpy(), tgt) \
                if c_u is not None else float("nan")
            vals[tgt] = (r, u)
        return vals

    point = point_estimate(sub)

    if curve_only or "image" not in sub.columns:
        for tgt in FCR_TARGETS:
            r, u = point[tgt]
            out.append(dict(dataset=dataset or "all", metric=metric,
                            fcr_target=tgt, risk_value=r, uniform_value=u,
                            delta=r - u if np.isfinite(r) and np.isfinite(u) else np.nan,
                            ci_lo=np.nan, ci_hi=np.nan, n_boot=0,
                            bootstrap_level="curve_only (sweep_curve.csv, "
                                            "no per-image identity)",
                            note=""))
        return out

    rng = np.random.default_rng(int(seed))
    images = sub["image"].astype(str).to_numpy()
    uniq = np.unique(images)
    boots = {tgt: [] for tgt in FCR_TARGETS}
    idx_by_img = {im: np.nonzero(images == im)[0] for im in uniq}
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_img[im] for im in pick])
        rs = sub.iloc[sel]
        vals = point_estimate(rs)
        for tgt in FCR_TARGETS:
            r, u = vals[tgt]
            boots[tgt].append(r - u if np.isfinite(r) and np.isfinite(u) else np.nan)
    for tgt in FCR_TARGETS:
        arr = np.asarray(boots[tgt], dtype=float)
        lo, hi = (np.nanpercentile(arr, [2.5, 97.5]) if np.isfinite(arr).any()
                  else (np.nan, np.nan))
        r, u = point[tgt]
        out.append(dict(dataset=dataset or "all", metric=metric, fcr_target=tgt,
                        risk_value=r, uniform_value=u,
                        delta=r - u if np.isfinite(r) and np.isfinite(u) else np.nan,
                        ci_lo=float(lo), ci_hi=float(hi), n_boot=int(n_boot),
                        bootstrap_level="image", note=""))
    return out


def build_fig3(results_dir: str = RESULTS_DIR, figs_dir: str = FIGS_DIR,
               n_boot: int = 1000, smoke_label: bool = False) -> Optional[Dict]:
    """Fig.3: FCR-TRR_recall and FCR-biomarker-benefit curves for
    probability-only / uniform / risk-guided, plus matched-FCR statistics
    written to ``results/fig3_matched_fcr.csv``."""
    import pandas as pd

    frames = _sweep_per_image_frames()
    used_curve_only = False
    if not frames:
        frames = _sweep_curve_fallback_frames()
        used_curve_only = True
        if not frames:
            print("[fig3][skip] no sweep_per_image.csv or sweep_curve.csv found "
                  "under runs/rigr/** (run `run_rigr.py --sweep`)")
            return None
        print("[fig3][warn] using sweep_curve.csv fallback: no per-image "
              "identity, so bootstrap CIs are unavailable (point estimates only)")
    df = pd.concat(frames, ignore_index=True)
    if "_curve_only" not in df.columns:
        df["_curve_only"] = used_curve_only

    datasets = sorted(df["dataset"].unique())
    modes_present = sorted(df["mode"].unique())
    print(f"[fig3] datasets={datasets} modes={modes_present}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("TRR_recall", "TRR_recall"), ("macro_mae_benefit", "biomarker benefit")]
    panels = datasets + ["all"]
    fig, axes = plt.subplots(len(metrics), len(panels),
                             figsize=(3.2 * len(panels), 3.0 * len(metrics)),
                             squeeze=False)
    for mi, (metric, ylabel) in enumerate(metrics):
        for pi, ds in enumerate(panels):
            ax = axes[mi][pi]
            ds_arg = None if ds == "all" else ds
            any_line = False
            for mode in ("prob", "uniform", "risk"):
                if mode not in modes_present:
                    continue
                c = _curve_for(df, ds_arg, mode, metric)
                if c is None or c.empty:
                    continue
                ax.plot(c["FCR"], c[metric], marker="o", ms=3,
                       color=MODE_COLORS[mode], label=MODE_LABELS[mode])
                any_line = True
            if not any_line:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8)
            else:
                # Mark the pre-registered operating points and say plainly that
                # they are unreachable.  Every arm bottoms out around FCR 0.3-0.7
                # (the only lower points are degenerate |A| = 0 ones), so the
                # 1/2.5/5% targets sit far to the left of any curve and
                # results/fig3_matched_fcr.csv is empty for that reason -- not
                # because runs are missing.  results/fig3_matched_fcr_posthoc.csv
                # redoes the comparison inside the achievable range.
                lo = float(np.nanmin(_all_fcr(df, ds_arg, modes_present)))
                hi = float(np.nanmax(_all_fcr(df, ds_arg, modes_present)))
                ax.axvspan(lo, hi, color="#2e7d32", alpha=0.07, zorder=0)
                # The union shaded above says where *some* arm can operate.
                # Conclusion 2 needs the intersection: the band on which all
                # three arms can be placed at the same FCR.  It is empty on
                # CHASE_DB1, and that is marked rather than left blank.
                band = _common_band(df, ds_arg, modes_present)
                if band is not None:
                    ax.axvspan(band[0], band[1], color="#1565c0", alpha=0.13,
                               zorder=0)
                elif mi == 0:
                    ax.text(0.98, 0.04, "no common band",
                            transform=ax.transAxes, fontsize=6, ha="right",
                            va="bottom", color="#1565c0")
                for tgt in FCR_TARGETS:
                    ax.axvline(tgt, color="#b71c1c", ls=":", lw=0.8, zorder=1)
                if mi == 0:
                    ax.text(0.02, 0.97,
                            "pre-spec FCR %s\nunreachable (min %.2f)"
                            % ("/".join(f"{t:g}" for t in FCR_TARGETS), lo),
                            transform=ax.transAxes, fontsize=6, va="top",
                            color="#b71c1c")
            ax.set_title(ds, fontsize=9)
            if mi == len(metrics) - 1:
                ax.set_xlabel("FCR")
            if pi == 0:
                ax.set_ylabel(ylabel)
            if mi == 0 and pi == len(panels) - 1:
                ax.legend(fontsize=7)
    # No suptitle: the figure is included in a LaTeX float that numbers and
    # titles it, and an embedded "Fig.3" contradicts whatever number the
    # document assigns.  Panel titles stay.  A provenance warning is still
    # stamped when the data are not the real thing.
    if smoke_label or used_curve_only:
        fig.text(0.5, 0.995,
                 "SMOKE DATA" if smoke_label else "curve-level data only: no CIs",
                 ha="center", va="top", fontsize=11, color="#b71c1c")
    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    out_path = os.path.join(figs_dir, "fig3_fcr_tradeoff.png")
    save_fig(fig, out_path, dpi=150)
    plt.close(fig)
    print(f"[fig3] wrote {out_path}")

    # ---- matched-FCR statistics ----------------------------------------
    rows = []
    for ds in panels:
        ds_arg = None if ds == "all" else ds
        for metric in ("TRR_recall", "macro_mae_benefit"):
            rows.extend(_bootstrap_matched_fcr(df, ds_arg, metric, n_boot=n_boot))
    stats = pd.DataFrame(rows)
    stats_path = os.path.join(results_dir, "fig3_matched_fcr.csv")
    os.makedirs(results_dir, exist_ok=True)
    stats.to_csv(stats_path, index=False)
    print(f"[fig3] wrote {stats_path}")
    return dict(fig=out_path, stats=stats)


# --------------------------------------------------------------------------- #
# Fig.4 -- robustness (delegates to src.robust.fig4)
# --------------------------------------------------------------------------- #
def build_fig4(results_dir: str = RESULTS_DIR, figs_dir: str = FIGS_DIR
              ) -> Optional[str]:
    """Fig.4: calls ``src.robust.fig4.main`` directly.

    Location reconciliation (schema item 5): ``src.robust.fig4`` expects
    ``fig4a_stability_*_per_image.csv`` and ``fig4b_resolution_<ds>_drift.csv``
    inside ``results_dir`` itself.  ``src.robust.run_stability`` /
    ``run_resolution`` now write those two summary files into ``results/``
    directly (in addition to wherever ``--out-dir`` points, e.g. a
    ``runs/robust/<name>/`` detail directory holding the bulkier per-view /
    per-image CSVs), so this copy step is normally a no-op.  It is kept as a
    defensive fallback for runs produced before that reconciliation (or with
    a hand-rolled ``--out-dir``): any matching file still sitting under
    ``runs/robust/**`` and missing from ``results_dir`` is copied in before
    calling ``fig4.main``, and the panels that end up available are reported.
    """
    import shutil

    os.makedirs(results_dir, exist_ok=True)
    copied = []
    for pat in ("fig4a_stability_*_per_image.csv", "fig4b_resolution_*_drift.csv"):
        for src in _live_glob(os.path.join(RUNS_DIR, "robust", "**", pat)):
            dst = os.path.join(results_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(dst)
    if copied:
        print(f"[fig4] copied {len(copied)} file(s) from runs/robust/** into "
              f"{results_dir} (see build_fig4 docstring for why this is needed): "
              + ", ".join(os.path.basename(c) for c in copied))

    # pick a resolution dataset that actually has a drift file, defaulting to hrf
    res_candidates = _glob(os.path.join(results_dir, "fig4b_resolution_*_drift.csv"))
    res_ds = "hrf"
    if res_candidates and not os.path.exists(
            os.path.join(results_dir, "fig4b_resolution_hrf_drift.csv")):
        import re

        m = re.search(r"fig4b_resolution_(.+)_drift\.csv$",
                      os.path.basename(res_candidates[0]))
        if m:
            res_ds = m.group(1)

    from src.robust import fig4 as fig4_mod

    out_path = os.path.join(figs_dir, "fig4_robustness.png")
    os.makedirs(figs_dir, exist_ok=True)
    rc = fig4_mod.main(["--results-dir", results_dir,
                        "--resolution-dataset", res_ds, "--out", out_path])
    if rc != 0:
        print(f"[fig4][skip] src.robust.fig4.main returned {rc}")
        return None
    print(f"[fig4] wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Fig.5 -- qualitative panels
# --------------------------------------------------------------------------- #
# Schema (schema item 6, now written by ``src.rigr.run_rigr``):
#
#   <out>/edges/<image>.csv, one row per **candidate** edge (accepted AND
#   rejected -- <out> is whatever --out the run used, e.g.
#   runs/rigr/<mode>/<ds>/seed<seed>), columns:
#     cand_id        int    matches cand_<image>.csv's cand_id
#     accepted       bool   True iff selected by the max-weight matching
#                           (and not vetoed)
#     is_true_repair bool   src.eval.trr_fcr.classify_edges's label, only
#                           meaningful (and only ever True) for accepted rows
#     dangerous      bool   a *rejected* candidate whose false-connection
#                           risk R_false is in the image's top decile, or
#                           that was vetoed (crossing / corridor overlap)
#     p, U           float  calibrated acceptance probability / utility
#     path_row, path_col    JSON-encoded list of the rasterised A* path's
#                           pixel rows / columns (``[]`` when A* failed)
#     radius         float  tube radius used to rasterise the path
#
#   ``run_rigr.py`` also now saves the **pre-repair** mask to
#   ``<out>/mask_before/<image>.png`` (alongside the post-repair
#   ``<out>/mask/<image>.png`` it always wrote), so Fig.5 can show both sides
#   of a RiGR repair without re-running the pipeline.
#
# ``src.baselines.run_baseline`` persists accepted-edge paths only
# (``edges/<image>.npz``: n, paths, lengths, radius, score) with no
# ``is_true_repair`` label stored (only computed inside ``evaluate_repair()``
# and discarded there) -- build_fig5 recomputes it from the accepted paths
# via ``src.eval.trr_fcr.classify_edges`` (exact, since those paths are
# available) for that source.
#
# build_fig5 therefore prefers RiGR's edges/*.csv (full accept/reject/danger
# picture) and falls back to the baseline npz path when no RiGR edges exist.
def _classify_baseline_edges(npz_path: str, ds: str, image_id: str,
                             base_mask_path: str):
    import cv2

    from src.data.datasets import load_dataset, read_binary
    from src.eval.trr_fcr import CandidateEdge, classify_edges

    recs = {r["image_id"]: r for r in
           (load_dataset(ds, split="train") + load_dataset(ds, split="test")
            + load_dataset(ds, split="val"))}
    rec = recs.get(image_id)
    if rec is None:
        return None
    gt = read_binary(rec["label_path"]) > 0
    base = (cv2.imread(base_mask_path, cv2.IMREAD_GRAYSCALE) > 0)

    d = np.load(npz_path, allow_pickle=True)
    n = int(d["n"])
    if n == 0:
        return [], gt, base, rec
    paths, lengths, radius, score = d["paths"], d["lengths"], d["radius"], d["score"]
    edges = []
    off = 0
    for i in range(n):
        ln = int(lengths[i])
        p = paths[off: off + ln]
        off += ln
        edges.append(CandidateEdge(path=p, radius=float(radius[i]), edge_id=i,
                                   score=float(score[i])))
    recs_cls = classify_edges(edges, base, gt)
    return recs_cls, gt, base, rec, edges


def _draw_case_panel(fig, gs_row, ds: str, image_id: str, image_path: str,
                     gt: np.ndarray, base: np.ndarray, repaired: np.ndarray,
                     accepted_edges: Sequence, rejected_edges: Sequence,
                     title_suffix: str = ""):
    import matplotlib.pyplot as plt

    from src.data.datasets import read_image

    img = read_image(image_path)
    axes = gs_row
    axes[0].imshow(img)
    axes[0].set_title(f"{ds}/{image_id} image", fontsize=8)
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT", fontsize=8)
    axes[2].imshow(base, cmap="gray")
    axes[2].set_title("baseline mask", fontsize=8)
    axes[3].imshow(repaired if repaired is not None else base, cmap="gray")
    for rec, edge in accepted_edges:
        color = "#2e7d32" if rec["is_true_repair"] else "#c62828"
        p = edge.path
        axes[3].plot(p[:, 1], p[:, 0], color=color, lw=1.5)
    for edge in rejected_edges:
        p = edge.path
        axes[3].plot(p[:, 1], p[:, 0], color="#c62828", lw=1.0, ls="--")
    axes[3].set_title("repaired: accepted (green=true/red=false), "
                      "rejected (dashed)" + title_suffix, fontsize=7)
    for ax in axes:
        ax.axis("off")


def _load_rigr_edges(csv_path: str) -> List[dict]:
    """Parse one ``edges/<image>.csv`` (schema item 6) into a list of dicts
    with ``path`` as an ``(n, 2)`` int array (``path_row``/``path_col``
    JSON-decoded and zipped back together)."""
    import json

    df = _read_csv(csv_path)
    if df is None:
        return []
    edges = []
    for r in df.itertuples(index=False):
        try:
            pr = json.loads(r.path_row) if isinstance(r.path_row, str) else []
            pc = json.loads(r.path_col) if isinstance(r.path_col, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            pr, pc = [], []
        path = (np.array(list(zip(pr, pc)), dtype=np.int64) if pr
               else np.zeros((0, 2), dtype=np.int64))
        edges.append(dict(
            cand_id=int(r.cand_id), accepted=bool(r.accepted),
            is_true_repair=bool(r.is_true_repair), dangerous=bool(r.dangerous),
            p=float(r.p), U=float(r.U), radius=float(r.radius), path=path,
        ))
    return edges


def _rigr_found(results_dir: str) -> List[Tuple[str, str, str, str]]:
    """``(dataset, image, edges_csv_path, run_dir)`` for every
    ``runs/rigr/**/edges/*.csv`` whose dataset can be resolved from the
    sibling ``per_image.csv``."""
    found = []
    for ep in _live_glob(os.path.join(RUNS_DIR, "rigr", "**", "edges", "*.csv")):
        image_id = os.path.splitext(os.path.basename(ep))[0]
        run_dir = os.path.dirname(os.path.dirname(ep))
        per_image = _read_csv(os.path.join(run_dir, "per_image.csv"))
        ds = None
        if per_image is not None and "dataset" in per_image.columns:
            key = "image" if "image" in per_image.columns else "image_id"
            if key in per_image.columns:
                row = per_image[per_image[key].astype(str) == image_id]
                if not row.empty:
                    ds = str(row.iloc[0]["dataset"])
        if ds is None:
            continue
        found.append((ds, image_id, ep, run_dir))
    return found


def _draw_rigr_panel(axes, ds: str, image_id: str, run_dir: str, edge_csv_path: str):
    """Draw one RiGR case: image / GT / pre-repair mask / repaired mask with
    the edges/*.csv overlay (green=true accept, red=false accept, amber
    dashed=dangerous reject)."""
    import cv2

    from src.data.datasets import load_dataset, read_binary, read_image

    edges = _load_rigr_edges(edge_csv_path)
    recs = {r["image_id"]: r for r in
           (load_dataset(ds, split="train") + load_dataset(ds, split="test")
            + load_dataset(ds, split="val"))}
    rec = recs.get(image_id)
    if rec is None:
        # FIVES records are keyed "test_134_G" while its runs write the bare
        # "134_G", so an exact lookup never resolves a FIVES case and Fig.5
        # could not show that dataset at all (it silently fell back to four
        # CHASE_DB1 panels).  Try the split-prefixed key before giving up.
        for split in ("test", "val", "train"):
            rec = recs.get("%s_%s" % (split, image_id))
            if rec is not None:
                break
    if rec is None:
        raise FileNotFoundError(f"no dataset record for {ds}/{image_id}")
    gt = read_binary(rec["label_path"]) > 0

    before_path = os.path.join(run_dir, "mask_before", image_id + ".png")
    after_path = os.path.join(run_dir, "mask", image_id + ".png")
    before = (cv2.imread(before_path, cv2.IMREAD_GRAYSCALE) > 0
             if os.path.exists(before_path) else np.zeros_like(gt))
    after = (cv2.imread(after_path, cv2.IMREAD_GRAYSCALE) > 0
            if os.path.exists(after_path) else before)

    img = read_image(rec["image_path"])
    axes[0].imshow(img)
    axes[0].set_title(f"{ds}/{image_id} image", fontsize=8)
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT", fontsize=8)
    axes[2].imshow(before, cmap="gray")
    axes[2].set_title("pre-repair mask", fontsize=8)
    axes[3].imshow(after, cmap="gray")
    n_dangerous = 0
    for e in edges:
        if e["path"].shape[0] == 0:
            continue
        if e["accepted"]:
            color = "#2e7d32" if e["is_true_repair"] else "#c62828"
            axes[3].plot(e["path"][:, 1], e["path"][:, 0], color=color, lw=1.5)
        elif e["dangerous"]:
            axes[3].plot(e["path"][:, 1], e["path"][:, 0], color="#f9a825",
                         lw=1.0, ls="--")
            n_dangerous += 1
    axes[3].set_title("RiGR repaired: accepted (green=true/red=false),\n"
                      f"dangerous rejected (amber dashed, n={n_dangerous})",
                      fontsize=7)
    for ax in axes:
        ax.axis("off")


def build_fig5(cases: Optional[Sequence[Tuple[str, str]]] = None,
              results_dir: str = RESULTS_DIR, figs_dir: str = FIGS_DIR
              ) -> Optional[str]:
    """Fig.5: qualitative panels for a handful of ``(dataset, image)`` cases.

    Prefers RiGR's ``runs/rigr/**/edges/*.csv`` (schema item 6: accepted,
    rejected and dangerous candidates, with the true/false-repair label);
    falls back to ``runs/repair/**/<method>/edges/*.npz`` (accepted edges
    only, baseline methods) for any remaining slot.  ``cases=None``
    auto-picks up to 4 cases, RiGR first.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rigr = [("rigr",) + f for f in _rigr_found(results_dir)]
    baseline = []
    for ep in _live_glob(os.path.join(RUNS_DIR, "repair", "**", "edges", "*.npz")):
        image_id = os.path.splitext(os.path.basename(ep))[0]
        method_dir = os.path.dirname(os.path.dirname(ep))
        mask_path = os.path.join(method_dir, "mask", image_id + ".png")
        per_image = _read_csv(os.path.join(method_dir, "per_image.csv"))
        ds = None
        if per_image is not None and "dataset" in per_image.columns:
            key = "image" if "image" in per_image.columns else "image_id"
            if key in per_image.columns:
                row = per_image[per_image[key].astype(str) == image_id]
                if not row.empty:
                    ds = str(row.iloc[0]["dataset"])
        if ds is None or not os.path.exists(mask_path):
            continue
        baseline.append(("baseline", ds, image_id, ep, mask_path))

    found = rigr + baseline
    if not found:
        print("[fig5][skip] no runs/rigr/**/edges/*.csv or "
              "runs/repair/**/edges/*.npz found")
        return None

    if cases is not None:
        wanted = set(cases)
        found = [f for f in found if (f[1], f[2]) in wanted]
        if not found:
            print(f"[fig5][skip] none of the requested cases {cases} have "
                  "edges output available")
            return None

    found = found[:4]
    n = len(found)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    ok = 0
    for row_i, entry in enumerate(found):
        kind, ds, image_id = entry[0], entry[1], entry[2]
        try:
            if kind == "rigr":
                _, _, _, edge_csv_path, run_dir = entry
                _draw_rigr_panel(axes[row_i], ds, image_id, run_dir, edge_csv_path)
            else:
                _, _, _, npz_path, mask_path = entry
                recs_cls, gt, base, rec, edges = _classify_baseline_edges(
                    npz_path, ds, image_id, mask_path)
                accepted = list(zip(recs_cls, edges))
                _draw_case_panel(fig, axes[row_i], ds, image_id, rec["image_path"],
                                 gt, base, None, accepted, [])
        except Exception as e:
            print(f"[fig5][warn] failed on {kind}/{ds}/{image_id}: {e}")
            for ax in axes[row_i]:
                ax.axis("off")
            continue
        ok += 1

    if ok == 0:
        plt.close(fig)
        print("[fig5][skip] no case could be rendered")
        return None

    # No suptitle -- the LaTeX float numbers and titles this figure.
    fig.tight_layout()
    os.makedirs(figs_dir, exist_ok=True)
    out_path = os.path.join(figs_dir, "fig5_qualitative.png")
    save_fig(fig, out_path, dpi=150)
    plt.close(fig)
    print(f"[fig5] wrote {out_path} ({ok}/{len(found)} case(s))")
    return out_path


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    print("=== Fig.2 ===")
    build_fig2()
    print("\n=== Fig.3 ===")
    # smoke_label stamps "(SMOKE DATA)" across the title; it was left on from an
    # early smoke run and put that stamp on the figure the paper includes.
    build_fig3(smoke_label=False)
    print("\n=== Fig.4 ===")
    build_fig4()
    print("\n=== Fig.5 ===")
    # Pinned, not auto-picked.  cases=None takes whatever _rigr_found()
    # enumerates first, which was all four CHASE_DB1 images -- one dataset, no
    # false connections, so the panel the paper needs most (a case where the
    # repair is wrong) was not in the figure at all.  These four are one per
    # dataset and span the behaviour the text describes; counts are accepted /
    # true / false / dangerous-rejected from runs/rigr/risk/*/seed0/edges:
    #   drive/04_test        8 /  8 /  0 / 49   clean success
    #   chasedb1/Image_14L   3 /  3 /  0 / 81   extreme conservatism
    #   fives/134_G          5 /  5 /  0 /  6   few candidates, all correct
    #   hrf/12_h            55 / 15 / 40 / 128  failure: most edges are false
    build_fig5(cases=FIG5_CASES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
