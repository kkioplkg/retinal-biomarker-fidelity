"""Fig.4: three-axis robustness figure (proposal v3 section 4.3).

Panel A -- within-image biomarker CV under acquisition perturbations,
          baseline vs repaired (:mod:`src.robust.run_stability` output).
Panel B -- HRF resolution drift, 75% / 50% / DRIVE-equivalent vs 100%
          (:mod:`src.robust.run_resolution` output).
Panel C -- annotation reproducibility: observer-observer vs model-observer
          biomarker discrepancy (:mod:`src.robust.observer_ceiling` output).

Every panel is **independently optional**: if its CSV(s) are missing the
panel is replaced with a "data not available" placeholder and a note is
printed, so this can be (and is, at authoring time) run before the GPU stages
that produce panels A/B have been executed.

CLI
---
    python -m src.robust.fig4 --results-dir results --out figs/fig4_robustness.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
from src.eval.savefig_util import save_fig

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
FIGS_DIR = os.path.join(EXP_ROOT, "figs")

PRIMARY = ("FD", "tortuosity", "density", "total_length")


def _primary_cols() -> List[str]:
    from src.bio.biomarkers import primary_columns

    return primary_columns(zone=False)


# --------------------------------------------------------------------------- #
# panel A: stability CV, baseline vs repaired
# --------------------------------------------------------------------------- #
def _read_csv_or_none(path: str):
    """Read a CSV, returning None for a missing/empty/unparsable file.

    Stage files are pre-created as empty stubs before the run that fills them
    (e.g. the fig4a ``rigr_cmd`` stability files exist at 2 bytes while the
    repaired sweep is still pending).  Without this guard pandas raises
    EmptyDataError and the whole figure fails to build, when the correct
    behaviour is to draw the panels whose data *is* ready.
    """
    import pandas as pd

    if not path or not os.path.exists(path) or os.path.getsize(path) < 3:
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[fig4] skipping {os.path.basename(path)}: {e}")
        return None
    return None if df.empty else df


def _load_panel_a(results_dir: str):
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(results_dir, "fig4a_stability_*_per_image.csv")))
    if not paths:
        return None
    pat = re.compile(r"fig4a_stability_(?P<dataset>.+)_(?P<repair>none|rigr_cmd)_per_image\.csv$")
    frames = []
    for p in paths:
        m = pat.search(os.path.basename(p))
        df = _read_csv_or_none(p)
        if df is None:
            continue
        if m and "dataset" not in df.columns:
            df["dataset"] = m.group("dataset")
        if m and "repair" not in df.columns:
            df["repair"] = m.group("repair")
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _draw_panel_a(ax, df):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "Fig.4A data not available\n(run src.robust.run_stability)",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.axis("off")
        return
    import pandas as pd

    cols = [c for c in _primary_cols() if c in df["biomarker"].unique().tolist()]
    biomarkers = cols if cols else sorted(df["biomarker"].unique())
    repairs = sorted(df["repair"].unique())
    x = np.arange(len(biomarkers))
    width = 0.8 / max(1, len(repairs))
    colors = {"none": "#9aa5b1", "rigr_cmd": "#2e7d6b"}
    for i, rep in enumerate(repairs):
        means, sds = [], []
        for b in biomarkers:
            v = pd.to_numeric(
                df[(df["repair"] == rep) & (df["biomarker"] == b)]["cv"], errors="coerce"
            ).dropna()
            means.append(float(v.mean()) if len(v) else np.nan)
            sds.append(float(v.std()) if len(v) > 1 else 0.0)
        ax.bar(x + i * width, means, width=width, yerr=sds, capsize=3,
               label=rep, color=colors.get(rep, None))
    ax.set_xticks(x + width * (len(repairs) - 1) / 2)
    ax.set_xticklabels(biomarkers, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("within-image CV")
    ax.set_title("A. acquisition-perturbation stability")
    ax.legend(fontsize=8, title="repair")


# --------------------------------------------------------------------------- #
# panel B: HRF resolution drift
# --------------------------------------------------------------------------- #
def _load_panel_b(results_dir: str, dataset: str):
    import pandas as pd

    path = os.path.join(results_dir, f"fig4b_resolution_{dataset}_drift.csv")
    if not os.path.exists(path):
        return None
    return _read_csv_or_none(path)


def _draw_panel_b(ax, df):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "Fig.4B data not available\n(run src.robust.run_resolution)",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.axis("off")
        return
    level_order = ["75", "50", "driveeq"]
    present = set(df["level"].tolist()) if "level" in df.columns else set()
    order = [l for l in level_order if l in present]
    df = df.set_index("level").reindex(order) if order else df.set_index("level")
    x = np.arange(len(df.index))
    for name in PRIMARY:
        col = f"mean_delta_{name}_macro"
        if col not in df.columns:
            continue
        ax.plot(x, df[col].to_numpy(dtype=float), marker="o", label=name)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index.tolist())
    ax.set_xlabel("resolution level (vs 100%)")
    ax.set_ylabel("mean signed delta")
    ax.set_title("B. HRF resolution drift")
    ax.legend(fontsize=8)


# --------------------------------------------------------------------------- #
# panel C: annotation reproducibility
# --------------------------------------------------------------------------- #
def _load_panel_c(results_dir: str):
    import pandas as pd

    obs_path = os.path.join(results_dir, "fig4c_observer.csv")
    if not os.path.exists(obs_path):
        return None, None
    obs = _read_csv_or_none(obs_path)
    if obs is None:
        return None
    model_path = os.path.join(results_dir, "fig4c_model_vs_observer_summary.csv")
    model = _read_csv_or_none(model_path)
    return obs, model


def _draw_panel_c(ax, obs, model):
    if obs is None or obs.empty:
        ax.text(0.5, 0.5, "Fig.4C data not available\n(run src.robust.observer_ceiling)",
                ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.axis("off")
        return
    import pandas as pd

    z_cols = [c for c in obs.columns if c.startswith("z_absdiff_")]
    if not z_cols:
        z_cols = [c for c in obs.columns if c.startswith("absdiff_")]
    datasets = sorted(obs["dataset"].unique())
    data = []
    for ds in datasets:
        g = obs[obs["dataset"] == ds]
        vals = pd.concat([pd.to_numeric(g[c], errors="coerce") for c in z_cols])
        data.append(vals.dropna().to_numpy(dtype=float))
    bp = ax.boxplot(data, tick_labels=datasets, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3dc")
    ax.set_ylabel("|obs1 - obs2| biomarker diff"
                  + (" (MAD-standardised)" if z_cols and z_cols[0].startswith("z_") else ""))
    ax.set_title("C. annotation reproducibility")
    if model is not None and not model.empty and "mean_diff" in model.columns:
        txt = "model vs obs2 (paired vs obs1):\n" + "\n".join(
            f"{r.metric}: {r.mean_diff:+.3f}" for r in model.itertuples()
        )
        ax.text(0.98, 0.98, txt, ha="right", va="top", fontsize=7, transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fig.4: robustness multi-panel figure")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--resolution-dataset", default="hrf")
    ap.add_argument("--out", default=os.path.join(FIGS_DIR, "fig4_robustness.png"))
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    a_df = _load_panel_a(args.results_dir)
    _draw_panel_a(axes[0], a_df)

    b_df = _load_panel_b(args.results_dir, args.resolution_dataset)
    _draw_panel_b(axes[1], b_df)

    c_obs, c_model = _load_panel_c(args.results_dir)
    _draw_panel_c(axes[2], c_obs, c_model)

    # No suptitle -- the LaTeX float numbers and titles this figure, and an
    # embedded "Fig.4" contradicts whatever number the document assigns.
    # Panel titles stay.
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_fig(fig, args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  panel A: {'ok' if a_df is not None else 'MISSING'}")
    print(f"  panel B: {'ok' if b_df is not None else 'MISSING'}")
    print(f"  panel C: {'ok' if c_obs is not None else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
