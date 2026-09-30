"""Bias diagnostics: does macro-MAE reward over-connection?

Hypothesis under test (coordinator, 2026-09-16). Retinal segmentations
systematically *under*-estimate length and density, so the signed error
``(B_pred - B_GT)/sigma`` is negative before repair. If that is so, then **any**
method that adds vessel pixels moves those biomarkers toward the ground truth
and lowers macro-MAE -- even when the added pixels are false connections. Under
that reading macro-MAE is not measuring repair quality, it is measuring
"how much did you add", and a method with FCR 0.9 can win it.

What this computes, per (dataset, method):

* (a) mean signed bias ``(B_pred - B_GT)/sigma`` per primary biomarker,
  before and after repair. ``before`` is the unrepaired prediction and is
  identical for every method, so it doubles as the ``no_repair`` row.
* (b) mean accepted edges and mean foreground pixels added per image.
* (c) Spearman correlation across images between pixels added and the change in
  macro-MAE. **Negative = adding pixels lowers macro-MAE**, which is the
  signature the hypothesis predicts.

sigma is the Gate A single-source MAD scale in
``results/gateA_biomarker_scales.csv`` -- the same lookup
``src.eval.biomarker_eval.macro_mae`` uses, so the units here are exactly the
units of the macro-MAE column in Tab.2.

CLI
---
    cd exp
    python -m src.eval.bias_diagnostics
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .biomarker_eval import GATE_A_SCALES_CSV, PRIMARY
from .tables import (EXP_ROOT, RESULTS_DIR, RUNS_DIR, _live_glob, _read_csv,
                     df_to_markdown)

PIPELINES = ("pvbm", "skan")
#: method -> glob of its per-image files. rigr modes come from runs/rigr/<mode>,
#: baselines from the canonical runs/repair/<method> layout (never ``**`` -- see
#: ``tables._load_baseline_method`` for why).
METHOD_GLOBS = {
    "geometric":      (RUNS_DIR, "repair", "geometric", "*", "seed*"),
    "rnca":           (RUNS_DIR, "repair", "rnca", "*", "seed*"),
    "evapore_e2e":    (RUNS_DIR, "repair", "evapore_e2e", "*", "seed*"),
    "evapore_scorer": (RUNS_DIR, "repair", "evapore_scorer", "*", "seed*"),
    "rigr_uniform":   (RUNS_DIR, "rigr", "uniform", "*", "seed*"),
    "rigr_risk":      (RUNS_DIR, "rigr", "risk", "*", "seed*"),
}
MAIN_DATASETS = ("drive", "chasedb1", "hrf", "fives")


def _sigma_lookup() -> Dict[str, Dict[str, float]]:
    """``{dataset_lower: {"<biomarker>_<pipeline>": sigma}}``."""
    # the canonical training-split table, i.e. the one macro_mae itself uses
    # (DECISIONS.md 2026-09-03 10:20), so the bias units match the Tab.2 column
    s = _read_csv(GATE_A_SCALES_CSV)
    out: Dict[str, Dict[str, float]] = {}
    if s is None:
        print(f"[bias] {GATE_A_SCALES_CSV} missing; every bias is NaN")
        return out
    for _, r in s.iterrows():
        # Gate A writes "CHASE_DB1"; every per_image.csv writes "chasedb1".
        # Lower-casing alone leaves "chase_db1" and silently yields no sigma at
        # all for that dataset (its whole bias row came out blank).
        key = str(r["dataset"]).lower().replace("_", "")
        out.setdefault(key, {})[str(r["biomarker"])] = float(r["sigma"])
    return out


def _load(method: str):
    import pandas as pd

    parts = METHOD_GLOBS[method]
    paths = _live_glob(os.path.join(*parts, "per_image.csv"))
    frames = []
    for p in paths:
        d = _read_csv(p)
        if d is None or "dataset" not in d.columns:
            continue
        d = d.copy()
        d["dataset"] = d["dataset"].astype(str).str.lower()
        d["_src"] = os.path.relpath(p, EXP_ROOT)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def build(results_dir: str = RESULTS_DIR) -> Optional["object"]:
    import pandas as pd
    from scipy.stats import spearmanr

    sig = _sigma_lookup()
    rows: List[dict] = []
    no_repair_done = set()

    for method in METHOD_GLOBS:
        df = _load(method)
        if df is None:
            print(f"[bias][skip] {method}: no per_image.csv")
            continue
        for ds, sub in df.groupby("dataset"):
            if ds not in MAIN_DATASETS:
                continue
            sg = sig.get(ds, {})

            def signed(stage: str, bm: str, pipe: str):
                c_pred, c_gt = f"bio_{stage}_{bm}_{pipe}", f"bio_gt_{bm}_{pipe}"
                s = sg.get(f"{bm}_{pipe}", np.nan)
                if c_pred not in sub.columns or c_gt not in sub.columns \
                        or not np.isfinite(s) or s <= 0:
                    return np.nan
                return float(np.nanmean((sub[c_pred] - sub[c_gt]) / s))

            px_added = (sub["after_n_pred_px"] - sub["before_n_pred_px"]) \
                if {"after_n_pred_px", "before_n_pred_px"} <= set(sub.columns) \
                else pd.Series(np.nan, index=sub.index)
            dmae = sub["macro_mae_delta"] if "macro_mae_delta" in sub.columns \
                else pd.Series(np.nan, index=sub.index)
            ok = np.isfinite(px_added) & np.isfinite(dmae)
            if ok.sum() >= 3 and px_added[ok].nunique() > 1 and dmae[ok].nunique() > 1:
                rho, pval = spearmanr(px_added[ok], dmae[ok])
            else:
                rho, pval = np.nan, np.nan

            base = dict(dataset=ds, method=method, n_images=len(sub),
                        n_accepted_mean=float(np.nanmean(sub["n_accepted"]))
                        if "n_accepted" in sub.columns else np.nan,
                        px_added_mean=float(np.nanmean(px_added)),
                        px_added_frac_of_pred=float(
                            np.nanmean(px_added / sub["before_n_pred_px"]))
                        if "before_n_pred_px" in sub.columns else np.nan,
                        macro_mae_before=float(np.nanmean(sub["macro_mae_before"])),
                        macro_mae_after=float(np.nanmean(sub["macro_mae_after"])),
                        spearman_pxadded_vs_dmacromae=rho,
                        spearman_p=pval,
                        src=sorted(set(sub["_src"]))[0])
            for bm in PRIMARY:
                for pipe in PIPELINES:
                    base[f"bias_after_{bm}_{pipe}"] = signed("after", bm, pipe)
            rows.append(base)

            # no_repair: the unrepaired state, identical for every method, so
            # emit it once per dataset from whichever method got there first.
            if ds not in no_repair_done:
                nr = dict(dataset=ds, method="no_repair", n_images=len(sub),
                          n_accepted_mean=0.0, px_added_mean=0.0,
                          px_added_frac_of_pred=0.0,
                          macro_mae_before=float(np.nanmean(sub["macro_mae_before"])),
                          macro_mae_after=float(np.nanmean(sub["macro_mae_before"])),
                          spearman_pxadded_vs_dmacromae=np.nan, spearman_p=np.nan,
                          src=sorted(set(sub["_src"]))[0])
                for bm in PRIMARY:
                    for pipe in PIPELINES:
                        nr[f"bias_after_{bm}_{pipe}"] = signed("before", bm, pipe)
                rows.append(nr)
                no_repair_done.add(ds)

    if not rows:
        print("[bias] nothing to build")
        return None

    out = pd.DataFrame(rows)
    order = ["no_repair", "geometric", "evapore_scorer", "evapore_e2e",
             "rnca", "rigr_uniform", "rigr_risk"]
    out["method"] = pd.Categorical(out["method"], categories=order, ordered=True)
    out["dataset"] = pd.Categorical(out["dataset"], categories=MAIN_DATASETS,
                                    ordered=True)
    out = out.sort_values(["dataset", "method"]).reset_index(drop=True)

    csv_path = os.path.join(results_dir, "bias_diagnostics.csv")
    out.to_csv(csv_path, index=False)
    print(f"[bias] wrote {csv_path}")

    # ---- markdown summary -------------------------------------------
    md = ["# Bias diagnostics -- is macro-MAE rewarding over-connection?\n",
          "Source: `results/bias_diagnostics.csv` (built by "
          "`python -m src.eval.bias_diagnostics`). Signed bias is "
          "`mean_images (B_pred - B_GT) / sigma` with the Gate A sigma; "
          "**negative = the mask under-states the biomarker**. "
          "`px_added` is `after_n_pred_px - before_n_pred_px`.\n"]

    md.append("\n## 1. Signed bias before repair (the `no_repair` row)\n")
    md.append("If these are negative, the unrepaired segmentation under-states "
              "the biomarker and anything that adds pixels will move it "
              "toward the truth.\n")
    nr = out[out.method == "no_repair"]
    cols = ["dataset"] + [f"bias_after_{b}_{p}" for b in PRIMARY for p in PIPELINES]
    md.append(df_to_markdown(nr[cols].rename(
        columns={c: c.replace("bias_after_", "") for c in cols})))

    md.append("\n## 2. Pixels added, edges accepted, and macro-MAE change\n")
    t2 = out[["dataset", "method", "n_accepted_mean", "px_added_mean",
              "px_added_frac_of_pred", "macro_mae_before", "macro_mae_after"]].copy()
    t2["macro_mae_delta"] = t2.macro_mae_after - t2.macro_mae_before
    md.append(df_to_markdown(t2))

    md.append("\n## 3. Spearman(pixels added, delta macro-MAE) across images\n")
    md.append("Negative = adding pixels *lowers* macro-MAE, i.e. the metric "
              "rewards over-connection.\n")
    t3 = out[out.method != "no_repair"][
        ["dataset", "method", "n_images", "spearman_pxadded_vs_dmacromae", "spearman_p"]]
    md.append(df_to_markdown(t3))

    md_path = os.path.join(results_dir, "bias_diagnostics.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"[bias] wrote {md_path}")
    return out


if __name__ == "__main__":
    build()
