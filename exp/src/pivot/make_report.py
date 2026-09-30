"""Render ``results/pivot/PROBE_REPORT.md`` from the three probes' CSVs.

Every number in the report comes from a file written by
``src.pivot.{build_table,p1_run,p2_calibration,p3_downstream}``; nothing is
typed in by hand.  The narrative paragraphs are constants at the top of this
module so the report can be regenerated after a rerun.

CLI
---
    python -m src.pivot.make_report
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
from datetime import datetime
from typing import List, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS

PRIMARY_FOUR = ("density_pvbm", "total_length_skan", "FD_skan", "tortuosity_skan")


def md_table(df: pd.DataFrame, cols: Sequence[str], nd: int = 3,
             rename: dict = None) -> str:
    d = df[list(cols)].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if not np.isfinite(v) else f"{v:.{nd}f}")
    if rename:
        d = d.rename(columns=rename)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def section_p1(out: List[str], d: pd.DataFrame) -> None:
    out.append("### P1.1 Ranking agreement, `surrogate(prob)` vs pipeline"
               " biomarker of the same prediction\n")
    piv = d.pivot_table(index=["surrogate", "biomarker"], columns="dataset",
                        values="spearman_prob_vs_pipepred").reset_index()
    order = ["surrogate", "biomarker", "drive", "chasedb1", "hrf", "fives", "POOLED"]
    piv = piv[[c for c in order if c in piv.columns]]
    out.append(md_table(piv, piv.columns) + "\n")

    out.append("### P1.2 The same, with the **hard mask** instead of the "
               "probability map (isolates the soft/hard gap)\n")
    piv2 = d.pivot_table(index=["surrogate", "biomarker"], columns="dataset",
                         values="spearman_hard_vs_pipepred").reset_index()
    piv2 = piv2[[c for c in order if c in piv2.columns]]
    out.append(md_table(piv2, piv2.columns) + "\n")

    out.append("### P1.3 Surrogate on the reference mask vs pipeline on the "
               "reference mask (pure definitional agreement)\n")
    piv3 = d.pivot_table(index=["surrogate", "biomarker"], columns="dataset",
                         values="spearman_gt_vs_pipegt").reset_index()
    piv3 = piv3[[c for c in order if c in piv3.columns]]
    out.append(md_table(piv3, piv3.columns) + "\n")

    out.append("### P1.4 Bias: does descending the surrogate move the pipeline "
               "biomarker the right way?\n")
    out.append("`bias_pipeline` is `(B_pipeline(pred) - B_pipeline(GT)) / sigma`; "
               "`bias_surrogate` is the same for the affinely unit-mapped "
               "surrogate of the probability map.  Sign agreement is what "
               "matters.\n")
    b = d[(d["dataset"] != "POOLED")].copy()
    b = b[b["biomarker"].isin(PRIMARY_FOUR)]
    out.append(md_table(b.sort_values(["biomarker", "dataset"]),
                        ["dataset", "surrogate", "biomarker",
                         "bias_pipeline_sigma", "bias_surrogate_prob_sigma",
                         "bias_surrogate_hard_sigma", "resid_fit_sigma"]) + "\n")


def section_p2(out: List[str], d: pd.DataFrame, models: Sequence[str]) -> None:
    prim = d[d["biomarker"].isin(PRIMARY_COLS)]

    out.append("### P2.0 Headline: macro-MAE over the eight primary columns "
               "(the `macro_mae` of `src/eval/biomarker_eval.py`)\n")
    agg = (prim[prim["dataset"] != "ALL"]
           .groupby("dataset")[["mae_uncal"] + ["mae_" + m for m in models]]
           .mean().reset_index())
    for m in models:
        agg["red_" + m] = 1.0 - agg["mae_" + m] / agg["mae_uncal"]
    out.append(md_table(agg, ["dataset", "mae_uncal"] + ["mae_" + m for m in models]
                        + ["red_" + m for m in models]) + "\n")

    out.append("### P2.1 Standardised MAE `mean |B_hat - B_GT| / sigma` on the "
               "held-out test predictions\n")
    cols = ["biomarker", "dataset", "n", "mae_uncal"] + ["mae_" + m for m in models]
    out.append(md_table(prim.sort_values(["biomarker", "dataset"]), cols) + "\n")

    out.append("### P2.2 Relative MAE reduction vs the uncalibrated prediction\n")
    cols = ["biomarker", "dataset", "n"] + ["red_" + m for m in models]
    out.append(md_table(prim.sort_values(["biomarker", "dataset"]), cols) + "\n")

    out.append("### P2.3 Signed bias, before and after\n")
    cols = ["biomarker", "dataset", "bias_uncal"] + ["bias_" + m for m in models]
    out.append(md_table(prim.sort_values(["biomarker", "dataset"]), cols) + "\n")

    out.append("### P2.4 Split-conformal 90 % intervals: empirical coverage and "
               "mean width (in sigma)\n")
    cols = ["biomarker", "dataset"] + [f(m) for m in models
                                       for f in (lambda x: "cov90_" + x,
                                                 lambda x: "width_" + x)]
    out.append(md_table(prim.sort_values(["biomarker", "dataset"]), cols) + "\n")


def section_p3(out: List[str], d: pd.DataFrame) -> None:
    out.append("### P3.1 Macro one-vs-rest AUC by feature source "
               "(5-fold CV x 3 repeats, 1000-sample image bootstrap CI)\n")
    d = d.copy()
    d["auc_ci"] = [f"{r.macro_auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
                   for r in d.itertuples(index=False)]
    d["gt_minus"] = [f"{r.gt_minus_this:+.3f} [{r.diff_ci_lo:+.3f}, "
                     f"{r.diff_ci_hi:+.3f}]" for r in d.itertuples(index=False)]
    piv = d.pivot_table(index=["dataset", "split", "featureset", "clf", "n"],
                        columns="source", values="auc_ci",
                        aggfunc="first").reset_index()
    out.append(md_table(piv, piv.columns) + "\n")
    out.append("### P3.2 Attenuation: macro-AUC(GT features) - macro-AUC(this "
               "source), paired bootstrap 95 % CI\n")
    piv2 = d[d["source"] != "gt"].pivot_table(
        index=["dataset", "split", "featureset", "clf"], columns="source",
        values="gt_minus", aggfunc="first").reset_index()
    out.append(md_table(piv2, piv2.columns) + "\n")


def section_p3_reliability(out: List[str], pred: pd.DataFrame,
                           models: Sequence[str]) -> None:
    """Pearson r(B_pred, B_GT) per dataset -- the reliability that drives
    attenuation.  A constant bias does not change it; per-image scatter does."""
    out.append("### P3.3 Reliability `r(B, B_GT)` within each dataset -- the "
               "quantity attenuation actually depends on\n")
    out.append("A calibration that removes a constant offset cannot change "
               "this number, which is why bias removal does not buy back "
               "downstream AUC.\n")
    rows = []
    for ds, g in pred.groupby("dataset"):
        for bio in PRIMARY_COLS:
            gt = g[bio + "__gt"].to_numpy(dtype=float)
            row = {"dataset": ds, "biomarker": bio}
            for tag, col in [("pred", bio + "__pred")] + [
                    (m, bio + "__cal_" + m) for m in models]:
                x = g[col].to_numpy(dtype=float)
                m = np.isfinite(x) & np.isfinite(gt)
                row["r_" + tag] = (float(np.corrcoef(x[m], gt[m])[0, 1])
                                   if m.sum() > 3 and np.std(x[m]) > 0
                                   else float("nan"))
            rows.append(row)
    r = pd.DataFrame(rows)
    out.append(md_table(r.sort_values(["biomarker", "dataset"]),
                        ["dataset", "biomarker", "r_pred"]
                        + ["r_" + m for m in models]) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=PIVOT_DIR)
    ap.add_argument("--narrative", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "report_text.py"))
    args = ap.parse_args(argv)

    from src.pivot import report_text as T

    p1 = pd.read_csv(os.path.join(args.dir, "p1_surrogate_agreement.csv"))
    p2 = pd.read_csv(os.path.join(args.dir, "p2_calibration.csv"))
    p3 = pd.read_csv(os.path.join(args.dir, "p3_downstream.csv"))
    master = pd.read_csv(os.path.join(args.dir, "bio_master.csv"))
    models = [c[4:] for c in p2.columns if c.startswith("mae_") and c != "mae_uncal"]

    out: List[str] = []
    out.append(T.HEADER.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        python=platform.python_version(),
        counts=md_table(master.groupby(["dataset", "source"]).size()
                        .rename("n_images").reset_index(),
                        ["dataset", "source", "n_images"], nd=0)))
    out.append(T.P1_INTRO)
    section_p1(out, p1)
    out.append(T.P1_VERDICT)
    out.append(T.P2_INTRO)
    section_p2(out, p2, models)
    out.append(T.P2_VERDICT)
    out.append(T.P3_INTRO)
    section_p3(out, p3)
    section_p3_reliability(out, pd.read_csv(
        os.path.join(args.dir, "p2_predictions.csv")), models)
    out.append(T.P3_VERDICT)
    out.append(T.CLOSING)

    path = os.path.join(args.dir, "PROBE_REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("[out]", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
