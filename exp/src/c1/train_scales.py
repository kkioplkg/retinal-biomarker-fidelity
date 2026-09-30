"""Training-split-only biomarker scales ``sigma_B`` (and the Gate A agreement re-check).

The decision scale a deployed system divides by must not be informed by the
images it is later judged on.  ``results/gateA_biomarker_scales.csv`` was
estimated over **all** Gate A reference masks, test images included, so this
module re-estimates it from training-split, observer-1 masks only and writes

``results/gateA_biomarker_scales_train.csv``
    same schema as the all-mask file (``dataset, biomarker, n, median, mad,
    sigma``) plus ``source`` and ``resolution``.

``results/gateA_pipeline_agreement_train.csv``
    the two-pipeline Spearman agreement recomputed on training-split masks, so
    the Gate A tortuosity decision (rho < 0.8 -> skan-only) can be re-checked
    without test-image information.

FIVES needs care.  Gate A measured FIVES on 100 **test** images, so it
contributes no training-split reference masks at all; for FIVES the scale is
taken from the C1 ``B0_*`` cache of the 60 FIVES *training* images, which is
measured at the C1 working resolution (1536 px long side) rather than natively.
That is the right choice -- C1's ``|dB|`` for FIVES is also measured at 1536, so
numerator and denominator stay commensurate -- but it means the FIVES row is not
comparable to a native-resolution scale, which the ``resolution`` column records.

Usage::

    python -m src.c1.train_scales [--results-dir results]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")

PRIMARY = ("FD", "tortuosity", "density", "total_length")
PIPELINES = ("pvbm", "skan")

#: Gate A's pre-registered two-pipeline agreement threshold.
AGREEMENT_RHO = 0.80


def _split_lut(datasets: Sequence[str]) -> Dict[Tuple[str, str], str]:
    from ..data.datasets import load_dataset

    out: Dict[Tuple[str, str], str] = {}
    for ds in datasets:
        try:
            recs = load_dataset(ds)
        except Exception:  # noqa: BLE001
            continue
        for r in recs:
            out[(ds, str(r["image_id"]))] = str(r["split"])
    return out


def _mad_scale(v):
    """Pre-registered robust scale ``1.4826 * MAD`` (NaN when undefined)."""
    import numpy as np

    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return len(a), float("nan"), float("nan"), float("nan")
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    sig = 1.4826 * mad
    return int(a.size), med, mad, (sig if sig > 0 else float("nan"))


def build(results_dir: str = RESULTS_DIR):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    gt_csv = os.path.join(results_dir, "gateA_biomarkers_gt.csv")
    im_csv = os.path.join(results_dir, "c1_images.csv")
    cand = [f"{b}_{p}" for b in PRIMARY for p in PIPELINES]

    rows, agree_rows = [], []

    # ---- (1) Gate A reference masks, training split, observer 1 -----------
    gt = pd.read_csv(gt_csv) if os.path.exists(gt_csv) else None
    gt_tr = None
    if gt is not None:
        g = gt.copy()
        if "split" not in g.columns:
            lut = _split_lut(sorted(g["dataset"].astype(str).unique()))
            g["split"] = [lut.get((d, i)) for d, i in
                          zip(g["dataset"].astype(str), g["image_id"].astype(str))]
        obs = g["observer"].astype(str) if "observer" in g.columns else "obs1"
        gt_tr = g[(g["split"].astype(str) == "train") & (obs == "obs1")]
        for ds, sub in gt_tr.groupby(g["dataset"].astype(str)):
            for col in cand:
                if col not in sub.columns:
                    continue
                n, med, mad, sig = _mad_scale(sub[col])
                if n == 0:
                    continue
                rows.append(dict(dataset=ds, biomarker=col, n=n, median=med,
                                 mad=mad, sigma=sig,
                                 source="gateA_train_obs1", resolution="native"))
            for b in PRIMARY:
                ca, cb = f"{b}_pvbm", f"{b}_skan"
                if ca in sub.columns and cb in sub.columns:
                    a = pd.to_numeric(sub[ca], errors="coerce")
                    c = pd.to_numeric(sub[cb], errors="coerce")
                    m = a.notna() & c.notna()
                    rho = (float(spearmanr(a[m], c[m])[0]) if m.sum() >= 4
                           else float("nan"))
                    agree_rows.append(dict(dataset=ds, biomarker=b, n=int(m.sum()),
                                           spearman=rho,
                                           passes=bool(np.isfinite(rho)
                                                       and rho >= AGREEMENT_RHO),
                                           source="gateA_train_obs1"))

    have = {(r["dataset"], r["biomarker"]) for r in rows}

    # ---- (2) datasets Gate A cannot cover on the training split ----------
    # src/eval/biomarker_eval scores predictions mapped back to NATIVE
    # resolution, so sigma_B must be native for every dataset and the C1 B0
    # cache (1536 px for HRF/FIVES) cannot serve here.  These datasets are
    # measured directly at native resolution with both pipelines:
    #
    #   FIVES  Gate A measured 100 *test* images only, so it contributes no
    #          training-split masks; the first 120 images of the fixed training
    #          split are measured instead (the subset S4's image_caps uses).
    #   STARE  not part of the Gate A corpus at all; its 10 training-split
    #          observer-1 (ah) masks are measured.
    NATIVE_TABLES = (
        ("FIVES", "fives_native_train_biomarkers.csv", "fives_native_train120_obs1"),
        ("STARE", "stare_native_train_biomarkers.csv", "stare_train_obs1"),
    )
    for ds_name, fname, src_tag in NATIVE_TABLES:
        fn_csv = os.path.join(results_dir, fname)
        if not os.path.exists(fn_csv):
            continue
        fn = pd.read_csv(fn_csv)
        for col in cand:
            if (ds_name, col) in have or col not in fn.columns:
                continue
            n, med, mad, sig = _mad_scale(fn[col])
            if n == 0:
                continue
            rows.append(dict(dataset=ds_name, biomarker=col, n=n, median=med,
                             mad=mad, sigma=sig, source=src_tag,
                             resolution="native"))
        if not any(a["dataset"] == ds_name for a in agree_rows):
            for b in PRIMARY:
                ca, cb = f"{b}_pvbm", f"{b}_skan"
                if ca in fn.columns and cb in fn.columns:
                    a = pd.to_numeric(fn[ca], errors="coerce")
                    c = pd.to_numeric(fn[cb], errors="coerce")
                    m = a.notna() & c.notna()
                    rho = (float(spearmanr(a[m], c[m])[0]) if m.sum() >= 4
                           else float("nan"))
                    agree_rows.append(dict(
                        dataset=ds_name, biomarker=b, n=int(m.sum()), spearman=rho,
                        passes=bool(np.isfinite(rho) and rho >= AGREEMENT_RHO),
                        source=src_tag))

    scales = pd.DataFrame(rows).sort_values(["dataset", "biomarker"])
    agree = pd.DataFrame(agree_rows).sort_values(["biomarker", "dataset"])
    return scales, agree


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="training-split-only sigma_B")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    scales, agree = build(args.results_dir)
    sp = os.path.join(args.results_dir, "gateA_biomarker_scales_train.csv")
    ap_ = os.path.join(args.results_dir, "gateA_pipeline_agreement_train.csv")
    scales.to_csv(sp, index=False)
    agree.to_csv(ap_, index=False)
    print(f"[out] {sp}  ({len(scales)} rows)")
    print(f"[out] {ap_} ({len(agree)} rows)")

    # ---- ratio against the all-mask scales --------------------------------
    allp = os.path.join(args.results_dir, "gateA_biomarker_scales.csv")
    if os.path.exists(allp):
        a = pd.read_csv(allp)[["dataset", "biomarker", "sigma"]].rename(
            columns={"sigma": "sigma_all"})
        m = scales.merge(a, on=["dataset", "biomarker"], how="left")
        m["ratio_train_over_all"] = m["sigma"] / m["sigma_all"]
        print("\n=== sigma ratio (train-only / all-mask) ===")
        piv = m.pivot_table(index="biomarker", columns="dataset",
                            values="ratio_train_over_all")
        print(piv.round(3).to_string())
        bad = m[(m["ratio_train_over_all"] < 0.5) | (m["ratio_train_over_all"] > 2.0)]
        if len(bad):
            print("\n  MATERIALLY DIFFERENT (ratio outside [0.5, 2]):")
            print(bad[["dataset", "biomarker", "sigma", "sigma_all",
                       "ratio_train_over_all", "source", "resolution"]]
                  .round(5).to_string(index=False))
        m.to_csv(os.path.join(args.results_dir,
                              "gateA_biomarker_scales_train_vs_all.csv"), index=False)

    print("\n=== two-pipeline agreement on the TRAINING split ===")
    if len(agree):
        print(agree.round(4).to_string(index=False))
        t = agree[agree["biomarker"] == "tortuosity"]
        if len(t):
            worst = float(np.nanmin(t["spearman"]))
            print(f"\n  tortuosity: rho per dataset = "
                  f"{dict(zip(t['dataset'], t['spearman'].round(3)))}")
            print(f"  min rho = {worst:.3f} vs threshold {AGREEMENT_RHO} -> "
                  f"skan-only decision {'HOLDS' if worst < AGREEMENT_RHO else 'DOES NOT hold'} "
                  "on the training split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
