"""R2 -- absolute pixel-level quality of the baseline segmenters, against the
contemporary published range.

Reviewer point 10 (``review/user_cmig_review_20260918.md``)
----------------------------------------------------------
"Show that the baseline segmenters are good enough: report absolute Dice, IoU,
clDice, sensitivity and specificity, and compare them with published baselines."

The audit's whole argument is that a *competent* segmenter still produces a
biomarker offset of order 1 sigma.  That argument is only interesting if the
segmenter is competent, so the absolute numbers have to be on the table and
they have to be placed next to what the field reports.

Part 1 (``r2_pixel_baselines.csv``)
    Per dataset x metric, over the held-out test split and segmentation seeds
    0-2, from ``results/seg_per_image.csv`` (one row per image x seed, written
    by the training/eval pipeline via ``src.topo.metrics.evaluate_all``):
    the mean over images and seeds, the between-image SD, the between-seed
    spread, and a 2000-draw image-clustered bootstrap CI of the mean.  The
    inference working resolution is carried on every row, because the HRF and
    FIVES numbers are produced at ``resize_longest = 1536`` rather than native
    and the published HRF range depends strongly on that choice.

Part 2 (``r2_pixel_vs_literature.csv``)
    ``results/pivot/r2/pixel_literature.csv`` -- a hand-collected table of
    contemporary published results on the same four datasets, every row
    carrying the URL it was read from -- is reduced to a per-dataset published
    range (min / 25th / median / 75th / max, and the same restricted to rows
    whose method name is a plain U-Net), and our baseline is placed inside it:
    the fraction of published entries we exceed, and whether we fall inside the
    published interquartile range.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_pixel_baselines
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import EXP_ROOT, PIVOT_DIR, RESULTS_DIR, RUNS_DIR

SEED = 0
N_BOOT = 2000
DATASETS = ("drive", "chasedb1", "hrf", "fives")
SEEDS = (0, 1, 2)
METRICS = ("dice", "iou", "cldice", "sen", "spe", "acc", "auc", "pr_auc")
#: our metric name -> the column name used in pixel_literature.csv
LIT_COL = {"dice": "dice", "iou": "iou", "cldice": "cldice",
           "sen": "sensitivity", "spe": "specificity", "acc": "accuracy",
           "auc": "auc_roc"}


def working_resolution(ds: str, seed: int) -> Dict[str, object]:
    p = os.path.join(RUNS_DIR, "seg", ds, f"seed{seed}", "pred",
                     "infer_meta.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        m = json.load(fh)
    return {"patch": m.get("patch"), "stride": m.get("stride"),
            "resize_longest": m.get("resize_longest"),
            "threshold": m.get("threshold"), "flips": m.get("flips")}


def boot_ci_images(df: pd.DataFrame, metric: str, n_boot: int = N_BOOT,
                   seed: int = SEED):
    """Image-clustered bootstrap: resample images, keep all three seeds of the
    resampled image together (the seeds are three fits of the same model on the
    same image and are not independent observations)."""
    groups = list(df.groupby("image").indices.values())
    n = len(groups)
    if n < 3:
        return (float("nan"), float("nan"))
    v = df[metric].to_numpy(float)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.randint(0, n, n)
        sel = np.concatenate([groups[j] for j in pick])
        m = np.nanmean(v[sel])
        if np.isfinite(m):
            out.append(m)
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def baselines(per_image_csv: str) -> pd.DataFrame:
    d = pd.read_csv(per_image_csv)
    d = d[(d["split"] == "test") & (d["seed"].isin(SEEDS))]
    rows: List[dict] = []
    for ds in DATASETS:
        g = d[d["dataset"] == ds]
        if not len(g):
            continue
        res = working_resolution(ds, 0)
        for m in METRICS:
            if m not in g.columns:
                continue
            per_seed = g.groupby("seed")[m].mean()
            lo, hi = boot_ci_images(g, m)
            rows.append(dict(
                dataset=ds, metric=m,
                n_images=int(g["image"].nunique()), n_seeds=int(g["seed"].nunique()),
                mean=float(g[m].mean()),
                sd_images=float(g.groupby("image")[m].mean().std(ddof=1)),
                ci_lo=lo, ci_hi=hi,
                seed0=float(per_seed.get(0, np.nan)),
                seed1=float(per_seed.get(1, np.nan)),
                seed2=float(per_seed.get(2, np.nan)),
                seed_spread=float(per_seed.max() - per_seed.min()),
                min_image=float(g.groupby("image")[m].mean().min()),
                max_image=float(g.groupby("image")[m].mean().max()),
                patch=res.get("patch"), stride=res.get("stride"),
                resize_longest=res.get("resize_longest"),
                threshold=res.get("threshold"),
                architecture="nnU-Net-style U-Net, 7.77 M parameters",
                source=("results/seg_per_image.csv (method=unet, split=test, "
                        "seeds 0-2; metrics from src.topo.metrics.evaluate_all "
                        "inside the FOV); working resolution from "
                        "runs/seg/<ds>/seed0/pred/infer_meta.json; 2000-draw "
                        "image-clustered bootstrap")))
    return pd.DataFrame(rows)


def is_plain_unet(name: str) -> bool:
    n = str(name).strip().lower().replace("-", "").replace(" ", "")
    return n in ("unet", "unet2d", "vanillaunet", "baselineunet")


def vs_literature(base: pd.DataFrame, lit_csv: str) -> pd.DataFrame:
    if not os.path.exists(lit_csv):
        print("[lit] %s missing -- skipping the literature comparison" % lit_csv)
        return pd.DataFrame()
    lit = pd.read_csv(lit_csv)
    lit["dataset"] = lit["dataset"].astype(str).str.strip().str.lower()
    rows: List[dict] = []
    for r in base.itertuples(index=False):
        col = LIT_COL.get(r.metric)
        if col is None or col not in lit.columns:
            continue
        g = lit[lit["dataset"] == r.dataset]
        v = pd.to_numeric(g[col], errors="coerce").dropna()
        vu = pd.to_numeric(
            g[g["method"].map(is_plain_unet)][col], errors="coerce").dropna()
        if not len(v):
            continue
        rows.append(dict(
            dataset=r.dataset, metric=r.metric, ours=r.mean,
            ours_ci_lo=r.ci_lo, ours_ci_hi=r.ci_hi,
            n_published=int(len(v)),
            pub_min=float(v.min()), pub_q25=float(v.quantile(0.25)),
            pub_median=float(v.median()), pub_q75=float(v.quantile(0.75)),
            pub_max=float(v.max()),
            frac_published_below_ours=float((v < r.mean).mean()),
            inside_pub_iqr=bool(v.quantile(0.25) <= r.mean <= v.quantile(0.75)),
            inside_pub_range=bool(v.min() <= r.mean <= v.max()),
            n_published_plain_unet=int(len(vu)),
            unet_min=float(vu.min()) if len(vu) else float("nan"),
            unet_median=float(vu.median()) if len(vu) else float("nan"),
            unet_max=float(vu.max()) if len(vu) else float("nan"),
            inside_unet_range=(bool(vu.min() <= r.mean <= vu.max())
                               if len(vu) else None),
            resize_longest=r.resize_longest,
            source=("ours: results/seg_per_image.csv; published: "
                    "results/pivot/r2/pixel_literature.csv (every row carries "
                    "its source URL; see its notes column for reporting "
                    "conventions -- F1 vs Dice wording, evaluation resolution, "
                    "split convention)")))
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_image",
                    default=os.path.join(RESULTS_DIR, "seg_per_image.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    ap.add_argument("--literature", default=None)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    lit_csv = args.literature or os.path.join(args.out_dir,
                                              "pixel_literature.csv")

    base = baselines(args.per_image)
    p = os.path.join(args.out_dir, "r2_pixel_baselines.csv")
    base.to_csv(p, index=False)
    print("wrote", p, base.shape)

    cmp_ = vs_literature(base, lit_csv)
    if len(cmp_):
        q = os.path.join(args.out_dir, "r2_pixel_vs_literature.csv")
        cmp_.to_csv(q, index=False)
        print("wrote", q, cmp_.shape)

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print("\n[absolute pixel metrics, seeds 0-2, test split]")
        print(base[["dataset", "metric", "n_images", "mean", "ci_lo", "ci_hi",
                    "sd_images", "seed_spread", "resize_longest"]]
              .round(4).to_string(index=False))
        if len(cmp_):
            print("\n[against the published range]")
            print(cmp_[["dataset", "metric", "ours", "n_published", "pub_min",
                        "pub_median", "pub_max", "frac_published_below_ours",
                        "inside_pub_iqr", "n_published_plain_unet",
                        "unet_min", "unet_median", "unet_max",
                        "inside_unet_range"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
