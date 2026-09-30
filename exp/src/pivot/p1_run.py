"""Probe P1 -- do differentiable surrogates track the real biomarker pipeline?

For every seed-0 test prediction we evaluate the four surrogates of
``src.pivot.surrogates`` three ways

    prob   surrogate(soft probability map)   <- what a loss would see
    hard   surrogate(thresholded mask)       <- discretisation-free reference
    gt     surrogate(reference mask)         <- used only to fit the unit map

and compare them with the pipeline biomarkers of the same masks, taken from
``results/pivot/bio_master.csv``.

Two questions are answered separately.

1. *Ranking*: Spearman / Pearson between ``surrogate(prob)`` and the pipeline
   value of the prediction, across images within a dataset.  This is the
   success criterion (rho >= 0.8 for density / length / FD).
2. *Bias*: the surrogate is in arbitrary units, so a per-dataset affine map
   ``a * surrogate + b`` is fitted by OLS **on the reference masks only**
   (never on predictions).  The calibrated surrogate of the prediction is then
   compared with the pipeline biomarker of the reference mask in sigma units
   (``results/gateA_biomarker_scales_train.csv``), next to the pipeline's own
   bias.  If the two biases agree in sign and size, descending the surrogate
   moves the pipeline biomarker the right way.

CLI
---
    python -m src.pivot.p1_run --device cuda:1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

from src.pivot.common import (DATASETS, PIVOT_DIR, binarize, load_fov,
                              pred_paths, read_gray, records, sigma_table)

#: surrogate -> pipeline columns it is meant to approximate
PAIRS: Dict[str, Tuple[str, ...]] = {
    "s_density": ("density_pvbm", "density_skan"),
    "s_length": ("total_length_skan", "total_length_pvbm"),
    "s_fd": ("FD_skan", "FD_pvbm"),
    "s_fd_auto": ("FD_skan", "FD_pvbm"),
    "s_fd_skel": ("FD_skan", "FD_pvbm"),
    "s_tortuosity": ("tortuosity_skan", "tortuosity_pvbm"),
    "s_tort_auto": ("tortuosity_skan", "tortuosity_pvbm"),
}


def _surrogates_for(dataset: str, seed: int, device: str, num_iter: int
                    ) -> "object":
    import pandas as pd
    import torch

    from src.pivot.surrogates import all_surrogates

    dev = torch.device(device)
    rows: List[dict] = []
    recs = [r for r in records(dataset) if r["split"] == "test"]
    for i, r in enumerate(recs):
        pp, mp = pred_paths(dataset, r["image_id"], "pred_test", seed)
        if not (os.path.exists(pp) and os.path.exists(mp)):
            continue
        prob = np.load(pp).astype(np.float32)
        hw = prob.shape[:2]
        fov = load_fov(r, hw)
        hard = binarize(read_gray(mp)).astype(np.float32)
        gt = binarize(read_gray(r["label_path"])).astype(np.float32)
        if gt.shape != hw:
            import cv2
            gt = (cv2.resize(gt * 255, (hw[1], hw[0]),
                             interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)

        fov_t = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
        row = dict(dataset=dataset, image_id=r["image_id"], seed=seed,
                   disease=r.get("disease"))
        with torch.no_grad():
            for tag, arr in (("prob", prob), ("hard", hard), ("gt", gt)):
                t = torch.from_numpy(arr)[None, None].to(dev)
                s = all_surrogates(t, fov_t, num_iter=num_iter)
                for k, v in s.items():
                    row[f"{tag}_{k}"] = float(v.item())
                del t
        rows.append(row)
        del fov_t
        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
            print("  [%s] %d/%d" % (dataset, i + 1, len(recs)), flush=True)
    return pd.DataFrame(rows)


def _corr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    from scipy.stats import pearsonr, spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan"), float("nan"), int(m.sum())
    return (float(spearmanr(a[m], b[m]).statistic),
            float(pearsonr(a[m], b[m])[0]), int(m.sum()))


def analyse(sur, master, sigmas, out_dir: str):
    import pandas as pd

    gt = master[master["source"] == "gt"].set_index(["dataset", "image_id"])
    pr = master[master["source"] == "pred_test"].set_index(["dataset", "image_id"])

    rows = []
    groups = [(ds, g) for ds, g in sur.groupby("dataset")]
    # a POOLED row: every quantity is replaced by its within-dataset rank first,
    # so the pooled correlation measures agreement *within* a domain and is not
    # inflated by the large between-dataset offsets in resolution and scale.
    pooled = sur.copy()
    num = [c for c in pooled.columns if c.startswith(("prob_", "hard_", "gt_"))]
    pooled[num] = pooled.groupby("dataset")[num].rank()
    groups.append(("POOLED", pooled))

    for ds, g in groups:
        idx = pd.MultiIndex.from_arrays([g["dataset"], g["image_id"]])
        sig = sigmas.get(ds, {})
        for s_name, bio_cols in PAIRS.items():
            for bio in bio_cols:
                if bio not in pr.columns:
                    continue
                pipe_pred = pr.reindex(idx)[bio].to_numpy(dtype=float)
                pipe_gt = gt.reindex(idx)[bio].to_numpy(dtype=float)
                if ds == "POOLED":
                    key = g["dataset"].to_numpy()
                    pipe_pred = pd.Series(pipe_pred).groupby(key).rank().to_numpy()
                    pipe_gt = pd.Series(pipe_gt).groupby(key).rank().to_numpy()
                sur_prob = g[f"prob_{s_name}"].to_numpy(dtype=float)
                sur_hard = g[f"hard_{s_name}"].to_numpy(dtype=float)
                sur_gt = g[f"gt_{s_name}"].to_numpy(dtype=float)

                r = dict(dataset=ds, surrogate=s_name, biomarker=bio)
                for tag, sv, pv in (("prob_vs_pipepred", sur_prob, pipe_pred),
                                    ("hard_vs_pipepred", sur_hard, pipe_pred),
                                    ("prob_vs_pipegt", sur_prob, pipe_gt),
                                    ("gt_vs_pipegt", sur_gt, pipe_gt)):
                    rho, pear, n = _corr(sv, pv)
                    r[f"spearman_{tag}"] = rho
                    r[f"pearson_{tag}"] = pear
                    r["n"] = n

                if ds == "POOLED":          # ranks have no units: no bias term
                    rows.append(r)
                    continue

                # --- unit map fitted on reference masks only ---------------
                m = np.isfinite(sur_gt) & np.isfinite(pipe_gt)
                if m.sum() >= 4 and np.std(sur_gt[m]) > 0:
                    a, b = np.polyfit(sur_gt[m], pipe_gt[m], 1)
                else:
                    a, b = np.nan, np.nan
                r["fit_a"], r["fit_b"] = float(a), float(b)
                sg = float(sig.get(bio, np.nan))
                cal_prob = a * sur_prob + b
                cal_hard = a * sur_hard + b
                with np.errstate(invalid="ignore"):
                    r["bias_pipeline_sigma"] = float(
                        np.nanmean((pipe_pred - pipe_gt) / sg))
                    r["bias_surrogate_prob_sigma"] = float(
                        np.nanmean((cal_prob - pipe_gt) / sg))
                    r["bias_surrogate_hard_sigma"] = float(
                        np.nanmean((cal_hard - pipe_gt) / sg))
                    r["resid_fit_sigma"] = float(
                        np.nanmean(np.abs(a * sur_gt + b - pipe_gt)) / sg)
                r["sigma"] = sg
                rows.append(r)

    res = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "p1_surrogate_agreement.csv"), index=False)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--num_iter", type=int, default=10)
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse results/pivot/p1_surrogates_per_image.csv")
    args = ap.parse_args(argv)

    import pandas as pd

    per_img = os.path.join(args.out_dir, "p1_surrogates_per_image.csv")
    if args.reuse and os.path.exists(per_img):
        sur = pd.read_csv(per_img)
    else:
        parts = []
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            t0 = time.perf_counter()
            parts.append(_surrogates_for(ds, args.seed, args.device, args.num_iter))
            print("[p1] %s done in %.0f s" % (ds, time.perf_counter() - t0), flush=True)
        sur = pd.concat(parts, ignore_index=True)
        os.makedirs(args.out_dir, exist_ok=True)
        sur.to_csv(per_img, index=False)

    master = pd.read_csv(os.path.join(PIVOT_DIR, "bio_master.csv"))
    res = analyse(sur, master, sigma_table(), args.out_dir)
    cols = ["dataset", "surrogate", "biomarker", "n", "spearman_prob_vs_pipepred",
            "spearman_hard_vs_pipepred", "spearman_prob_vs_pipegt",
            "bias_pipeline_sigma", "bias_surrogate_prob_sigma"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(res[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
