"""Fig.4B: HRF resolution stress test (100% / 75% / 50% / DRIVE-equivalent).

Plan reference: exp/EXPERIMENT_PLAN.md S5 / proposal v3 section 4.3 -- "(B) HRF
100/75/50/DRIVE-equivalent resolution drift (incl. worst-case deviation)".

Feeding convention (documented here because it is the one non-obvious design
choice): the segmenter always expects an input whose **longest side equals the
dataset's training convention** (``src.seg.data.DATASET_CFG[...]["resize_longest"]``,
1536 for HRF). So for every resolution level we

    1. degrade: resize the *native* image/FOV down to the level's target
       longest side (100% = native, 75%/50% = that fraction of native, the
       DRIVE-equivalent level = 565 px) -- this is where information is
       actually discarded, with area-averaging downsampling;
    2. feed: resize the *degraded* image back up to the model's fixed input
       longest side (bilinear) -- so the network always receives an
       appropriately-sized tensor and the only thing that varies between
       levels is the information content, not the input resolution the model
       was trained to expect;
    3. predict at that fixed input size, then resize the probability map back
       down to the **native** resolution (a single common coordinate frame)
       before thresholding and computing biomarkers/topology metrics, so
       every level's mask lives on the same pixel grid as the GT and is
       directly comparable.

Outputs (under ``--out-dir``, default ``exp/results``)
-------------------------------------------------------
``fig4b_resolution_<dataset>_per_image.csv``
    one row per (image, resolution level): the 8 primary biomarker columns,
    the topology/pixel metrics vs GT (all at native resolution), timings.
``fig4b_resolution_<dataset>_drift.csv``
    one row per non-100% resolution level: mean signed delta (level - 100%)
    of FD / tortuosity / density / total_length (per pipeline, plus a
    pipeline-averaged "macro" column), and the worst-case (max |delta|)
    across images; each per-pipeline column also gets a ``..._z`` sibling
    standardised by the single source of sigma (schema item 3,
    ``results/gateA_biomarker_scales.csv`` via ``src.eval.biomarker_eval``)
    when that file is available.  This file (plus the stability module's
    ``fig4a_stability_*_per_image.csv``) is also always written straight into
    ``results/`` regardless of ``--out-dir``, since that is where
    ``src.robust.fig4`` looks for it (schema item 5).

CLI
---
    python -m src.robust.run_resolution --dataset hrf \\
        --ckpt runs/seg/hrf/seed0/best.pt --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")

#: (level name, target longest side or None; None = "same as native", used by 100%)
DRIVE_EQUIV_LONGEST = 565
LEVELS: Tuple[Tuple[str, Optional[float]], ...] = (
    ("100", 1.0),
    ("75", 0.75),
    ("50", 0.50),
    ("driveeq", None),  # resolved to DRIVE_EQUIV_LONGEST / native_longest below
)


def _resize(arr: np.ndarray, longest: int, interp) -> np.ndarray:
    import cv2

    h, w = arr.shape[:2]
    if max(h, w) == longest:
        return arr
    scale = float(longest) / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(arr, (nw, nh), interpolation=interp)


def _degrade_and_feed(image: np.ndarray, fov: np.ndarray, native_hw: Tuple[int, int],
                       degrade_longest: int, feed_longest: int):
    """Step 1 (degrade) + step 2 (feed) of the module docstring."""
    import cv2

    if degrade_longest >= max(native_hw):
        small_img, small_fov = image, fov  # "100%": no artificial degradation
    else:
        small_img = _resize(image, degrade_longest, cv2.INTER_AREA)
        small_fov = (_resize(fov.astype(np.uint8) * 255, degrade_longest, cv2.INTER_NEAREST) > 127
                     ).astype(np.uint8)
    feed_img = _resize(small_img, feed_longest, cv2.INTER_LINEAR)
    feed_fov = (_resize(small_fov.astype(np.uint8) * 255, feed_longest, cv2.INTER_NEAREST) > 127
               ).astype(np.uint8)
    return feed_img, feed_fov


def run_one_level(model, image, fov, native_hw, *, degrade_longest: int, feed_longest: int,
                   patch: int, device, threshold: float, stride: int, amp: bool, sw_batch: int):
    import torch

    from src.seg import data as segdata
    from src.seg.infer import sliding_window_predict, to_native

    feed_img, feed_fov = _degrade_and_feed(image, fov, native_hw, degrade_longest, feed_longest)

    m = feed_fov > 0
    if m.sum() < 16:
        m = np.ones_like(feed_fov, bool)
    pix = feed_img[m].astype(np.float32)
    mean = pix.mean(axis=0).astype(np.float32)
    std = np.maximum(pix.std(axis=0), 1e-3).astype(np.float32)

    x = torch.from_numpy(segdata.normalize(feed_img, mean, std))
    prob_feed = sliding_window_predict(model, x, patch, device, stride=stride, amp=amp,
                                        batch_size=sw_batch)
    prob_native = to_native(prob_feed, native_hw)
    mask_native = (prob_native >= threshold) & (fov > 0)
    return mask_native, prob_native


def _level_longests(native_longest: int, model_input_longest: Optional[int]) -> Dict[str, Tuple[int, int]]:
    """dataset-independent resolution -> (degrade_longest, feed_longest)."""
    feed_default = int(model_input_longest) if model_input_longest else int(native_longest)
    out = {}
    for name, frac in LEVELS:
        if name == "driveeq":
            degrade = min(DRIVE_EQUIV_LONGEST, native_longest)
        else:
            degrade = max(1, int(round(native_longest * frac)))
        out[name] = (degrade, feed_default)
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fig.4B: HRF (or any dataset) resolution stress test")
    ap.add_argument("--dataset", default="hrf")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0, help="-1 forces CPU")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride-div", type=int, default=2)
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--repair", choices=["none", "rigr_cmd"], default="none",
                     help="repair every level's predicted mask before the "
                          "biomarkers are computed (Fig.4B 'repaired' curve)")
    ap.add_argument("--repair-cmd", default=None,
                     help="shell template with {input}/{output} PNG paths")
    ap.add_argument("--repair-fn", default=None,
                     help="'module:function(mask, fov, image, prob=None) -> mask'; "
                          "src.rigr.repair_hook:repair is the RiGR hook")
    ap.add_argument("--tag", default="",
                     help="suffix appended to <dataset> in every output file "
                          "name, so a repaired run cannot overwrite the "
                          "baseline one (e.g. '_rigr'). Empty (the default) "
                          "reproduces the previous file names exactly.")
    args = ap.parse_args(argv)

    import pandas as pd
    import torch

    from src.seg import data as segdata
    from src.seg.infer import load_model, load_native_fov
    from src.topo.metrics import evaluate_all
    from src.bio.biomarkers import compute_all, primary_columns
    from src.data.datasets import read_binary

    device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available()
                           else f"cuda:{args.gpu}")
    cfg = segdata.dataset_cfg(args.dataset)
    patch, model_longest = int(cfg["patch"]), cfg["resize_longest"]
    stride = max(1, patch // max(1, args.stride_div))

    split_seed = args.split_seed if args.split_seed is not None else segdata.DEFAULT_SPLIT_SEED
    recs = segdata.get_records(args.dataset)
    tr, va, te, _info = segdata.make_splits(recs, split_seed)
    subset = {"test": te, "val": va, "train": tr, "all": recs}[args.split]
    if args.limit:
        subset = subset[: args.limit]
    if not subset:
        print(f"[run_resolution] split '{args.split}' empty for {args.dataset}")
        return 2

    from src.robust.run_stability import apply_repair

    model, _ck = load_model(args.ckpt, device)
    cols = primary_columns(zone=False)
    ds_key = segdata.canon(args.dataset)
    #: file-name key: the canonical dataset plus --tag, so a repaired run and
    #: the baseline run can coexist under results/
    out_key = ds_key + str(args.tag or "")

    rows: List[dict] = []
    t_all = time.time()
    for ri, rec in enumerate(subset):
        image = segdata._imread_color(rec["image_path"])
        native_hw = image.shape[:2]
        fov = load_native_fov(rec, native_hw)
        gt = read_binary(rec["label_path"]) if rec.get("label_path") else None
        if gt is not None and gt.shape != native_hw:
            import cv2
            gt = cv2.resize(gt.astype(np.uint8) * 255, (native_hw[1], native_hw[0]),
                             interpolation=cv2.INTER_NEAREST) > 127

        levels = _level_longests(max(native_hw), model_longest)
        for level, (degrade_longest, feed_longest) in levels.items():
            t0 = time.time()
            row = dict(dataset=ds_key, image_id=rec["image_id"], level=level,
                       degrade_longest=degrade_longest, feed_longest=feed_longest,
                       repair=args.repair, repaired=0, error="")
            try:
                mask, prob = run_one_level(model, image, fov, native_hw,
                                            degrade_longest=degrade_longest, feed_longest=feed_longest,
                                            patch=patch, device=device, threshold=args.threshold,
                                            stride=stride, amp=not args.no_amp, sw_batch=args.sw_batch)
                mask, repaired = apply_repair(mask, fov, image, mode=args.repair,
                                               cmd_template=args.repair_cmd,
                                               fn_spec=args.repair_fn, prob=prob)
                row["repaired"] = int(repaired)
                bio = compute_all(mask, fov, image=None, disc=None)
                for c in cols:
                    row[c] = bio.get(c, np.nan)
                if gt is not None:
                    row.update(evaluate_all(prob, mask, gt, fov))
                row["pred_fg_frac"] = float(mask.sum()) / max(1.0, float(fov.sum()))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["seconds"] = round(time.time() - t0, 3)
            rows.append(row)
        print(f"[{ri + 1}/{len(subset)}] {rec['image_id']} ({time.time() - t_all:.1f}s elapsed)",
              flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    pi_path = os.path.join(args.out_dir, f"fig4b_resolution_{out_key}_per_image.csv")
    df.to_csv(pi_path, index=False)
    print(f"wrote {len(df)} rows -> {pi_path}")

    # schema item 3: single source of sigma (results/gateA_biomarker_scales.csv,
    # GT masks at native resolution -- see src.eval.biomarker_eval), used only
    # to add MAD-standardised drift columns; nothing here re-estimates sigma.
    from src.eval.biomarker_eval import load_gt_scales

    sigma = load_gt_scales(ds_key)
    drift_df = build_drift_table(df, cols, sigma=sigma)
    dr_path = os.path.join(args.out_dir, f"fig4b_resolution_{out_key}_drift.csv")
    drift_df.to_csv(dr_path, index=False)
    print(f"wrote drift table -> {dr_path}")
    print(drift_df)
    # schema item 5: src.robust.fig4 looks for the drift table under results/
    # specifically; when --out-dir points elsewhere (e.g. a
    # runs/robust/<name>/ detail directory), also drop a copy there.
    if os.path.abspath(args.out_dir) != os.path.abspath(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fallback_path = os.path.join(RESULTS_DIR, f"fig4b_resolution_{out_key}_drift.csv")
        drift_df.to_csv(fallback_path, index=False)
        print(f"[run_resolution] also wrote the Fig.4 summary -> {fallback_path}")

    meta = dict(dataset=ds_key, out_key=out_key, repair=args.repair,
                repair_fn=args.repair_fn, repair_cmd=args.repair_cmd,
                ckpt=os.path.abspath(args.ckpt), split=args.split,
                model_input_longest=model_longest, drive_equiv_longest=DRIVE_EQUIV_LONGEST,
                n_images=len(subset), device=str(device), total_seconds=round(time.time() - t_all, 2))
    with open(os.path.join(args.out_dir, f"fig4b_resolution_{out_key}_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return 0


def build_drift_table(df, cols: List[str], sigma: Optional[Dict[str, float]] = None):
    """Mean/worst signed delta of every primary column vs the 100% level.

    ``sigma`` (schema item 3), when given, is the single source-of-truth MAD
    scale per biomarker column (``results/gateA_biomarker_scales.csv`` via
    ``src.eval.biomarker_eval.load_gt_scales``); every ``mean_delta_<c>`` /
    ``worst_abs_delta_<c>`` then also gets a ``..._z`` sibling standardised by
    that scale.  Omitted (``None``) -> only the raw, un-standardised columns
    this function has always written.
    """
    import pandas as pd

    ok = df[df["error"] == ""] if "error" in df.columns else df
    base = ok[ok["level"] == "100"].set_index("image_id")
    rows = []
    macro_pairs = {
        "FD": ("FD_pvbm", "FD_skan"),
        "tortuosity": ("tortuosity_pvbm", "tortuosity_skan"),
        "density": ("density_pvbm", "density_skan"),
        "total_length": ("total_length_pvbm", "total_length_skan"),
    }
    for level in [l for l, _ in LEVELS if l != "100"]:
        g = ok[ok["level"] == level].set_index("image_id")
        common = g.index.intersection(base.index)
        row = dict(level=level, n=len(common))
        deltas_macro: Dict[str, np.ndarray] = {}
        for c in cols:
            if c not in g.columns:
                continue
            d = pd.to_numeric(g.loc[common, c], errors="coerce").to_numpy(dtype=float) - \
                pd.to_numeric(base.loc[common, c], errors="coerce").to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            row[f"mean_delta_{c}"] = float(np.mean(d)) if d.size else np.nan
            row[f"worst_abs_delta_{c}"] = float(np.max(np.abs(d))) if d.size else np.nan
            if sigma:
                s = float(sigma.get(c, np.nan))
                if np.isfinite(s) and s > 0:
                    row[f"mean_delta_{c}_z"] = row[f"mean_delta_{c}"] / s
                    row[f"worst_abs_delta_{c}_z"] = row[f"worst_abs_delta_{c}"] / s
        for name, (pk, sk) in macro_pairs.items():
            a = pd.to_numeric(g.loc[common, pk], errors="coerce").to_numpy(dtype=float) - \
                pd.to_numeric(base.loc[common, pk], errors="coerce").to_numpy(dtype=float) \
                if pk in g.columns else np.full(len(common), np.nan)
            b = pd.to_numeric(g.loc[common, sk], errors="coerce").to_numpy(dtype=float) - \
                pd.to_numeric(base.loc[common, sk], errors="coerce").to_numpy(dtype=float) \
                if sk in g.columns else np.full(len(common), np.nan)
            macro = np.nanmean(np.vstack([a, b]), axis=0)
            macro = macro[np.isfinite(macro)]
            row[f"mean_delta_{name}_macro"] = float(np.mean(macro)) if macro.size else np.nan
            row[f"worst_abs_delta_{name}_macro"] = float(np.max(np.abs(macro))) if macro.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
