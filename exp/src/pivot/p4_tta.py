"""Probe P4b -- dihedral (D4) test-time augmentation at the *probability* level.

The Fig.4A views of :mod:`src.pivot.p4_views` are acquisition **degradations**
(blur, noise, JPEG, vignette, ...), so averaging their biomarkers mostly undoes
the damage the perturbation did rather than adding information over the clean
prediction.  The standard, strictly anatomy-preserving test-time aggregation is
the dihedral group D4 (4 rotations x optional flip): the image is transformed,
the network run, and the probability map transformed back, so all 8 predictions
are pixel-aligned and can be averaged before a *single* biomarker computation.

This writes the same file layout as ``src/seg/infer.py`` under
``runs/pivot/pred_<ds>_tta8`` so ``src.pivot.p5_eval bio`` can score it.

CLI
---
    python -m src.pivot.p4_tta --dataset hrf --gpu 1 \
        --ckpt runs/seg/hrf/seed0/best.pt --tag tta8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch

from src.seg import data as segdata
from src.seg.infer import (load_model, load_native_fov, sliding_window_predict,
                           to_native)

cv2.setNumThreads(0)


def d4_forward(t: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.flip(t, dims=[-1]) if flip else t
    return torch.rot90(x, k, dims=(-2, -1))


def d4_inverse(a: np.ndarray, k: int, flip: bool) -> np.ndarray:
    a = np.rot90(a, -k)
    return np.fliplr(a) if flip else a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hrf")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="tta8")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ds = segdata.canon(args.dataset)
    ckpt = args.ckpt or os.path.join("runs", "seg", ds, f"seed{args.seed}", "best.pt")
    cfg = segdata.dataset_cfg(ds)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]
    recs = segdata.get_records(ds)
    _tr, _va, te, _ = segdata.make_splits(recs, segdata.DEFAULT_SPLIT_SEED)
    cache = os.path.join("runs", "_cache", f"{ds}_{longest}") if longest else None

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, _ck = load_model(ckpt, device)

    out_dir = os.path.join("runs", "pivot", f"pred_{ds}_{args.tag}")
    os.makedirs(os.path.join(out_dir, "prob"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mask"), exist_ok=True)

    from src.seg.evaluate import read_fov, read_gt
    from src.topo.metrics import cldice as cldice_fn
    from src.topo.metrics import dice as dice_fn

    fids = segdata.FullImageDataset(te, longest, cache_dir=cache, mem_cache=2)
    rows, px = [], []
    t_all = time.time()
    for i in range(len(fids)):
        item = fids[i]
        t0 = time.time()
        acc = None
        for flip in (False, True):
            for k in range(4):
                img = d4_forward(item["image"], k, flip).contiguous()
                pr = sliding_window_predict(model, img, patch, device,
                                            stride=patch // 2, amp=True, batch_size=4)
                pr = d4_inverse(pr, k, flip)
                acc = pr.astype(np.float64) if acc is None else acc + pr
        prob_r = (acc / 8.0).astype(np.float32)
        native_hw = item["native_hw"]
        prob = to_native(prob_r, native_hw)
        fov = load_native_fov({"fov_path": item["fov_path"],
                               "image_path": item["image_path"]}, native_hw)
        mask = ((prob >= 0.5) & (fov > 0)).astype(np.uint8)
        key = item["key"]
        np.save(os.path.join(out_dir, "prob", key + ".npy"), prob.astype(np.float16))
        cv2.imwrite(os.path.join(out_dir, "mask", key + ".png"), mask * 255)
        rows.append({"dataset": ds, "seed": args.tag, "split": "test", "image": key,
                     "subject_id": item["subject_id"],
                     "image_path": item["image_path"], "label_path": item["label_path"],
                     "label2_path": item["label2_path"], "fov_path": item["fov_path"],
                     "prob_path": os.path.abspath(os.path.join(out_dir, "prob", key + ".npy")),
                     "mask_path": os.path.abspath(os.path.join(out_dir, "mask", key + ".png")),
                     "native_h": native_hw[0], "native_w": native_hw[1],
                     "patch": patch, "stride": patch // 2, "threshold": 0.5,
                     "pred_fg_frac": float(mask.sum()) / max(1.0, float(fov.sum())),
                     "seconds": round(time.time() - t0, 3)})
        gt = read_gt(rows[-1], native_hw)
        fv = read_fov(rows[-1], native_hw)
        px.append({"image": key, "dice": float(dice_fn(mask, gt, fv)),
                   "cldice": float(cldice_fn(mask, gt, fv)),
                   "pred_fg_frac": rows[-1]["pred_fg_frac"]})
        print("[{}/{}] {} {:.1f}s dice={:.4f} cldice={:.4f}".format(
            i + 1, len(fids), key, rows[-1]["seconds"], px[-1]["dice"],
            px[-1]["cldice"]), flush=True)

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import pandas as pd
    pxdf = pd.DataFrame(px)
    pxdf.to_csv(os.path.join(out_dir, "pixel_metrics.csv"), index=False)
    meta = {"dataset": ds, "ckpt": os.path.abspath(ckpt), "tag": args.tag,
            "tta": "D4 (4 rotations x 2 flips), probability-level mean",
            "n_images": len(rows), "patch": patch, "stride": patch // 2,
            "resize_longest": longest, "threshold": 0.5,
            "mean_dice": float(pxdf["dice"].mean()),
            "mean_cldice": float(pxdf["cldice"].mean()),
            "total_seconds": round(time.time() - t_all, 1)}
    with open(os.path.join(out_dir, "infer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
