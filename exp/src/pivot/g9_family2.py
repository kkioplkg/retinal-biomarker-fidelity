"""Review point 9(b) -- a SECOND segmenter family for the C1 audit.

The paper's only segmenter is the S2 baseline U-Net (``src/seg/unet.py``,
7.77 M params, trained from scratch).  Reviewer point 9 asks whether the C1
measurement-calibration audit is a property of *that* network or of the
image-to-biomarker chain.  This module trains a **transformer** segmenter --
SegFormer-B0 (MiT-B0 hierarchical-attention encoder + all-MLP decoder, Xie et
al., NeurIPS 2021; 3.71 M parameters against the U-Net's 7.77 M, so the second
family is not also a capacity increase), via ``segmentation_models_pytorch``
0.5.0 -- on **exactly the
same splits, resolution convention, augmentation, loss and epoch budget** as
the baseline U-Net, and writes its predictions in exactly the layout
``src/pivot/p5_eval.py`` expects, so the biomarker stage and the whole C1 audit
run on it unchanged.

What is held identical to ``src/seg/train.py``
---------------------------------------------
* records / train / val / test splits (``segdata.make_splits``, split seed 1337)
* geometry: ``DATASET_CFG`` -- longest side 1536 + patch 768 (HRF, FIVES),
  native + patch 512 (DRIVE, CHASE_DB1)
* per-image FOV z-score normalisation, the same on-disk resized cache
* ``VesselPatchDataset`` with ``oversample_p`` 0.7, the same augmentation
* loss: ``SegLoss(w_bce=1, w_dice=1, w_cldice=0)``  (BCE + soft Dice)
* 100 iterations/epoch, batch 8 (DRIVE/CHASE) or 4 (HRF/FIVES),
  300 / 150 epochs, early stopping on val Dice with patience 60
* AMP on, ``cudnn.benchmark`` off, deterministic seeding
* inference: sliding window = patch, stride = patch/2, Gaussian blending,
  no flips, threshold 0.5 inside the FOV, probability resized to native
  resolution *before* thresholding

The two deliberate differences, both recorded in ``config.json``
----------------------------------------------------------------
1. **architecture** -- that is the point of the experiment.
2. **optimiser** -- Adam 1e-3 (the U-Net recipe) does not train a ViT-style
   encoder; it diverges within a few epochs.  SegFormer's published recipe is
   AdamW with a small base LR, so the arm uses ``AdamW(lr=3e-4, wd=0.01)`` with
   the *same* poly(0.9) decay and the same gradient clipping.  The LR was
   chosen on a 40-epoch DRIVE probe scored on the **validation** split only
   (3e-4 -> best val Dice 0.7989; 6e-5 -> 0.7941; the U-Net's 1e-3 diverges);
   no test split was consulted.  The encoder is
   ImageNet-pretrained (``smp-hub/mit_b0.imagenet``), which the U-Net is not --
   a transformer trained from scratch on 15-600 fundus images is not a fair
   "different family", it is a broken model.  Both facts are reported.

CLI
---
    python -m src.pivot.g9_family2 train --dataset fives --arch segformer_b0 \\
        --seed 0 --gpu 0
    python -m src.pivot.g9_family2 infer --dataset fives --arch segformer_b0 \\
        --seed 0 --gpu 0 --split test
    # then the usual CPU biomarker stage:
    python -m src.pivot.p5_eval bio --dataset fives --tag r2_segformer_b0_fives \\
        --root runs/pivot/r2/segformer_b0/fives/pred --procs 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.seg import data as segdata
from src.seg.infer import predict_records, sliding_window_predict
from src.seg.losses import SegLoss
from src.seg.train import make_scaler, poly_lr, set_determinism, validate

R2_ROOT = os.path.join("runs", "pivot", "r2")

#: architecture id -> (smp constructor name, encoder name, pretty label)
ARCHS: Dict[str, dict] = {
    # PRIMARY (locked 2026-09-18 after the methods consultation): MiT-B0 is
    # 3.7 M parameters against the baseline U-Net's 7.77 M, so the second
    # family is not also a capacity increase -- the comparison isolates the
    # architecture, not the parameter count.
    "segformer_b0": dict(cls="Segformer", encoder="mit_b0", weights="imagenet",
                         family="transformer",
                         label="SegFormer-B0 (MiT-B0 encoder, all-MLP decoder)"),
    # kept for reference; not part of the locked plan
    "segformer_b2": dict(cls="Segformer", encoder="mit_b2", weights="imagenet",
                         family="transformer",
                         label="SegFormer (MiT-B2 encoder, all-MLP decoder)"),
    # fallback / second opinion: CNN but a completely different topology
    # (dilated ResNet-50 + ASPP, no U-shaped skip ladder)
    "deeplabv3plus_r50": dict(cls="DeepLabV3Plus", encoder="resnet50",
                              weights="imagenet", family="cnn-aspp",
                              label="DeepLabV3+ (ResNet-50 encoder, ASPP)"),
}

#: optimiser for the transformer arm (see the module docstring)
FT_LR = 3e-4
FT_WD = 0.01


def build_model(arch: str) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    spec = ARCHS[arch]
    ctor = getattr(smp, spec["cls"])
    return ctor(encoder_name=spec["encoder"], encoder_weights=spec["weights"],
                in_channels=3, classes=1)


def run_dir(arch: str, dataset: str, seed: int = 0) -> str:
    """``runs/pivot/r2/<arch>/<ds>`` for seed 0, ``<ds>_s<k>`` for the others.

    The suffix is the one :mod:`src.pivot.g9_bio_watch` parses back off, so the
    biomarker tag of a directory is always ``r2_<arch>_<ds>[_s<k>]`` and the
    dataset it is measured against is always the bare ``<ds>``.
    """
    d = os.path.join(R2_ROOT, arch, segdata.canon(dataset))
    return d if seed == 0 else d + "_s%d" % seed


def tag_of(arch: str, dataset: str, seed: int = 0, suffix: str = "") -> str:
    t = "r2_%s_%s" % (arch, segdata.canon(dataset))
    if seed:
        t += "_s%d" % seed
    return t + suffix


# --------------------------------------------------------------------------
def cmd_train(args) -> int:
    ds_name = segdata.canon(args.dataset)
    cfg = segdata.dataset_cfg(ds_name)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]
    epochs = args.epochs if args.epochs is not None else segdata.DEFAULT_EPOCHS[ds_name]
    batch = args.batch_size if args.batch_size is not None else segdata.DEFAULT_BATCH[ds_name]
    out_dir = args.out or run_dir(args.arch, ds_name, args.seed)
    os.makedirs(out_dir, exist_ok=True)

    recs = segdata.get_records(ds_name)
    tr_recs, va_recs, te_recs, split_info = segdata.make_splits(recs, args.split_seed)
    if not va_recs:
        raise RuntimeError("empty validation split for %s" % ds_name)

    set_determinism(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.gpu if use_cuda else "cpu")
    amp = (not args.no_amp) and use_cuda
    cache = args.cache_dir
    if cache is None and longest is not None:
        cache = os.path.join("runs", "_cache", "%s_%s" % (ds_name, longest))

    train_ds = segdata.VesselPatchDataset(
        tr_recs, patch, longest,
        samples_per_epoch=args.iters_per_epoch * batch,
        augment=not args.no_augment, oversample_p=args.oversample_p,
        seed=args.seed, cache_dir=cache,
        mem_cache=8 if longest is not None else 64)
    val_sel = sorted(va_recs, key=lambda r: str(r["image_path"]))
    cap = args.val_max_images
    if cap < 0:
        cap = min(len(val_sel), 30)
    if cap:
        val_sel = val_sel[:cap]
    val_ds = segdata.FullImageDataset(val_sel, longest, cache_dir=cache, mem_cache=4)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=False,
                        num_workers=args.workers, pin_memory=use_cuda,
                        drop_last=True, worker_init_fn=segdata.seed_worker,
                        generator=g, persistent_workers=(args.workers > 0))

    model = build_model(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = SegLoss(w_bce=args.w_bce, w_dice=args.w_dice, w_cldice=args.w_cldice)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = make_scaler(amp)

    spec = ARCHS[args.arch]
    config = {
        "review_point": "9b-second-segmenter-family",
        "arch": args.arch, "arch_label": spec["label"], "family": spec["family"],
        "encoder": spec["encoder"], "encoder_weights": spec["weights"],
        "dataset": ds_name, "seed": args.seed, "gpu": args.gpu, "epochs": epochs,
        "iters_per_epoch": args.iters_per_epoch, "batch_size": batch,
        "patch": patch, "resize_longest": longest, "lr": args.lr,
        "lr_schedule": "poly(0.9)", "optimizer": "AdamW",
        "weight_decay": args.weight_decay, "amp": amp,
        "loss": {"bce": args.w_bce, "dice": args.w_dice, "cldice": args.w_cldice},
        "oversample_p": args.oversample_p, "augment": not args.no_augment,
        "patience": args.patience, "n_params": n_params,
        "split_seed": args.split_seed,
        "baseline_reference": "runs/seg/%s/seed%d" % (ds_name, args.seed),
        "differences_from_baseline": [
            "architecture (%s vs 5-level U-Net, 7.77M params)" % spec["label"],
            "optimizer AdamW(lr=%g, wd=%g) instead of Adam(lr=1e-3, wd=0) -- "
            "the U-Net LR diverges on a ViT-style encoder" % (args.lr,
                                                              args.weight_decay),
            "ImageNet-pretrained encoder (%s); the baseline U-Net is trained "
            "from scratch" % spec["weights"],
        ],
        "val_images_used": [
            os.path.splitext(os.path.basename(str(r["image_path"])))[0]
            for r in val_sel],
        "torch": torch.__version__, "device": str(device),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("[g9] %s %s seed=%d device=%s params=%.2fM" % (
        args.arch, ds_name, args.seed, device, n_params / 1e6), flush=True)
    print("[g9] patch=%d batch=%d iters/ep=%d epochs=%d amp=%s lr=%g" % (
        patch, batch, args.iters_per_epoch, epochs, amp, args.lr), flush=True)
    print("[g9] train imgs=%d val imgs=%d (of %d) test imgs=%d" % (
        len(tr_recs), len(val_sel), len(va_recs), len(te_recs)), flush=True)

    best_dice, best_epoch = -1.0, -1
    log: List[dict] = []
    t_start = time.time()
    for epoch in range(epochs):
        lr = poly_lr(args.lr, epoch, epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "cldice": 0.0}
        n_seen = 0
        for batch_data in loader:
            x = batch_data["image"].to(device, non_blocking=True)
            y = batch_data["label"].to(device, non_blocking=True)
            m = batch_data["fov"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                logits = model(x)
                loss, parts = criterion(logits, y, m)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.detach())
            for k in ("bce", "dice", "cldice"):
                if k in parts:
                    sums[k] += float(parts[k])
            n_seen += 1

        entry = {"epoch": epoch, "lr": lr,
                 "train_loss": sums["loss"] / max(1, n_seen),
                 "train_bce": sums["bce"] / max(1, n_seen),
                 "train_dice_loss": sums["dice"] / max(1, n_seen),
                 "train_seconds": round(time.time() - t0, 2)}

        if ((epoch + 1) % max(1, args.val_every) == 0) or (epoch == epochs - 1):
            tv = time.time()
            vm = validate(model, val_ds, patch, device, amp, args.sw_batch)
            entry.update(vm)
            entry["val_seconds"] = round(time.time() - tv, 2)
            if vm["val_dice"] > best_dice:
                best_dice, best_epoch = vm["val_dice"], epoch
                torch.save({"model": model.state_dict(), "config": config,
                            "epoch": epoch, "val_dice": best_dice},
                           os.path.join(out_dir, "best.pt"))
                entry["is_best"] = True

        entry["epoch_seconds"] = round(time.time() - t0, 2)
        entry["elapsed_seconds"] = round(time.time() - t_start, 2)
        log.append(entry)
        torch.save({"model": model.state_dict(), "config": config,
                    "epoch": epoch, "best_dice": best_dice,
                    "best_epoch": best_epoch},
                   os.path.join(out_dir, "last.pt"))
        with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "split": split_info, "log": log,
                       "best_dice": best_dice, "best_epoch": best_epoch}, f,
                      indent=2)
        msg = "ep %4d/%d lr %.2e loss %.4f" % (epoch + 1, epochs, lr,
                                               entry["train_loss"])
        if "val_dice" in entry:
            msg += " | val Dice %.4f (best %.4f @ep%d)" % (
                entry["val_dice"], best_dice, best_epoch + 1)
        msg += " | %.1fs" % entry["epoch_seconds"]
        print(msg, flush=True)

        if best_epoch >= 0 and (epoch - best_epoch) >= args.patience:
            print("early stopping: no val Dice improvement for %d epochs"
                  % args.patience, flush=True)
            break

    total = time.time() - t_start
    summary = {"arch": args.arch, "dataset": ds_name, "seed": args.seed,
               "epochs_run": len(log), "best_dice": best_dice,
               "best_epoch": best_epoch, "total_seconds": round(total, 1),
               "mean_epoch_seconds": round(total / max(1, len(log)), 2),
               "best_ckpt": os.path.abspath(os.path.join(out_dir, "best.pt")),
               "last_ckpt": os.path.abspath(os.path.join(out_dir, "last.pt")),
               "n_params": n_params}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------
def load_family2(ckpt_path: str, device: torch.device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    arch = cfg.get("arch")
    if arch not in ARCHS:
        raise SystemExit("checkpoint %s has no known 'arch' in its config"
                         % ckpt_path)
    model = build_model(arch)
    missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
    if missing:
        print("[warn] missing keys: %s" % (list(missing)[:6],))
    model.to(device).eval()
    return model, ck


def cmd_infer(args) -> int:
    """Sliding-window inference, byte-for-byte the ``p5_eval infer`` layout."""
    import pandas as pd

    from src.pivot.common import read_gray
    from src.seg.evaluate import read_fov, read_gt
    from src.topo.metrics import cldice as cldice_fn
    from src.topo.metrics import dice as dice_fn

    ds = segdata.canon(args.dataset)
    cfg = segdata.dataset_cfg(ds)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]
    if args.infer_patch:
        patch = int(args.infer_patch)
    if args.infer_longest is not None:
        longest = int(args.infer_longest) or None

    recs = segdata.get_records(ds)
    tr, va, te, _ = segdata.make_splits(recs, args.split_seed)
    subset = {"test": te, "train": tr, "val": va}[args.split]
    cache = os.path.join("runs", "_cache", "%s_%s" % (ds, longest)) if longest else None

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    ckpt = args.ckpt or os.path.join(run_dir(args.arch, ds, args.seed), "best.pt")
    model, ck = load_family2(ckpt, device)

    tag = args.tag or tag_of(args.arch, ds, args.seed)
    od = args.root or os.path.join(run_dir(args.arch, ds, args.seed), "pred")
    os.makedirs(od, exist_ok=True)
    rows, meta = predict_records(model, subset, patch, longest, device, od,
                                 dataset_name=ds, seed_tag=tag,
                                 split_label=args.split, threshold=0.5,
                                 stride=max(1, patch // 2), amp=True,
                                 sw_batch=4, save_prob=True, cache_dir=cache,
                                 manifest_name="manifest.csv")
    meta = {"dataset": ds, "ckpt": os.path.abspath(ckpt), "tag": tag,
            "arch": ck.get("config", {}).get("arch"),
            "arch_label": ck.get("config", {}).get("arch_label"),
            "ckpt_epoch": ck.get("epoch"), "ckpt_val_dice": ck.get("val_dice"),
            "split": args.split, **meta}

    px = []
    for r in rows:
        native_hw = (int(r["native_h"]), int(r["native_w"]))
        gt = read_gt(r, native_hw)
        fov = read_fov(r, native_hw)
        pred = (read_gray(r["mask_path"]) > 127).astype(np.uint8)
        px.append({"image": r["image"],
                   "dice": float(dice_fn(pred, gt, fov)),
                   "cldice": float(cldice_fn(pred, gt, fov)),
                   "pred_fg_frac": r["pred_fg_frac"]})
        print("  px %s dice=%.4f cldice=%.4f" % (px[-1]["image"], px[-1]["dice"],
                                                 px[-1]["cldice"]), flush=True)
    pxdf = pd.DataFrame(px)
    pxdf.to_csv(os.path.join(od, "pixel_metrics.csv"), index=False)
    meta["mean_dice"] = float(pxdf["dice"].mean())
    meta["mean_cldice"] = float(pxdf["cldice"].mean())
    with open(os.path.join(od, "infer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--arch", default="segformer_b0", choices=sorted(ARCHS))
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--out", default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--iters-per-epoch", type=int, default=100)
    t.add_argument("--lr", type=float, default=FT_LR)
    t.add_argument("--weight-decay", type=float, default=FT_WD)
    t.add_argument("--patience", type=int, default=60)
    t.add_argument("--w-bce", type=float, default=1.0)
    t.add_argument("--w-dice", type=float, default=1.0)
    t.add_argument("--w-cldice", type=float, default=0.0)
    t.add_argument("--oversample-p", type=float, default=0.7)
    t.add_argument("--split-seed", type=int, default=segdata.DEFAULT_SPLIT_SEED)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--val-every", type=int, default=1)
    t.add_argument("--val-max-images", type=int, default=-1)
    t.add_argument("--sw-batch", type=int, default=4)
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--cache-dir", default=None)
    t.set_defaults(fn=cmd_train)

    i = sub.add_parser("infer")
    i.add_argument("--dataset", required=True)
    i.add_argument("--arch", default="segformer_b0", choices=sorted(ARCHS))
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--gpu", type=int, default=0)
    i.add_argument("--ckpt", default=None)
    i.add_argument("--tag", default=None)
    i.add_argument("--root", default=None)
    i.add_argument("--split", default="test", choices=["test", "train", "val"])
    i.add_argument("--split-seed", type=int, default=segdata.DEFAULT_SPLIT_SEED)
    i.add_argument("--infer-longest", type=int, default=None)
    i.add_argument("--infer-patch", type=int, default=0)
    i.set_defaults(fn=cmd_infer)

    a = ap.parse_args(argv)
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..")))
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
