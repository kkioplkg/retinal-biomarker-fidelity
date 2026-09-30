"""Train the S2 baseline U-Net segmenter.

    cd exp
    python -m src.seg.train --dataset drive --seed 0 --gpu 0 \
        --out runs/seg/drive/seed0

Defaults follow exp/EXPERIMENT_PLAN.md S2: Adam lr 1e-3 with poly decay,
BCE + soft Dice, strong augmentation, mixed precision, 100 iterations per epoch,
300 epochs for DRIVE/CHASE/STARE and 150 for HRF/FIVES, early stopping on the
validation Dice with patience 60.  Everything is seeded deterministically and
``cudnn.benchmark`` is off.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.seg import data as segdata
from src.seg.infer import load_model as infer_load_model
from src.seg.infer import predict_records, sliding_window_predict
from src.seg.losses import SegLoss
from src.seg.unet import UNet


# --------------------------------------------------------------------------
def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def poly_lr(base_lr: float, epoch: int, max_epochs: int, exponent: float = 0.9) -> float:
    return base_lr * (1.0 - float(epoch) / float(max(1, max_epochs))) ** exponent


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def dice_score(pred_bin: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> float:
    m = fov > 0
    p = (pred_bin > 0) & m
    g = (gt > 0) & m
    inter = float(np.logical_and(p, g).sum())
    denom = float(p.sum() + g.sum())
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


# --------------------------------------------------------------------------
@torch.no_grad()
def validate(model, val_ds, patch, device, amp, sw_batch, threshold=0.5) -> Dict[str, float]:
    model.eval()
    dices: List[float] = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        prob = sliding_window_predict(
            model, item["image"], patch, device,
            stride=patch // 2, amp=amp, batch_size=sw_batch,
        )
        fov = item["fov"][0].numpy()
        gt = item["label"][0].numpy()
        dices.append(dice_score(prob >= threshold, gt, fov))
    return {"val_dice": float(np.mean(dices)) if dices else 0.0,
            "val_dice_std": float(np.std(dices)) if dices else 0.0,
            "n_val": len(dices)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train the baseline vessel U-Net")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="default 300 (DRIVE/CHASE/STARE) or 150 (HRF/FIVES)")
    ap.add_argument("--out", default=None, help="default runs/seg/<ds>/seed<k>")
    ap.add_argument("--iters-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=None, help="default 8 / 4 by dataset")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--w-bce", type=float, default=1.0)
    ap.add_argument("--w-dice", type=float, default=1.0)
    ap.add_argument("--w-cldice", type=float, default=0.0)
    ap.add_argument("--deep-supervision", action="store_true")
    ap.add_argument("--oversample-p", type=float, default=0.7)
    ap.add_argument("--split-seed", type=int, default=segdata.DEFAULT_SPLIT_SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--val-max-images", type=int, default=-1,
                    help="-1 = min(all, 30); 0 = all")
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--crossfit", type=int, default=0,
                    help="K-fold cross-fit for OOF predictions (0/off = normal training; "
                         "requires --fold)")
    ap.add_argument("--fold", type=int, default=None,
                    help="held-out fold index in [0, K) when --crossfit is set")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the dataset records/splits, print fold membership counts, "
                         "and exit without touching the GPU (no model/DataLoader/CUDA)")
    args = ap.parse_args(argv)

    ds_name = segdata.canon(args.dataset)
    cfg = segdata.dataset_cfg(ds_name)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]
    epochs = args.epochs if args.epochs is not None else segdata.DEFAULT_EPOCHS[ds_name]
    batch = args.batch_size if args.batch_size is not None else segdata.DEFAULT_BATCH[ds_name]

    crossfit = int(args.crossfit)
    if crossfit and args.fold is None:
        raise SystemExit("--fold is required when --crossfit is set (K={})".format(crossfit))

    if crossfit:
        default_out = os.path.join(
            "runs", "seg", ds_name,
            "crossfit{}_fold{}_seed{}".format(crossfit, args.fold, args.seed),
        )
    else:
        default_out = os.path.join("runs", "seg", ds_name, "seed{}".format(args.seed))
    out_dir = args.out or default_out

    # ---- data (built before touching the GPU, so --dry-run never does) -
    recs = segdata.get_records(ds_name)
    if crossfit:
        tr_recs, va_recs, held_recs, te_recs, split_info = segdata.make_crossfit_splits(
            recs, crossfit, args.fold, split_seed=args.split_seed)
    else:
        tr_recs, va_recs, te_recs, split_info = segdata.make_splits(recs, args.split_seed)
        held_recs = []
    if not va_recs:
        raise RuntimeError("empty validation split for {}".format(ds_name))
    if crossfit and not held_recs:
        raise RuntimeError("empty held-out fold {} for {}".format(args.fold, ds_name))

    if args.dry_run:
        summary = {
            "dry_run": True,
            "dataset": ds_name,
            "crossfit": crossfit,
            "fold": args.fold,
            "split_seed": args.split_seed,
            "n_train": len(tr_recs),
            "n_val": len(va_recs),
            "n_held_out": len(held_recs),
            "n_test": len(te_recs),
            "out_dir": out_dir,
        }
        print(json.dumps(summary, indent=2))
        print("[dry-run] no GPU / model / DataLoader touched; exiting.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    set_determinism(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")
    amp = (not args.no_amp) and use_cuda

    cache = args.cache_dir
    if cache is None and longest is not None:
        cache = os.path.join("runs", "_cache", "{}_{}".format(ds_name, longest))

    train_ds = segdata.VesselPatchDataset(
        tr_recs, patch, longest,
        samples_per_epoch=args.iters_per_epoch * batch,
        augment=not args.no_augment,
        oversample_p=args.oversample_p,
        seed=args.seed, cache_dir=cache,
        mem_cache=8 if longest is not None else 64,
    )
    val_sel = sorted(va_recs, key=lambda r: str(r["image_path"]))
    cap = args.val_max_images
    if cap < 0:
        cap = min(len(val_sel), 30)
    if cap:
        val_sel = val_sel[:cap]
    val_ds = segdata.FullImageDataset(val_sel, longest, cache_dir=cache, mem_cache=4)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        train_ds, batch_size=batch, shuffle=False, num_workers=args.workers,
        pin_memory=use_cuda, drop_last=True, worker_init_fn=segdata.seed_worker,
        generator=g, persistent_workers=(args.workers > 0),
    )

    # ---- model ---------------------------------------------------------
    model = UNet(in_channels=3, out_channels=1, base_channels=32, num_levels=5,
                 max_channels=512, deep_supervision=args.deep_supervision).to(device)
    n_params = model.num_parameters()
    criterion = SegLoss(w_bce=args.w_bce, w_dice=args.w_dice, w_cldice=args.w_cldice)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scaler = make_scaler(amp)

    config = {
        "dataset": ds_name, "seed": args.seed, "gpu": args.gpu, "epochs": epochs,
        "iters_per_epoch": args.iters_per_epoch, "batch_size": batch,
        "patch": patch, "resize_longest": longest, "lr": args.lr,
        "lr_schedule": "poly(0.9)", "optimizer": "Adam",
        "weight_decay": args.weight_decay, "amp": amp,
        "loss": {"bce": args.w_bce, "dice": args.w_dice, "cldice": args.w_cldice},
        "deep_supervision": bool(args.deep_supervision),
        "oversample_p": args.oversample_p, "augment": not args.no_augment,
        "patience": args.patience, "n_params": n_params,
        "in_channels": 3, "out_channels": 1, "base_channels": 32,
        "num_levels": 5, "max_channels": 512,
        "crossfit": crossfit, "fold": (args.fold if crossfit else None),
        "val_images_used": [
            os.path.splitext(os.path.basename(str(r["image_path"])))[0] for r in val_sel
        ],
        "torch": torch.__version__,
        "device": str(device),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print("dataset={} seed={} device={} params={:,} ({:.2f}M)".format(
        ds_name, args.seed, device, n_params, n_params / 1e6))
    print("patch={} batch={} iters/epoch={} epochs={} amp={}".format(
        patch, batch, args.iters_per_epoch, epochs, amp))
    print("train imgs={} val imgs={} (of {}) test imgs={} split_seed={}".format(
        len(tr_recs), len(val_sel), len(va_recs), len(te_recs), args.split_seed))
    if crossfit:
        print("crossfit K={} fold={} held_out imgs={}".format(
            crossfit, args.fold, len(held_recs)))

    start_epoch, best_dice, best_epoch = 0, -1.0, -1
    log: List[dict] = []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_dice = float(ck.get("best_dice", -1.0))
        best_epoch = int(ck.get("best_epoch", -1))
        log = list(ck.get("log", []))
        print("resumed from {} at epoch {}".format(args.resume, start_epoch))

    log_path = os.path.join(out_dir, "log.json")
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
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

        train_time = time.time() - t0
        entry = {
            "epoch": epoch, "lr": lr,
            "train_loss": sums["loss"] / max(1, n_seen),
            "train_bce": sums["bce"] / max(1, n_seen),
            "train_dice_loss": sums["dice"] / max(1, n_seen),
            "train_cldice_loss": sums["cldice"] / max(1, n_seen),
            "train_seconds": round(train_time, 2),
        }

        do_val = ((epoch + 1) % max(1, args.val_every) == 0) or (epoch == epochs - 1)
        if do_val:
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

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "epoch": epoch, "best_dice": best_dice,
                    "best_epoch": best_epoch, "log": log},
                   os.path.join(out_dir, "last.pt"))
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"config": config, "split": split_info, "log": log,
                       "best_dice": best_dice, "best_epoch": best_epoch}, f, indent=2)

        msg = "ep {:>4}/{} lr {:.2e} loss {:.4f}".format(
            epoch + 1, epochs, lr, entry["train_loss"])
        if "val_dice" in entry:
            msg += " | val Dice {:.4f} (best {:.4f} @ep{})".format(
                entry["val_dice"], best_dice, best_epoch + 1)
        msg += " | {:.1f}s".format(entry["epoch_seconds"])
        print(msg, flush=True)

        if best_epoch >= 0 and (epoch - best_epoch) >= args.patience:
            print("early stopping: no val Dice improvement for {} epochs".format(
                args.patience))
            break

    total = time.time() - t_start
    summary = {
        "dataset": ds_name, "seed": args.seed, "epochs_run": len(log),
        "best_dice": best_dice, "best_epoch": best_epoch,
        "total_seconds": round(total, 1),
        "mean_epoch_seconds": round(total / max(1, len(log)), 2),
        "best_ckpt": os.path.abspath(os.path.join(out_dir, "best.pt")),
        "last_ckpt": os.path.abspath(os.path.join(out_dir, "last.pt")),
        "n_params": n_params,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- crossfit: OOF inference on the held-out fold's *training* images --
    if crossfit:
        best_ckpt = os.path.join(out_dir, "best.pt")
        if not os.path.exists(best_ckpt):
            print("[crossfit] no best.pt found ({} epochs run); skipping OOF inference".format(
                len(log)))
        else:
            print("crossfit: OOF inference on held-out fold {} ({} images) -> {}".format(
                args.fold, len(held_recs), os.path.join(out_dir, "pred_oof")))
            oof_model, oof_ck = infer_load_model(best_ckpt, device)
            oof_dir = os.path.join(out_dir, "pred_oof")
            oof_rows, oof_meta = predict_records(
                oof_model, held_recs, patch, longest, device, oof_dir,
                dataset_name=ds_name,
                seed_tag=oof_ck.get("config", {}).get("seed", args.seed),
                split_label="train_oof",
                threshold=0.5,
                stride=max(1, patch // 2),
                amp=amp,
                sw_batch=args.sw_batch,
                save_prob=True,
                cache_dir=cache,
                manifest_name="oof_manifest.csv",
            )
            oof_meta = {
                "dataset": ds_name,
                "ckpt": os.path.abspath(best_ckpt),
                "split": "train_oof",
                "crossfit_k": crossfit,
                "fold": args.fold,
                **oof_meta,
            }
            with open(os.path.join(oof_dir, "oof_infer_meta.json"), "w", encoding="utf-8") as f:
                json.dump(oof_meta, f, indent=2)
            print("wrote {} OOF predictions -> {}".format(len(oof_rows), oof_dir))
            print(json.dumps(oof_meta, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
