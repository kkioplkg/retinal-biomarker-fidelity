"""Probe P5 -- measurement-aware fine-tuning of an existing segmenter.

Fine-tunes ``runs/seg/<ds>/seed0/best.pt`` with

    L = CE-Dice  +  lambda * C

where ``C`` is a *biomarker-consistency* term computed on the training patch
with the differentiable surrogates validated in probe P1
(``src/pivot/surrogates.py``).  Two variants:

``--loss-kind measure``   (ours)  density + length + FD, with FD taken on the
                          **soft skeleton** using the resolution-adaptive box
                          ladder ``2 .. min(H, W)/4`` -- the only FD surrogate
                          that tracked the pipeline in P1 (rho 0.85-1.00).
                          Tortuosity is deliberately excluded (P1: rho ~0).

``--loss-kind cfloss``    (closest prior, CF-Loss-style) density + FD only, FD
                          on the **mask** with the fixed ``2..64`` ladder.

Each term is a *relative* error ``|s(p) - s(y)| / (|s(y)| + eps)`` so the three
have comparable magnitude and a single ``lambda`` controls the whole block;
the per-term values are logged so the "consistency is 10-30 % of the loss"
condition can be checked after the fact.

CLI
---
    python -m src.pivot.p5_finetune --dataset hrf --gpu 1 \
        --loss-kind measure --lam 0.3 --epochs 50 \
        --out runs/pivot/ft_hrf_seed0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.seg import data as segdata
from src.seg.infer import load_model, sliding_window_predict
from src.seg.losses import SegLoss, soft_skeletonize
from src.seg.train import dice_score, make_scaler, poly_lr, set_determinism
from src.pivot.surrogates import FD_BOX_SIZES, auto_box_sizes, box_counts, _loglog_slope
from src.pivot.cf_loss import CF_ALPHA, CF_BETA, CF_GAMMA, cf_fd_vd

EPS = 1e-6


# --------------------------------------------------------------------------
def _fd_from(x: torch.Tensor, sizes) -> torch.Tensor:
    return -_loglog_slope(box_counts(x, sizes), sizes)


ALL_TERMS = ("density", "length", "fd")


def consistency_terms(p: torch.Tensor, y: torch.Tensor, fov: torch.Tensor,
                      kind: str, num_iter: int = 10,
                      terms: Sequence[str] = ALL_TERMS,
                      fd_ladder: str = "adaptive") -> Dict[str, torch.Tensor]:
    """Per-sample relative surrogate errors, shape (B,) each.

    ``p`` soft probability, ``y`` GT label, ``fov`` binary, all (B, 1, H, W).
    The GT side is detached (it is a constant target).

    ``terms`` selects which surrogates enter ``C`` (E2 leave-one-out ablation);
    ``fd_ladder`` selects the box ladder of the FD surrogate in the ``measure``
    variant -- ``adaptive`` = ``2 .. min(H, W)/4`` (ours), ``fixed`` = the
    CF-Loss ``2..64`` ladder (E2 ladder ablation).  ``kind='cfloss'`` is the
    frozen prior baseline and ignores both.
    """
    pf, yf = p * fov, y * fov
    h, w = p.shape[-2:]
    out: Dict[str, torch.Tensor] = {}
    want = set(terms)

    # ---- density (both variants) ----
    if "density" in want or kind == "cfloss":
        denom = fov.sum(dim=(1, 2, 3)).clamp_min(1.0)
        d_p = pf.sum(dim=(1, 2, 3)) / denom
        d_y = (yf.sum(dim=(1, 2, 3)) / denom).detach()
        out["density"] = (d_p - d_y).abs() / (d_y + EPS)

    if kind == "cfloss":
        # CF-Loss style: density + fractal dimension of the *mask*, fixed ladder
        fd_p = _fd_from(pf, FD_BOX_SIZES)
        fd_y = _fd_from(yf, FD_BOX_SIZES).detach()
        out["fd"] = (fd_p - fd_y).abs() / (fd_y.abs() + EPS)
        return out

    # ---- ours: length + FD on the soft skeleton, adaptive ladder ----
    if not ({"length", "fd"} & want):
        return out
    sk_p = soft_skeletonize(pf, num_iter=num_iter)
    with torch.no_grad():
        sk_y = soft_skeletonize(yf, num_iter=num_iter)
    if "length" in want:
        L_p = sk_p.sum(dim=(1, 2, 3))
        L_y = sk_y.sum(dim=(1, 2, 3)).detach()
        out["length"] = (L_p - L_y).abs() / (L_y + 1.0)

    if "fd" in want:
        sizes = auto_box_sizes(h, w) if fd_ladder == "adaptive" else FD_BOX_SIZES
        fd_p = _fd_from(sk_p, sizes)
        fd_y = _fd_from(sk_y, sizes).detach()
        out["fd"] = (fd_p - fd_y).abs() / (fd_y.abs() + EPS)
    return out


def consistency_loss(p, y, fov, kind, min_gt_density=0.005,
                     terms: Sequence[str] = ALL_TERMS,
                     fd_ladder: str = "adaptive"):
    """Mean over the batch of the summed relative errors.

    Patches whose GT vessel fraction is below ``min_gt_density`` are dropped:
    FD / length of a near-empty patch is numerically meaningless.
    """
    tt = consistency_terms(p, y, fov, kind, terms=terms, fd_ladder=fd_ladder)
    if not tt:
        raise ValueError("consistency loss has no active term (terms=%r)" % (terms,))
    gt_d = (y * fov).sum(dim=(1, 2, 3)) / fov.sum(dim=(1, 2, 3)).clamp_min(1.0)
    keep = (gt_d > min_gt_density).float().detach()
    n = keep.sum().clamp_min(1.0)
    parts = {k: float((v.detach() * keep).sum() / n) for k, v in tt.items()}
    total = sum((v * keep).sum() / n for v in tt.values())
    return total, parts


# --------------------------------------------------------------------------
@torch.no_grad()
def _surrogate_measures(x: torch.Tensor, fov: torch.Tensor) -> Dict[str, float]:
    """density / skeleton length / adaptive-ladder FD of one binary map.

    The P1-validated differentiable surrogates, evaluated on a *hard* mask --
    the same three quantities the consistency loss optimises, so a checkpoint
    can be selected on validation *measurement fidelity* instead of val Dice.
    """
    xf = (x * fov)[None, None]
    h, w = x.shape[-2:]
    sk = soft_skeletonize(xf, num_iter=10)
    fd = -_loglog_slope(box_counts(sk, auto_box_sizes(h, w)), auto_box_sizes(h, w))
    return {"density": float(xf.sum() / fov.sum().clamp_min(1.0)),
            "length": float(sk.sum()),
            "fd": float(fd.reshape(-1)[0])}


@torch.no_grad()
def validate(model, val_ds, patch, device, amp, sw_batch, threshold=0.5,
             fidelity: bool = False):
    """Mean val Dice, and (optionally) the validation measurement fidelity.

    Returns ``(dice, fid)`` where ``fid`` holds ``r_<s>`` (Pearson across the
    val images, NaN when fewer than 4 images), ``mae_<s>`` (mean relative
    error), and their means ``fid_r`` / ``fid_mae``.
    """
    model.eval()
    dices, meas = [], []
    for i in range(len(val_ds)):
        item = val_ds[i]
        prob = sliding_window_predict(model, item["image"], patch, device,
                                      stride=patch // 2, amp=amp, batch_size=sw_batch)
        lab = item["label"][0].numpy()
        fovn = item["fov"][0].numpy()
        dices.append(dice_score(prob >= threshold, lab, fovn))
        if fidelity:
            f = torch.from_numpy(fovn.astype(np.float32)).to(device)
            pm = torch.from_numpy((prob >= threshold).astype(np.float32)).to(device)
            gm = torch.from_numpy(lab.astype(np.float32)).to(device)
            meas.append((_surrogate_measures(pm, f), _surrogate_measures(gm, f)))
    dice = float(np.mean(dices)) if dices else 0.0
    if not fidelity:
        return dice, {}
    fid: Dict[str, float] = {}
    rs, maes = [], []
    for k in ("density", "length", "fd"):
        a = np.array([m[0][k] for m in meas], dtype=float)
        b = np.array([m[1][k] for m in meas], dtype=float)
        if len(a) >= 4 and np.std(a) > 0 and np.std(b) > 0:
            r = float(np.corrcoef(a, b)[0, 1])
        else:
            r = float("nan")
        mae = float(np.mean(np.abs(a - b) / (np.abs(b) + 1e-6)))
        fid["r_" + k], fid["mae_" + k] = r, mae
        rs.append(r); maes.append(mae)
    fid["fid_r"] = float(np.mean(rs)) if np.all(np.isfinite(rs)) else float("nan")
    fid["fid_mae"] = float(np.mean(maes))
    return dice, fid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hrf")
    ap.add_argument("--ckpt", default=None, help="default runs/seg/<ds>/seed0/best.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--iters-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--loss-kind", default="measure",
                    choices=["measure", "cfloss", "cf_faithful", "cf_on_base"],
                    help="measure = ReliSeg surrogates; cf_faithful = the "
                         "published CF-Loss objective ported verbatim "
                         "(beta*CE + alpha*L_FD + gamma*L_vd, no Dice); "
                         "cf_on_base = the published feature terms bolted onto "
                         "our base loss (isolates the terms from the base "
                         "objective); cfloss = the retired 2..64-ladder variant")
    ap.add_argument("--terms", default="density,length,fd",
                    help="which consistency surrogates enter C (E2 leave-one-out "
                         "ablation); ignored by --loss-kind cfloss")
    ap.add_argument("--fd-ladder", default="adaptive", choices=["adaptive", "fixed"],
                    help="box ladder of the FD surrogate in the measure variant")
    ap.add_argument("--cf-scale", type=float, default=1.0,
                    help="multiplier on the published CF-Loss feature block "
                         "(alpha*L_FD + gamma*L_vd).  1.0 = the published "
                         "weighting; a larger value is the 'was the baseline "
                         "simply under-weighted?' control, calibrated so the "
                         "block is ~25%% of the loss at initialisation, exactly "
                         "as lambda was calibrated for ReliSeg")
    ap.add_argument("--select-by", default="auto", choices=["auto", "r", "mae"],
                    help="validation measurement-fidelity rule for best_fid.pt: "
                         "'r' = mean Pearson r over density/length/FD across the "
                         "val images, 'mae' = mean relative surrogate error "
                         "(the only well-defined one when n_val < 4); "
                         "'auto' picks r when n_val >= 4 else mae")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="cap the per-epoch validation set to N evenly spaced "
                         "images (0 = all); FIVES has 90 val images and "
                         "validating all of them every epoch dominates wall clock")
    ap.add_argument("--w-cldice", type=float, default=0.0,
                    help="keep a soft-clDice term in the base loss (0 = the "
                         "original CE-Dice objective)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--oversample-p", type=float, default=0.7)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="run 5 batches, print the loss decomposition, exit")
    args = ap.parse_args(argv)

    terms = tuple(t.strip() for t in str(args.terms).split(",") if t.strip())
    bad = [t for t in terms if t not in ALL_TERMS]
    if bad:
        raise SystemExit("unknown --terms entries: %r (known %r)" % (bad, ALL_TERMS))

    ds_name = segdata.canon(args.dataset)
    cfg = segdata.dataset_cfg(ds_name)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]
    batch = args.batch_size or segdata.DEFAULT_BATCH[ds_name]
    ckpt = args.ckpt or os.path.join("runs", "seg", ds_name, f"seed{args.seed}", "best.pt")
    out_dir = args.out or os.path.join("runs", "pivot",
                                       f"ft_{ds_name}_seed{args.seed}_{args.loss_kind}")
    os.makedirs(out_dir, exist_ok=True)

    set_determinism(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and torch.cuda.is_available()

    recs = segdata.get_records(ds_name)
    tr_recs, va_recs, te_recs, split_info = segdata.make_splits(recs, segdata.DEFAULT_SPLIT_SEED)
    cache = os.path.join("runs", "_cache", f"{ds_name}_{longest}") if longest else None

    train_ds = segdata.VesselPatchDataset(
        tr_recs, patch, longest, samples_per_epoch=args.iters_per_epoch * batch,
        augment=True, oversample_p=args.oversample_p, seed=args.seed,
        cache_dir=cache, mem_cache=8 if longest is not None else 64)
    va_sorted = sorted(va_recs, key=lambda r: str(r["image_path"]))
    if args.val_limit and len(va_sorted) > args.val_limit:
        step = max(1, len(va_sorted) // args.val_limit)
        va_sorted = va_sorted[::step][:args.val_limit]
    val_ds = segdata.FullImageDataset(va_sorted, longest, cache_dir=cache, mem_cache=4)

    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        worker_init_fn=segdata.seed_worker, generator=g,
                        persistent_workers=(args.workers > 0))

    model, ck = load_model(ckpt, device)
    model.train()
    cf_kind = args.loss_kind in ("cf_faithful", "cf_on_base")
    if args.loss_kind == "cf_faithful":
        # published objective: beta * CE + alpha * L_FD + gamma * L_vd.  No
        # Dice and no clDice -- this is the baseline exactly as it is published.
        seg_crit = SegLoss(w_bce=CF_BETA, w_dice=0.0, w_cldice=0.0)
    else:
        seg_crit = SegLoss(w_bce=1.0, w_dice=1.0, w_cldice=args.w_cldice)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = make_scaler(amp)

    config = {"probe": "P5", "dataset": ds_name, "init_ckpt": os.path.abspath(ckpt),
              "loss_kind": args.loss_kind, "lam": args.lam, "lr": args.lr,
              "terms": list(terms), "fd_ladder": args.fd_ladder,
              "cf_alpha": CF_ALPHA, "cf_beta": CF_BETA, "cf_gamma": CF_GAMMA,
              "cf_scale": args.cf_scale,
              "val_limit": int(args.val_limit), "n_val_used": len(va_sorted),
              "w_cldice": args.w_cldice,
              "epochs": args.epochs, "batch": batch, "patch": patch,
              "iters_per_epoch": args.iters_per_epoch, "seed": args.seed,
              "n_train": len(tr_recs), "n_val": len(va_recs), "n_test": len(te_recs),
              "amp": amp, "torch": torch.__version__, "device": str(device),
              "in_channels": ck.get("config", {}).get("in_channels", 3),
              "out_channels": 1, "base_channels": 32, "num_levels": 5,
              "max_channels": 512}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.probe_only:
        model.train()
        for i, b in enumerate(loader):
            x = b["image"].to(device); y = b["label"].to(device); m = b["fov"].to(device)
            with torch.autocast("cuda", enabled=amp):
                logits = model(x)
            seg, _ = seg_crit(logits, y, m)
            p = torch.sigmoid(logits.float())
            if cf_kind:
                pm, ym = p * m.float(), y.float() * m.float()
                l_fd, l_vd = cf_fd_vd(pm, ym.detach())
                c = args.cf_scale * (CF_ALPHA * l_fd + CF_GAMMA * l_vd)
                parts = {"cf_fd": float(l_fd.detach()), "cf_vd": float(l_vd.detach())}
            else:
                c, parts = consistency_loss(p, y.float(), m.float(), args.loss_kind,
                                            terms=terms, fd_ladder=args.fd_ladder)
            lam_eff = 1.0 if cf_kind else args.lam
            print(f"batch {i}: seg={float(seg):.4f} consist={float(c):.4f} "
                  f"lam*C={lam_eff * float(c):.4f} "
                  f"frac={lam_eff * float(c) / (float(seg) + lam_eff * float(c)):.3f} "
                  f"parts={ {k: round(v, 4) for k, v in parts.items()} }", flush=True)
            if i >= 4:
                break
        return 0

    rule = args.select_by
    if rule == "auto":
        rule = "r" if len(va_sorted) >= 4 else "mae"
    config["select_by"] = rule
    base_val, base_fid = validate(model, val_ds, patch, device, amp, args.sw_batch,
                                  fidelity=True)
    print(f"[init] val Dice of the starting checkpoint = {base_val:.4f}; "
          f"val measurement fidelity fid_r={base_fid.get('fid_r'):.4f} "
          f"fid_mae={base_fid.get('fid_mae'):.4f} (selection rule: {rule})",
          flush=True)

    def fid_key(f):
        """Higher is better."""
        return f.get("fid_r", float("nan")) if rule == "r" else -f.get("fid_mae")

    best, best_ep, log = base_val, -1, []
    best_fid, best_fid_ep = fid_key(base_fid), -1
    torch.save({"model": model.state_dict(), "config": config,
                "epoch": -1, "val_dice": base_val}, os.path.join(out_dir, "best.pt"))
    torch.save({"model": model.state_dict(), "config": config, "epoch": -1,
                "val_dice": base_val, "val_fid": base_fid},
               os.path.join(out_dir, "best_fid.pt"))
    t_start = time.time()
    for epoch in range(args.epochs):
        lr = poly_lr(args.lr, epoch, args.epochs)
        for pg in opt.param_groups:
            pg["lr"] = lr
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        s = {"loss": 0.0, "seg": 0.0, "consist": 0.0, "frac": 0.0}
        pacc: Dict[str, float] = {}
        n = 0
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            m = b["fov"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                logits = model(x)
            seg, _sp = seg_crit(logits, y, m)
            p = torch.sigmoid(logits.float())
            if cf_kind:
                pm, ym = p * m.float(), y.float() * m.float()
                l_fd, l_vd = cf_fd_vd(pm, ym.detach())
                c = args.cf_scale * (CF_ALPHA * l_fd + CF_GAMMA * l_vd)
                parts = {"cf_fd": float(l_fd.detach()), "cf_vd": float(l_vd.detach())}
            else:
                c, parts = consistency_loss(p, y.float(), m.float(), args.loss_kind,
                                            terms=terms, fd_ladder=args.fd_ladder)
            loss = seg.float() + (1.0 if cf_kind else args.lam) * c
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt); scaler.update()
            s["loss"] += float(loss.detach()); s["seg"] += float(seg.detach())
            s["consist"] += float(c.detach())
            s["frac"] += (1.0 if cf_kind else args.lam) * float(c.detach())                 / max(float(loss.detach()), 1e-8)
            for k, v in parts.items():
                pacc[k] = pacc.get(k, 0.0) + v
            n += 1
        vd, vfid = validate(model, val_ds, patch, device, amp, args.sw_batch,
                            fidelity=True)
        entry = {"epoch": epoch, "lr": lr,
                 **{k: v / max(1, n) for k, v in s.items()},
                 **{"term_" + k: v / max(1, n) for k, v in pacc.items()},
                 "val_dice": vd, **{"val_" + k: v for k, v in vfid.items()},
                 "seconds": round(time.time() - t0, 1)}
        if fid_key(vfid) > best_fid:
            best_fid, best_fid_ep = fid_key(vfid), epoch
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "val_dice": vd, "val_fid": vfid},
                       os.path.join(out_dir, "best_fid.pt"))
            entry["is_best_fid"] = True
        if vd > best:
            best, best_ep = vd, epoch
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "val_dice": vd}, os.path.join(out_dir, "best.pt"))
            entry["is_best"] = True
        log.append(entry)
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                    "val_dice": vd}, os.path.join(out_dir, "last.pt"))
        with open(os.path.join(out_dir, "log.json"), "w", encoding="utf-8") as f:
            json.dump({"config": config, "split": split_info, "base_val_dice": base_val,
                       "base_val_fid": base_fid, "log": log, "best_dice": best,
                       "best_epoch": best_ep, "select_by": rule,
                       "best_fid": best_fid, "best_fid_epoch": best_fid_ep},
                      f, indent=2)
        print("ep {:>3}/{} lr {:.2e} loss {:.4f} (seg {:.4f} + {:.2f}*{:.4f}, "
              "consist {:.0%} of loss) val Dice {:.4f} (best {:.4f} @{}) "
              "fid[{}] r {:.4f} mae {:.4f} (best @{}) {:.0f}s".format(
                  epoch + 1, args.epochs, lr, entry["loss"], entry["seg"],
                  (1.0 if cf_kind else args.lam),
                  entry["consist"], entry["frac"], vd, best, best_ep + 1,
                  rule, float(vfid.get("fid_r", float("nan"))),
                  float(vfid.get("fid_mae", float("nan"))), best_fid_ep + 1,
                  entry["seconds"]), flush=True)

    summary = {"out_dir": os.path.abspath(out_dir), "base_val_dice": base_val,
               "best_dice": best, "best_epoch": best_ep,
               "select_by": rule, "best_fid": best_fid, "best_fid_epoch": best_fid_ep,
               "final_val_fid_r": log[-1].get("val_fid_r"),
               "final_val_fid_mae": log[-1].get("val_fid_mae"),
               "base_val_fid_r": base_fid.get("fid_r"),
               "base_val_fid_mae": base_fid.get("fid_mae"),
               "epochs_run": len(log), "total_seconds": round(time.time() - t_start, 1),
               "mean_consist_frac": float(np.mean([e["frac"] for e in log])),
               **{k: float(np.mean([e[k] for e in log])) for k in log[0] if k.startswith("term_")}}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
