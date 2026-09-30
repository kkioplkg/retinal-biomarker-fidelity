"""Sanity check: our PyTorch rNCA on the upstream repo's own bundled data.

Runs :mod:`src.baselines.rnca.rnca_core` against
``third_party/rnca/src/data/{train,val}`` -- the 64x64 synthetic set the
official JAX implementation ships -- with **their** ``main.py::Config``
hyper-parameters (state_channels 16, hidden_dim 128, dropout 0.5,
alive_threshold 0.1, pool_size 256, batch_size 16, replace_n 2, nca_steps 64,
lr 1e-4).  Their data layout is

    images/<name>.png   the conditioning image  (grayscale, img_channels = 1)
    labels/<name>.png   the ground-truth mask   (the target)
    states/<name>.png   the *imperfect* mask    (the alpha seed)

which is the layout our adapter mirrors.  Because ``img_channels == 1`` here,
this is also the one setting where deviation **D2** does not apply, so
``--delta_width img`` reproduces upstream's literal ``state + delta``
broadcast and can be compared against our ``state`` default.

The check is qualitative and answers three questions:

1. does the training MSE fall (is the operator learning to repair at all)?
2. does the rolled-out mask move **towards** the label -- IoU(out, label)
   rising above IoU(seed, label)?
3. is alive masking doing its job -- does the alive set stay finite and
   anchored on the seed rather than flooding the frame?

    python -m src.baselines.rnca.sanity_upstream_data --gpu 1 --iters 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.baselines.rnca.rnca_core import (
    RNCA, RNCAConfig, SamplePool, loss_at_random_late_step, seed_state,
)

RNCA_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "rnca"
DATA_ROOT = RNCA_ROOT / "src" / "data"


def load_split(split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Their ``Dataset.__getitem__``, in numpy: ``(X, Y, S_vis)``."""
    from PIL import Image

    d = DATA_ROOT / split
    names = sorted(p.name for p in (d / "images").glob("*.png"))
    if not names:
        raise SystemExit(f"no images under {d / 'images'}")
    X, Y, S = [], [], []
    for n in names:
        img = np.asarray(Image.open(d / "images" / n)).astype(np.float32) / 255.0
        lab = (np.asarray(Image.open(d / "labels" / n)) > 128).astype(np.float32)
        st = np.asarray(Image.open(d / "states" / n)).astype(np.float32) / 255.0
        st = np.clip(st, 0.1, 1.0)              # their non-alive clip
        if img.ndim == 2:
            img = img[..., None]
        X.append(np.transpose(img, (2, 0, 1)))  # channels-first
        Y.append(lab[None])
        S.append(st[None])
    return (torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y)),
            torch.from_numpy(np.stack(S)))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.baselines.rnca.sanity_upstream_data")
    ap.add_argument("--gpu", default="cpu")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--pool_size", type=int, default=256)
    ap.add_argument("--replace_n", type=int, default=2)
    ap.add_argument("--nca_steps", type=int, default=64)
    ap.add_argument("--infer_steps", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta_width", default="state", choices=["state", "img"])
    ap.add_argument("--log_every", type=int, default=100)
    a = ap.parse_args(argv)

    device = "cpu" if str(a.gpu).lower() in ("cpu", "-1", "") else f"cuda:{a.gpu}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    Xtr, Ytr, Str = load_split("train")
    print(f"[sanity] upstream data: train X{tuple(Xtr.shape)} Y{tuple(Ytr.shape)}",
          flush=True)
    try:
        Xva, Yva, Sva = load_split("val")
        print(f"[sanity]                val   X{tuple(Xva.shape)}", flush=True)
    except SystemExit:
        Xva, Yva, Sva = Xtr[:16], Ytr[:16], Str[:16]

    cfg = RNCAConfig(img_channels=Xtr.shape[1], state_channels=16,
                     hidden_dim=128, dropout_rate=0.5, alive_threshold=0.1,
                     delta_width=a.delta_width)
    model = RNCA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    n = min(a.pool_size, Xtr.shape[0])
    sel = rng.choice(Xtr.shape[0], size=n, replace=False)
    X, Y, S0 = Xtr[sel], Ytr[sel], Str[sel]
    fresh = seed_state(S0, cfg.state_channels)
    pool = SamplePool(X, Y, fresh.clone())

    gen = torch.Generator(device=dev).manual_seed(a.seed)
    model.train()
    losses, first = [], None
    t0 = time.time()
    for it in range(1, int(a.iters) + 1):
        idx, xb, yb, sb = pool.sample(a.batch_size, rng)
        r = rng.choice(a.batch_size, size=min(a.replace_n, a.batch_size),
                       replace=False)
        src = rng.choice(pool.size, size=len(r), replace=False)
        sb[torch.as_tensor(r, dtype=torch.long)] = fresh[torch.as_tensor(src, dtype=torch.long)]

        xb, yb, sb = xb.to(dev), yb.to(dev), sb.to(dev)
        loss, evolved = loss_at_random_late_step(model, xb, yb, sb,
                                                 a.nca_steps, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        pool.write(idx, evolved)

        losses.append(float(loss.item()))
        if first is None:
            first = losses[0]
        if it % a.log_every == 0 or it == 1:
            print(f"  it {it:5d}/{a.iters}  mse {np.mean(losses[-a.log_every:]):.5f}"
                  f"  {time.time()-t0:6.1f}s", flush=True)

    # ---- qualitative rollout on held-out samples -------------------------
    model.eval()
    k = min(16, Xva.shape[0])
    xv, yv, sv = Xva[:k].to(dev), Yva[:k], Sva[:k]
    seed_bin = (sv.numpy() > cfg.alive_threshold)
    gt = yv.numpy() > 0.5
    iou_seed = float(np.mean([iou(seed_bin[i], gt[i]) for i in range(k)]))
    frac_gt = float(gt.mean())

    # NCAs are only supervised on steps [nca_steps/2, nca_steps); rolling out
    # far beyond that horizon is a known stability failure mode, so report the
    # whole curve rather than a single length.
    ladder = sorted({a.nca_steps // 2, a.nca_steps, 2 * a.nca_steps,
                     a.infer_steps})
    curve = []
    for ns in ladder:
        with torch.no_grad():
            st = seed_state(sv.to(dev), cfg.state_channels)
            st = model(st, xv, num_steps=int(ns))
            al = st[:, -1:, :, :].cpu().numpy()
        o = al > cfg.alive_threshold
        curve.append((int(ns),
                      float(np.mean([iou(o[i], gt[i]) for i in range(k)])),
                      float(o.mean())))
    best_ns, iou_out, frac_alive = max(curve, key=lambda t: t[1])
    out = None

    print(f"\n== rNCA (our re-implementation) on upstream bundled data ==")
    print(f"  hyper-params: THEIR main.py Config (delta_width={a.delta_width})")
    print(f"  iters {a.iters}, {time.time()-t0:.0f}s on {device}")
    print(f"  MSE   first {first:.5f} -> last-{a.log_every} "
          f"{np.mean(losses[-a.log_every:]):.5f}")
    print(f"  IoU vs GT   seed {iou_seed:.4f}")
    for ns, iu, fa in curve:
        tag = "  <- trained horizon" if ns == a.nca_steps else ""
        print(f"     rollout {ns:4d} steps -> IoU {iu:.4f}  alive {fa:.4f}{tag}")
    print(f"  best rollout {best_ns} steps: IoU {iou_out:.4f}"
          f"   ({'IMPROVED over seed' if iou_out > iou_seed else 'still below seed'})")
    print(f"  alive fraction {frac_alive:.4f} vs GT foreground {frac_gt:.4f}"
          f"   ({'bounded' if frac_alive < 5 * frac_gt + 0.05 else 'FLOODING'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
