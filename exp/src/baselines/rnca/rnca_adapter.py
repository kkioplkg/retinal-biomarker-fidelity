"""rNCA baseline adapter -- training on our data, inference on our predictions.

The upstream repo (``exp/third_party/rnca``, commit ``54f681b``) is JAX/Flax
NNX, ships no weights and no corruption code, and pins ``jax[cuda12]`` (no
Windows GPU wheels).  It is therefore **re-implemented** in
:mod:`src.baselines.rnca.rnca_core`; this baseline must be reported as
**"rNCA (our re-implementation)"**.  See that module's docstring for the
faithfully-reproduced parts and the five documented deviations.

Training
--------
    python -m src.baselines.rnca.rnca_adapter --dataset drive --gpu 1 \
        --epochs 200 --out runs/repair/ckpt/rnca_drive.pt

Pairs are ``(imperfect mask, GT)`` built from the *training* split: ``M_hat``
is the GT with random capsule severances (``src.baselines.synth``), which is
the degradation this study is about; the conditioning image is the green
channel plus the probability map.

Inference
---------
``repair_detailed`` seeds the NCA alpha channel with ``M_hat``, rolls out
``steps`` updates and binarises at ``alive_threshold``.  Because rNCA emits a
whole mask rather than edges, the accepted edges needed for TRR / FCR are
recovered post hoc by :func:`src.baselines.common.extract_added_edges`:
components of ``M_rep \\ M_hat`` that join two previously separate components
of ``M_hat`` each count as one accepted edge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.topo import skeleton as sk
from src.baselines.common import RepairOutput, as_bool, extract_added_edges
from src.baselines.rnca.rnca_core import (
    RNCA, RNCAConfig, SamplePool, binarize, loss_at_random_late_step, seed_state,
)

__all__ = ["repair", "repair_detailed", "train", "build_conditioning", "main"]

DEFAULT_STEPS = 256          # upstream validation rollout length
_CACHE: Dict[str, Tuple[RNCA, RNCAConfig]] = {}


# --------------------------------------------------------------------------
# conditioning input
# --------------------------------------------------------------------------


def build_conditioning(image: Optional[np.ndarray], prob: Optional[np.ndarray],
                       mask: np.ndarray) -> np.ndarray:
    """The conditioning image ``X``: ``[green channel, probability]`` in [0, 1].

    Upstream conditions on a single grayscale image separate from the mask
    seed.  Our equivalent evidence is the fundus green channel (the standard
    vessel channel) and the segmenter's probability map; when either is absent
    it is replaced by the mask itself, so the model always sees 2 channels.
    """
    m = as_bool(mask).astype(np.float32)
    if image is None:
        green = m
    else:
        a = np.asarray(image)
        green = a[..., 1] if a.ndim == 3 else a
        green = green.astype(np.float32)
        mx = float(green.max())
        green = green / mx if mx > 1.0 else green
    p = m if prob is None else np.asarray(prob, dtype=np.float32)
    if p.max() > 1.0:
        p = p / 255.0
    return np.stack([green, p], axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# checkpoint io
# --------------------------------------------------------------------------


def save_ckpt(path: str, model: RNCA, cfg: RNCAConfig, meta: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), cfg=cfg.to_dict(), meta=meta), p)


def load_ckpt(path: str, device: str = "cpu") -> Tuple[RNCA, RNCAConfig]:
    key = f"{os.path.abspath(path)}::{device}"
    if key in _CACHE:
        return _CACHE[key]
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = RNCAConfig(**blob["cfg"])
    model = RNCA(cfg).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    _CACHE[key] = (model, cfg)
    return model, cfg


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


def repair_detailed(
    image: Optional[np.ndarray],
    prob: Optional[np.ndarray],
    mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    ckpt: Optional[str] = None,
    device: str = "cpu",
    steps: int = DEFAULT_STEPS,
    threshold: Optional[float] = None,
    keep_base: bool = True,
    **kw: Any,
) -> RepairOutput:
    """Run the trained rNCA on one image.

    ``keep_base=True`` unions the NCA output with ``M_hat``.  The NCA is free
    to *delete* pixels, but this baseline is being compared as a **repair**
    stage, and the post-hoc edge extraction is only defined for additions; the
    union keeps the comparison like-for-like with the other two baselines.
    Set it False to report the raw NCA output.
    """
    if ckpt is None:
        raise ValueError(
            "rnca needs --ckpt: train one first with "
            "`python -m src.baselines.rnca.rnca_adapter --dataset <ds> --gpu 1 "
            "--epochs 200 --out <ckpt.pt>`")
    model, cfg = load_ckpt(ckpt, device)
    thr = float(cfg.alive_threshold if threshold is None else threshold)

    m = as_bool(mask)
    f = sk.fov_or_true(fov, m.shape)
    m = m & f

    x = build_conditioning(image, prob, m)
    xt = torch.from_numpy(x)[None].to(device)
    mt = torch.from_numpy(m.astype(np.float32))[None, None].to(device)

    t0 = time.time()
    with torch.no_grad():
        state = seed_state(mt, cfg.state_channels)
        state = model(state, xt, num_steps=int(steps))
        alpha = state[:, -1:, :, :]
        out = binarize(alpha, thr)[0, 0].cpu().numpy()
    dt = time.time() - t0

    out = out & f
    if keep_base:
        out = out | m

    edges = extract_added_edges(m, out, fov=f)
    info = dict(method="rnca_reimpl", steps=int(steps), threshold=thr,
                device=str(device), keep_base=bool(keep_base),
                seconds_nca=round(dt, 3),
                n_added_px=int(np.count_nonzero(out & ~m)),
                n_removed_px=int(np.count_nonzero(m & ~out)),
                n_edges=len(edges), ckpt=str(ckpt))
    return RepairOutput(mask=out, edges=edges, info=info)


def repair(image, prob, mask, fov=None, **kw) -> np.ndarray:
    return repair_detailed(image, prob, mask, fov, **kw).mask


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def _build_pool_tensors(
    dataset: str, split: str, patch: int, pool_size: int, n_cuts: int,
    seed: int, limit: Optional[int] = None, cut_scale: float = 1.0,
    skip: int = 0, pred_dir: Optional[str] = None, extra_cuts: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``pool_size`` (X, Y, S) patches from the training split.

    One severed version is built per source image (capsule cuts on the GT),
    then random ``patch x patch`` crops that contain vessel are drawn from it.
    """
    from src.data.datasets import load_dataset, read_binary, read_image
    from src.baselines.run_baseline import pred_key as _pred_key
    from src.baselines.synth import training_inputs

    recs = load_dataset(dataset, split=split)
    if skip:
        recs = recs[int(skip):]
    if limit:
        recs = recs[: int(limit)]
    if not recs:
        raise SystemExit(f"no {split} images for {dataset}")
    rng = np.random.default_rng(seed)

    per_img = max(1, int(np.ceil(pool_size / len(recs))))
    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    Ss: List[np.ndarray] = []

    for k, r in enumerate(recs):
        gt = read_binary(r["label_path"])
        fov = read_binary(r["fov_path"]) if r.get("fov_path") else np.ones(gt.shape, bool)
        image = np.asarray(read_image(r["image_path"]))
        # scale the number of cuts with image area so HRF/FIVES are not undercut
        area_scale = float(gt.shape[0] * gt.shape[1]) / (584.0 * 565.0)
        n = max(4, int(round(n_cuts * cut_scale * max(1.0, area_scale) ** 0.5)))
        mhat, prob, _c, src_kind = training_inputs(
            gt, fov, image_id=[_pred_key(r), r["image_id"]],
            pred_dir=pred_dir, n_cuts=n,
            seed=seed + k, extra_cuts=extra_cuts)
        x = build_conditioning(image, prob, mhat)
        y = as_bool(gt).astype(np.float32)
        s = as_bool(mhat).astype(np.float32)

        h, w = y.shape
        if h < patch or w < patch:
            continue
        tries = 0
        got = 0
        while got < per_img and tries < per_img * 30:
            tries += 1
            i = int(rng.integers(0, h - patch + 1))
            j = int(rng.integers(0, w - patch + 1))
            sl = (slice(i, i + patch), slice(j, j + patch))
            if not fov[sl].any() or y[sl].sum() < 0.005 * patch * patch:
                continue
            Xs.append(x[:, sl[0], sl[1]])
            Ys.append(y[sl][None])
            Ss.append(s[sl][None])
            got += 1
        print(f"  [{k+1}/{len(recs)}] {r['image_id']}: {src_kind}, {n} cuts, "
              f"{got} patches", flush=True)

    if not Xs:
        raise SystemExit("no usable patches -- is --patch larger than the images?")
    idx = rng.permutation(len(Xs))[:pool_size]
    X = torch.from_numpy(np.stack([Xs[i] for i in idx]))
    Y = torch.from_numpy(np.stack([Ys[i] for i in idx]))
    S = torch.from_numpy(np.stack([Ss[i] for i in idx]))
    return X, Y, S


def train(
    dataset: str,
    out: str,
    device: str = "cpu",
    iters: int = 2000,
    patch: int = 96,
    pool_size: int = 256,
    batch_size: int = 16,
    replace_n: int = 2,
    nca_steps: int = 64,
    lr: float = 1e-4,
    seed: int = 0,
    n_cuts: int = 20,
    limit: Optional[int] = None,
    skip: int = 0,
    log_every: int = 25,
    pred_dir: Optional[str] = None,
    extra_cuts: bool = True,
    state_channels: int = 16,
    hidden_dim: int = 128,
    dropout_rate: float = 0.5,
    alive_threshold: float = 0.1,
) -> str:
    """Train an rNCA with the upstream pool recipe.  Returns the ckpt path.

    ``iters`` counts optimiser steps (upstream counts epochs over a 512-sample
    synthetic set; an explicit step budget is easier to hold to a GPU-time
    limit while the S2 training jobs are running).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    print(f"[rnca] building sample pool from {dataset}/train "
          f"(pool={pool_size}, patch={patch}) ...", flush=True)
    X, Y, S0 = _build_pool_tensors(dataset, "train", patch, pool_size, n_cuts,
                                   seed, limit, skip=skip, pred_dir=pred_dir,
                                   extra_cuts=extra_cuts)
    print(f"[rnca] pool: X{tuple(X.shape)} Y{tuple(Y.shape)}", flush=True)

    cfg = RNCAConfig(img_channels=X.shape[1], state_channels=state_channels,
                     hidden_dim=hidden_dim, dropout_rate=dropout_rate,
                     alive_threshold=alive_threshold, delta_width="state")
    model = RNCA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # the pool holds evolving *states*; fresh seeds come from S0
    pool = SamplePool(X, Y, seed_state(S0, cfg.state_channels))
    fresh = seed_state(S0, cfg.state_channels)

    gen = torch.Generator(device=dev).manual_seed(seed)
    model.train()
    losses: List[float] = []
    t0 = time.time()
    for it in range(1, int(iters) + 1):
        idx, xb, yb, sb = pool.sample(batch_size, rng)
        # reseed replace_n slots with the un-evolved imperfect mask
        r = rng.choice(batch_size, size=min(replace_n, batch_size), replace=False)
        src = rng.choice(pool.size, size=len(r), replace=False)
        sb[torch.as_tensor(r, dtype=torch.long)] = fresh[torch.as_tensor(src, dtype=torch.long)]

        xb, yb, sb = xb.to(dev), yb.to(dev), sb.to(dev)
        loss, evolved = loss_at_random_late_step(model, xb, yb, sb, nca_steps, gen)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        pool.write(idx, evolved)
        losses.append(float(loss.item()))
        if it % log_every == 0 or it == 1:
            print(f"  it {it:5d}/{iters}  mse {np.mean(losses[-log_every:]):.5f}"
                  f"  {time.time()-t0:6.1f}s", flush=True)

    meta = dict(dataset=dataset, iters=int(iters), patch=int(patch),
                pool_size=int(pool_size), batch_size=int(batch_size),
                replace_n=int(replace_n), nca_steps=int(nca_steps), lr=float(lr),
                seed=int(seed), n_cuts=int(n_cuts), device=str(device),
                skip=int(skip), pred_dir=str(pred_dir or ""),
                extra_cuts=bool(extra_cuts),
                train_input=("oof_pred" + ("+uniform_cuts" if extra_cuts else "")
                             if pred_dir else "gt+uniform_cuts"),
                final_mse=float(np.mean(losses[-25:])),
                seconds=round(time.time() - t0, 1),
                upstream="maltesilber/rnca@54f681b (JAX); this is our PyTorch "
                         "re-implementation, see rnca_core docstring")
    save_ckpt(out, model, cfg, meta)
    print(f"[rnca] saved {out}  (final mse {meta['final_mse']:.5f}, "
          f"{meta['seconds']}s)", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.baselines.rnca.rnca_adapter",
        description="Train rNCA (our re-implementation) on our data.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", default="cpu",
                    help="GPU index (e.g. 1), or 'cpu'")
    ap.add_argument("--epochs", type=int, default=2000,
                    help="optimiser steps (upstream counts epochs; see train())")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--pool_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--replace_n", type=int, default=2)
    ap.add_argument("--nca_steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_cuts", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N training images (holds out a smoke-test subset)")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--pred_dir", default=None,
                    help="out-of-fold prediction dir (runs/seg_oof/<ds>/pred): "
                         "train on the segmenter's real failures instead of "
                         "GT + synthetic cuts, which is the S4 protocol")
    ap.add_argument("--no_extra_cuts", action="store_true",
                    help="with --pred_dir, do NOT add the uniform capsule "
                         "severances on top of the predicted mask")
    a = ap.parse_args(argv)

    device = "cpu" if str(a.gpu).lower() in ("cpu", "-1", "") else f"cuda:{a.gpu}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[rnca] CUDA unavailable, falling back to CPU", flush=True)
        device = "cpu"

    train(dataset=a.dataset, out=a.out, device=device, iters=a.epochs,
          patch=a.patch, pool_size=a.pool_size, batch_size=a.batch_size,
          replace_n=a.replace_n, nca_steps=a.nca_steps, lr=a.lr, seed=a.seed,
          n_cuts=a.n_cuts, limit=a.limit, skip=a.skip, log_every=a.log_every,
          pred_dir=a.pred_dir, extra_cuts=not a.no_extra_cuts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
