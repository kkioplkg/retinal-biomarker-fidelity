"""Faithful port of the published CF-Loss (Zhou et al.), for the E2 baseline.

Reference implementation: ``third_party/feature-loss/scripts/loss.py``
(https://github.com/rmaphoh/feature-loss, ``CF_Loss``), cloned at the commit
recorded in ``third_party/feature-loss/.git``.  The published loss is

    L = beta * CE  +  alpha * L_FD  +  gamma * L_vd

with, verbatim from ``CF_Loss``:

*   **box ladder** ``sizes = 2 ** arange(floor(log2(W)), 1, -1)`` -- dyadic and
    *scaled to the image size*: for a 768-px patch, ``[512, 256, ..., 4]``
    (note the ladder stops at 4, not 2).
*   **soft box counts** per size: zero-pad the right/bottom edge up to a
    multiple of the size, ``AvgPool2d(kernel=stride=size)``, then keep only
    boxes with ``S > 0`` (the published mask ``(S > 0) & (S < size*size)`` is
    vacuous on the upper side because average pooling gives ``S <= 1``).
*   **raw-count regression, not a log-log slope**: the per-size discrepancy is
    ``|S_pred - S_gt|.sum() / (S_gt > 0).sum()`` -- summed over the whole
    batch, exactly as published -- and the sizes are combined as
    ``sqrt(sum(size * count^2)) / sqrt(sum(size^2)) / B``.
*   **density (``L_vd``)**: ``|sum(p) - sum(gt)| / (B*H*W)``, i.e. the L1 error
    of the class pixel ratio.

The published model is a 4-class artery / vein / crossing segmenter and
therefore sums the FD term over its two vessel classes.  Our segmenter is
binary, so the port runs the identical arithmetic over a single foreground
class (the two-class sum degenerates to a 0/0 for an always-empty vein
channel).  ``verify_against_reference`` checks the port against the original
module on a genuine two-class input, where nothing degenerates.

CLI
---
    python -m src.pivot.cf_loss verify --gpu 1
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence, Tuple

import torch
import torch.nn as nn

#: published defaults, README training command of the reference repo
CF_ALPHA = 0.5      # weight of L_FD
CF_BETA = 1.0       # weight of CE
CF_GAMMA = 1.0      # weight of L_vd


def cf_sizes(width: int, device=None) -> torch.Tensor:
    """``2 ** arange(floor(log2(W)), 1, -1)`` -- the published ladder."""
    n = int(torch.floor(torch.log2(torch.tensor(float(width)))).item())
    return 2 ** torch.arange(n, 1, -1, dtype=torch.int, device=device)


def cf_counts(stack: torch.Tensor, width: int, sizes: torch.Tensor) -> torch.Tensor:
    """Per-size soft box-count discrepancy, shape (n_classes, n_sizes).

    ``stack`` is ``(B, 2*C, H, W)`` = ``cat(pred[:, :C], gt[:, :C], dim=1)``,
    mirroring the reference's ``cat(masks_pred_softmax, encode_tensor, 1)``.
    """
    c = stack.shape[1] // 2
    p = torch.tensor(float(width))
    out = []
    for size in sizes:
        size_i = int(size.item())
        pad_size = 0 if (width % size_i) == 0 else (size_i - width % size_i)
        pad = nn.ZeroPad2d((0, pad_size, 0, pad_size))
        pool = nn.AvgPool2d(kernel_size=(size_i, size_i), stride=(size_i, size_i))
        S = pool(pad(stack))
        S = S * ((S > 0) & (S < (size_i * size_i)))
        col = []
        for k in range(c):
            num = (S[:, k, ...] - S[:, c + k, ...]).abs().sum()
            den = (S[:, c + k, ...] > 0).sum()
            col.append(num / den.clamp_min(1))
        out.append(torch.stack(col))
    del p
    return torch.stack(out, dim=1)                       # (C, n_sizes)


def cf_fd_vd(prob_fg: torch.Tensor, gt_fg: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(L_FD, L_vd)`` of the published loss for ``C`` foreground classes.

    ``prob_fg`` / ``gt_fg`` are ``(B, C, H, W)``; for our binary segmenter
    ``C = 1``.  Both are probabilities / one-hot in ``[0, 1]``.
    """
    b, c, h, w = prob_fg.shape
    sizes = cf_sizes(w, device=prob_fg.device)
    l_vd = (prob_fg - gt_fg).sum(dim=(0, 2, 3)).abs().sum() / (b * h * w)
    stack = torch.cat((prob_fg, gt_fg), dim=1)
    counts = cf_counts(stack, w, sizes)                  # (C, n_sizes)
    s = sizes.to(counts.dtype)
    per_class = torch.sqrt((s * counts.pow(2)).sum(dim=1))
    size_t = torch.sqrt((s ** 2).sum())
    l_fd = per_class.sum() / size_t / b
    return l_fd, l_vd


# --------------------------------------------------------------------------
def verify_against_reference(gpu: int = 1, seed: int = 0) -> int:
    """Run the *original* ``CF_Loss`` and this port on the same mask pair.

    Uses a real HRF prediction/GT pair split into two foreground classes (top
    half = "artery", bottom half = "vein") so the reference's two-class code
    path is exercised with nothing empty.
    """
    import numpy as np

    sys.path.insert(0, os.path.abspath("third_party/feature-loss"))
    from scripts.loss import CF_Loss                       # noqa: E402

    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(seed)

    # a genuine mask pair, cropped to the 720x720 the reference repo trains at
    from src.pivot.common import binarize, read_gray
    gt_p = "runs/pivot/hrf/seed0/baseline/pred/mask/01_dr.png"
    if not os.path.exists(gt_p):
        gt_p = sorted(__import__("glob").glob(
            "runs/pivot/hrf/seed0/baseline/pred/mask/*.png"))[0]
    m = binarize(read_gray(gt_p)).astype(np.float32)[500:1220, 500:1220]
    pred = torch.from_numpy(m).to(device)[None, None]
    # a perturbed version stands in for the network output
    noise = torch.rand_like(pred)
    prob = (pred * 0.85 + 0.12 * noise).clamp(0, 1)

    h = pred.shape[-2] // 2
    gt2 = torch.zeros(1, 2, *pred.shape[-2:], device=device)
    gt2[:, 0, :h] = pred[:, 0, :h]
    gt2[:, 1, h:] = pred[:, 0, h:]
    pr2 = torch.zeros_like(gt2)
    pr2[:, 0, :h] = prob[:, 0, :h]
    pr2[:, 1, h:] = prob[:, 0, h:]

    # ---- reference: rebuild its internals with our (already softmaxed) input
    ref = CF_Loss((720, 720), beta=CF_BETA, alpha=CF_ALPHA, gamma=CF_GAMMA)
    ref_sizes = ref.sizes.to(device)
    ours_sizes = cf_sizes(720, device=device)
    assert torch.equal(ref_sizes.cpu(), ours_sizes.cpu()), \
        f"ladder mismatch {ref_sizes.tolist()} vs {ours_sizes.tolist()}"

    stack = torch.cat((pr2, gt2), dim=1)
    ref_counts = ref.get_count(ref.sizes, ref.p, stack).to(device)   # (B,n_sizes,2)
    our_counts = cf_counts(stack, 720, ours_sizes)                   # (2,n_sizes)
    d_counts = (ref_counts[0, :, 0] - our_counts[0]).abs().max().item() + \
               (ref_counts[0, :, 1] - our_counts[1]).abs().max().item()

    artery_ = torch.sqrt(torch.sum(ref.sizes.to(device) * (ref_counts[..., 0] ** 2)))
    vein_ = torch.sqrt(torch.sum(ref.sizes.to(device) * (ref_counts[..., 1] ** 2)))
    size_t = torch.sqrt(torch.sum(ref.sizes.to(device).float() ** 2))
    ref_fd = ((artery_ + vein_) / size_t / stack.shape[0]).item()
    ref_vd = ((torch.abs(pr2[:, 0].sum() - gt2[:, 0].sum())
               + torch.abs(pr2[:, 1].sum() - gt2[:, 1].sum()))
              / (pr2.shape[0] * pr2.shape[2] * pr2.shape[3])).item()

    our_fd, our_vd = cf_fd_vd(pr2, gt2)
    print(f"ladder         : {ours_sizes.tolist()}")
    print(f"max |d count|  : {d_counts:.3e}")
    print(f"L_FD reference : {ref_fd:.10f}")
    print(f"L_FD port      : {our_fd.item():.10f}   d = {abs(ref_fd-our_fd.item()):.3e}")
    print(f"L_vd reference : {ref_vd:.10f}")
    print(f"L_vd port      : {our_vd.item():.10f}   d = {abs(ref_vd-our_vd.item()):.3e}")
    ok = (abs(ref_fd - our_fd.item()) < 1e-6 and abs(ref_vd - our_vd.item()) < 1e-6
          and d_counts < 1e-5)
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return verify_against_reference(a.gpu)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
