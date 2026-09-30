"""Differentiable biomarker surrogates on soft probability maps (probe P1 / D1).

Every function takes ``p`` of shape ``(B, 1, H, W)`` with values in [0, 1] (a
sigmoid output, not a thresholded mask) and an optional binary ``fov`` of the
same shape, and returns one scalar per batch element with a gradient path back
to ``p``.  Nothing here thresholds, calls ``skimage`` or leaves the GPU.

The soft skeleton is the clDice soft-skeletonisation of Shit et al. (CVPR 2021)
already implemented in ``src/seg/losses.py`` -- it is imported, not re-written,
so the surrogate and the clDice loss cannot drift apart.

Definitions
-----------
soft_density      mean of p over the FOV                     ~ vessel_density
soft_length       sum of the soft skeleton over the FOV      ~ skeleton_length_total
soft_fd           -slope of log N(s) vs log s, where N(s) is the summed
                  occupancy of s x s max-pooled boxes, s = 2..64 (powers of 2)
                                                             ~ box-counting FD
soft_tortuosity   (s * N_skel(s)) / soft_length, the ratio of the skeleton
                  length measured with s-pixel boxes to its fine-scale length.
                  A straight vessel fills one box per s pixels so the ratio is
                  ~1; a vessel that wanders fills boxes in several rows per s
                  columns, so the ratio grows -- an arc/chord-like proxy.
                  EXPLORATORY: it is confounded by branch density, because
                  coarse boxes merge neighbouring branches.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from src.seg.losses import soft_skeletonize

__all__ = ["soft_density", "soft_length", "soft_fd", "soft_tortuosity",
           "box_counts", "all_surrogates", "FD_BOX_SIZES"]

FD_BOX_SIZES: Sequence[int] = (2, 4, 8, 16, 32, 64)


def _apply_fov(p: torch.Tensor, fov: Optional[torch.Tensor]) -> torch.Tensor:
    return p if fov is None else p * fov


def soft_density(p: torch.Tensor, fov: Optional[torch.Tensor] = None) -> torch.Tensor:
    dims = (1, 2, 3)
    if fov is None:
        return p.mean(dim=dims)
    return (p * fov).sum(dim=dims) / fov.sum(dim=dims).clamp_min(1.0)


def soft_length(p: torch.Tensor, fov: Optional[torch.Tensor] = None,
                num_iter: int = 10) -> torch.Tensor:
    skel = soft_skeletonize(_apply_fov(p, fov), num_iter=num_iter)
    return skel.sum(dim=(1, 2, 3))


def box_counts(p: torch.Tensor, sizes: Sequence[int] = FD_BOX_SIZES
               ) -> torch.Tensor:
    """(B, len(sizes)) soft occupancy counts: sum of the s x s max-pooled map."""
    out = []
    for s in sizes:
        s = int(s)
        if s == 1:
            out.append(p.sum(dim=(1, 2, 3)))
            continue
        pooled = F.max_pool2d(p, kernel_size=s, stride=s, ceil_mode=True)
        out.append(pooled.sum(dim=(1, 2, 3)))
    return torch.stack(out, dim=1)


def _loglog_slope(counts: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
    x = torch.log(torch.tensor([float(s) for s in sizes], device=counts.device,
                               dtype=counts.dtype))
    y = torch.log(counts.clamp_min(1e-8))
    xm = x.mean()
    ym = y.mean(dim=1, keepdim=True)
    num = ((x - xm).unsqueeze(0) * (y - ym)).sum(dim=1)
    den = ((x - xm) ** 2).sum()
    return num / den


def soft_fd(p: torch.Tensor, fov: Optional[torch.Tensor] = None,
            sizes: Sequence[int] = FD_BOX_SIZES) -> torch.Tensor:
    """Box-counting fractal dimension: ``-d log N(s) / d log s``."""
    counts = box_counts(_apply_fov(p, fov), sizes)
    return -_loglog_slope(counts, sizes)


def soft_tortuosity(p: torch.Tensor, fov: Optional[torch.Tensor] = None,
                    box: int = 4, num_iter: int = 10) -> torch.Tensor:
    skel = soft_skeletonize(_apply_fov(p, fov), num_iter=num_iter)
    fine = skel.sum(dim=(1, 2, 3))
    coarse = F.max_pool2d(skel, kernel_size=box, stride=box,
                          ceil_mode=True).sum(dim=(1, 2, 3)) * float(box)
    return coarse / fine.clamp_min(1e-6)


FD_N_SIZES = 12


def auto_box_sizes(h: int, w: int, n_sizes: int = FD_N_SIZES,
                   min_box: int = 2) -> Sequence[int]:
    """Geometric box sizes ``min_box .. min(h, w) // 4``.

    This is the range ``src/bio/skan_pipe.box_counting_fd`` uses.  The fixed
    2..64 ladder of :data:`FD_BOX_SIZES` covers only a thin slice of it on a
    2048 px FIVES image, which is why the fixed-ladder surrogate loses its
    agreement with the pipeline on the large datasets.
    """
    import numpy as _np

    max_box = max(min_box * 2, int(min(h, w) // 4))
    sizes = _np.unique(_np.round(_np.geomspace(min_box, max_box, n_sizes))
                       .astype(int))
    return [int(s) for s in sizes]


def all_surrogates(p: torch.Tensor, fov: Optional[torch.Tensor] = None,
                   num_iter: int = 10) -> dict:
    """Every surrogate for one batch; values are plain tensors of shape (B,).

    ``s_fd`` uses the fixed 2..64 box ladder; ``s_fd_auto`` and ``s_fd_skel``
    use the resolution-adaptive ladder of :func:`auto_box_sizes`, on the mask
    (matching PVBM's segmentation D0) and on the soft skeleton (matching
    ``skan_pipe``'s skeleton box-count) respectively.
    """
    pf = _apply_fov(p, fov)
    skel = soft_skeletonize(pf, num_iter=num_iter)
    fine = skel.sum(dim=(1, 2, 3))
    coarse4 = F.max_pool2d(skel, kernel_size=4, stride=4,
                           ceil_mode=True).sum(dim=(1, 2, 3)) * 4.0
    h, w = pf.shape[-2:]
    auto = auto_box_sizes(h, w)
    box_t = max(4, int(round(max(h, w) / 128.0)))
    coarse_t = F.max_pool2d(skel, kernel_size=box_t, stride=box_t,
                            ceil_mode=True).sum(dim=(1, 2, 3)) * float(box_t)
    return {
        "s_density": soft_density(p, fov),
        "s_length": fine,
        "s_fd": -_loglog_slope(box_counts(pf, FD_BOX_SIZES), FD_BOX_SIZES),
        "s_fd_auto": -_loglog_slope(box_counts(pf, auto), auto),
        "s_fd_skel": -_loglog_slope(box_counts(skel, auto), auto),
        "s_tortuosity": coarse4 / fine.clamp_min(1e-6),
        "s_tort_auto": coarse_t / fine.clamp_min(1e-6),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    # a straight line and a zig-zag of equal pixel count: the tortuosity proxy
    # must order them correctly, the density/length surrogates must not.
    a = torch.zeros(1, 1, 128, 128)
    a[0, 0, 64, 10:110] = 1.0
    b = torch.zeros(1, 1, 128, 128)
    for i, x in enumerate(range(10, 110)):
        b[0, 0, 64 + (i % 6) - 3, x] = 1.0
    for name, t in (("straight", a), ("zigzag", b)):
        s = all_surrogates(t)
        print(name, {k: round(float(v), 4) for k, v in s.items()})
    x = a.clone().requires_grad_(True)
    out = all_surrogates(x)
    g = torch.autograd.grad(sum(v.sum() for v in out.values()), x)[0]
    print("gradient flows:", bool(torch.isfinite(g).all() and g.abs().sum() > 0))
