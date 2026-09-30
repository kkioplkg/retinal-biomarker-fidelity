"""Segmentation losses: BCE + soft Dice (CE-Dice), optional soft-clDice.

soft-clDice follows Shit et al., "clDice - a Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation", CVPR 2021.  Soft skeletonisation
is the differentiable min/max-pool erosion/dilation scheme from that paper.

The clDice weight defaults to 0.0, so the default objective is exactly
BCE + soft Dice; the term is exposed for the later ablations of the plan.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# soft morphology / soft skeleton
# ---------------------------------------------------------------------------
def soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Min-filter with a cross structuring element (3x1 and 1x3 min-pools)."""
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """Differentiable skeletonisation (Shit et al. 2021, Algorithm 1)."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(num_iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ---------------------------------------------------------------------------
# individual terms
# ---------------------------------------------------------------------------
def _masked(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x
    return x * mask


def soft_dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    smooth: float = 1.0,
    batch_dice: bool = True,
) -> torch.Tensor:
    """1 - soft Dice.  ``mask`` (e.g. the FOV) zeroes out excluded pixels."""
    p = _masked(probs, mask)
    t = _masked(target, mask)
    dims = tuple(range(1, p.dim()))
    inter = (p * t).sum(dim=dims)
    denom = p.sum(dim=dims) + t.sum(dim=dims)
    if batch_dice:
        inter, denom = inter.sum(), denom.sum()
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    if mask is None:
        return loss.mean()
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def soft_cldice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    num_iter: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """1 - soft clDice (Shit et al. 2021)."""
    p = _masked(probs, mask)
    t = _masked(target, mask)
    skel_p = soft_skeletonize(p, num_iter=num_iter)
    skel_t = soft_skeletonize(t, num_iter=num_iter)
    dims = tuple(range(1, p.dim()))
    # topology precision: predicted skeleton inside the GT mask
    tprec = ((skel_p * t).sum(dim=dims) + smooth) / (skel_p.sum(dim=dims) + smooth)
    # topology sensitivity: GT skeleton inside the prediction
    tsens = ((skel_t * p).sum(dim=dims) + smooth) / (skel_t.sum(dim=dims) + smooth)
    cldice = 2.0 * tprec * tsens / (tprec + tsens)
    return 1.0 - cldice.mean()


# ---------------------------------------------------------------------------
# combined criterion
# ---------------------------------------------------------------------------
class SegLoss(nn.Module):
    """w_bce * BCE + w_dice * softDice + w_cldice * softClDice.

    Accepts either a single logit tensor or, with deep supervision, a list of
    logit tensors ordered highest-resolution first; deep-supervision weights
    halve per level and are renormalised to sum to one.
    """

    def __init__(
        self,
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_cldice: float = 0.0,
        cldice_iter: int = 10,
        smooth: float = 1.0,
        batch_dice: bool = True,
        pos_weight: Optional[float] = None,
        deep_supervision_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.w_cldice = float(w_cldice)
        self.cldice_iter = int(cldice_iter)
        self.smooth = float(smooth)
        self.batch_dice = bool(batch_dice)
        self.ds_weights = deep_supervision_weights
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    # -- single scale ------------------------------------------------------
    def _single(self, logits, target, mask):
        parts = {}
        total = logits.float().new_zeros(())
        if self.w_bce:
            bce = masked_bce_with_logits(logits, target, mask, self.pos_weight)
            parts["bce"] = bce.detach()
            total = total + self.w_bce * bce.float()
        if self.w_dice or self.w_cldice:
            # clDice / Dice are defined on probabilities; keep them in fp32
            probs = torch.sigmoid(logits.float())
            tgt = target.float()
            msk = None if mask is None else mask.float()
        if self.w_dice:
            d = soft_dice_loss(probs, tgt, msk, self.smooth, self.batch_dice)
            parts["dice"] = d.detach()
            total = total + self.w_dice * d
        if self.w_cldice:
            c = soft_cldice_loss(probs, tgt, msk, self.cldice_iter, self.smooth)
            parts["cldice"] = c.detach()
            total = total + self.w_cldice * c
        return total, parts

    def forward(self, logits, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Returns (loss, dict of the individual terms at full resolution)."""
        if torch.is_tensor(logits):
            return self._single(logits, target, mask)

        n = len(logits)
        if self.ds_weights is not None:
            w = list(self.ds_weights)[:n]
        else:
            w = [1.0 / (2 ** i) for i in range(n)]
        s = float(sum(w))
        w = [x / s for x in w]

        total = None
        parts0 = {}
        for i, lg in enumerate(logits):
            if i == 0:
                t, m = target, mask
            else:
                sz = lg.shape[-2:]
                t = F.interpolate(target, size=sz, mode="nearest")
                m = None if mask is None else F.interpolate(mask, size=sz, mode="nearest")
            li, pi = self._single(lg, t, m)
            if i == 0:
                parts0 = pi
            total = li * w[i] if total is None else total + li * w[i]
        return total, parts0


def build_loss(w_cldice: float = 0.0, **kwargs) -> SegLoss:
    return SegLoss(w_cldice=w_cldice, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    lg = torch.randn(2, 1, 64, 64, requires_grad=True)
    tg = (torch.rand(2, 1, 64, 64) > 0.9).float()
    fov = torch.ones_like(tg)
    for wc in (0.0, 0.5):
        crit = SegLoss(w_cldice=wc)
        loss, parts = crit(lg, tg, fov)
        loss.backward(retain_graph=True)
        msg = ", ".join("{}={:.4f}".format(k, v.item()) for k, v in parts.items())
        print("w_cldice={}: loss={:.4f} parts=[{}]".format(wc, loss.item(), msg))
