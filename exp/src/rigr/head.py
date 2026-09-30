"""Micro multi-task head for RiGR (proposal v3 section 3.2.2).

The head is a **black box bolted onto the frozen backbone**: its only inputs
are the RGB image (per-image z-scored inside the FOV) and the backbone's
probability map ``P``, stacked into 4 channels.  It never sees the backbone's
features, so the "pluggable post-processing stage" claim of section 3.2 holds.

Outputs
-------
``V_I(x)``                 one logit per pixel: **vessel evidence**, supervised
                           on *all* pixels with BCE + soft Dice against the
                           reference mask.  This is the appearance term the A*
                           reads inside a gap, and it is deliberately *not*
                           ``P`` itself: inside a real break ``P ~ 0``, so
                           ``-log P`` would punish the true path hardest.
``q(theta | x, vessel)``   ``K = 16`` axial orientation bins,
                           ``theta_k = k * pi / K``, returned as a log-softmax.
                           Supervised **only inside the reference vessel
                           dilated by 1 px**, with a KL to a von Mises soft
                           target ``q*_k ~ exp(kappa_o cos 2(theta_k - theta))``
                           (``kappa_o = 4``) built from the reference skeleton
                           tangent (local PCA over a 7 px window).  The loss
                           weight is multiplied by 0.2 within ``2 r`` of a
                           junction (mixture of axial modes there), and
                           background pixels get a uniform (maximum-entropy)
                           target at weight 0.05 so the network cannot emit a
                           confident but meaningless direction in the very
                           region the A* has to search.

Augmentation
------------
Only **flips and rot90** are used, because they are the isometries of the pixel
lattice under which the orientation target transforms exactly and without
resampling:

  * a flip about either axis maps the axial angle ``theta -> -theta (mod pi)``
    (a reflection reverses the sign of the angle whichever axis it is about);
  * ``np.rot90(a, k)`` maps ``theta -> theta + k * pi/2 (mod pi)``.

Affine / elastic warps would need the target angle to be pushed forward through
the Jacobian, which is exact only for rigid maps; they are therefore excluded
from the head's augmentation (the *backbone* keeps its full augmentation).

Analytic alternative
--------------------
``frangi_evidence`` supplies ``V_I`` from ``skimage.filters.frangi`` on the
green channel (black ridges) for the Tab.3 ablation "learned ``V_I`` vs
analytic Frangi", which answers whether the gain comes from risk-guided repair
or from implicitly training a second segmentation network.

Run ``cd exp && python -m src.rigr.head`` for the parameter count and a shape
check.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "K_BINS",
    "KAPPA_O",
    "BIN_CENTRES",
    "MicroUNet",
    "build_head",
    "bin_centres",
    "von_mises_target",
    "orientation_targets",
    "head_loss",
    "predict_head",
    "frangi_evidence",
    "flip_rot_theta",
]

#: number of axial orientation bins (section 3.2.2)
K_BINS = 16
#: concentration of the axial von Mises soft target
KAPPA_O = 4.0
#: junction down-weighting of the orientation loss
JUNCTION_WEIGHT = 0.2
#: background maximum-entropy regulariser weight
BACKGROUND_WEIGHT = 0.05
#: junction neighbourhood, in units of the local radius
JUNCTION_RADIUS_FACTOR = 2.0


def bin_centres(k: int = K_BINS, device=None, dtype=torch.float32) -> torch.Tensor:
    """``theta_k = k * pi / K`` for ``k = 0 .. K-1`` (axial bins)."""
    return torch.arange(k, device=device, dtype=dtype) * (math.pi / float(k))


BIN_CENTRES = np.arange(K_BINS, dtype=np.float64) * (math.pi / K_BINS)


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------
def _cna(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=True),
        nn.InstanceNorm2d(cout, affine=True, eps=1e-5),
        nn.LeakyReLU(1e-2, inplace=True),
    )


class MicroUNet(nn.Module):
    """~0.3 M parameter U-Net, 4 levels, base 16.

    Widths ``[16, 32, 64, 96]`` (the cap at 96 rather than 128 is what keeps
    the head inside the ~0.3 M budget of section 3.2.8); two conv-IN-LReLU
    units per encoder level, one per decoder level, transposed-conv upsampling.
    A single 1x1 head emits ``1 + K`` channels: the vessel-evidence logit and
    the ``K`` orientation logits.
    """

    def __init__(
        self,
        in_channels: int = 4,
        k_bins: int = K_BINS,
        base_channels: int = 16,
        num_levels: int = 4,
        max_channels: int = 96,
    ):
        super().__init__()
        self.k_bins = int(k_bins)
        self.in_channels = int(in_channels)
        chans = [min(base_channels * (2 ** i), max_channels) for i in range(num_levels)]
        self.channels = chans

        enc = []
        prev = in_channels
        for i, c in enumerate(chans):
            enc.append(nn.Sequential(_cna(prev, c, 1 if i == 0 else 2), _cna(c, c, 1)))
            prev = c
        self.encoder = nn.ModuleList(enc)

        ups, dec = [], []
        for i in range(num_levels - 1, 0, -1):
            deep, shallow = chans[i], chans[i - 1]
            ups.append(nn.ConvTranspose2d(deep, shallow, 2, stride=2, bias=True))
            dec.append(_cna(2 * shallow, shallow, 1))
        self.ups = nn.ModuleList(ups)
        self.decoder = nn.ModuleList(dec)
        self.out = nn.Conv2d(chans[0], 1 + self.k_bins, 1, bias=True)

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=1e-2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(v_logit [B,1,H,W], q_logit [B,K,H,W])``."""
        skips = []
        for e in self.encoder:
            x = e(x)
            skips.append(x)
        x = skips[-1]
        for i, (up, dc) in enumerate(zip(self.ups, self.decoder)):
            x = up(x)
            s = skips[-(i + 2)]
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = dc(torch.cat([x, s], dim=1))
        y = self.out(x)
        return y[:, :1], y[:, 1:]

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = [p for p in self.parameters() if (p.requires_grad or not trainable_only)]
        return int(sum(p.numel() for p in ps))


def build_head(**kw) -> MicroUNet:
    return MicroUNet(**kw)


# --------------------------------------------------------------------------
# orientation targets
# --------------------------------------------------------------------------
def orientation_targets(
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    win: int = 7,
    junction_factor: float = JUNCTION_RADIUS_FACTOR,
    junction_weight: float = JUNCTION_WEIGHT,
    min_branch_px: int = 3,
) -> Dict[str, np.ndarray]:
    """Per-pixel orientation supervision derived from the reference mask.

    Returns a dict of ``(H, W)`` arrays:

    ``theta``   axial tangent angle in ``[0, pi)`` -- from the local PCA of the
                reference skeleton (``win`` px window), then propagated to every
                pixel by nearest skeleton pixel (so a non-skeleton vessel pixel
                inherits the target of its nearest centreline pixel).
    ``valid``   bool, the reference vessel **dilated by 1 px** -- the only
                region where the directional KL is applied at full weight.
    ``weight``  float32, ``1.0`` inside ``valid``, ``junction_weight`` (0.2)
                within ``junction_factor * r`` of a junction, and
                ``BACKGROUND_WEIGHT`` (0.05) on background pixels, where the
                target is the uniform distribution.
    """
    from scipy import ndimage as ndi

    from src.rigr.candidates import local_orientation_field, propagate_to_mask
    from src.topo import skeleton as sk

    g = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, g.shape)
    g = g & f
    skel = sk.skeletonize_mask(g, min_branch_px=min_branch_px, prune=True, fov=f)
    theta_s, _ = local_orientation_field(skel, win=win)
    theta = propagate_to_mask(theta_s, skel)

    valid = ndi.binary_dilation(g, structure=np.ones((3, 3), bool), iterations=1) & f

    w = np.where(valid, 1.0, BACKGROUND_WEIGHT).astype(np.float32)
    junc = sk.junctions(skel)
    if junc.size:
        jm = np.zeros(g.shape, dtype=bool)
        jm[junc[:, 0], junc[:, 1]] = True
        radius = sk.local_radius(g)
        dist_j, idx_j = ndi.distance_transform_edt(~jm, return_indices=True)
        r_at_j = radius[idx_j[0], idx_j[1]]
        near_j = dist_j <= float(junction_factor) * np.maximum(r_at_j, 1.0)
        w = np.where(valid & near_j, float(junction_weight), w).astype(np.float32)
    w = (w * f).astype(np.float32)
    return dict(theta=theta.astype(np.float32), valid=valid, weight=w, skel=skel)


def von_mises_target(
    theta: torch.Tensor, kappa: float = KAPPA_O, k_bins: int = K_BINS
) -> torch.Tensor:
    """Axial von Mises soft target ``q*_k ~ exp(kappa cos 2(theta_k - theta))``.

    ``theta`` is ``[B, 1, H, W]``; the result is ``[B, K, H, W]`` and sums to 1
    over the bin axis.  The factor 2 inside the cosine encodes the axial
    equivalence ``theta == theta + pi``.
    """
    tk = bin_centres(k_bins, device=theta.device, dtype=theta.dtype).view(1, -1, 1, 1)
    logits = float(kappa) * torch.cos(2.0 * (tk - theta))
    return torch.softmax(logits, dim=1)


def head_loss(
    v_logit: torch.Tensor,
    q_logit: torch.Tensor,
    gt: torch.Tensor,
    fov: torch.Tensor,
    theta: torch.Tensor,
    ori_weight: torch.Tensor,
    valid: torch.Tensor,
    kappa: float = KAPPA_O,
    lambda_ori: float = 1.0,
    w_bce: float = 1.0,
    w_dice: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Total head loss: ``(BCE + Dice)(V_I) + lambda_ori * weighted-KL(q)``.

    All tensors are ``[B, 1, H, W]`` except ``q_logit`` (``[B, K, H, W]``).
    ``valid`` marks the dilated reference vessel; inside it the target is the
    axial von Mises built from ``theta``, outside it the uniform distribution
    (the maximum-entropy regulariser).  ``ori_weight`` already carries the 1.0 /
    0.2 / 0.05 weighting from :func:`orientation_targets`.
    """
    from src.seg.losses import masked_bce_with_logits, soft_dice_loss

    l_bce = masked_bce_with_logits(v_logit, gt, mask=fov)
    l_dice = soft_dice_loss(torch.sigmoid(v_logit), gt, mask=fov)

    logq = F.log_softmax(q_logit, dim=1)
    k = q_logit.shape[1]
    tgt_vm = von_mises_target(theta, kappa=kappa, k_bins=k)
    tgt_u = torch.full_like(tgt_vm, 1.0 / float(k))
    tgt = torch.where(valid > 0.5, tgt_vm, tgt_u)
    kl = (tgt * (torch.log(tgt.clamp_min(1e-12)) - logq)).sum(dim=1, keepdim=True)

    w = ori_weight * fov
    denom = w.sum().clamp_min(1.0)
    l_ori = (w * kl).sum() / denom

    total = w_bce * l_bce + w_dice * l_dice + float(lambda_ori) * l_ori
    return {"loss": total, "bce": l_bce.detach(), "dice": l_dice.detach(),
            "ori": l_ori.detach()}


# --------------------------------------------------------------------------
# augmentation bookkeeping
# --------------------------------------------------------------------------
def flip_rot_theta(theta: np.ndarray, k_rot: int, flip_r: bool, flip_c: bool) -> np.ndarray:
    """Transform an axial angle map to match ``np.rot90(x, k)`` + flips.

    Order of application is: ``rot90(k_rot)`` first, then the flips (this is
    the order the arrays are transformed in :class:`train_head.HeadDataset`).
    A reflection about *either* axis sends ``theta -> -theta``; a single
    ``rot90`` sends ``theta -> theta + pi/2``.  Two flips compose to a 180
    degree rotation, which leaves an axial angle unchanged -- consistent.
    """
    t = np.asarray(theta, dtype=np.float32) + (int(k_rot) % 4) * (math.pi / 2.0)
    if bool(flip_r) != bool(flip_c):
        t = -t
    elif bool(flip_r) and bool(flip_c):
        pass                      # 180 degree rotation: axial angle unchanged
    return np.mod(t, math.pi).astype(np.float32)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def _gaussian_window(size: int, sigma_scale: float = 0.125) -> np.ndarray:
    sigma = size * sigma_scale
    c = (size - 1) / 2.0
    ax = np.arange(size, dtype=np.float64) - c
    g1 = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = np.outer(g1, g1)
    g = g / g.max()
    return np.maximum(g, g.max() * 1e-3).astype(np.float32)


def _tile_starts(length: int, window: int, stride: int):
    if length <= window:
        return [0]
    n = int(np.ceil((length - window) / float(stride))) + 1
    if n == 1:
        return [0]
    step = (length - window) / float(n - 1)
    return [int(round(i * step)) for i in range(n)]


def load_head(ckpt_path: str, device) -> Tuple[MicroUNet, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    net = MicroUNet(
        in_channels=cfg.get("in_channels", 4),
        k_bins=cfg.get("k_bins", K_BINS),
        base_channels=cfg.get("base_channels", 16),
        num_levels=cfg.get("num_levels", 4),
        max_channels=cfg.get("max_channels", 96),
    )
    net.load_state_dict(ck.get("model", ck))
    net.to(device).eval()
    return net, ck


@torch.no_grad()
def predict_head(
    image: np.ndarray,
    prob: np.ndarray,
    ckpt: str,
    gpu: int = 0,
    fov: Optional[np.ndarray] = None,
    patch: Optional[int] = None,
    stride_div: int = 2,
    batch_size: int = 4,
    amp: bool = True,
    model: Optional[MicroUNet] = None,
    device=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window inference of the head.

    Parameters
    ----------
    image : (H, W, 3) uint8 RGB (native resolution of whatever ``prob`` is on)
    prob  : (H, W) float in [0, 1], the frozen backbone's probability map
    ckpt  : path to a ``train_head`` checkpoint (ignored when ``model`` is given)
    fov   : (H, W) bool; the z-score statistics are taken inside it

    Returns
    -------
    (v_prob, q) : ((H, W) float32 in [0,1], (K, H, W) float32 summing to 1)
    """
    from src.seg.data import derive_fov, normalize

    if device is None:
        device = torch.device("cuda:%d" % int(gpu) if torch.cuda.is_available() else "cpu")
    ck: dict = {}
    if model is None:
        model, ck = load_head(ckpt, device)
    model.eval()
    patch = int(patch or ck.get("config", {}).get("patch", 512))

    img = np.ascontiguousarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if fov is None:
        fov = derive_fov(img.astype(np.uint8)) > 0
    m = np.asarray(fov, dtype=bool)
    pix = img[m].astype(np.float32) if m.sum() > 16 else img.reshape(-1, 3).astype(np.float32)
    mean = pix.mean(axis=0)
    std = np.maximum(pix.std(axis=0), 1e-3)
    x = normalize(img.astype(np.uint8), mean, std)                 # CHW float32
    p = np.asarray(prob, dtype=np.float32)[None]
    x = np.concatenate([x, p], axis=0)

    c, h, w = x.shape
    ph, pw = max(patch, h), max(patch, w)
    if (ph, pw) != (h, w):
        pad = np.zeros((c, ph, pw), dtype=np.float32)
        pad[:, :h, :w] = x
        x = pad
    xt = torch.from_numpy(x).to(device)

    stride = max(1, patch // max(1, int(stride_div)))
    ys = _tile_starts(ph, patch, stride)
    xs = _tile_starts(pw, patch, stride)
    gw = torch.from_numpy(_gaussian_window(patch)).to(device)
    k = model.k_bins

    acc_v = torch.zeros((ph, pw), dtype=torch.float32, device=device)
    acc_q = torch.zeros((k, ph, pw), dtype=torch.float32, device=device)
    wsum = torch.zeros((ph, pw), dtype=torch.float32, device=device)

    coords = [(y, xx) for y in ys for xx in xs]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        tiles = torch.stack([xt[:, y:y + patch, xx:xx + patch] for y, xx in chunk], 0)
        with torch.autocast("cuda", enabled=(amp and device.type == "cuda")):
            vl, ql = model(tiles)
        vp = torch.sigmoid(vl.float())[:, 0]
        qp = torch.softmax(ql.float(), dim=1)
        for j, (y, xx) in enumerate(chunk):
            acc_v[y:y + patch, xx:xx + patch] += vp[j] * gw
            acc_q[:, y:y + patch, xx:xx + patch] += qp[j] * gw
            wsum[y:y + patch, xx:xx + patch] += gw

    ws = wsum.clamp_min(1e-8)
    v = (acc_v / ws)[:h, :w]
    q = (acc_q / ws[None])[:, :h, :w]
    q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return (v.cpu().numpy().astype(np.float32), q.cpu().numpy().astype(np.float32))


# --------------------------------------------------------------------------
# analytic alternative (Tab.3 ablation)
# --------------------------------------------------------------------------
def frangi_evidence(
    image: np.ndarray,
    fov: Optional[np.ndarray] = None,
    sigmas: Sequence[float] = (1, 2, 3, 4, 5),
    lo: float = 1.0,
    hi: float = 99.0,
) -> np.ndarray:
    """Analytic vessel evidence: Frangi vesselness of the green channel.

    Retinal vessels are *darker* than the background on the green channel, so
    ``black_ridges=True``.  The response is contrast-normalised inside the FOV
    by the ``[lo, hi]`` percentiles and clipped to ``[0, 1]``, which is the
    range ``V_I`` must live in for ``w_I = -log V_I`` to be non-negative.

    ``hi = p99`` is the measured optimum on DRIVE: the Frangi response is
    extremely heavy-tailed (FOV p50 = 7.5e-5, p99.9 = 0.64), so a higher
    percentile crushes real vessels towards 0 and a lower one lifts the
    background.  Measured mean ``V_I`` inside / outside the reference mask on
    DRIVE 21_training: p99.5 -> 0.102 / 0.010 (ratio 10.6), **p99 -> 0.221 /
    0.016 (ratio 13.7)**, p95 -> 0.552 / 0.056 (ratio 9.8).
    """
    from skimage.filters import frangi

    img = np.asarray(image)
    if img.ndim == 3:
        green = img[..., 1].astype(np.float32)
    else:
        green = img.astype(np.float32)
    green = green / max(float(green.max()), 1e-6)

    resp = frangi(green, sigmas=sigmas, black_ridges=True)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    m = np.ones(resp.shape, bool) if fov is None else np.asarray(fov, bool)
    if m.sum() < 16:
        m = np.ones(resp.shape, bool)
    a = float(np.percentile(resp[m], lo))
    b = float(np.percentile(resp[m], hi))
    if not np.isfinite(b) or b <= a:
        b = a + 1e-6
    out = np.clip((resp - a) / (b - a), 0.0, 1.0).astype(np.float32)
    out[~m] = 0.0
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - smoke demo
    net = MicroUNet()
    n = net.num_parameters()
    print("MicroUNet channels   : %s" % net.channels)
    print("trainable parameters : %d (%.3f M)" % (n, n / 1e6))
    with torch.no_grad():
        v, q = net(torch.randn(2, 4, 128, 128))
    print("V_I logit shape      : %s" % (tuple(v.shape),))
    print("q logit shape        : %s" % (tuple(q.shape),))

    # loss + target sanity
    gt = (torch.rand(2, 1, 128, 128) > 0.9).float()
    fov = torch.ones_like(gt)
    theta = torch.rand(2, 1, 128, 128) * math.pi
    ow = torch.full_like(gt, 0.05) + gt * 0.95
    out = head_loss(v, q, gt, fov, theta, ow, gt)
    print("loss terms           : %s"
          % {k: round(float(x), 4) for k, x in out.items()})
    t = von_mises_target(theta[:1, :, :2, :2])
    print("vM target sums to 1  : %.6f" % float(t.sum(dim=1).mean()))

    # axial angle transform table
    for kr, fr, fc in [(0, False, False), (1, False, False), (0, True, False),
                       (0, False, True), (0, True, True), (1, True, False)]:
        th = np.array([0.0, math.pi / 4, math.pi / 2], dtype=np.float32)
        print("rot%d flip_r=%d flip_c=%d -> %s"
              % (kr, fr, fc, np.round(flip_rot_theta(th, kr, fr, fc), 4).tolist()))
