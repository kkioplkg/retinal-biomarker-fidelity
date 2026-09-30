"""rNCA (our re-implementation) -- the essential NCA refinement operator, in PyTorch.

Faithful port of ``maltesilber/rnca`` ("rNCA: Self-Repairing Segmentation
Masks", commit ``54f681b6aa352ff7134232fac0d6c3a25b052ebc``, README says MIT,
no LICENSE file).  The upstream repo is **JAX / Flax NNX** (``requirements.txt``
pins ``jax[cuda12]``, for which no Windows GPU wheels exist) and ships neither
pretrained weights nor the mask-corruption code, so it cannot be imported or
reused here.  Everything below re-derives the model from their source.

Faithfully reproduced
---------------------
* state ``(B, C, H, W)`` with ``C = state_channels = 16``; the **last** channel
  is the alpha / alive / output channel, seeded from the imperfect mask and
  clipped to ``[0.1, 1]``; the other 15 start at zero.
* perception = **fixed, non-trainable depthwise** conv with the identity kernel
  plus the two L1-normalised Sobel-like gradient kernels (``grad_kernel``),
  giving ``signal_size = 3 * state_channels = 48`` channels.  No Laplacian.
* the conditioning image is embedded **once**, outside the recurrent loop, by a
  learned ``3x3`` conv into ``signal_size`` channels, and reused every step.
* update MLP = ``concat(x_emb, signal) -> 1x1 conv to hidden_dim=128 -> ReLU ->
  1x1 conv (zero-initialised) -> Dropout(0.5)``; the zero init makes
  ``delta = 0`` at step 0.  Dropout on the delta is their stochastic per-cell
  update rule (no separate fire-rate mask).
* alive masking: ``maxpool3x3(alpha) > 0.1`` computed **before and AND-ed with
  after** the update, then multiplied into the new state.
* training: Growing-NCA persistent sample pool (``pool_size=256``,
  ``batch_size=16``, ``replace_n=2`` reseeded per step), ``nca_steps=64``
  fully-differentiable unrolled steps, MSE between the alpha channel and the GT
  taken at a **random step in the second half** of the rollout, AdamW
  ``lr=1e-4`` with global-norm clipping at 1.0.
* inference: 256 steps, binarise ``alpha > alive_threshold`` (0.1, *not* 0.5).

Deviations (all forced, all documented)
---------------------------------------
D1. **Framework**: JAX/Flax NNX -> PyTorch.  Numerically equivalent ops; the
    fixed perception kernel is registered as a buffer rather than an
    ``nnx.Variable``.
D2. **Delta width**: upstream's last conv emits ``img_channels`` channels and
    relies on broadcasting ``state + delta`` -- which only works because their
    bundled dataset has ``img_channels == 1``.  With a multi-channel
    conditioning input (we use green + probability) that broadcast is a shape
    error.  We emit ``state_channels`` channels, the obviously intended
    behaviour and the standard NCA form.  ``delta_width="img"`` restores the
    upstream expression for ``img_channels == 1``.
D3. **Corruption**: not shipped upstream.  We build the imperfect mask with the
    project's capsule severances (``src.baselines.synth``), which is the
    degradation our study is about.
D4. **Patching**: their data is 64x64 whole images.  Fundus images are far
    larger, so we train on random ``patch`` crops (default 96) and run
    inference fully convolutionally on the whole image (optionally tiled).
D5. **Checkpoints**: Orbax -> ``torch.save`` of a plain ``state_dict`` plus a
    config dict (loadable under torch>=2.6's ``weights_only=True`` default).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RNCA", "RNCAConfig", "seed_state", "rollout", "binarize"]


# --------------------------------------------------------------------------
# fixed perception kernels (verbatim port of nca.py::identity_kernel/grad_kernel)
# --------------------------------------------------------------------------


def _identity_kernel() -> torch.Tensor:
    k = torch.zeros(3, 3)
    k[1, 1] = 1.0
    return k


def _grad_kernels(normalize: bool = True) -> torch.Tensor:
    """The two directional Sobel-like kernels, L1-normalised (2, 3, 3)."""
    grad = torch.tensor([-1.0, 0.0, 1.0])
    smooth = torch.tensor([1.0, 2.0, 1.0])
    ky = torch.outer(grad, smooth)   # gradient along axis 0
    kx = torch.outer(smooth, grad)   # gradient along axis 1
    ks = torch.stack([ky, kx], dim=0)
    if normalize:
        ks = ks / ks.abs().sum(dim=(1, 2), keepdim=True)
    return ks


def perception_kernel(state_channels: int) -> torch.Tensor:
    """Depthwise kernel of shape ``(3 * state_channels, 1, 3, 3)``.

    Channel ``3*c + {0,1,2}`` holds identity / grad-y / grad-x of state channel
    ``c``, matching upstream's ``concat([kernel] * state_channels)`` layout
    under a grouped conv with ``feature_group_count = state_channels``.
    """
    base = torch.cat([_identity_kernel().unsqueeze(0), _grad_kernels()], dim=0)  # (3,3,3)
    k = base.repeat(state_channels, 1, 1)                                        # (3C,3,3)
    return k.unsqueeze(1).contiguous()                                           # (3C,1,3,3)


# --------------------------------------------------------------------------


class RNCAConfig:
    """Hyper-parameters, defaulting to upstream ``main.py::Config``."""

    def __init__(self, img_channels: int = 2, state_channels: int = 16,
                 img_kernel_size: int = 3, hidden_dim: int = 128,
                 dropout_rate: float = 0.5, alive_threshold: float = 0.1,
                 delta_width: str = "state", **extra):
        self.img_channels = int(img_channels)
        self.state_channels = int(state_channels)
        self.img_kernel_size = int(img_kernel_size)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.alive_threshold = float(alive_threshold)
        self.delta_width = str(delta_width)  # "state" (D2) or "img" (upstream)
        self.extra = dict(extra)

    def to_dict(self) -> Dict:
        d = dict(img_channels=self.img_channels, state_channels=self.state_channels,
                 img_kernel_size=self.img_kernel_size, hidden_dim=self.hidden_dim,
                 dropout_rate=self.dropout_rate, alive_threshold=self.alive_threshold,
                 delta_width=self.delta_width)
        d.update(self.extra)
        return d


class RNCA(nn.Module):
    """The refinement neural cellular automaton.

    ``state`` is ``(B, C, H, W)``, ``x`` (the conditioning image) is
    ``(B, img_channels, H, W)``.  Channels-first throughout, unlike upstream's
    channels-last JAX arrays.
    """

    def __init__(self, cfg: RNCAConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.state_channels
        signal = 3 * C

        self.register_buffer("perceive_kernel", perception_kernel(C), persistent=False)
        self.cond_embedder = nn.Conv2d(
            cfg.img_channels, signal, cfg.img_kernel_size,
            padding=cfg.img_kernel_size // 2, bias=False)

        out_ch = C if cfg.delta_width == "state" else cfg.img_channels
        self.fc1 = nn.Conv2d(signal * 2, cfg.hidden_dim, 1)
        self.fc2 = nn.Conv2d(cfg.hidden_dim, out_ch, 1)
        nn.init.zeros_(self.fc2.weight)          # upstream: kernel_init=zeros
        nn.init.zeros_(self.fc2.bias)
        self.drop = nn.Dropout(cfg.dropout_rate)

    # -- pieces -----------------------------------------------------------
    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        """Fixed depthwise identity + Sobel perception -> (B, 3C, H, W)."""
        return F.conv2d(state, self.perceive_kernel.to(state.dtype),
                        padding=1, groups=self.cfg.state_channels)

    def alive_mask(self, state: torch.Tensor) -> torch.Tensor:
        alpha = state[:, -1:, :, :]
        pooled = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
        return pooled > self.cfg.alive_threshold

    def update(self, signal: torch.Tensor, x_emb: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_emb, signal], dim=1)
        h = F.relu(self.fc1(h))
        return self.drop(self.fc2(h))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Conditioning embedding, computed once per rollout (as upstream)."""
        return self.cond_embedder(x)

    def step(self, state: torch.Tensor, x_emb: torch.Tensor) -> torch.Tensor:
        signal = self.perceive(state)
        alive = self.alive_mask(state)
        delta = self.update(signal, x_emb)
        new_state = state + delta
        alive = alive & self.alive_mask(new_state)
        return new_state * alive.to(new_state.dtype)

    def forward(self, state: torch.Tensor, x: torch.Tensor, num_steps: int = 64,
                return_traj: bool = False):
        """Run ``num_steps`` NCA updates.  Returns the final state, or the
        whole trajectory ``(num_steps, B, C, H, W)`` when ``return_traj``."""
        x_emb = self.embed(x)
        traj = []
        for _ in range(int(num_steps)):
            state = self.step(state, x_emb)
            if return_traj:
                traj.append(state)
        return torch.stack(traj, 0) if return_traj else state

    def render(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, -1:, :, :] > self.cfg.alive_threshold


# --------------------------------------------------------------------------
# state seeding / rollout helpers
# --------------------------------------------------------------------------


def seed_state(mask: torch.Tensor, state_channels: int = 16,
               clip_lo: float = 0.1) -> torch.Tensor:
    """Seed the NCA state from an imperfect mask ``(B, 1, H, W)`` in ``[0, 1]``.

    Upstream clips the seed to ``[0.1, 1]`` so background cells are not exactly
    dead; that floor is why ``alive_threshold`` is 0.1 and not 0.5.
    """
    b, _, h, w = mask.shape
    state = mask.new_zeros((b, state_channels, h, w))
    state[:, -1:, :, :] = mask.clamp(min=clip_lo, max=1.0)
    return state


@torch.no_grad()
def rollout(model: RNCA, mask: torch.Tensor, x: torch.Tensor,
            num_steps: int = 256) -> torch.Tensor:
    """Inference rollout -> the final alpha channel ``(B, 1, H, W)``."""
    model.eval()
    state = seed_state(mask, model.cfg.state_channels)
    state = model(state, x, num_steps=num_steps)
    return state[:, -1:, :, :]


def binarize(alpha: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    return alpha > threshold


# --------------------------------------------------------------------------
# persistent sample pool (Growing-NCA trick, upstream dataset.py)
# --------------------------------------------------------------------------


class SamplePool:
    """Fixed-size pool of evolving ``(X, Y, S)`` triples.

    Each training step draws ``batch_size`` slots, overwrites ``replace_n`` of
    them with freshly seeded samples, and writes the evolved states back --
    upstream ``sample_pool`` / ``random_batch_update`` / ``update_pool``.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
        self.X, self.Y, self.S = x, y, s
        self.size = x.shape[0]

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.choice(self.size, size=int(batch_size), replace=False)
        idx = torch.as_tensor(idx, dtype=torch.long)
        return idx, self.X[idx].clone(), self.Y[idx].clone(), self.S[idx].clone()

    def write(self, idx: torch.Tensor, s: torch.Tensor) -> None:
        self.S[idx] = s.detach().to(self.S.dtype).cpu()


def loss_at_random_late_step(
    model: RNCA, x: torch.Tensor, y: torch.Tensor, s: torch.Tensor,
    num_steps: int, generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Upstream ``loss_fun``: MSE against the GT at a random step in the
    second half of the rollout, one independent step index per batch element.

    The trajectory is kept so the per-sample index can be gathered; that is
    what upstream does too (``states[jnp.arange(B), idx]``).
    """
    traj = model(s, x, num_steps=num_steps, return_traj=True)      # (T,B,C,H,W)
    b = x.shape[0]
    lo = num_steps // 2
    idx = torch.randint(lo, num_steps, (b,), generator=generator, device=traj.device)
    picked = traj[idx, torch.arange(b, device=traj.device)]        # (B,C,H,W)
    loss = F.mse_loss(picked[:, -1:, :, :], y)
    return loss, picked
