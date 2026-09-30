"""nnU-Net-style 2D U-Net for retinal vessel segmentation.

Design (per exp/EXPERIMENT_PLAN.md S2):
  * 5 resolution levels, base 32 channels, doubling per level, capped at 512
    -> [32, 64, 128, 256, 512]
  * two (conv 3x3 -> InstanceNorm -> LeakyReLU(0.01)) blocks per level
  * downsampling by strided convolution (stride 2 on the first conv of a level)
  * upsampling by transposed convolution (kernel 2, stride 2)
  * skip connections concatenated
  * optional deep supervision (OFF by default)
  * input : 3-channel RGB, z-scored per image inside the FOV
  * output: 1 logit channel

Parameter count is ~7.76 M, inside the 7-8 M target of the plan.
Run `python -m src.seg.unet` to print the exact count and a shape check.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_norm_act(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.InstanceNorm2d(out_ch, affine=True, eps=1e-5),
        nn.LeakyReLU(negative_slope=1e-2, inplace=True),
    )


class StackedConvBlock(nn.Module):
    """Two conv-IN-LeakyReLU units; the first may be strided (downsampling)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            _conv_norm_act(in_ch, out_ch, stride=stride),
            _conv_norm_act(out_ch, out_ch, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """nnU-Net-style 2D U-Net.

    Args:
        in_channels: number of input channels (3 for RGB).
        out_channels: number of output logit channels (1 for binary vessels).
        base_channels: channels at the highest resolution (32).
        num_levels: number of resolution levels (5).
        max_channels: channel cap (512).
        deep_supervision: if True, ``forward`` returns a list of logits, one per
            decoder stage, ordered from the highest resolution to the lowest.
            Default False -> returns a single full-resolution logit tensor.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 5,
        max_channels: int = 512,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.deep_supervision = deep_supervision
        self.out_channels = out_channels

        chans = [min(base_channels * (2 ** i), max_channels) for i in range(num_levels)]
        self.channels = chans

        # ---- encoder -----------------------------------------------------
        encoder = []
        prev = in_channels
        for i, c in enumerate(chans):
            stride = 1 if i == 0 else 2
            encoder.append(StackedConvBlock(prev, c, stride=stride))
            prev = c
        self.encoder = nn.ModuleList(encoder)

        # ---- decoder -----------------------------------------------------
        ups, decoder, heads = [], [], []
        for i in range(num_levels - 1, 0, -1):
            deep, shallow = chans[i], chans[i - 1]
            ups.append(nn.ConvTranspose2d(deep, shallow, kernel_size=2, stride=2, bias=True))
            decoder.append(StackedConvBlock(2 * shallow, shallow, stride=1))
            heads.append(nn.Conv2d(shallow, out_channels, kernel_size=1, bias=True))
        self.ups = nn.ModuleList(ups)
        self.decoder = nn.ModuleList(decoder)
        # heads[0] is the coarsest decoder stage, heads[-1] the full-resolution one
        self.heads = nn.ModuleList(heads)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=1e-2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)

        x = skips[-1]
        logits = []
        for i, (up, dec, head) in enumerate(zip(self.ups, self.decoder, self.heads)):
            skip = skips[-(i + 2)]
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            if self.deep_supervision:
                logits.append(head(x))

        if self.deep_supervision:
            # return highest resolution first
            return logits[::-1]
        return self.heads[-1](x)

    # -- convenience -------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        if trainable_only:
            ps = (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in ps)


def build_unet(deep_supervision: bool = False, **kwargs) -> UNet:
    return UNet(deep_supervision=deep_supervision, **kwargs)


if __name__ == "__main__":
    net = UNet()
    n = net.num_parameters()
    print(f"UNet channels        : {net.channels}")
    print(f"trainable parameters : {n:,} ({n / 1e6:.2f} M)")
    with torch.no_grad():
        y = net(torch.randn(2, 3, 256, 256))
    print(f"output shape         : {tuple(y.shape)}")
    net_ds = UNet(deep_supervision=True)
    with torch.no_grad():
        ys = net_ds(torch.randn(1, 3, 256, 256))
    print(f"deep supervision     : {[tuple(t.shape) for t in ys]}")
