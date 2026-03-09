from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int | tuple[int, int], padding: int | tuple[int, int]):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class SOEPBlock(nn.Module):
    """A lightweight SOEP block inspired by the paper section 3.3 description."""

    def __init__(self, channels: int):
        super().__init__()
        self.large_square = DepthwiseSeparableConv(channels, kernel_size=31, padding=15)
        self.large_h = DepthwiseSeparableConv(channels, kernel_size=(1, 31), padding=(0, 15))
        self.large_v = DepthwiseSeparableConv(channels, kernel_size=(31, 1), padding=(15, 0))

        self.global_fca = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.global_sca = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.global_freq = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.global_spatial = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.local_branch = DepthwiseSeparableConv(channels, kernel_size=1, padding=0)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _channel_attention(x: torch.Tensor, projection: nn.Conv2d) -> torch.Tensor:
        attn = projection(x.mean(dim=(2, 3), keepdim=True)).sigmoid()
        return attn

    def _global_branch(self, x: torch.Tensor) -> torch.Tensor:
        fca_attn = self._channel_attention(x, self.global_fca)
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fca = torch.fft.ifft2(x_fft * fca_attn, norm='ortho').real

        sca_attn = self._channel_attention(x_fca, self.global_sca)
        x_dcam = x_fca * sca_attn

        freq_feat = self.global_freq(x_dcam)
        spatial_attn = self.global_spatial(x_dcam).sigmoid()
        x_fsam = torch.fft.ifft2(torch.fft.fft2(freq_feat, norm='ortho') * spatial_attn, norm='ortho').real
        return x_fsam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        large = self.large_square(x) + self.large_h(x) + self.large_v(x)
        global_feat = self._global_branch(x)
        local = self.local_branch(x)
        fused = self.fuse(torch.cat([large, global_feat, local], dim=1))
        return fused + x
