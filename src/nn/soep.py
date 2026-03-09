"""
SOEP: Small Object Enhancement Processing module.
Based on paper Section 3.3: SPDConv + CSP-OmniKernel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Standard Conv + BN + Activation"""
    def __init__(self, in_c, out_c, k=1, s=1, p=0, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution.
    Input: H x W x C
    Output: H/s x W/s x (s^2*C) -> Conv1x1 -> H/s x W/s x out_c
    Scale s=2 by default.
    """
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        mid_c = in_c * scale * scale
        self.conv = ConvBNAct(mid_c, out_c, k=1, s=1, p=0)

    def forward(self, x):
        s = self.scale
        B, C, H, W = x.shape
        x = x.view(B, C, H // s, s, W // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * s * s, H // s, W // s)
        return self.conv(x)


class LocalBranch(nn.Module):
    """Local branch: 1x1 depthwise separable conv"""
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class LargeKernelBranch(nn.Module):
    """Large kernel branch: 31x31 + 31x1 + 1x31 DW-separable convs"""
    def __init__(self, channels, large_k=31):
        super().__init__()
        pad = large_k // 2
        self.dw_sq = nn.Conv2d(channels, channels, large_k, 1, pad, groups=channels, bias=False)
        self.dw_h = nn.Conv2d(channels, channels, (large_k, 1), 1, (pad, 0), groups=channels, bias=False)
        self.dw_v = nn.Conv2d(channels, channels, (1, large_k), 1, (0, pad), groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw_sq(x) + self.dw_h(x) + self.dw_v(x))))


class DCAM(nn.Module):
    """
    Dual-domain Channel Attention Module.
    X_FCA = IFFT(FFT(X) * W_1x1(GAP(X)))
    X_DCAM = X_FCA * W_1x1(GAP(X_FCA))
    """
    def __init__(self, channels):
        super().__init__()
        self.w1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.w2 = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        gap = x.mean(dim=[2, 3], keepdim=True)
        weight1 = torch.sigmoid(self.w1(gap))
        x_fft = torch.fft.rfft2(x)
        x_fca = torch.fft.irfft2(x_fft * weight1, s=x.shape[-2:])
        gap2 = x_fca.mean(dim=[2, 3], keepdim=True)
        weight2 = torch.sigmoid(self.w2(gap2))
        return x_fca * weight2


class FSAM(nn.Module):
    """
    Frequency-Based Spatial Attention Module.
    X_FSAM = IFFT(FFT(W_1x1(X_DCAM)) * W_1x1(X_DCAM))
    """
    def __init__(self, channels):
        super().__init__()
        self.w1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.w2 = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        w1_x = self.w1(x)
        w2_x = self.w2(x)
        x_fft = torch.fft.rfft2(w1_x)
        x_fsam = torch.fft.irfft2(x_fft * torch.fft.rfft2(w2_x), s=x.shape[-2:])
        return x_fsam


class GlobalBranch(nn.Module):
    """Global branch: DCAM + FSAM"""
    def __init__(self, channels):
        super().__init__()
        self.dcam = DCAM(channels)
        self.fsam = FSAM(channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x_dcam = self.dcam(x)
        x_fsam = self.fsam(x_dcam)
        return self.act(self.bn(x_fsam))


class OmniKernel(nn.Module):
    """OmniKernel: Local + LargeKernel + Global branches, concatenated then projected"""
    def __init__(self, channels, large_k=31):
        super().__init__()
        self.local_br = LocalBranch(channels)
        self.large_br = LargeKernelBranch(channels, large_k)
        self.global_br = GlobalBranch(channels)
        self.proj = ConvBNAct(channels * 3, channels, k=1)

    def forward(self, x):
        local_out = self.local_br(x)
        large_out = self.large_br(x)
        global_out = self.global_br(x)
        return self.proj(torch.cat([local_out, large_out, global_out], dim=1))


class CSPOmniKernel(nn.Module):
    """
    CSP-OmniKernel: split channels, process 0.25 through OmniKernel, concat with 0.75 bypass.
    """
    def __init__(self, in_c, out_c, large_k=31):
        super().__init__()
        mid_c = in_c
        self.omni_c = max(1, mid_c // 4)
        self.bypass_c = mid_c - self.omni_c

        self.input_conv = ConvBNAct(in_c, mid_c, k=1)
        self.omni = OmniKernel(self.omni_c, large_k)
        self.output_conv = ConvBNAct(mid_c, out_c, k=1)

    def forward(self, x):
        x = self.input_conv(x)
        omni_part = x[:, :self.omni_c, ...]
        bypass = x[:, self.omni_c:, ...]
        omni_out = self.omni(omni_part)
        out = torch.cat([omni_out, bypass], dim=1)
        return self.output_conv(out)


class SOEPModule(nn.Module):
    """
    SOEP (Small Object Enhancement Processing) Module.
    Integrates P2 backbone feature with P3 using:
    1. SPDConv on P2 to match P3 spatial resolution
    2. CSPOmniKernel fusion of SPDConv(P2) + P3

    Args:
        p2_channels: channels of P2 feature (from backbone, e.g., 256)
        p3_channels: channels of P3 feature (e.g., 512)
        out_channels: output channels (should match p3_channels for drop-in)
        spd_scale: scale factor for SPDConv (default 2)
        large_k: large kernel size for OmniKernel (default 31)
    """
    def __init__(self, p2_channels, p3_channels, out_channels, spd_scale=2, large_k=31):
        super().__init__()
        self.spd = SPDConv(p2_channels, p3_channels, scale=spd_scale)
        self.fusion = CSPOmniKernel(p3_channels * 2, out_channels, large_k=large_k)

    def forward(self, p2, p3):
        """
        p2: [B, p2_channels, H*2, W*2]
        p3: [B, p3_channels, H, W]
        Returns: enhanced p3 [B, out_channels, H, W]
        """
        p2_down = self.spd(p2)
        if p2_down.shape[-2:] != p3.shape[-2:]:
            p2_down = F.interpolate(p2_down, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([p2_down, p3], dim=1)
        return self.fusion(fused)
