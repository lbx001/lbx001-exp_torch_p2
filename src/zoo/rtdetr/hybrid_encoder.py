"""HybridEncoder: AIFI + CCFM neck for RT-DETR."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['HybridEncoder']


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act='silu'):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = self._build_act(act)

    @staticmethod
    def _build_act(act):
        acts = {'silu': nn.SiLU(), 'relu': nn.ReLU(inplace=True),
                'gelu': nn.GELU(), 'identity': nn.Identity()}
        return acts.get(act, nn.SiLU())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepC3(nn.Module):
    """RepC3 / CSP-like block for feature fusion."""
    def __init__(self, in_c, out_c, n=3, e=1.0, act='silu'):
        super().__init__()
        mid_c = int(out_c * e)
        self.cv1 = ConvBNAct(in_c, mid_c, 1, act=act)
        self.cv2 = ConvBNAct(in_c, mid_c, 1, act=act)
        self.bottlenecks = nn.Sequential(*[
            nn.Sequential(
                ConvBNAct(mid_c, mid_c, 3, act=act),
                ConvBNAct(mid_c, mid_c, 3, act=act),
            )
            for _ in range(n)
        ])
        self.cv3 = ConvBNAct(mid_c * 2, out_c, 1, act=act)

    def forward(self, x):
        y1 = self.bottlenecks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


# ---------------------------------------------------------------------------
# Transformer encoder (AIFI)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.0, act: str = 'gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True)

    def forward(self, src, src_key_padding_mask=None, pos=None):
        q = k = src if pos is None else src + pos
        src2, _ = self.self_attn(q, k, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class AIFI(nn.Module):
    """Attentional Injection Feature Integration: applies transformer encoder to one FPN level."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0, act: str = 'gelu',
                 num_layers: int = 1, pe_temperature: float = 10000):
        super().__init__()
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, act)
            for _ in range(num_layers)
        ])
        self.pe_temperature = pe_temperature

    def _build_2d_sincos_pe(self, H: int, W: int, C: int, device) -> torch.Tensor:
        """[1, H*W, C] sine-cosine positional encoding."""
        assert C % 2 == 0
        half = C // 2
        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        dim_t = self.pe_temperature ** (
            2 * torch.arange(half // 2, device=device).float() / half)
        y_embed = ys[:, None] / dim_t[None, :]  # [H, half/2]
        x_embed = xs[:, None] / dim_t[None, :]  # [W, half/2]
        pe_y = torch.stack([y_embed.sin(), y_embed.cos()], dim=-1).flatten(1)  # [H, half]
        pe_x = torch.stack([x_embed.sin(), x_embed.cos()], dim=-1).flatten(1)  # [W, half]
        pe_y = pe_y.unsqueeze(1).expand(-1, W, -1)  # [H, W, half]
        pe_x = pe_x.unsqueeze(0).expand(H, -1, -1)  # [H, W, half]
        pe = torch.cat([pe_y, pe_x], dim=-1).view(1, H * W, C)  # [1, H*W, C]
        return pe

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        pe = self._build_2d_sincos_pe(H, W, C, feat.device)
        src = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        for layer in self.encoder:
            src = layer(src, pos=pe)
        return src.transpose(1, 2).view(B, C, H, W)


# ---------------------------------------------------------------------------
# HybridEncoder
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    """
    Hybrid Encoder = AIFI (transformer on high-level features) + CCFM (top-down FPN with RepC3).
    Input:  list of backbone features [C3, C4, C5] with channels = in_channels
    Output: list of enhanced features [P3, P4, P5] all with hidden_dim channels
    """

    def __init__(
        self,
        in_channels=None,
        feat_strides=None,
        hidden_dim: int = 256,
        use_encoder_idx=None,
        num_encoder_layers: int = 1,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        enc_act: str = 'gelu',
        pe_temperature: float = 10000,
        expansion: float = 1.0,
        depth_mult: float = 1.0,
        act: str = 'silu',
        eval_spatial_size=None,
    ):
        super().__init__()
        if in_channels is None:
            in_channels = [512, 1024, 2048]
        if feat_strides is None:
            feat_strides = [8, 16, 32]
        if use_encoder_idx is None:
            use_encoder_idx = [2]

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_levels = len(in_channels)

        # 1x1 lateral convs to project each backbone level to hidden_dim
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(c, hidden_dim, 1, act=act) for c in in_channels
        ])

        # AIFI transformer on selected levels
        self.encoder = nn.ModuleList()
        for _ in use_encoder_idx:
            self.encoder.append(AIFI(
                hidden_dim, nhead, dim_feedforward, dropout, enc_act,
                num_encoder_layers, pe_temperature,
            ))

        # CCFM: top-down path
        n = max(round(3 * depth_mult), 1)
        mid = int(hidden_dim * expansion)

        # Top-down FPN
        self.top_down_blocks = nn.ModuleList()
        self.upsample_projs = nn.ModuleList()
        for i in range(self.num_levels - 1, 0, -1):
            # upsample from level i to level i-1
            self.upsample_projs.append(ConvBNAct(hidden_dim, hidden_dim, 1, act=act))
            self.top_down_blocks.append(RepC3(hidden_dim * 2, hidden_dim, n, act=act))

        # Bottom-up path
        self.bottom_up_convs = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.bottom_up_convs.append(ConvBNAct(hidden_dim, hidden_dim, 3, s=2, act=act))
            self.bottom_up_blocks.append(RepC3(hidden_dim * 2, hidden_dim, n, act=act))

    def forward(self, feats):
        """feats: list of backbone features [C3, C4, C5]."""
        assert len(feats) == self.num_levels

        # Lateral projections
        inner = [proj(f) for proj, f in zip(self.lateral_convs, feats)]

        # AIFI on selected levels
        for enc_idx, enc in zip(self.use_encoder_idx, self.encoder):
            inner[enc_idx] = enc(inner[enc_idx])

        # Top-down FPN
        td = list(inner)
        for i in range(self.num_levels - 2, -1, -1):
            proj_idx = self.num_levels - 2 - i
            up = F.interpolate(
                self.upsample_projs[proj_idx](td[i + 1]),
                size=td[i].shape[-2:], mode='nearest',
            )
            td[i] = self.top_down_blocks[proj_idx](torch.cat([up, td[i]], dim=1))

        # Bottom-up path
        out = [td[0]]
        for i in range(self.num_levels - 1):
            down = self.bottom_up_convs[i](out[-1])
            out.append(self.bottom_up_blocks[i](torch.cat([down, td[i + 1]], dim=1)))

        return out  # [P3, P4, P5]
