"""RT-DETR Transformer decoder (pure PyTorch implementation)."""
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MLP, inverse_sigmoid, bias_init_with_prob
from .denoising import get_contrastive_denoising_training_group

__all__ = ['RTDETRTransformer']


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def get_sine_pos_embed(pos: torch.Tensor, num_pos_feats: int = 128,
                       temperature: int = 10000, scale: float = 2 * math.pi):
    """Sinusoidal positional encoding from 2-D normalized coords [0,1]."""
    assert pos.shape[-1] == 2
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    x_embed = pos[..., 0:1] * scale / dim_t
    y_embed = pos[..., 1:2] * scale / dim_t
    pos_x = torch.stack([x_embed[..., 0::2].sin(), x_embed[..., 1::2].cos()], dim=-1).flatten(-2)
    pos_y = torch.stack([y_embed[..., 0::2].sin(), y_embed[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([pos_x, pos_y], dim=-1)  # [..., num_pos_feats*2]


# ---------------------------------------------------------------------------
# Multi-scale deformable attention (pure PyTorch)
# ---------------------------------------------------------------------------

class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention (pure-PyTorch, no CUDA kernel)."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_levels: int = 4, num_points: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [H, 2]
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).expand(
            self.num_heads, self.num_levels, self.num_points, 2)
        # Scale by point index
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.flatten())

        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query, reference_points, value, value_spatial_shapes,
                value_level_start_index, attn_mask=None):
        """
        query: [B, Q, C]
        reference_points: [B, Q, L, 2]  (normalized [0,1])
        value: [B, sum(H*W), C]
        value_spatial_shapes: list of (H_l, W_l)
        value_level_start_index: list of start indices
        """
        B, Q, _ = query.shape
        B, Sv, _ = value.shape
        H = self.num_heads
        L = self.num_levels
        P = self.num_points

        # Project values
        value = self.value_proj(value)  # [B, Sv, C]
        value = value.view(B, Sv, H, self.head_dim)  # [B, Sv, H, Dh]

        # Sampling offsets
        offsets = self.sampling_offsets(query)  # [B, Q, H*L*P*2]
        offsets = offsets.view(B, Q, H, L, P, 2)

        # Attention weights
        attn_w = self.attention_weights(query)  # [B, Q, H*L*P]
        attn_w = attn_w.view(B, Q, H, L * P).softmax(-1)
        attn_w = attn_w.view(B, Q, H, L, P)  # [B, Q, H, L, P]

        # Sampling locations: reference + offset / spatial_shape
        # offsets: [B, Q, H, L, P, 2]; reference_points: [B, Q, L, 2]
        ref = reference_points  # [B, Q, L, 2]
        sampling_locs = torch.zeros(B, Q, L, H, P, 2, device=query.device, dtype=query.dtype)
        for l_idx, (H_l, W_l) in enumerate(value_spatial_shapes):
            denom = torch.tensor([W_l, H_l], dtype=torch.float32, device=query.device)
            # ref_l: [B, Q, 1, 1, 2], offsets_l: [B, Q, H, P, 2]
            ref_l = ref[:, :, l_idx, :].unsqueeze(2).unsqueeze(2)
            offsets_l = offsets[:, :, :, l_idx, :, :]
            sampling_locs[:, :, l_idx, :, :, :] = ref_l + offsets_l / denom

        # Sample features from each level
        out = torch.zeros(B, Q, H, self.head_dim, device=query.device, dtype=query.dtype)
        for l_idx, (H_l, W_l) in enumerate(value_spatial_shapes):
            start = value_level_start_index[l_idx]
            end = value_level_start_index[l_idx + 1] if l_idx + 1 < L else Sv
            val_l = value[:, start:end, :, :]  # [B, H_l*W_l, H, Dh]
            val_l = val_l.permute(0, 2, 3, 1).reshape(B * H, self.head_dim, H_l, W_l)

            loc_l = sampling_locs[:, :, l_idx, :, :, :]  # [B, Q, H, P, 2]
            loc_l = loc_l.permute(0, 2, 1, 3, 4).reshape(B * H, Q, P, 2)
            # grid_sample expects xy in [-1, 1]
            loc_l = loc_l * 2 - 1

            sampled = F.grid_sample(
                val_l, loc_l, mode='bilinear', padding_mode='zeros', align_corners=False
            )  # [B*H, Dh, Q, P]
            sampled = sampled.view(B, H, self.head_dim, Q, P).permute(0, 3, 1, 4, 2)
            # [B, Q, H, P, Dh]
            w_l = attn_w[:, :, :, l_idx, :].unsqueeze(-1)  # [B, Q, H, P, 1]
            out = out + (sampled * w_l).sum(3)  # [B, Q, H, Dh]

        out = out.reshape(B, Q, H * self.head_dim)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dim_feedforward: int = 1024,
                 dropout: float = 0.0, num_levels: int = 3, num_points: int = 4):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention (deformable)
        self.cross_attn = MSDeformableAttention(hidden_dim, num_heads, num_levels, num_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes,
                memory_level_start_index, self_attn_mask=None, query_pos=None):
        # Self-attention
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        # Cross-attention
        tgt2 = self.cross_attn(
            tgt if query_pos is None else tgt + query_pos,
            reference_points, memory, memory_spatial_shapes,
            memory_level_start_index,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # FFN
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout4(tgt2))
        return tgt


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, decoder_layer, num_layers: int, eval_idx: int = -1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.eval_idx = eval_idx % num_layers if eval_idx < 0 else eval_idx

    def forward(self, tgt, reference_points, memory, memory_spatial_shapes,
                memory_level_start_index, bbox_head, cls_head, query_pos=None,
                self_attn_mask=None):
        output = tgt
        ref = reference_points  # [B, Q, 4] cxcywh

        intermediate = []
        intermediate_ref = []

        for i, layer in enumerate(self.layers):
            ref_xy = ref[..., :2]  # [B, Q, 2]
            # Create per-level reference points
            ref_per_level = ref_xy.unsqueeze(2).expand(
                -1, -1, len(memory_spatial_shapes), -1)  # [B, Q, L, 2]

            output = layer(
                output, ref_per_level, memory,
                memory_spatial_shapes, memory_level_start_index,
                self_attn_mask=self_attn_mask, query_pos=query_pos,
            )

            # Predict box delta
            delta = bbox_head[i](output)  # [B, Q, 4]
            new_ref = torch.sigmoid(inverse_sigmoid(ref) + delta)
            ref = new_ref.detach()

            if self.training or i == self.eval_idx:
                intermediate.append(output)
                intermediate_ref.append(new_ref)

        return intermediate, intermediate_ref


# ---------------------------------------------------------------------------
# RTDETRTransformer (main)
# ---------------------------------------------------------------------------

class RTDETRTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        position_embed_type: str = 'sine',
        feat_channels=None,
        feat_strides=None,
        num_levels: int = 3,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_idx: int = -1,
        eval_spatial_size=None,
    ):
        super().__init__()
        if feat_channels is None:
            feat_channels = [512, 1024, 2048]
        if feat_strides is None:
            feat_strides = [8, 16, 32]

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_levels = num_levels
        self.num_decoder_layers = num_decoder_layers
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.eval_spatial_size = eval_spatial_size

        # Input projections for each feature level
        self.input_proj = nn.ModuleList()
        for ch in feat_channels:
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            ))

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        # Decoder
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, num_heads, dim_feedforward, dropout, num_levels)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # Per-layer box/class heads
        self.bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3)
                                         for _ in range(num_decoder_layers)])
        self.cls_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes)
                                        for _ in range(num_decoder_layers)])

        # For DN: a simple label embedding
        self.dn_label_embed = nn.Embedding(num_classes, hidden_dim)

        # Encoder-level position embedding
        self.level_embed = nn.Embedding(num_levels, hidden_dim)

        # Two-stage reference point initialisation from encoder output
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_cls = bias_init_with_prob(prior_prob)
        for head in self.cls_head:
            nn.init.constant_(head.bias, bias_cls)
        nn.init.constant_(self.enc_score_head.bias, bias_cls)
        for head in self.bbox_head:
            nn.init.constant_(head.layers[-1].bias, 0.0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0.0)

    def _build_memory(self, feats):
        """Flatten multi-scale feature maps into memory."""
        projs = []
        spatial_shapes = []
        for i, (feat, proj) in enumerate(zip(feats, self.input_proj)):
            p = proj(feat)  # [B, C, H, W]
            B, C, H, W = p.shape
            spatial_shapes.append((H, W))
            # Add level embedding
            level_emb = self.level_embed.weight[i].view(1, C, 1, 1)
            p = p + level_emb
            p = p.flatten(2).transpose(1, 2)  # [B, H*W, C]
            projs.append(p)

        memory = torch.cat(projs, dim=1)  # [B, sum(H*W), C]
        level_start_index = [0]
        for (H, W) in spatial_shapes[:-1]:
            level_start_index.append(level_start_index[-1] + H * W)

        return memory, spatial_shapes, level_start_index

    def _get_reference_points(self, spatial_shapes, device):
        """Generate grid reference points for the encoder outputs."""
        refs = []
        for H, W in spatial_shapes:
            ys = torch.linspace(0.5 / H, 1 - 0.5 / H, H, device=device)
            xs = torch.linspace(0.5 / W, 1 - 0.5 / W, W, device=device)
            y, x = torch.meshgrid(ys, xs, indexing='ij')
            refs.append(torch.stack([x, y], dim=-1).view(-1, 2))  # [H*W, 2]
        return torch.cat(refs, dim=0)  # [sum(H*W), 2]

    def _two_stage_init(self, memory, spatial_shapes, level_start_index):
        """Two-stage: select top-K positions as initial reference points."""
        output = self.enc_output(memory)  # [B, S, C]
        enc_logits = self.enc_score_head(output)  # [B, S, num_classes]
        enc_boxes = self.enc_bbox_head(output).sigmoid()  # [B, S, 4]

        # Select top-num_queries positions
        scores = enc_logits.max(-1).values  # [B, S]
        _, topk_idx = scores.topk(self.num_queries, dim=1)
        topk_idx = topk_idx.unsqueeze(-1)

        ref_pts = enc_boxes.gather(1, topk_idx.expand(-1, -1, 4))  # [B, Q, 4]
        return ref_pts, output

    def forward(self, feats, targets=None):
        memory, spatial_shapes, level_start_index = self._build_memory(feats)
        B = memory.shape[0]
        device = memory.device

        # Two-stage initial reference points
        ref_pts, enc_memory = self._two_stage_init(memory, spatial_shapes, level_start_index)

        # Learnable queries
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]
        query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # DN group
        dn_meta, dn_cls, dn_bbox, attn_mask = None, None, None, None
        if self.training and targets is not None and self.num_denoising > 0:
            dn_meta, dn_cls, dn_bbox, attn_mask = get_contrastive_denoising_training_group(
                targets, self.num_classes, self.num_queries,
                self.dn_label_embed, self.num_denoising,
                self.label_noise_ratio, self.box_noise_scale,
            )

        if dn_meta is not None:
            dn_num = dn_meta['dn_num']
            # Concat DN queries with regular queries
            tgt = torch.cat([dn_cls, tgt], dim=1)           # [B, dn+Q, C]
            query_pos = torch.cat([
                torch.zeros_like(dn_cls), query_pos], dim=1)
            dn_ref = dn_bbox  # [B, dn, 4]
            ref_pts = torch.cat([dn_ref, ref_pts], dim=1)   # [B, dn+Q, 4]

        # Run decoder
        intermediates, intermediate_refs = self.decoder(
            tgt, ref_pts, memory, spatial_shapes, level_start_index,
            self.bbox_head, self.cls_head,
            query_pos=query_pos, self_attn_mask=attn_mask,
        )

        # Build outputs
        all_logits = [self.cls_head[i](feat) for i, feat in enumerate(intermediates)]
        all_boxes = [intermediate_refs[i] for i in range(len(intermediate_refs))]

        if dn_meta is not None:
            dn_num = dn_meta['dn_num']
            # Split DN outputs
            dn_logits = [l[:, :dn_num] for l in all_logits]
            dn_boxes = [b[:, :dn_num] for b in all_boxes]
            all_logits = [l[:, dn_num:] for l in all_logits]
            all_boxes = [b[:, dn_num:] for b in all_boxes]
            dn_out = {
                'pred_logits': dn_logits[-1],
                'pred_boxes': dn_boxes[-1],
            }
        else:
            dn_out = None

        out = {
            'pred_logits': all_logits[-1],
            'pred_boxes': all_boxes[-1],
            'aux_outputs': [
                {'pred_logits': l, 'pred_boxes': b}
                for l, b in zip(all_logits[:-1], all_boxes[:-1])
            ],
            'dn_meta': dn_meta,
            'dn_out': dn_out,
        }
        return out
