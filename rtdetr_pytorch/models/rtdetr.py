from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models._utils import IntermediateLayerGetter

from .soep import SOEPBlock


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_embed = torch.linspace(0, 1, steps=h, device=x.device).view(1, h, 1).expand(b, h, w)
        x_embed = torch.linspace(0, 1, steps=w, device=x.device).view(1, 1, w).expand(b, h, w)
        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)


class MultiScaleBackbone(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        if backbone_name == 'resnet18':
            backbone = resnet18(weights=None)
            channels = [128, 256, 512]
        elif backbone_name == 'resnet34':
            backbone = resnet34(weights=None)
            channels = [128, 256, 512]
        else:
            backbone = resnet50(weights=None)
            channels = [512, 1024, 2048]
        self.body = IntermediateLayerGetter(backbone, return_layers={'layer2': '0', 'layer3': '1', 'layer4': '2'})
        self.channels = channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.body(x)
        return [outputs[str(idx)] for idx in range(3)]


class RTDETR(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any], num_classes: int):
        super().__init__()
        hidden_dim = int(model_cfg.get('hidden_dim', 256))
        pool_sizes = model_cfg.get('token_pool_sizes', [16, 8, 4])
        self.num_queries = int(model_cfg.get('num_queries', 100))
        self.num_classes = num_classes
        self.backbone = MultiScaleBackbone(model_cfg.get('backbone', 'resnet18'))
        self.input_proj = nn.ModuleList(
            [nn.Conv2d(ch, hidden_dim, kernel_size=1) for ch in self.backbone.channels]
        )
        self.pool_layers = nn.ModuleList([nn.AdaptiveAvgPool2d((size, size)) for size in pool_sizes])
        self.use_soep = bool(model_cfg.get('use_soep', True))
        self.soep_blocks = nn.ModuleList([SOEPBlock(hidden_dim) for _ in pool_sizes]) if self.use_soep else None
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(model_cfg.get('nheads', 8)),
            dim_feedforward=int(model_cfg.get('dim_feedforward', hidden_dim * 4)),
            dropout=float(model_cfg.get('dropout', 0.1)),
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=int(model_cfg.get('nheads', 8)),
            dim_feedforward=int(model_cfg.get('dim_feedforward', hidden_dim * 4)),
            dropout=float(model_cfg.get('dropout', 0.1)),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(model_cfg.get('num_encoder_layers', 2)))
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(model_cfg.get('num_decoder_layers', 3)))
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def _encode_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        tokens = []
        for idx, feature in enumerate(features):
            feature = self.input_proj[idx](feature)
            feature = self.pool_layers[idx](feature)
            if self.soep_blocks is not None:
                feature = self.soep_blocks[idx](feature)
            pos = self.position_embedding(feature)
            feature = feature + pos
            b, c, h, w = feature.shape
            tokens.append(feature.flatten(2).transpose(1, 2))
        return torch.cat(tokens, dim=1)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.encoder(self._encode_features(images))
        query = self.query_embed.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        hs = self.decoder(query, memory)
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()
        return {'pred_logits': logits, 'pred_boxes': boxes}



def build_model(config: Dict[str, Any], num_classes: int) -> RTDETR:
    return RTDETR(config.get('model', {}), num_classes=num_classes)
