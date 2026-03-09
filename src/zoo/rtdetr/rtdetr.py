"""RT-DETR top-level model."""
import torch
import torch.nn as nn

__all__ = ['RTDETR']


class RTDETR(nn.Module):
    """
    RT-DETR object detector.

    Args:
        backbone: feature extractor returning [C2, C3, C4, C5] (or subset)
        encoder:  HybridEncoder neck, expects [C3, C4, C5]
        decoder:  RTDETRTransformer, expects [P3, P4, P5]
        soep:     optional SOEPModule for small-object enhancement
        multi_scale: optional list of image sizes for multi-scale training
    """

    def __init__(self, backbone, encoder, decoder, soep=None, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.soep = soep
        self.multi_scale = multi_scale

    def forward(self, x: torch.Tensor, targets=None):
        # Optional multi-scale resize during training
        if self.training and self.multi_scale is not None:
            import random
            sz = random.choice(self.multi_scale)
            x = torch.nn.functional.interpolate(x, size=(sz, sz))

        # Backbone
        backbone_feats = self.backbone(x)  # list of tensors

        # SOEP: enhance P3 using P2 detail
        if self.soep is not None and len(backbone_feats) >= 4:
            # backbone returns [C2, C3, C4, C5] when return_idx=[0,1,2,3]
            p2, p3, p4, p5 = backbone_feats
            p3_enhanced = self.soep(p2, p3)
            encoder_feats = [p3_enhanced, p4, p5]
        elif self.soep is not None and len(backbone_feats) == 3:
            # backbone returns [C3, C4, C5]; assume C3 is also P2 (first stage output)
            # In this case SOEP is skipped gracefully
            encoder_feats = backbone_feats
        else:
            # Default: use last 3 features as [P3, P4, P5]
            encoder_feats = backbone_feats[-3:] if len(backbone_feats) >= 3 else backbone_feats

        # Encoder (neck)
        neck_feats = self.encoder(encoder_feats)  # [P3', P4', P5']

        # Decoder
        out = self.decoder(neck_feats, targets)
        return out
