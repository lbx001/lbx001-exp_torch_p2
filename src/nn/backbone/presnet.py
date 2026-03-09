"""PResNet (ResNet-vd variant) backbone."""
import torch
import torch.nn as nn
from src.core.config import register

__all__ = ['PResNet']

# Bottleneck channel configs per depth
_STAGE_CHANNELS = {
    50: [256, 512, 1024, 2048],
    101: [256, 512, 1024, 2048],
}
_STAGE_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
}


class ConvNormLayer(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=None, g=1, bias=False, freeze_norm=False):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=bias)
        self.norm = nn.BatchNorm2d(out_c)
        if freeze_norm:
            for param in self.norm.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.norm(self.conv(x))


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_c, mid_c, stride=1, shortcut=True, variant='d', freeze_norm=False):
        super().__init__()
        out_c = mid_c * self.expansion
        self.conv1 = ConvNormLayer(in_c, mid_c, 1, freeze_norm=freeze_norm)
        self.conv2 = ConvNormLayer(mid_c, mid_c, 3, s=stride, freeze_norm=freeze_norm)
        self.conv3 = ConvNormLayer(mid_c, out_c, 1, freeze_norm=freeze_norm)
        self.act = nn.ReLU(inplace=True)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                # avg pool + 1x1 conv
                self.short = nn.Sequential(
                    nn.AvgPool2d(2, stride=2, ceil_mode=True),
                    ConvNormLayer(in_c, out_c, 1, freeze_norm=freeze_norm),
                )
            else:
                self.short = ConvNormLayer(in_c, out_c, 1, s=stride, freeze_norm=freeze_norm)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.conv3(out)
        if self.shortcut:
            skip = x
        else:
            skip = self.short(x)
        return self.act(out + skip)


class Blocks(nn.Module):
    def __init__(self, in_c, mid_c, count, stride=1, variant='d', freeze_norm=False):
        super().__init__()
        layers = []
        for i in range(count):
            layers.append(BottleNeck(
                in_c if i == 0 else mid_c * BottleNeck.expansion,
                mid_c,
                stride=stride if i == 0 else 1,
                shortcut=(i != 0),
                variant=variant,
                freeze_norm=freeze_norm,
            ))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


@register
class PResNet(nn.Module):
    """ResNet-vd backbone (PResNet)."""

    def __init__(self, depth=50, variant='d', freeze_at=0, return_idx=None,
                 num_stages=4, freeze_norm=False, pretrained=False):
        super().__init__()
        if return_idx is None:
            return_idx = [1, 2, 3]
        self.return_idx = return_idx
        self.freeze_at = freeze_at

        stage_channels = _STAGE_CHANNELS[depth]
        stage_blocks = _STAGE_BLOCKS[depth]
        mid_channels = [c // 4 for c in stage_channels]  # bottleneck mid

        # Stem: 3 × Conv3x3 (stride 2, 1, 1) instead of one 7x7
        self.stem = nn.Sequential(
            ConvNormLayer(3, 32, 3, s=2, freeze_norm=freeze_norm),
            nn.ReLU(inplace=True),
            ConvNormLayer(32, 32, 3, s=1, freeze_norm=freeze_norm),
            nn.ReLU(inplace=True),
            ConvNormLayer(32, 64, 3, s=1, freeze_norm=freeze_norm),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 4 stages
        self.stages = nn.ModuleList()
        in_c = 64
        for i in range(num_stages):
            stride = 1 if i == 0 else 2
            stage = Blocks(in_c, mid_channels[i], stage_blocks[i],
                           stride=stride, variant=variant, freeze_norm=freeze_norm)
            self.stages.append(stage)
            in_c = stage_channels[i]

        self._out_channels = [stage_channels[i] for i in return_idx]
        self._freeze()

        if pretrained:
            self._load_pretrained(depth)

    def _freeze(self):
        # Freeze stem always if freeze_at >= 1
        if self.freeze_at >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
        # Freeze stages
        for i, stage in enumerate(self.stages):
            if i < self.freeze_at - 1:
                for p in stage.parameters():
                    p.requires_grad = False

    def _load_pretrained(self, depth):
        import torchvision.models as tvm
        if depth == 50:
            ref = tvm.resnet50(pretrained=True)
        elif depth == 101:
            ref = tvm.resnet101(pretrained=True)
        else:
            return
        # Copy layer weights where shapes match
        ref_sd = ref.state_dict()
        own_sd = self.state_dict()
        for k in list(own_sd.keys()):
            # Map stages → layerN
            for si, sname in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
                prefix = f'stages.{si}.blocks.'
                if k.startswith(prefix):
                    ref_k = k.replace(prefix, f'{sname}.')
                    if ref_k in ref_sd and ref_sd[ref_k].shape == own_sd[k].shape:
                        own_sd[k] = ref_sd[ref_k]
        self.load_state_dict(own_sd, strict=False)

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx:
                outs.append(x)
        return outs
