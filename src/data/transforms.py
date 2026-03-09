"""Data transforms for detection (image, target) pairs.

All transforms expect target['boxes'] in [x1,y1,x2,y2] pixel format during
the transform pipeline. CocoDetection stores boxes as cxcywh-normalized, so
callers must convert before applying these transforms (or use the pipeline
returned by build_transforms which handles the conversion).
"""
import random
import math
from typing import List, Tuple, Optional

import torch
import torchvision.transforms.functional as F
from PIL import Image

__all__ = [
    'Compose', 'RandomHorizontalFlip', 'RandomPhotometricDistort',
    'RandomZoomOut', 'RandomIoUCrop', 'Resize', 'ToTensor', 'Normalize',
    'ConvertBoxFormat',
]


# ---------------------------------------------------------------------------
# Helper: box format conversions (pixel coords)
# ---------------------------------------------------------------------------

def _cxcywh_norm_to_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return torch.stack([cx, cy, bw, bh], dim=-1)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ConvertBoxFormat:
    """Convert boxes from cxcywh-normalized → xyxy pixel coords (or back)."""

    def __init__(self, direction: str = 'norm_cxcywh_to_xyxy'):
        assert direction in ('norm_cxcywh_to_xyxy', 'xyxy_to_norm_cxcywh')
        self.direction = direction

    def __call__(self, image, target):
        w, h = image.size if isinstance(image, Image.Image) else (
            image.shape[-1], image.shape[-2])
        boxes = target.get('boxes')
        if boxes is None or boxes.numel() == 0:
            return image, target
        if self.direction == 'norm_cxcywh_to_xyxy':
            target['boxes'] = _cxcywh_norm_to_xyxy(boxes, w, h)
        else:
            target['boxes'] = _xyxy_to_cxcywh_norm(boxes, w, h)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            w = image.width if isinstance(image, Image.Image) else image.shape[-1]
            boxes = target.get('boxes')
            if boxes is not None and boxes.numel() > 0:
                x1, y1, x2, y2 = boxes.unbind(-1)
                boxes = torch.stack([w - x2, y1, w - x1, y2], dim=-1)
                target['boxes'] = boxes
        return image, target


class RandomPhotometricDistort:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            brightness = random.uniform(0.875, 1.125)
            contrast = random.uniform(0.5, 1.5)
            saturation = random.uniform(0.5, 1.5)
            hue = random.uniform(-0.05, 0.05)
            image = F.adjust_brightness(image, brightness)
            image = F.adjust_contrast(image, contrast)
            image = F.adjust_saturation(image, saturation)
            image = F.adjust_hue(image, hue)
        return image, target


class RandomZoomOut:
    """Pad image (zoom out) by a random factor."""

    def __init__(self, fill=(123, 117, 104), side_range=(1.0, 4.0), p: float = 0.5):
        self.fill = fill
        self.side_range = side_range
        self.p = p

    def __call__(self, image, target):
        if random.random() > self.p:
            return image, target

        orig_w, orig_h = image.size
        ratio = random.uniform(*self.side_range)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        left = random.randint(0, new_w - orig_w)
        top = random.randint(0, new_h - orig_h)

        canvas = Image.new('RGB', (new_w, new_h), self.fill)
        canvas.paste(image, (left, top))
        image = canvas

        boxes = target.get('boxes')
        if boxes is not None and boxes.numel() > 0:
            boxes = boxes + torch.tensor([left, top, left, top], dtype=boxes.dtype)
            target['boxes'] = boxes

        target['size'] = torch.as_tensor([new_h, new_w], dtype=torch.int64)
        return image, target


class RandomIoUCrop:
    """Random crop whose IoU with all boxes is above a threshold."""

    def __init__(self, min_scale: float = 0.3, max_scale: float = 1.0,
                 min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2.0,
                 sampler_options=None, num_trials: int = 40):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.sampler_options = sampler_options or [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, None]
        self.num_trials = num_trials

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        boxes = target.get('boxes')
        if boxes is None or boxes.numel() == 0:
            return image, target

        option = random.choice(self.sampler_options)
        if option is None:
            return image, target

        for _ in range(self.num_trials):
            scale = random.uniform(self.min_scale, self.max_scale)
            aspect = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
            crop_w = int(orig_w * scale)
            crop_h = int(crop_w / aspect)
            crop_h = min(crop_h, orig_h)
            crop_w = min(crop_w, orig_w)
            if crop_w <= 0 or crop_h <= 0:
                continue

            left = random.randint(0, orig_w - crop_w)
            top = random.randint(0, orig_h - crop_h)
            right = left + crop_w
            bottom = top + crop_h

            # Compute IoU with crop region
            crop_box = torch.tensor([[left, top, right, bottom]], dtype=torch.float32)
            inter_x1 = torch.max(boxes[:, 0], crop_box[:, 0])
            inter_y1 = torch.max(boxes[:, 1], crop_box[:, 1])
            inter_x2 = torch.min(boxes[:, 2], crop_box[:, 2])
            inter_y2 = torch.min(boxes[:, 3], crop_box[:, 3])
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h
            box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iou = inter_area / box_area.clamp(min=1e-6)

            if iou.min().item() < option:
                continue

            # Crop
            image = image.crop((left, top, right, bottom))
            new_boxes = boxes.clone()
            new_boxes[:, 0] = (boxes[:, 0] - left).clamp(min=0)
            new_boxes[:, 1] = (boxes[:, 1] - top).clamp(min=0)
            new_boxes[:, 2] = (boxes[:, 2] - left).clamp(max=crop_w)
            new_boxes[:, 3] = (boxes[:, 3] - top).clamp(max=crop_h)

            # Filter degenerate boxes
            keep = ((new_boxes[:, 2] - new_boxes[:, 0]) > 1) & \
                   ((new_boxes[:, 3] - new_boxes[:, 1]) > 1)
            target['boxes'] = new_boxes[keep]
            target['labels'] = target['labels'][keep]
            target['size'] = torch.as_tensor([crop_h, crop_w], dtype=torch.int64)
            return image, target

        return image, target


class Resize:
    def __init__(self, size: int):
        self.size = size  # target longest side

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        if isinstance(self.size, (list, tuple)):
            new_h, new_w = self.size
        else:
            if orig_w > orig_h:
                new_w = self.size
                new_h = int(orig_h * self.size / orig_w)
            else:
                new_h = self.size
                new_w = int(orig_w * self.size / orig_h)

        image = F.resize(image, (new_h, new_w))

        boxes = target.get('boxes')
        if boxes is not None and boxes.numel() > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
            target['boxes'] = boxes

        target['size'] = torch.as_tensor([new_h, new_w], dtype=torch.int64)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        # Convert boxes from xyxy pixel → cxcywh normalized
        boxes = target.get('boxes')
        if boxes is not None and boxes.numel() > 0:
            h, w = image.shape[-2], image.shape[-1]
            target['boxes'] = _xyxy_to_cxcywh_norm(boxes, w, h)
        return image, target
