from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, PILToTensor
import torchvision.transforms.functional as F


class CocoDetectionDataset(Dataset):
    def __init__(self, image_dir: str | Path, annotation_file: str | Path, training: bool, augmentation_cfg: Dict[str, Any] | None = None):
        self.image_dir = Path(image_dir)
        payload = json.loads(Path(annotation_file).read_text(encoding='utf-8'))
        self.images = payload.get('images', [])
        self.categories = payload.get('categories', [])
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in payload.get('annotations', []):
            self.annotations_by_image.setdefault(int(ann['image_id']), []).append(ann)
        self.training = training
        self.augmentation_cfg = augmentation_cfg or {}
        self.to_tensor = PILToTensor()
        jitter_cfg = self.augmentation_cfg.get('color_jitter', {})
        self.color_jitter = None
        if training and self.augmentation_cfg.get('enabled', False) and jitter_cfg:
            self.color_jitter = ColorJitter(
                brightness=jitter_cfg.get('brightness', 0.0),
                contrast=jitter_cfg.get('contrast', 0.0),
                saturation=jitter_cfg.get('saturation', 0.0),
                hue=jitter_cfg.get('hue', 0.0),
            )

    def __len__(self) -> int:
        return len(self.images)

    def _load_boxes(self, image_info: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        anns = self.annotations_by_image.get(int(image_info['id']), [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann['category_id']) - 1)
        if boxes:
            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image_info = self.images[index]
        image_path = self.image_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        boxes, labels = self._load_boxes(image_info)
        width, height = image.size

        if self.training and self.augmentation_cfg.get('enabled', False):
            flip_prob = float(self.augmentation_cfg.get('horizontal_flip_prob', 0.0))
            if flip_prob > 0 and random.random() < flip_prob:
                image = F.hflip(image)
                if len(boxes) > 0:
                    x1 = width - boxes[:, 2]
                    x2 = width - boxes[:, 0]
                    boxes[:, 0] = x1
                    boxes[:, 2] = x2
            if self.color_jitter is not None:
                image = self.color_jitter(image)

        image_tensor = self.to_tensor(image).float() / 255.0
        if len(boxes) > 0:
            boxes_xyxy = boxes.clone()
            boxes_cxcywh = boxes.clone()
            boxes_cxcywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0 / width
            boxes_cxcywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0 / height
            boxes_cxcywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
            boxes_cxcywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
        else:
            boxes_xyxy = boxes
            boxes_cxcywh = boxes

        target = {
            'image_id': int(image_info['id']),
            'boxes': boxes_cxcywh,
            'boxes_xyxy': boxes_xyxy,
            'labels': labels,
            'orig_size': torch.tensor([height, width], dtype=torch.int64),
            'size': torch.tensor([height, width], dtype=torch.int64),
        }
        return image_tensor, target



def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)



def build_dataloaders(config: Dict[str, Any], dataset_paths: Dict[str, Path]):
    training_cfg = config.get('training', {})
    eval_cfg = config.get('evaluation', {})
    train_dataset = CocoDetectionDataset(
        image_dir=dataset_paths['train_images'],
        annotation_file=dataset_paths['train_annotations'],
        training=True,
        augmentation_cfg=training_cfg.get('augmentation', {}),
    )
    val_dataset = CocoDetectionDataset(
        image_dir=dataset_paths['val_images'],
        annotation_file=dataset_paths['val_annotations'],
        training=False,
        augmentation_cfg=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get('batch_size', 1)),
        shuffle=True,
        num_workers=int(training_cfg.get('num_workers', 0)),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(eval_cfg.get('batch_size', 1)),
        shuffle=False,
        num_workers=int(eval_cfg.get('num_workers', 0)),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, train_loader, val_loader
