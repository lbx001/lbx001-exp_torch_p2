"""COCO-format detection dataset."""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO


class CocoDetection(Dataset):
    """Dataset that loads images and annotations from a COCO-format JSON file."""

    def __init__(self, img_folder, ann_file, transforms=None, return_masks=False):
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        self.return_masks = return_masks
        self.transforms = transforms

        # Keep only images that have at least one annotation
        all_img_ids = list(self.coco.imgs.keys())
        self.ids = [
            img_id for img_id in all_img_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_folder / img_info['file_name']
        image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        w, h = image.size

        boxes = []
        labels = []
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            x, y, bw, bh = ann['bbox']
            # clamp to image bounds
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(w), x + bw)
            y2 = min(float(h), y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            # Convert xywh → cxcywh normalized
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h
            boxes.append([cx, cy, nw, nh])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.as_tensor([h, w], dtype=torch.int64),
            'size': torch.as_tensor([h, w], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
