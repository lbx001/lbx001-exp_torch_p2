"""DataLoader builders for detection."""
import torch
from torch.utils.data import DataLoader

from .coco import CocoDetection
from .transforms import (
    Compose, ConvertBoxFormat, RandomHorizontalFlip, RandomPhotometricDistort,
    RandomZoomOut, RandomIoUCrop, Resize, ToTensor, Normalize,
)

__all__ = ['build_dataset', 'build_dataloader', 'collate_fn']


def build_transforms(cfg, split: str):
    """Build the transform pipeline for a given split."""
    img_size = cfg.get('img_size', 640)
    if isinstance(img_size, int):
        img_size = [img_size, img_size]

    if split == 'train':
        transforms = Compose([
            # Boxes are stored as cxcywh-norm in the dataset; convert for spatial ops
            ConvertBoxFormat('norm_cxcywh_to_xyxy'),
            RandomPhotometricDistort(p=0.5),
            RandomZoomOut(p=0.5),
            RandomIoUCrop(),
            RandomHorizontalFlip(p=0.5),
            Resize(img_size),
            ToTensor(),
            Normalize(),
            # Normalize() converts boxes back to cxcywh-norm
        ])
    else:
        transforms = Compose([
            ConvertBoxFormat('norm_cxcywh_to_xyxy'),
            Resize(img_size),
            ToTensor(),
            Normalize(),
        ])
    return transforms


def build_dataset(cfg, split: str = 'train') -> CocoDetection:
    """Build a CocoDetection dataset for the given split."""
    if split == 'train':
        img_folder = cfg.get('train_img_dir', 'data/train/images')
        ann_file = cfg.get('train_ann_file', 'data/train/annotations.json')
    else:
        img_folder = cfg.get('val_img_dir', 'data/val/images')
        ann_file = cfg.get('val_ann_file', 'data/val/annotations.json')

    transforms = build_transforms(cfg, split)
    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    return dataset


def collate_fn(batch):
    """Collate variable-size images into a padded batch tensor."""
    images, targets = list(zip(*batch))

    # Determine max H, W in the batch
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded = torch.zeros(len(images), 3, max_h, max_w)
    for i, img in enumerate(images):
        h, w = img.shape[-2], img.shape[-1]
        padded[i, :, :h, :w] = img

    return padded, list(targets)


def build_dataloader(dataset, batch_size: int, num_workers: int,
                     shuffle: bool = False, collate_fn=collate_fn) -> DataLoader:
    """Wrap a dataset in a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(shuffle),  # drop_last only for training
    )
