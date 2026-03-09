"""COCO utility functions."""
import json
from pathlib import Path

from pycocotools.coco import COCO

__all__ = ['get_coco_api_from_dataset', 'convert_to_coco_api']


def get_coco_api_from_dataset(dataset) -> COCO:
    """Return the COCO API object from a CocoDetection dataset."""
    if hasattr(dataset, 'coco'):
        return dataset.coco
    return convert_to_coco_api(dataset)


def convert_to_coco_api(ds) -> COCO:
    """Build a COCO API object from an arbitrary dataset."""
    coco_ds = COCO()

    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()

    for idx in range(len(ds)):
        img, target = ds[idx]
        image_id = target['image_id'].item() if hasattr(target['image_id'], 'item') else int(target['image_id'])
        orig_size = target['orig_size']
        h, w = int(orig_size[0]), int(orig_size[1])

        dataset['images'].append({'id': image_id, 'height': h, 'width': w})

        boxes = target['boxes']  # cxcywh normalized
        labels = target['labels']

        for box, label in zip(boxes.tolist(), labels.tolist()):
            cx, cy, nw, nh = box
            x1 = (cx - nw / 2) * w
            y1 = (cy - nh / 2) * h
            bw = nw * w
            bh = nh * h
            area = bw * bh
            dataset['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': int(label),
                'bbox': [x1, y1, bw, bh],
                'area': area,
                'iscrowd': 0,
            })
            categories.add(int(label))
            ann_id += 1

    dataset['categories'] = [{'id': c} for c in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
