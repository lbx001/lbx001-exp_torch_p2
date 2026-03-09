from __future__ import annotations

import glob
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

from ..utils import dump_json, ensure_dir


@dataclass
class ImageRecord:
    image: Dict[str, Any]
    annotations: List[Dict[str, Any]]
    categories: Dict[int, str]
    source_path: Path


@dataclass
class CropWindow:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top



def _normalize_glob(pattern: str) -> str:
    return f"**{pattern}" if pattern.startswith('/') else pattern



def _collect_annotation_files(input_root: Path, patterns: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(str(input_root / _normalize_glob(pattern)), recursive=True):
            path = Path(match)
            if path.is_file():
                files.add(path)
    return sorted(files)



def _category_allowed(category_id: int, category_name: str, dataset_cfg: Dict[str, Any]) -> bool:
    whitelist = dataset_cfg.get('category_id_whitelist', []) or []
    blacklist = dataset_cfg.get('category_id_blacklist', []) or []
    name_blacklist = dataset_cfg.get('category_name_blacklist', []) or []
    if whitelist:
        return category_id in whitelist
    if category_id in blacklist:
        return False
    if category_name in name_blacklist:
        return False
    return True



def _choose_crop_window(width: int, height: int, crop_cfg: Dict[str, Any], rng: random.Random) -> CropWindow:
    if not crop_cfg.get('enabled', False):
        return CropWindow(0, 0, width, height)

    size = min(int(crop_cfg['size']), width, height)
    mode = crop_cfg.get('mode', 'center')
    if mode == 'left':
        left = 0
    elif mode == 'right':
        left = width - size
    elif mode == 'random':
        left = rng.randint(0, max(width - size, 0))
    else:
        left = max((width - size) // 2, 0)

    if mode == 'top':
        top = 0
    elif mode == 'bottom':
        top = height - size
    else:
        top = max((height - size) // 2, 0)
        if mode == 'random':
            top = rng.randint(0, max(height - size, 0))

    return CropWindow(left=left, top=top, right=left + size, bottom=top + size)



def _clip_bbox_to_crop(bbox: list[float], crop: CropWindow) -> list[float] | None:
    x, y, w, h = bbox
    x1 = max(x, crop.left)
    y1 = max(y, crop.top)
    x2 = min(x + w, crop.right)
    y2 = min(y + h, crop.bottom)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 - crop.left, y1 - crop.top, x2 - x1, y2 - y1]



def _load_records(annotation_files: list[Path]) -> list[ImageRecord]:
    all_records: list[ImageRecord] = []
    for ann_path in annotation_files:
        payload = json.loads(ann_path.read_text(encoding='utf-8'))
        categories = {int(cat['id']): cat['name'] for cat in payload.get('categories', [])}
        anns_by_image: dict[int, list[Dict[str, Any]]] = defaultdict(list)
        for ann in payload.get('annotations', []):
            anns_by_image[int(ann['image_id'])].append(dict(ann))
        image_dir = ann_path.parent
        for image in payload.get('images', []):
            source_path = image_dir / image['file_name']
            if not source_path.exists():
                fallback = ann_path.parent / 'images' / image['file_name']
                if fallback.exists():
                    source_path = fallback
            if not source_path.exists():
                continue
            all_records.append(
                ImageRecord(
                    image=dict(image),
                    annotations=anns_by_image.get(int(image['id']), []),
                    categories=categories,
                    source_path=source_path,
                )
            )
    return all_records



def _deduplicate(records: list[ImageRecord], delim: str) -> list[ImageRecord]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        base_name = Path(record.image['file_name']).name
        key = base_name.split(delim)[0] if delim and delim in base_name else Path(base_name).stem
        grouped[key].append(record)
    selected: list[ImageRecord] = []
    for _, group in sorted(grouped.items()):
        best = sorted(group, key=lambda item: (-len(item.annotations), item.source_path.name))[0]
        selected.append(best)
    return selected



def _sample_records(records: list[ImageRecord], ratio: float, seed: int) -> list[ImageRecord]:
    if ratio >= 1.0:
        return list(records)
    if not records:
        return []
    rng = random.Random(seed)
    count = max(2, int(math.ceil(len(records) * ratio))) if len(records) > 1 else 1
    count = min(count, len(records))
    return sorted(rng.sample(records, count), key=lambda item: item.source_path.name)



def _split_records(records: list[ImageRecord], train_ratio: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    train_count = min(max(1, int(len(shuffled) * train_ratio)), len(shuffled) - 1)
    return shuffled[:train_count], shuffled[train_count:]



def processed_dataset_paths(dataset_cfg: Dict[str, Any]) -> Dict[str, Path]:
    output_root = Path(dataset_cfg['output_root'])
    return {
        'root': output_root,
        'train_images': output_root / 'train' / 'images',
        'val_images': output_root / 'val' / 'images',
        'train_annotations': output_root / 'train' / 'annotations.json',
        'val_annotations': output_root / 'val' / 'annotations.json',
        'metadata': output_root / 'metadata.json',
    }



def prepare_dataset(dataset_cfg: Dict[str, Any], logger=None) -> Dict[str, Path]:
    paths = processed_dataset_paths(dataset_cfg)
    if paths['metadata'].exists() and not dataset_cfg.get('force_rebuild', False):
        return paths

    if paths['root'].exists() and dataset_cfg.get('force_rebuild', False):
        shutil.rmtree(paths['root'])

    for key in ('train_images', 'val_images'):
        ensure_dir(paths[key])

    input_root = Path(dataset_cfg['input_root'])
    annotation_files = _collect_annotation_files(input_root, dataset_cfg.get('annotation_globs', []))
    if not annotation_files:
        raise FileNotFoundError(f'未找到 COCO 标注文件: {input_root}')

    records = _load_records(annotation_files)
    if not records:
        raise RuntimeError('未从标注文件中解析到有效图像记录。')

    deduped = _deduplicate(records, dataset_cfg.get('dedup_delim', '.rf.'))
    sampled = _sample_records(deduped, float(dataset_cfg.get('sample_ratio', 1.0)), int(dataset_cfg.get('split_seed', 0)))
    train_records, val_records = _split_records(sampled, float(dataset_cfg.get('train_ratio', 0.8)), int(dataset_cfg.get('split_seed', 0)))

    crop_cfg = dataset_cfg.get('crop', {})
    crop_rng = random.Random(int(crop_cfg.get('seed', dataset_cfg.get('split_seed', 0))))

    category_names: dict[int, str] = {}
    kept_category_ids: set[int] = set()
    for record in sampled:
        category_names.update(record.categories)
        for ann in record.annotations:
            cat_id = int(ann['category_id'])
            cat_name = record.categories.get(cat_id, str(cat_id))
            if _category_allowed(cat_id, cat_name, dataset_cfg):
                kept_category_ids.add(cat_id)
    allowed_categories = [cat_id for cat_id in sorted(kept_category_ids) if cat_id in category_names]
    category_id_map = {old_id: new_id for new_id, old_id in enumerate(allowed_categories, start=1)}
    categories = [{'id': new_id, 'name': category_names[old_id]} for old_id, new_id in category_id_map.items()]

    def convert_split(split_name: str, split_records: list[ImageRecord], image_dir: Path, ann_path: Path) -> None:
        images_payload: list[Dict[str, Any]] = []
        anns_payload: list[Dict[str, Any]] = []
        ann_id = 1
        max_ann = int(dataset_cfg.get('max_ann_per_image', 200))
        min_box_area = float(dataset_cfg.get('min_box_area', 0))
        for image_id, record in enumerate(split_records, start=1):
            with Image.open(record.source_path) as image:
                image = image.convert('RGB')
                crop = _choose_crop_window(image.width, image.height, crop_cfg, crop_rng)
                cropped = image.crop((crop.left, crop.top, crop.right, crop.bottom))
                output_name = f"{Path(record.source_path).stem}_{split_name}.jpg"
                cropped.save(image_dir / output_name, quality=95)

            filtered_annotations = []
            for ann in record.annotations:
                old_cat = int(ann['category_id'])
                if old_cat not in category_id_map:
                    continue
                clipped = _clip_bbox_to_crop(list(map(float, ann['bbox'])), crop)
                if clipped is None:
                    continue
                area = clipped[2] * clipped[3]
                if area < min_box_area:
                    continue
                new_ann = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': category_id_map[old_cat],
                    'bbox': [round(value, 4) for value in clipped],
                    'area': round(area, 4),
                    'iscrowd': int(ann.get('iscrowd', 0)),
                }
                ann_id += 1
                filtered_annotations.append(new_ann)

            filtered_annotations.sort(key=lambda item: item['area'], reverse=True)
            filtered_annotations = filtered_annotations[:max_ann]
            anns_payload.extend(filtered_annotations)
            images_payload.append(
                {
                    'id': image_id,
                    'file_name': output_name,
                    'width': crop.width,
                    'height': crop.height,
                    'source_file': str(record.source_path),
                    'annotation_count': len(filtered_annotations),
                }
            )

        payload = {'images': images_payload, 'annotations': anns_payload, 'categories': categories}
        ann_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    convert_split('train', train_records, paths['train_images'], paths['train_annotations'])
    convert_split('val', val_records, paths['val_images'], paths['val_annotations'])

    metadata = {
        'input_root': str(input_root),
        'source_annotations': [str(path) for path in annotation_files],
        'total_images_before_dedup': len(records),
        'total_images_after_dedup': len(deduped),
        'total_images_after_sample': len(sampled),
        'train_images': len(train_records),
        'val_images': len(val_records),
        'categories': categories,
    }
    dump_json(metadata, paths['metadata'])
    if logger:
        logger.info('数据集处理完成: %s', metadata)
    return paths
