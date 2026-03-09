"""
prepare_data.py — 数据准备脚本
=================================
将多个 COCO 格式标注文件合并、去重、采样、分割、裁剪，
并输出符合训练要求的 COCO 格式数据集。

使用方法::

    python prepare_data.py [--config config.yaml]

所有参数均从 config.yaml 的 ``dataset`` 节读取。
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR 数据准备脚本")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_cfg(config_path: str) -> dict:
    with open(config_path) as f:
        full = yaml.safe_load(f) or {}
    return full.get("dataset", full)


# ---------------------------------------------------------------------------
# Step 1: Merge annotation files
# ---------------------------------------------------------------------------

def find_annotation_files(input_root: str, globs: list) -> list:
    """按 glob 模式在 input_root 下查找标注文件。"""
    found = []
    for pattern in globs:
        # pattern 以 / 开头时去掉，拼接到 input_root
        full_pattern = input_root.rstrip("/") + "/" + pattern.lstrip("/")
        matched = glob.glob(full_pattern, recursive=True)
        found.extend(matched)
    # 去重
    return sorted(set(found))


def merge_coco_jsons(ann_files: list) -> dict:
    """合并多个 COCO JSON 文件，统一 image_id 和 annotation_id。"""
    merged = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    cat_name2id = {}   # 类别名 -> 新 id
    cat_id_map = {}    # (file, old_id) -> new_id

    img_id_offset = 0
    ann_id_offset = 0

    for ann_file in ann_files:
        with open(ann_file) as f:
            data = json.load(f)

        src_dir = str(Path(ann_file).parent)  # 图像所在目录

        # --- categories ---
        for cat in data.get("categories", []):
            name = cat["name"]
            if name not in cat_name2id:
                new_id = len(cat_name2id)
                cat_name2id[name] = new_id
                merged["categories"].append({"id": new_id, "name": name, "supercategory": cat.get("supercategory", "")})
            cat_id_map[(ann_file, cat["id"])] = cat_name2id[name]

        # --- images ---
        old2new_img = {}
        max_img_id = 0
        for img in data.get("images", []):
            old_id = img["id"]
            new_id = old_id + img_id_offset
            old2new_img[old_id] = new_id
            max_img_id = max(max_img_id, old_id)

            new_img = dict(img)
            new_img["id"] = new_id
            # 记录图像实际目录（用于后续复制）
            new_img["_src_dir"] = src_dir
            merged["images"].append(new_img)

        # --- annotations ---
        max_ann_id = 0
        for ann in data.get("annotations", []):
            old_cat_id = ann["category_id"]
            new_cat_id = cat_id_map.get((ann_file, old_cat_id), old_cat_id)
            new_ann = dict(ann)
            new_ann["id"] = ann["id"] + ann_id_offset
            new_ann["image_id"] = old2new_img.get(ann["image_id"], ann["image_id"] + img_id_offset)
            new_ann["category_id"] = new_cat_id
            merged["annotations"].append(new_ann)
            max_ann_id = max(max_ann_id, ann["id"])

        img_id_offset += max_img_id + 1
        ann_id_offset += max_ann_id + 1

    return merged


# ---------------------------------------------------------------------------
# Step 2: Deduplicate by .rf. prefix
# ---------------------------------------------------------------------------

def dedup_by_prefix(images: list, dedup_delim: str) -> list:
    """
    按文件名前缀去重。
    例：abc.rf.123.jpg 与 abc.rf.456.jpg 都映射到前缀 abc，只保留一个。
    """
    prefix2img = {}
    for img in images:
        fname = img.get("file_name", "")
        basename = os.path.basename(fname)
        if dedup_delim and dedup_delim in basename:
            prefix = basename.split(dedup_delim)[0]
        else:
            prefix = basename
        if prefix not in prefix2img:
            prefix2img[prefix] = img
    return list(prefix2img.values())


# ---------------------------------------------------------------------------
# Step 3: Sample
# ---------------------------------------------------------------------------

def sample_images(images: list, ratio: float, seed: int) -> list:
    if ratio >= 1.0:
        return images
    rng = random.Random(seed)
    k = max(1, int(len(images) * ratio))
    return rng.sample(images, k)


# ---------------------------------------------------------------------------
# Step 4: Filter by category whitelist / blacklist / name_blacklist
# ---------------------------------------------------------------------------

def filter_categories(merged: dict, whitelist: list, blacklist: list, name_blacklist: list):
    """过滤标注中的类别，返回有效 category_id 集合。"""
    valid_ids = set(c["id"] for c in merged["categories"])
    if whitelist:
        # 只保留白名单中 id 的子类别
        # 白名单可以是父类别 id，需要递归找子类别
        # 简单处理：只保留 id 在 whitelist 中（及其 supercategory 对应的子类）
        # 如果白名单是 [1]，则保留所有 supercategory_id=1 的类，或直接 id=1
        # 由于 COCO 格式 supercategory 是名称，不是 id，需特殊处理
        # 这里简单处理：保留 id 在 whitelist 中的 + 名称中带有对应父名的
        whitelist_names = {c["name"] for c in merged["categories"] if c["id"] in whitelist}
        valid_ids = set()
        for c in merged["categories"]:
            if c["id"] in whitelist:
                valid_ids.add(c["id"])
            # 保留 supercategory 为白名单名称之一的类别
            if c.get("supercategory", "") in whitelist_names:
                valid_ids.add(c["id"])
    if blacklist:
        valid_ids -= set(blacklist)
    if name_blacklist:
        name_bl_set = set(name_blacklist)
        valid_ids -= {c["id"] for c in merged["categories"] if c["name"] in name_bl_set}
    return valid_ids


def remap_categories(merged: dict, valid_ids: set):
    """将有效 category_id 重新映射为从 0 开始的连续整数。"""
    old_cats = [c for c in merged["categories"] if c["id"] in valid_ids]
    old_cats_sorted = sorted(old_cats, key=lambda c: c["id"])
    old2new = {c["id"]: new_id for new_id, c in enumerate(old_cats_sorted)}
    new_cats = [{"id": old2new[c["id"]], "name": c["name"], "supercategory": c.get("supercategory", "")}
                for c in old_cats_sorted]
    return new_cats, old2new


# ---------------------------------------------------------------------------
# Step 5: Crop images and adjust bboxes
# ---------------------------------------------------------------------------

def crop_bbox(bbox, crop_x, crop_y, crop_w, crop_h, min_area):
    """
    将 COCO 格式 bbox (x, y, w, h) 按裁剪窗口调整。
    返回裁剪后的 bbox，若完全在窗口外则返回 None。
    """
    bx, by, bw, bh = bbox
    # 转换到绝对坐标
    x1, y1, x2, y2 = bx, by, bx + bw, by + bh
    # 裁剪
    cx1 = max(x1, crop_x) - crop_x
    cy1 = max(y1, crop_y) - crop_y
    cx2 = min(x2, crop_x + crop_w) - crop_x
    cy2 = min(y2, crop_y + crop_h) - crop_y
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    nw = cx2 - cx1
    nh = cy2 - cy1
    if nw * nh < min_area:
        return None
    return [float(cx1), float(cy1), float(nw), float(nh)]


def get_crop_offset(img_w, img_h, crop_size, mode, rng):
    """计算裁剪的左上角偏移 (x_offset, y_offset)。"""
    max_x = max(0, img_w - crop_size)
    max_y = max(0, img_h - crop_size)
    if mode == "center":
        return max_x // 2, max_y // 2
    elif mode == "left":
        return 0, max_y // 2
    elif mode == "right":
        return max_x, max_y // 2
    elif mode == "random":
        return rng.randint(0, max_x), rng.randint(0, max_y)
    else:
        return max_x // 2, max_y // 2


# ---------------------------------------------------------------------------
# Step 6: Build final dataset
# ---------------------------------------------------------------------------

def build_split(images, annotations_by_imgid, valid_cat_ids, old2new_cat,
                max_ann_per_image, min_box_area,
                crop_cfg, img_src_dirs,
                out_img_dir, out_ann_file):
    """
    生成一个 split（train 或 val）的数据集文件。
    复制图像（可选裁剪），写入 COCO JSON。
    """
    os.makedirs(out_img_dir, exist_ok=True)

    crop_enabled = crop_cfg.get("enabled", False)
    crop_size = crop_cfg.get("size", 1536)
    crop_mode = crop_cfg.get("mode", "center")
    crop_seed = crop_cfg.get("seed", 42)
    crop_rng = random.Random(crop_seed)

    new_images = []
    new_annotations = []
    new_ann_id = 0
    new_img_id = 0

    for img in images:
        old_img_id = img["id"]
        fname = img.get("file_name", "")
        basename = os.path.basename(fname)
        src_dir = img.get("_src_dir", "")

        # 找到图像源文件
        src_path = os.path.join(src_dir, basename)
        if not os.path.exists(src_path):
            # 也可能是子路径
            src_path = os.path.join(src_dir, fname)
        if not os.path.exists(src_path):
            print(f"  [WARN] 图像文件不存在，跳过 (id={old_img_id}, file={basename}): {src_path}", file=sys.stderr)
            continue

        # 打开图像
        try:
            pil_img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] 打开图像失败 {src_path}: {e}", file=sys.stderr)
            continue

        img_w, img_h = pil_img.size

        # 裁剪
        if crop_enabled:
            cx_off, cy_off = get_crop_offset(img_w, img_h, crop_size, crop_mode, crop_rng)
            pil_img = pil_img.crop((cx_off, cy_off, cx_off + crop_size, cy_off + crop_size))
            out_w, out_h = crop_size, crop_size
        else:
            cx_off, cy_off = 0, 0
            out_w, out_h = img_w, img_h

        # 收集该图的标注
        anns = annotations_by_imgid.get(old_img_id, [])
        valid_anns = []
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id not in valid_cat_ids:
                continue
            new_cat_id = old2new_cat.get(cat_id)
            if new_cat_id is None:
                continue

            bbox = ann.get("bbox", [0, 0, 0, 0])
            if crop_enabled:
                new_bbox = crop_bbox(bbox, cx_off, cy_off, crop_size, crop_size, min_box_area)
            else:
                area = bbox[2] * bbox[3]
                new_bbox = bbox if area >= min_box_area else None

            if new_bbox is None:
                continue

            new_ann = {
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": new_cat_id,
                "bbox": new_bbox,
                "area": float(new_bbox[2] * new_bbox[3]),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": [],
            }
            valid_anns.append(new_ann)
            new_ann_id += 1

            if len(valid_anns) >= max_ann_per_image:
                break

        # 图像有有效标注才保存（目标检测一般要求）
        # 若需要保留无标注图像可注释掉以下判断
        if not valid_anns:
            continue

        # 保存图像
        dst_fname = basename
        dst_path = os.path.join(out_img_dir, dst_fname)
        pil_img.save(dst_path)

        new_img_record = {
            "id": new_img_id,
            "file_name": dst_fname,
            "width": out_w,
            "height": out_h,
        }
        new_images.append(new_img_record)
        new_annotations.extend(valid_anns)
        new_img_id += 1

    return new_images, new_annotations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    input_root = cfg.get("input_root", "")
    output_root = cfg.get("output_root", "data_processed")
    force_rebuild = cfg.get("force_rebuild", False)
    train_ratio = cfg.get("train_ratio", 0.8)
    split_seed = cfg.get("split_seed", 42)
    sample_ratio = cfg.get("sample_ratio", 1.0)
    dedup_delim = cfg.get("dedup_delim", ".rf.")
    annotation_globs = cfg.get("annotation_globs", ["/*/_annotations.coco.json"])
    cat_whitelist = cfg.get("category_id_whitelist", [])
    cat_blacklist = cfg.get("category_id_blacklist", [])
    name_blacklist = cfg.get("category_name_blacklist", [])
    max_ann_per_image = cfg.get("max_ann_per_image", 200)
    min_box_area = cfg.get("min_box_area", 25)
    crop_cfg = cfg.get("crop", {"enabled": False})

    train_out = os.path.join(output_root, "train")
    val_out = os.path.join(output_root, "val")
    train_img_dir = os.path.join(train_out, "images")
    val_img_dir = os.path.join(val_out, "images")
    train_ann = os.path.join(train_out, "_annotations.coco.json")
    val_ann = os.path.join(val_out, "_annotations.coco.json")

    # 检查是否需要重新生成
    if not force_rebuild and os.path.exists(train_ann) and os.path.exists(val_ann):
        print("输出文件已存在，跳过（使用 force_rebuild: true 强制重新生成）")
        return

    print("=" * 60)
    print("RT-DETR 数据准备脚本")
    print("=" * 60)

    # Step 1: Find and merge
    print(f"\n[1/6] 查找标注文件: {input_root}")
    ann_files = find_annotation_files(input_root, annotation_globs)
    if not ann_files:
        print(f"  错误：在 {input_root} 下未找到标注文件！", file=sys.stderr)
        sys.exit(1)
    print(f"  找到 {len(ann_files)} 个标注文件:")
    for f in ann_files:
        print(f"    {f}")

    print("\n[2/6] 合并标注文件...")
    merged = merge_coco_jsons(ann_files)
    print(f"  合并后: {len(merged['images'])} 张图像, "
          f"{len(merged['annotations'])} 个标注, "
          f"{len(merged['categories'])} 个类别")
    print(f"  类别: {[c['name'] for c in merged['categories']]}")

    # Step 2: Dedup
    print("\n[3/6] 按前缀去重...")
    images = dedup_by_prefix(merged["images"], dedup_delim)
    print(f"  去重后: {len(images)} 张图像")

    # Step 3: Sample
    print("\n[4/6] 采样...")
    images = sample_images(images, sample_ratio, split_seed)
    print(f"  采样后 (ratio={sample_ratio}): {len(images)} 张图像")

    # Step 4: Filter categories
    print("\n[5/6] 过滤类别...")
    valid_cat_ids = filter_categories(merged, cat_whitelist, cat_blacklist, name_blacklist)
    new_cats, old2new_cat = remap_categories(merged, valid_cat_ids)
    print(f"  有效类别 ids: {sorted(valid_cat_ids)}")
    print(f"  重映射后类别: {[(c['id'], c['name']) for c in new_cats]}")

    # Build annotation index
    ann_by_imgid = defaultdict(list)
    for ann in merged["annotations"]:
        ann_by_imgid[ann["image_id"]].append(ann)

    # Step 5: Train/val split
    print("\n[6/6] 分割训练/验证集并处理图像...")
    rng = random.Random(split_seed)
    shuffled = list(images)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    train_imgs = shuffled[:n_train]
    val_imgs = shuffled[n_train:]
    print(f"  训练集: {len(train_imgs)} 张, 验证集: {len(val_imgs)} 张")

    # Build datasets
    print("\n  处理训练集...")
    os.makedirs(train_img_dir, exist_ok=True)
    t_imgs, t_anns = build_split(
        train_imgs, ann_by_imgid, valid_cat_ids, old2new_cat,
        max_ann_per_image, min_box_area, crop_cfg,
        img_src_dirs=None, out_img_dir=train_img_dir, out_ann_file=train_ann
    )
    train_coco = {
        "images": t_imgs,
        "annotations": t_anns,
        "categories": new_cats,
    }
    with open(train_ann, "w") as f:
        json.dump(train_coco, f, ensure_ascii=False)
    print(f"  训练集保存: {len(t_imgs)} 张图像, {len(t_anns)} 个标注 -> {train_ann}")

    print("\n  处理验证集...")
    os.makedirs(val_img_dir, exist_ok=True)
    v_imgs, v_anns = build_split(
        val_imgs, ann_by_imgid, valid_cat_ids, old2new_cat,
        max_ann_per_image, min_box_area, crop_cfg,
        img_src_dirs=None, out_img_dir=val_img_dir, out_ann_file=val_ann
    )
    val_coco = {
        "images": v_imgs,
        "annotations": v_anns,
        "categories": new_cats,
    }
    with open(val_ann, "w") as f:
        json.dump(val_coco, f, ensure_ascii=False)
    print(f"  验证集保存: {len(v_imgs)} 张图像, {len(v_anns)} 个标注 -> {val_ann}")

    print("\n" + "=" * 60)
    print("数据准备完成！")
    print(f"  输出目录: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
