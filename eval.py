"""
eval.py — RT-DETR 评估入口脚本
================================
直接运行::

    python eval.py [--config eval_config.yaml]

所有参数均从 eval_config.yaml 读取，无需命令行传参。
自动在 work_dir 下查找 best_map50.pth（若 checkpoint 未指定）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR 评估脚本")
    parser.add_argument("--config", default="eval_config.yaml", help="评估配置文件路径")
    parser.add_argument("--train-config", default="config.yaml", help="训练配置文件路径（用于模型结构）")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_best_checkpoint(work_dir: str) -> str:
    """在 work_dir 下递归查找最新的 best_map50.pth。"""
    candidates = sorted(Path(work_dir).rglob("best_map50.pth"))
    if candidates:
        # 选最新的
        return str(candidates[-1])
    return ""


# ---------------------------------------------------------------------------
# Build model (复用 train.py 中的逻辑)
# ---------------------------------------------------------------------------

def build_model_from_cfg(train_cfg: dict):
    """根据训练配置重建模型结构（不加载权重）。"""
    from train import flatten_cfg, build_model
    flat = flatten_cfg(train_cfg)
    return flat, build_model(flat)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 加载评估配置
    eval_raw = load_yaml(args.config)
    eval_cfg = eval_raw.get("eval", eval_raw)
    ds_eval_cfg = eval_raw.get("dataset", {})

    # 加载训练配置（用于重建模型结构）
    train_raw = load_yaml(args.train_config)

    # 确定 checkpoint 路径
    checkpoint_path = eval_cfg.get("checkpoint", "").strip()
    if not checkpoint_path:
        work_dir = eval_cfg.get("work_dir", train_raw.get("project", {}).get("work_dir", "work_dirs/rtdetr_l"))
        checkpoint_path = find_best_checkpoint(work_dir)
        if not checkpoint_path:
            print(f"[ERROR] 在 {work_dir} 下未找到 best_map50.pth，请在 eval_config.yaml 中指定 checkpoint 路径")
            sys.exit(1)
    print(f"[INFO] 使用权重: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 重建模型
    print("[INFO] 重建模型结构...")
    flat, model = build_model_from_cfg(train_raw)
    model = model.to(device)

    # 加载权重
    print("[INFO] 加载权重...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # 优先使用 EMA 权重
    state_dict = ckpt.get("ema", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] 缺失键: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] 多余键: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()

    # 获取评估参数
    img_size = eval_cfg.get("imgsz", [640, 640])
    if isinstance(img_size, int):
        img_size = [img_size, img_size]
    batch_size = eval_cfg.get("batch_size", 4)
    num_workers = eval_cfg.get("num_workers", 4)

    # 数据集路径：优先 eval_config，其次推断
    train_ds_cfg = train_raw.get("dataset", {})
    output_root = train_ds_cfg.get("output_root", "data_processed")

    val_img_dir = ds_eval_cfg.get("val_img_dir", "").strip()
    val_ann_file = ds_eval_cfg.get("val_ann_file", "").strip()
    if not val_img_dir:
        val_img_dir = os.path.join(output_root, "val", "images")
    if not val_ann_file:
        val_ann_file = os.path.join(output_root, "val", "_annotations.coco.json")

    if not os.path.exists(val_ann_file):
        print(f"[ERROR] 验证集标注文件不存在: {val_ann_file}")
        sys.exit(1)

    print(f"[INFO] 验证集: {val_img_dir}")
    print(f"[INFO] 标注文件: {val_ann_file}")

    # 构建 eval 用的 flat 配置
    eval_flat = dict(flat)
    eval_flat["val_img_dir"] = val_img_dir
    eval_flat["val_ann_file"] = val_ann_file
    eval_flat["img_size"] = img_size

    # 构建验证集 dataloader
    print("[INFO] 构建验证集...")
    from src.data.dataloader import build_dataset, build_dataloader, collate_fn
    val_ds = build_dataset(eval_flat, split="val")
    val_loader = build_dataloader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=collate_fn
    )
    print(f"[INFO] 验证集图像数: {len(val_ds)}")

    # 构建 criterion 和 postprocessor（评估时也需要 criterion 计算 val loss）
    from train import build_criterion, build_postprocessor
    criterion = build_criterion(eval_flat)
    criterion = criterion.to(device)
    postprocessor = build_postprocessor(eval_flat)

    # 构建 COCO evaluator
    from src.data.coco import CocoEvaluator
    from src.data.coco.coco_utils import get_coco_api_from_dataset
    coco_gt = get_coco_api_from_dataset(val_ds)
    evaluator = CocoEvaluator(coco_gt, ["bbox"])

    # 运行评估
    print("\n[INFO] 开始评估...")
    from src.solver.det_engine import evaluate
    stats = evaluate(model, criterion, postprocessor, val_loader, evaluator, device)

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    coco_bbox = stats.get("coco_eval_bbox", [])
    metric_names = [
        "AP@[0.50:0.95]", "AP@0.50", "AP@0.75",
        "AP@small",       "AP@medium", "AP@large",
        "AR@1",           "AR@10",     "AR@100",
        "AR@small",       "AR@medium", "AR@large",
    ]
    for name, val in zip(metric_names, coco_bbox):
        print(f"  {name:<25}: {val:.4f}")

    # 保存结果
    work_dir = eval_cfg.get("work_dir", "work_dirs/rtdetr_l")
    os.makedirs(work_dir, exist_ok=True)
    result_path = os.path.join(work_dir, "eval_results.json")
    result = {
        "checkpoint": checkpoint_path,
        "metrics": {name: float(val) for name, val in zip(metric_names, coco_bbox)},
        "val_stats": {k: float(v) for k, v in stats.items() if k != "coco_eval_bbox"},
    }
    with open(result_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 评估结果已保存到: {result_path}")

    if len(coco_bbox) > 1:
        print(f"\n主要指标: mAP@0.5 = {coco_bbox[1]:.4f}, mAP@0.5:0.95 = {coco_bbox[0]:.4f}")


if __name__ == "__main__":
    main()
