"""
train.py — RT-DETR 训练入口脚本
=================================
直接运行::

    python train.py [--config config.yaml]

所有超参数均从 config.yaml 读取，无需命令行传参。
"""

import argparse
import copy
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR 训练脚本")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def flatten_cfg(cfg: dict) -> dict:
    """
    将嵌套的 config.yaml（含 project/dataset/train 节）拍平为训练脚本使用的扁平字典。
    同时保留原始嵌套结构以便读取 dataset 等节。
    """
    flat = {}
    # project
    proj = cfg.get("project", {})
    flat["project_name"] = proj.get("name", "rtdetr")
    flat["work_dir"] = proj.get("work_dir", "work_dirs/rtdetr_l")
    flat["seed"] = proj.get("seed", 237)

    # dataset
    ds = cfg.get("dataset", {})
    flat["train_img_dir"] = os.path.join(ds.get("output_root", "data_processed"), "train", "images")
    flat["train_ann_file"] = os.path.join(ds.get("output_root", "data_processed"), "train", "_annotations.coco.json")
    flat["val_img_dir"] = os.path.join(ds.get("output_root", "data_processed"), "val", "images")
    flat["val_ann_file"] = os.path.join(ds.get("output_root", "data_processed"), "val", "_annotations.coco.json")
    flat["auto_prepare"] = ds.get("auto_prepare", False)
    flat["force_rebuild"] = ds.get("force_rebuild", False)

    # train
    tr = cfg.get("train", {})
    flat["epoches"] = tr.get("epoches", 150)
    flat["batch_size"] = tr.get("batch_size", 2)
    flat["val_batch_size"] = tr.get("val_batch_size", 4)
    flat["num_workers"] = tr.get("num_workers", 4)
    flat["lr"] = tr.get("lr", 1e-4)
    flat["backbone_lr"] = tr.get("backbone_lr", 1e-5)
    flat["weight_decay"] = tr.get("weight_decay", 1e-4)
    flat["clip_max_norm"] = tr.get("clip_max_norm", 0.1)
    flat["use_ema"] = tr.get("use_ema", True)
    flat["use_amp"] = tr.get("use_amp", True)
    flat["img_size"] = tr.get("imgsz", [640, 640])
    flat["early_stopping"] = tr.get("early_stopping", {"enabled": True, "patience": 30})

    # model
    model_cfg = tr.get("model", {})
    flat["num_classes"] = model_cfg.get("num_classes", 3)
    flat["backbone_depth"] = model_cfg.get("backbone_depth", 50)
    flat["encoder_hidden_dim"] = model_cfg.get("encoder_hidden_dim", 256)
    flat["num_decoder_layers"] = model_cfg.get("num_decoder_layers", 6)
    flat["num_queries"] = model_cfg.get("num_queries", 300)

    # soep
    soep_cfg = tr.get("soep", {})
    flat["use_soep"] = soep_cfg.get("enabled", True)
    flat["soep_spd_scale"] = soep_cfg.get("spd_scale", 2)
    flat["soep_large_k"] = soep_cfg.get("omnikernel_large_k", 31)

    # augmentation
    aug_cfg = tr.get("augmentation", {})
    flat["aug_photometric"] = aug_cfg.get("random_photometric_distort", True)
    flat["aug_zoom_out"] = aug_cfg.get("random_zoom_out", True)
    flat["aug_iou_crop"] = aug_cfg.get("random_iou_crop", True)
    flat["aug_hflip"] = aug_cfg.get("random_horizontal_flip", True)

    # defaults for optim helpers
    flat["min_lr"] = 1e-6
    flat["warmup_epochs"] = 1

    return flat


# ---------------------------------------------------------------------------
# Logger: tee to console + file
# ---------------------------------------------------------------------------

class TeeLogger:
    """将 stdout 同时写入日志文件。"""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ---------------------------------------------------------------------------
# Set random seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(flat: dict):
    """根据配置构建 RTDETR 模型（含可选 SOEP 模块）。"""
    from src.nn.backbone.presnet import PResNet
    from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
    from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
    from src.zoo.rtdetr.rtdetr import RTDETR

    use_soep = flat.get("use_soep", True)
    num_classes = flat["num_classes"]
    hidden_dim = flat["encoder_hidden_dim"]
    num_decoder_layers = flat["num_decoder_layers"]
    num_queries = flat["num_queries"]
    img_size = flat["img_size"]
    if isinstance(img_size, int):
        img_size = [img_size, img_size]

    # Backbone
    # 若启用 SOEP 则需要 P2 特征，return_idx=[0,1,2,3]（C2~C5）
    # 否则 return_idx=[1,2,3]（C3~C5）
    return_idx = [0, 1, 2, 3] if use_soep else [1, 2, 3]
    backbone = PResNet(
        depth=flat.get("backbone_depth", 50),
        variant="d",
        freeze_at=0,
        return_idx=return_idx,
        pretrained=False,   # 用户可自行加载预训练权重
    )

    # HybridEncoder 输入通道：C3=512, C4=1024, C5=2048（ResNet50）
    encoder = HybridEncoder(
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=hidden_dim,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=img_size,
    )

    # Decoder (RTDETRTransformer)
    decoder = RTDETRTransformer(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        feat_channels=[hidden_dim] * 3,
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_decoder_layers=num_decoder_layers,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_spatial_size=img_size,
    )

    # SOEP 模块（可选）
    soep = None
    if use_soep:
        from src.nn.soep import SOEPModule
        # P2 (C2) 输出通道 = 256, P3 (C3) = 512
        soep = SOEPModule(
            p2_channels=256,
            p3_channels=512,
            out_channels=512,
            spd_scale=flat.get("soep_spd_scale", 2),
            large_k=flat.get("soep_large_k", 31),
        )

    model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder, soep=soep)
    return model


# ---------------------------------------------------------------------------
# Build criterion and postprocessor
# ---------------------------------------------------------------------------

def build_criterion(flat: dict):
    from src.zoo.rtdetr.rtdetr_criterion import RTDETRCriterion
    from src.zoo.rtdetr.matcher import HungarianMatcher

    num_classes = flat["num_classes"]
    matcher = HungarianMatcher(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, use_focal_loss=True
    )
    weight_dict = {
        "loss_cls": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }
    # DN 辅助输出损失
    aux_weight_dict = {}
    for i in range(flat["num_decoder_layers"] - 1):
        aux_weight_dict.update({f"{k}_aux{i}": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    criterion = RTDETRCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=("labels", "boxes"),
        alpha=0.25,
        gamma=2.0,
    )
    return criterion


def build_postprocessor(flat: dict):
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    return RTDETRPostProcessor(
        num_classes=flat["num_classes"],
        use_focal_loss=True,
        num_top_queries=flat["num_queries"],
    )


# ---------------------------------------------------------------------------
# Build data loaders
# ---------------------------------------------------------------------------

def build_loaders(flat: dict):
    from src.data.dataloader import build_dataset, build_dataloader, collate_fn

    train_ds = build_dataset(flat, split="train")
    val_ds = build_dataset(flat, split="val")

    train_loader = build_dataloader(
        train_ds,
        batch_size=flat["batch_size"],
        num_workers=flat["num_workers"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=flat.get("val_batch_size", flat["batch_size"]),
        num_workers=flat["num_workers"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load config ---
    raw_cfg = load_yaml(args.config)
    flat = flatten_cfg(raw_cfg)

    # --- Set seed ---
    set_seed(flat["seed"])

    # --- Auto prepare data ---
    if flat.get("auto_prepare", False):
        print("[train.py] auto_prepare=true，正在运行数据准备脚本...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "prepare_data.py", "--config", args.config],
            check=False,
        )
        if result.returncode != 0:
            print("[ERROR] 数据准备失败，请检查 prepare_data.py 的输出")
            sys.exit(1)

    # --- Create timestamped run directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rtdetr_train_{timestamp}"
    run_dir = Path(flat["work_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup tee logger ---
    log_path = run_dir / "train.log"
    tee = TeeLogger(str(log_path))
    sys.stdout = tee
    sys.stderr = tee

    print("=" * 60)
    print(f"RT-DETR 训练开始")
    print(f"  运行目录: {run_dir}")
    print(f"  配置文件: {args.config}")
    print(f"  时间戳:   {timestamp}")
    print("=" * 60)

    # --- Copy config to run_dir ---
    shutil.copy2(args.config, run_dir / "config.yaml")
    print(f"[INFO] 配置文件已复制到: {run_dir / 'config.yaml'}")

    # --- Check data ---
    train_ann = flat["train_ann_file"]
    val_ann = flat["val_ann_file"]
    if not os.path.exists(train_ann):
        print(f"[ERROR] 训练集标注文件不存在: {train_ann}")
        print("        请先运行 prepare_data.py 或设置 auto_prepare: true")
        sys.exit(1)
    if not os.path.exists(val_ann):
        print(f"[ERROR] 验证集标注文件不存在: {val_ann}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- Build components ---
    print("\n[INFO] 构建模型...")
    model = build_model(flat)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] 可训练参数量: {n_params / 1e6:.2f} M")

    criterion = build_criterion(flat)
    postprocessor = build_postprocessor(flat)

    print("[INFO] 构建数据集...")
    train_loader, val_loader = build_loaders(flat)
    print(f"[INFO] 训练集: {len(train_loader.dataset)} 张图像")
    print(f"[INFO] 验证集: {len(val_loader.dataset)} 张图像")

    from src.optim.optim import build_optimizer, build_lr_scheduler
    optimizer = build_optimizer(model, flat)
    lr_scheduler = build_lr_scheduler(optimizer, flat, len(train_loader))

    from src.data.coco import CocoEvaluator
    from src.data.coco.coco_utils import get_coco_api_from_dataset
    coco_gt = get_coco_api_from_dataset(val_loader.dataset)
    evaluator = CocoEvaluator(coco_gt, ["bbox"])

    # --- DetSolver ---
    from src.solver.det_solver import DetSolver
    solver = DetSolver(
        model=model,
        criterion=criterion,
        postprocessor=postprocessor,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        cfg=flat,
        run_dir=str(run_dir),
    )

    print("\n[INFO] 开始训练...")
    try:
        solver.fit()
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断")

    print("\n" + "=" * 60)
    print(f"训练结束！最佳 mAP50: {solver.best_map50:.4f}")
    print(f"结果目录: {run_dir}")
    print("=" * 60)

    # Restore stdout
    sys.stdout = tee.terminal
    sys.stderr = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
