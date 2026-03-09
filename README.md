# RT-DETR 钢轨缺陷检测系统

基于 [RT-DETR](https://arxiv.org/abs/2304.08069)（Real-Time Detection Transformer）的钢轨缺陷智能检测系统，集成了 **SOEP**（小目标增强处理模块），提升对细小缺陷的检测精度。

---

## 目录

- [项目介绍](#项目介绍)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [模型结构说明](#模型结构说明)
- [常见问题](#常见问题)

---

## 项目介绍

本项目针对钢轨表面缺陷检测任务，在 RT-DETR 基础上进行了以下改进：

1. **SOEP 模块**（Small Object Enhancement Processing）：融合 P2 高分辨率特征与 P3 特征，通过 SPDConv 降维后利用 CSP-OmniKernel 进行多感受野融合，显著提升对小尺寸缺陷（如细裂纹、轻微划伤）的检测能力。
2. **PResNet-vd 骨干**：采用 ResNet-vd 改进的 stem（三个 3×3 卷积替换 7×7），下采样使用均值池化 + 1×1 卷积，提升特征提取质量。
3. **HybridEncoder 颈部**：AIFI（Attentional Injection Feature Integration）+ CCFM（Cross-scale Feature Fusion Module），实现跨尺度特征高效融合。
4. **DN（去噪）训练**：通过在 GT 标注上添加噪声生成辅助查询，加速收敛并提升定位精度。

---

## 目录结构

```
lbx001-exp_torch_p2/
├── config.yaml              # 训练配置文件
├── eval_config.yaml         # 评估配置文件
├── requirements.txt         # Python 依赖列表
├── README.md                # 本文档
├── work_dirs/               # 训练输出目录（自动创建）
│   └── rail_defect_rtdetr/
│       ├── checkpoint.pth   # 最新 checkpoint
│       ├── best_map50.pth   # 最优 mAP50 模型
│       └── log.txt          # 训练日志
└── src/
    ├── core/                # 配置系统
    ├── data/                # 数据集与预处理
    │   └── coco/            # COCO 格式数据集支持
    ├── misc/                # 工具函数（日志、分布式）
    ├── nn/
    │   ├── backbone/        # PResNet 骨干网络
    │   └── soep.py          # SOEP 小目标增强模块
    ├── optim/               # 优化器与学习率调度
    ├── solver/              # 训练 / 评估引擎
    └── zoo/rtdetr/          # RT-DETR 模型实现
        ├── rtdetr.py        # 顶层模型
        ├── hybrid_encoder.py
        ├── rtdetr_decoder.py
        ├── rtdetr_criterion.py
        ├── rtdetr_postprocessor.py
        ├── matcher.py
        ├── denoising.py
        └── box_ops.py
```

---

## 环境配置

### 1. 创建 Python 虚拟环境（推荐）

```bash
conda create -n rtdetr python=3.10 -y
conda activate rtdetr
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：请根据 CUDA 版本选择对应的 PyTorch 安装命令。例如，CUDA 11.8：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 数据准备

本项目使用 **COCO 格式**标注文件。请按以下结构组织数据集：

```
data/
├── train/
│   ├── images/          # 训练图像（.jpg 或 .png）
│   └── annotations.json # COCO 格式训练标注
└── val/
    ├── images/          # 验证图像
    └── annotations.json # COCO 格式验证标注
```

### COCO JSON 格式说明

`annotations.json` 应包含以下字段：

```json
{
  "images": [
    {"id": 1, "file_name": "rail_001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "crack"},
    {"id": 2, "name": "scratch"}
  ]
}
```

> `bbox` 格式为 `[x_min, y_min, width, height]`（左上角坐标 + 宽高）。

### 修改配置文件

编辑 `config.yaml`，将数据路径修改为实际路径：

```yaml
train_img_dir: /path/to/your/train/images
train_ann_file: /path/to/your/train/annotations.json
val_img_dir: /path/to/your/val/images
val_ann_file: /path/to/your/val/annotations.json
num_classes: 10   # 修改为实际类别数
```

---

## 训练

### 快速开始

```python
import sys
sys.path.insert(0, '.')

from src.core import YAMLConfig
from src.nn.backbone.presnet import PResNet
from src.nn.soep import SOEPModule
from src.zoo.rtdetr import (
    RTDETR, HybridEncoder, RTDETRTransformer,
    RTDETRCriterion, RTDETRPostProcessor,
)
from src.zoo.rtdetr.matcher import HungarianMatcher
from src.data.dataloader import build_dataset, build_dataloader, collate_fn
from src.optim.optim import build_optimizer, build_lr_scheduler
from src.solver import DetSolver

# 加载配置
cfg = YAMLConfig('config.yaml')

# 构建骨干网络
backbone = PResNet(
    depth=cfg.backbone_depth,
    variant=cfg.backbone_variant,
    return_idx=cfg.backbone_return_idx,
    freeze_at=cfg.freeze_at,
    pretrained=cfg.pretrained_backbone,
)

# 构建 SOEP 模块
soep = SOEPModule(
    p2_channels=cfg.soep_p2_channels,
    p3_channels=cfg.soep_p3_channels,
    out_channels=cfg.soep_out_channels,
) if cfg.use_soep else None

# 构建 HybridEncoder
encoder = HybridEncoder(
    in_channels=[cfg.soep_out_channels, 1024, 2048],
    hidden_dim=cfg.encoder_hidden_dim,
    nhead=cfg.encoder_nhead,
    dim_feedforward=cfg.encoder_dim_feedforward,
)

# 构建解码器
decoder = RTDETRTransformer(
    num_classes=cfg.num_classes,
    hidden_dim=cfg.hidden_dim,
    num_queries=cfg.num_queries,
    num_decoder_layers=cfg.num_decoder_layers,
    num_denoising=cfg.num_denoising,
)

model = RTDETR(backbone, encoder, decoder, soep=soep)

# 损失函数
matcher = HungarianMatcher(
    cost_class=cfg.matcher_cost_class,
    cost_bbox=cfg.matcher_cost_bbox,
    cost_giou=cfg.matcher_cost_giou,
)
weight_dict = {
    'loss_cls': cfg.loss_weight_cls,
    'loss_bbox': cfg.loss_weight_bbox,
    'loss_giou': cfg.loss_weight_giou,
}
criterion = RTDETRCriterion(cfg.num_classes, matcher, weight_dict)
postprocessor = RTDETRPostProcessor(num_classes=cfg.num_classes)

# 数据集
train_ds = build_dataset(cfg, split='train')
val_ds = build_dataset(cfg, split='val')
train_loader = build_dataloader(
    train_ds, cfg.batch_size, cfg.num_workers, shuffle=True, collate_fn=collate_fn)
val_loader = build_dataloader(
    val_ds, cfg.val_batch_size, cfg.num_workers, shuffle=False, collate_fn=collate_fn)

# 优化器与学习率调度
optimizer = build_optimizer(model, cfg)
lr_scheduler = build_lr_scheduler(optimizer, cfg, len(train_loader))

# 启动训练
solver = DetSolver(
    model, criterion, postprocessor, optimizer, lr_scheduler,
    train_loader, val_loader, evaluator=None, cfg=cfg._cfg,
    run_dir=cfg.output_dir,
)
solver.fit()
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epoches` | 150 | 总训练轮数 |
| `batch_size` | 8 | 训练批次大小 |
| `lr` | 1e-4 | Head 学习率 |
| `backbone_lr` | 1e-5 | Backbone 学习率 |
| `use_amp` | true | 混合精度训练 |
| `use_ema` | true | 指数移动平均 |
| `early_stopping.patience` | 30 | 早停耐心值 |

---

## 评估

```python
import torch
from src.core import YAMLConfig
from src.data.dataloader import build_dataset, build_dataloader, collate_fn
from src.data.coco import CocoEvaluator
from src.data.coco.coco_utils import get_coco_api_from_dataset
from src.solver.det_engine import evaluate

cfg = YAMLConfig('eval_config.yaml')

# 构建模型（与训练代码相同）
# ... 构建 model、criterion、postprocessor ...

# 加载权重
ckpt = torch.load(cfg.checkpoint, map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

val_ds = build_dataset(cfg, split='val')
val_loader = build_dataloader(
    val_ds, cfg.val_batch_size, cfg.num_workers, shuffle=False, collate_fn=collate_fn)

coco_gt = get_coco_api_from_dataset(val_ds)
evaluator = CocoEvaluator(coco_gt, ['bbox'])

stats = evaluate(model, criterion, postprocessor, val_loader, evaluator, device)
print('mAP@0.5:0.95 =', stats['coco_eval_bbox'][0])
print('mAP@0.50     =', stats['coco_eval_bbox'][1])
```

---

## 模型结构说明

### SOEP 模块

```
P2 (H×2, W×2, 256ch)
    │
    ▼ SPDConv (scale=2)
P2_down (H, W, 512ch)
    │
    ├─── concat ─── P3 (H, W, 512ch)
    │
    ▼ CSP-OmniKernel (1024ch → 512ch)
P3_enhanced (H, W, 512ch)
```

**CSP-OmniKernel** 内部三分支：
- **Local branch**：1×1 深度可分离卷积，捕获局部精细纹理
- **Large Kernel branch**：31×31 + 31×1 + 1×31 大核卷积，扩大感受野
- **Global branch**：DCAM（频域通道注意力）+ FSAM（频域空间注意力）

### HybridEncoder

```
C3 → Lateral Conv → AIFI (可选) ─┐
C4 → Lateral Conv ─────────────── Top-Down FPN ── Bottom-Up Path → [P3', P4', P5']
C5 → Lateral Conv → AIFI ─────────┘
```

---

## 常见问题

**Q: 训练时 GPU 内存不足怎么办？**

A: 可以减小 `batch_size`（如改为 4 或 2），或关闭 `use_amp: false` 临时调试。

**Q: 如何使用自定义类别？**

A: 修改 `config.yaml` 中的 `num_classes`，确保与标注 JSON 中的类别数量一致。

**Q: 预训练权重在哪里下载？**

A: 设置 `pretrained_backbone: true` 后，代码会自动从 torchvision 下载 ResNet50 ImageNet 预训练权重。

**Q: 如何在多 GPU 上训练？**

A: 使用 `torch.distributed.launch`：
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py
```
并在代码中初始化分布式环境（`torch.distributed.init_process_group`）。