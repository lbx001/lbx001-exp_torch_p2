# RT-DETR 钢轨缺陷检测系统

基于 [RT-DETR](https://arxiv.org/abs/2304.08069)（Real-Time Detection Transformer）的钢轨缺陷智能检测系统，集成了 **SOEP**（小目标增强处理模块），专为检测细小轨道缺陷（chipping / scrach / seams）而设计。

---

## 目录

- [项目介绍](#项目介绍)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [模型结构说明](#模型结构说明)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 项目介绍

本项目针对钢轨表面缺陷检测任务，在 RT-DETR 基础上进行了以下改进：

1. **SOEP 模块**（Small Object Enhancement Processing）：融合 P2 高分辨率特征与 P3 特征，通过 SPDConv 降维后利用 CSP-OmniKernel 进行多感受野融合，显著提升对小尺寸缺陷的检测能力。
2. **PResNet-vd 骨干**：采用 ResNet-vd 改进的 stem（三个 3×3 卷积替换 7×7），下采样使用均值池化 + 1×1 卷积，提升特征提取质量。
3. **HybridEncoder 颈部**：AIFI（注意力注入特征集成）+ CCFM（跨尺度特征融合模块），实现跨尺度特征高效融合。
4. **DN（去噪）训练**：在 GT 标注上添加噪声生成辅助查询，加速收敛并提升定位精度。
5. **早停机制**：监控 mAP50，连续多个 epoch 无提升时自动停止训练。
6. **完整数据处理流水线**：合并多 split 标注、去重、采样、裁剪、类别过滤、类别 ID 重映射。

---

## 目录结构

```
lbx001-exp_torch_p2/
├── config.yaml              # 训练配置文件（含详细中文注释）
├── eval_config.yaml         # 评估配置文件
├── train.py                 # 训练入口：python train.py
├── eval.py                  # 评估入口：python eval.py
├── prepare_data.py          # 数据准备脚本：python prepare_data.py
├── requirements.txt         # Python 依赖列表
├── README.md                # 本文档
├── work_dirs/               # 训练输出目录（自动创建）
│   └── rtdetr_l/
│       └── rtdetr_train_YYYYMMDD_HHMMSS/
│           ├── config.yaml      # 本次训练使用的配置（副本）
│           ├── train.log        # 训练日志（含 console 输出）
│           ├── log.txt          # 每 epoch 指标（JSON Lines 格式）
│           ├── checkpoint.pth   # 最新 checkpoint
│           └── best_map50.pth   # mAP50 最优模型
└── src/
    ├── __init__.py
    ├── core/                # 配置系统（register/create 模式）
    │   ├── config.py
    │   ├── yaml_config.py
    │   └── yaml_utils.py
    ├── data/                # 数据集与预处理
    │   ├── coco/            # COCO 格式数据集支持
    │   │   ├── coco_dataset.py
    │   │   ├── coco_eval.py
    │   │   └── coco_utils.py
    │   ├── dataloader.py
    │   └── transforms.py
    ├── misc/                # 工具函数（日志、分布式）
    │   ├── dist.py
    │   └── logger.py
    ├── nn/
    │   ├── soep.py          # SOEP 小目标增强模块（SPDConv + CSP-OmniKernel）
    │   ├── arch/
    │   └── backbone/
    │       └── presnet.py   # PResNet（ResNet-vd）骨干网络
    ├── optim/
    │   └── optim.py         # AdamW 优化器 + Cosine+Warmup 调度
    ├── solver/              # 训练 / 评估引擎
    │   ├── solver.py
    │   ├── det_solver.py    # DetSolver（含 EMA、早停、checkpoint 保存）
    │   └── det_engine.py    # train_one_epoch / evaluate
    └── zoo/rtdetr/          # RT-DETR 模型实现
        ├── rtdetr.py        # 顶层模型（RTDETR）
        ├── hybrid_encoder.py   # HybridEncoder（AIFI + CCFM）
        ├── rtdetr_decoder.py   # RTDETRTransformer（含 MSDeformableAttention）
        ├── rtdetr_criterion.py # 损失函数（Focal + L1 + GIoU + DN）
        ├── rtdetr_postprocessor.py
        ├── matcher.py       # Hungarian Matcher
        ├── denoising.py     # DN 训练辅助查询生成
        ├── box_ops.py
        └── utils.py
```

---

## 环境配置

### 1. 创建虚拟环境（推荐）

```bash
conda create -n rtdetr python=3.10 -y
conda activate rtdetr
```

### 2. 安装 PyTorch（根据 CUDA 版本选择）

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

---

## 数据准备

### 方式一：使用 prepare_data.py（推荐）

将原始 COCO 格式标注文件放置于 `dataset.input_root` 目录下（支持 train / test / valid 子目录），然后运行：

```bash
python prepare_data.py --config config.yaml
```

脚本会自动完成：
1. **合并**多个 split 的标注文件
2. **去重**（按 `.rf.` 分隔符前缀去重）
3. **采样**（由 `sample_ratio` 控制）
4. **分割**训练/验证集（由 `train_ratio` 控制）
5. **裁剪**图像（2048×1536 → 1536×1536，可配置 center/left/right/random）
6. **过滤**小 bbox（面积 < `min_box_area`）
7. **类别过滤**与 **ID 重映射**（contiguous from 0）
8. 输出 COCO 格式数据到 `dataset.output_root`

### 方式二：手动准备数据

按以下结构组织数据：

```
<output_root>/
├── train/
│   ├── images/                  # 训练图像
│   └── _annotations.coco.json   # COCO 格式训练标注
└── val/
    ├── images/                  # 验证图像
    └── _annotations.coco.json   # COCO 格式验证标注
```

修改 `config.yaml` 中的 `dataset.output_root` 指向上述目录。

---

## 训练

```bash
python train.py
```

或指定配置文件：

```bash
python train.py --config config.yaml
```

脚本将自动：
- 创建带时间戳的运行目录（如 `work_dirs/rtdetr_l/rtdetr_train_20260309_120000/`）
- 复制配置文件到运行目录（便于复现）
- 将控制台输出同时写入 `train.log`
- 每 epoch 保存 `checkpoint.pth`（最新）和 `best_map50.pth`（最优）
- 当 mAP50 连续 `patience` 个 epoch 无提升时自动早停

### 训练日志

训练期间，每 epoch 的指标以 JSON Lines 格式写入 `log.txt`：

```json
{"epoch": 0, "train": {"loss": 45.2, "lr": 0.0001}, "val": {"coco_eval_bbox": [0.45, 0.72, ...]}, "map50": 0.72}
```

---

## 评估

```bash
python eval.py
```

或指定配置文件：

```bash
python eval.py --config eval_config.yaml --train-config config.yaml
```

脚本将自动在 `work_dir` 下查找最新的 `best_map50.pth`，输出 COCO 标准指标：

```
AP@[0.50:0.95]           : 0.4512
AP@0.50                  : 0.7234
AP@0.75                  : 0.4891
...
```

评估结果保存到 `work_dir/eval_results.json`。

---

## 模型结构说明

### SOEP 模块（论文 Section 3.3）

```
P2 (H×2, W×2, 256ch)          ← backbone C2 输出（高分辨率）
    │
    ▼ SPDConv (scale=2)         ← Space-to-Depth + 1×1 Conv
P2_down (H, W, 512ch)
    │
    ├─── concat ───── P3 (H, W, 512ch)   ← backbone C3 输出
    │
    ▼ CSP-OmniKernel (1024ch → 512ch)
P3_enhanced (H, W, 512ch)               → 送入 HybridEncoder
```

**SPDConv（3.3.1）**：将 H×W×C 的特征图通过空间-深度变换分解为 H/s×W/s×(s²C)，再经 1×1 卷积降维，相比步幅卷积/池化保留更多小目标信息。

**CSP-OmniKernel（3.3.2）**：输入特征分成两路：
- 0.25 路径 → **OmniKernel**（三分支）：
  - **Local branch**：1×1 深度可分离卷积（局部精细纹理）
  - **Large-kernel branch**：31×31 + 31×1 + 1×31 DW 卷积（大感受野）
  - **Global branch**：DCAM（频域通道注意力）+ FSAM（频域空间注意力）
- 0.75 路径 → bypass
- Concat → 输出 Conv

### HybridEncoder

```
C3 ──── Lateral Conv ──── AIFI ──┐
                                  ▼ Top-Down FPN
C4 ──── Lateral Conv ────────────── Bottom-Up Path → [P3', P4', P5']
                                  ▲
C5 ──── Lateral Conv ──── AIFI ──┘
```

RepC3 CSP 块在 FPN 各尺度进行特征融合。

---

## 配置说明

### config.yaml 主要参数

| 参数路径 | 默认值 | 说明 |
|---------|--------|------|
| `project.work_dir` | `work_dirs/rtdetr_l` | 训练结果根目录 |
| `dataset.output_root` | `...` | 处理后数据路径 |
| `dataset.auto_prepare` | `false` | 是否自动运行 prepare_data.py |
| `train.epoches` | `150` | 总训练轮数 |
| `train.batch_size` | `2` | 批大小（4090+1536px适用）|
| `train.lr` | `0.0001` | 检测头学习率 |
| `train.backbone_lr` | `0.00001` | Backbone 学习率 |
| `train.use_amp` | `true` | 混合精度（4090推荐）|
| `train.use_ema` | `true` | EMA（推荐开启）|
| `train.early_stopping.patience` | `30` | 早停耐心值 |
| `train.model.num_classes` | `3` | 类别数 |
| `train.soep.enabled` | `true` | SOEP 模块开关 |
| `train.soep.omnikernel_large_k` | `31` | OmniKernel 大核尺寸 |

---

## 常见问题

**Q: 训练时 GPU 内存不足怎么办？**
A: 减小 `batch_size`（如 1），或降低 `imgsz`（如 `[512, 512]`），或将 `train.soep.omnikernel_large_k` 改为 15。

**Q: 如何使用自定义类别数？**
A: 修改 `config.yaml` 中的 `train.model.num_classes`，确保与处理后标注文件中的类别数一致。

**Q: 如何加载预训练骨干权重？**
A: 在 `train.py` 的 `build_model` 中设置 `pretrained=True`，会自动从 torchvision 下载 ResNet50 ImageNet 权重。

**Q: 早停触发后如何继续训练？**
A: 增大 `train.early_stopping.patience`（如 50），或直接设置 `enabled: false` 关闭早停，从 `checkpoint.pth` 恢复。

**Q: 如何在多 GPU 上训练？**
A: 当前版本以单 GPU 为主；可使用 `torch.distributed.launch` 启动，代码已包含 `dist.py` 分布式工具函数。
