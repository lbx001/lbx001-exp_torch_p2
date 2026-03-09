from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from ..data import build_dataloaders, prepare_dataset, processed_dataset_paths
from ..models import build_model
from ..utils import create_run_dir, latest_run_dir, save_runtime_config, seed_everything, setup_logger
from .evaluator import box_cxcywh_to_xyxy, evaluate_map50, generalized_box_iou


class GreedyMatcher:
    def __init__(self, class_cost: float, bbox_cost: float, giou_cost: float):
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]):
        matches = []
        for batch_idx, target in enumerate(targets):
            tgt_labels = target['labels'].to(outputs['pred_logits'].device)
            tgt_boxes = target['boxes'].to(outputs['pred_boxes'].device)
            if tgt_labels.numel() == 0:
                matches.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue
            pred_probs = outputs['pred_logits'][batch_idx].softmax(-1)[:, :-1]
            pred_boxes = outputs['pred_boxes'][batch_idx]
            class_cost = -pred_probs[:, tgt_labels]
            bbox_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)
            giou_cost = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            cost = self.class_cost * class_cost + self.bbox_cost * bbox_cost + self.giou_cost * giou_cost
            cost = cost.detach().cpu()
            pred_indices = []
            tgt_indices = []
            used_pred = set()
            used_tgt = set()
            flat_indices = torch.argsort(cost.flatten())
            num_preds, num_tgts = cost.shape
            for flat_idx in flat_indices.tolist():
                pred_idx = flat_idx // num_tgts
                tgt_idx = flat_idx % num_tgts
                if pred_idx in used_pred or tgt_idx in used_tgt:
                    continue
                used_pred.add(pred_idx)
                used_tgt.add(tgt_idx)
                pred_indices.append(pred_idx)
                tgt_indices.append(tgt_idx)
                if len(used_tgt) == num_tgts:
                    break
            matches.append((torch.tensor(pred_indices, dtype=torch.int64), torch.tensor(tgt_indices, dtype=torch.int64)))
        return matches


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher: GreedyMatcher, eos_coef: float, weights: Dict[str, float]):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weights = weights
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            target_classes[batch_idx, src_idx.to(src_logits.device)] = targets[batch_idx]['labels'][tgt_idx].to(src_logits.device)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        loss_bbox = torch.tensor(0.0, device=src_logits.device)
        loss_giou = torch.tensor(0.0, device=src_logits.device)
        total_boxes = 0
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            src_boxes = outputs['pred_boxes'][batch_idx, src_idx.to(src_logits.device)]
            tgt_boxes = targets[batch_idx]['boxes'][tgt_idx].to(src_logits.device)
            loss_bbox = loss_bbox + F.l1_loss(src_boxes, tgt_boxes, reduction='sum')
            loss_giou = loss_giou + (1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))).sum()
            total_boxes += len(src_idx)
        total_boxes = max(total_boxes, 1)
        loss_bbox = loss_bbox / total_boxes
        loss_giou = loss_giou / total_boxes
        total = self.weights['loss_ce'] * loss_ce + self.weights['loss_bbox'] * loss_bbox + self.weights['loss_giou'] * loss_giou
        return {'loss': total, 'loss_ce': loss_ce, 'loss_bbox': loss_bbox, 'loss_giou': loss_giou}



def _num_classes_from_metadata(dataset_paths: Dict[str, Path]) -> int:
    import json

    payload = json.loads(dataset_paths['metadata'].read_text(encoding='utf-8'))
    return max(len(payload.get('categories', [])), 1)



def _resolve_checkpoint(config: Dict[str, Any], run_dir: Path | None = None) -> Path:
    checkpoint_cfg = config.get('checkpoint', {})
    if checkpoint_cfg.get('path'):
        return Path(checkpoint_cfg['path']).resolve()
    if run_dir is not None:
        candidate = run_dir / 'best_map50.pt'
        if candidate.exists():
            return candidate
    latest = latest_run_dir(config['project'])
    if latest is None:
        raise FileNotFoundError('未找到可用的 checkpoint。请先训练或在 eval_config.yaml 中指定 checkpoint.path。')
    candidate = latest / 'best_map50.pt'
    if not candidate.exists():
        raise FileNotFoundError(f'最新运行目录中没有 best_map50.pt: {candidate}')
    return candidate



def _build_optimizer(model: nn.Module, training_cfg: Dict[str, Any]):
    optimizer_cfg = training_cfg.get('optimizer', {})
    name = optimizer_cfg.get('name', 'adamw').lower()
    lr = float(optimizer_cfg.get('lr', 1e-4))
    weight_decay = float(optimizer_cfg.get('weight_decay', 1e-4))
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(optimizer_cfg.get('momentum', 0.9)), weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)



def _build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: Dict[str, Any]):
    scheduler_cfg = training_cfg.get('scheduler', {})
    name = scheduler_cfg.get('name', 'step').lower()
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(training_cfg.get('epochs', 1))))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(scheduler_cfg.get('step_size', 10)), gamma=float(scheduler_cfg.get('gamma', 0.1)))



def _train_one_epoch(model, criterion, data_loader, optimizer, device, scaler, amp_enabled):
    model.train()
    running = {'loss': 0.0, 'loss_ce': 0.0, 'loss_bbox': 0.0, 'loss_giou': 0.0}
    for images, targets in data_loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['loss']
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for key in running:
            running[key] += float(loss_dict[key].detach().cpu())
    batches = max(len(data_loader), 1)
    return {key: value / batches for key, value in running.items()}



def _prepare_runtime(config: Dict[str, Any], config_path: Path):
    seed_everything(int(config['project'].get('seed', 0)))
    run_dir = create_run_dir(config['project'])
    logger = setup_logger(run_dir)
    save_runtime_config(config, config_path, run_dir)
    logger.info('运行目录: %s', run_dir)
    return run_dir, logger



def train_model(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    run_dir, logger = _prepare_runtime(config, config_path)
    dataset_cfg = config['dataset']
    if dataset_cfg.get('auto_prepare', False) or not processed_dataset_paths(dataset_cfg)['metadata'].exists():
        prepare_dataset(dataset_cfg, logger=logger)
    dataset_paths = processed_dataset_paths(dataset_cfg)
    num_classes = int(config.get('model', {}).get('num_classes') or _num_classes_from_metadata(dataset_paths))
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config, dataset_paths)
    logger.info('训练/验证样本数: %s / %s', len(train_dataset), len(val_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config, num_classes=num_classes).to(device)
    matcher = GreedyMatcher(
        class_cost=float(config['training']['matcher'].get('class_cost', 2.0)),
        bbox_cost=float(config['training']['matcher'].get('bbox_cost', 5.0)),
        giou_cost=float(config['training']['matcher'].get('giou_cost', 2.0)),
    )
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        eos_coef=float(config['training'].get('eos_coef', 0.1)),
        weights={
            'loss_ce': float(config['training']['loss_weights'].get('loss_ce', 1.0)),
            'loss_bbox': float(config['training']['loss_weights'].get('loss_bbox', 5.0)),
            'loss_giou': float(config['training']['loss_weights'].get('loss_giou', 2.0)),
        },
    ).to(device)
    optimizer = _build_optimizer(model, config['training'])
    scheduler = _build_scheduler(optimizer, config['training'])
    amp_enabled = bool(config['training'].get('amp', True) and torch.cuda.is_available())
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    patience = int(config['training'].get('early_stopping', {}).get('patience', 10))
    best_map50 = -1.0
    best_epoch = -1
    stale_epochs = 0
    history = []

    for epoch in range(1, int(config['training'].get('epochs', 1)) + 1):
        epoch_start = time.time()
        train_stats = _train_one_epoch(model, criterion, train_loader, optimizer, device, scaler, amp_enabled)
        metrics = evaluate_map50(model, val_loader, device, score_threshold=float(config['evaluation'].get('score_threshold', 0.05)))
        scheduler.step()
        epoch_time = time.time() - epoch_start

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_map50': max(best_map50, metrics['map50']),
            'num_classes': num_classes,
            'config': config,
        }
        torch.save(state, run_dir / 'last.pt')
        logger.info('Epoch %s | loss=%.4f ce=%.4f bbox=%.4f giou=%.4f | mAP50=%.4f | %.1fs', epoch, train_stats['loss'], train_stats['loss_ce'], train_stats['loss_bbox'], train_stats['loss_giou'], metrics['map50'], epoch_time)
        history.append({'epoch': epoch, 'train': train_stats, 'val': metrics})

        if metrics['map50'] > best_map50:
            best_map50 = metrics['map50']
            best_epoch = epoch
            stale_epochs = 0
            torch.save(state, run_dir / 'best_map50.pt')
            logger.info('新的最佳权重已保存: epoch=%s, mAP50=%.4f', best_epoch, best_map50)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info('触发早停: patience=%s, best_epoch=%s, best_mAP50=%.4f', patience, best_epoch, best_map50)
                break

    summary = {'run_dir': str(run_dir), 'best_epoch': best_epoch, 'best_map50': best_map50, 'history': history}
    logger.info('训练完成: %s', summary)
    return summary



def evaluate_only(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    dataset_cfg = config['dataset']
    if dataset_cfg.get('auto_prepare', False) or not processed_dataset_paths(dataset_cfg)['metadata'].exists():
        prepare_dataset(dataset_cfg)
    dataset_paths = processed_dataset_paths(dataset_cfg)
    num_classes = int(config.get('model', {}).get('num_classes') or _num_classes_from_metadata(dataset_paths))
    _, val_dataset, _, val_loader = build_dataloaders(config, dataset_paths)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config, num_classes=num_classes).to(device)
    checkpoint_path = _resolve_checkpoint(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    metrics = evaluate_map50(model, val_loader, device, score_threshold=float(config['evaluation'].get('score_threshold', 0.05)))
    run_dir = checkpoint_path.parent
    logger = setup_logger(run_dir, name='rtdetr_eval')
    save_runtime_config(config, config_path, run_dir)
    logger.info('评估 checkpoint: %s', checkpoint_path)
    logger.info('验证样本数: %s', len(val_dataset))
    logger.info('评估结果: %s', metrics)
    return {'checkpoint': str(checkpoint_path), 'metrics': metrics}
