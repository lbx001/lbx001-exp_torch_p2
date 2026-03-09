"""Detection solver with early stopping support."""
import copy
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from src.misc.dist import is_main_process

logger = logging.getLogger(__name__)


class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


class DetSolver(BaseSolver):
    def __init__(self, model, criterion, postprocessor, optimizer, lr_scheduler,
                 train_loader, val_loader, evaluator, cfg, run_dir):
        super().__init__(model, criterion, optimizer, lr_scheduler,
                         train_loader, val_loader, cfg)
        self.postprocessor = postprocessor
        self.evaluator = evaluator
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler() if cfg.get('use_amp', False) else None
        self.ema = ModelEMA(model) if cfg.get('use_ema', False) else None

        es_cfg = cfg.get('early_stopping') or {}
        self.es_enabled = es_cfg.get('enabled', False)
        self.es_patience = es_cfg.get('patience', 30)
        self.es_counter = 0
        self.best_map50 = 0.0

        self.log_path = self.run_dir / 'log.txt'

    # ------------------------------------------------------------------
    def fit(self):
        from src.data.coco import CocoEvaluator
        from src.data.coco.coco_utils import get_coco_api_from_dataset

        cfg = self.cfg
        epoches = cfg.get('epoches', 150)
        max_norm = cfg.get('clip_max_norm', 0.1)

        for epoch in range(epoches):
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_loader,
                self.optimizer, self.device, epoch,
                max_norm=max_norm, scaler=self.scaler, ema=self.ema,
            )
            self.lr_scheduler.step()

            eval_model = self.ema.ema if self.ema is not None else self.model
            coco_gt = get_coco_api_from_dataset(self.val_loader.dataset)
            evaluator = CocoEvaluator(coco_gt, ['bbox'])

            val_stats = evaluate(
                eval_model, self.criterion, self.postprocessor,
                self.val_loader, evaluator, self.device,
            )

            map50 = val_stats['coco_eval_bbox'][1] if len(val_stats['coco_eval_bbox']) > 1 else 0.0

            log_entry = {'epoch': epoch, 'train': train_stats, 'val': val_stats, 'map50': map50}
            if is_main_process():
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

                self.save_checkpoint(self.run_dir / 'checkpoint.pth', epoch=epoch, map50=map50)

                if map50 > self.best_map50:
                    self.best_map50 = map50
                    self.save_checkpoint(
                        self.run_dir / 'best_map50.pth', epoch=epoch, map50=map50)
                    print(f'[Epoch {epoch}] New best mAP50: {map50:.4f} -> saved best_map50.pth')
                    self.es_counter = 0
                else:
                    self.es_counter += 1

                if self.es_enabled and self.es_counter >= self.es_patience:
                    print(f'Early stopping at epoch {epoch}, best mAP50: {self.best_map50:.4f}')
                    break

    def save_checkpoint(self, path, **kwargs):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.ema is not None:
            state['ema'] = self.ema.ema.state_dict()
        state.update(kwargs)
        torch.save(state, path)
