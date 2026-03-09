"""Training and evaluation engine."""
import math
import sys
import time

import torch
import torch.nn as nn

from src.misc.logger import MetricLogger, SmoothedValue
from src.misc.dist import reduce_dict, is_main_process


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch,
                    max_norm=0.1, scaler=None, ema=None, print_freq=100, lr_scheduler=None):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch [{epoch}]'

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            losses = sum(loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss_dict_reduced.values())
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, evaluator, device):
    model.eval()
    criterion.eval()
    metric_logger = MetricLogger(delimiter='  ')
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        loss_dict_reduced = reduce_dict(loss_dict)
        metric_logger.update(**loss_dict_reduced)

        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)
        res = {t['image_id'].item(): out for t, out in zip(targets, results)}
        if evaluator is not None:
            evaluator.update(res)

    if evaluator is not None:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if evaluator is not None:
        eval_stats = evaluator.get_stats()
        stats['coco_eval_bbox'] = eval_stats.get('bbox', [0] * 12)
    else:
        stats['coco_eval_bbox'] = [0] * 12

    return stats
