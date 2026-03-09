"""Optimizer and LR scheduler builders."""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

__all__ = ['build_optimizer', 'build_lr_scheduler']


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """Build AdamW with per-group learning rates.

    Parameters named with 'backbone' use cfg.backbone_lr; all others use cfg.lr.
    Bias and normalization parameters are excluded from weight decay.
    """
    lr = cfg.get('lr', 1e-4)
    backbone_lr = cfg.get('backbone_lr', lr * 0.1)
    weight_decay = cfg.get('weight_decay', 1e-4)

    # Separate parameters into groups
    no_wd_names = ('bias', 'norm', 'bn')

    def _is_no_wd(name: str) -> bool:
        return any(s in name for s in no_wd_names)

    backbone_params, backbone_params_no_wd = [], []
    head_params, head_params_no_wd = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            if _is_no_wd(name):
                backbone_params_no_wd.append(param)
            else:
                backbone_params.append(param)
        else:
            if _is_no_wd(name):
                head_params_no_wd.append(param)
            else:
                head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': backbone_params_no_wd, 'lr': backbone_lr, 'weight_decay': 0.0},
        {'params': head_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': head_params_no_wd, 'lr': lr, 'weight_decay': 0.0},
    ]
    # Remove empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    return optimizer


def build_lr_scheduler(optimizer, cfg, num_steps_per_epoch: int):
    """Build cosine annealing scheduler with linear warmup.

    Warms up for ~1 epoch then cosines over (total_epochs - warmup_epochs).
    """
    total_epochs = cfg.get('epoches', 150)
    warmup_epochs = cfg.get('warmup_epochs', 1)

    warmup_steps = max(1, warmup_epochs * num_steps_per_epoch)
    total_steps = total_epochs  # scheduler steps once per epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=cfg.get('min_lr', 1e-6),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler
