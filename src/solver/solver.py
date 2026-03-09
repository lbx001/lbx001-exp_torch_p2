"""Base solver class."""
import torch


class BaseSolver:
    def __init__(self, model, criterion, optimizer, lr_scheduler,
                 train_loader, val_loader, cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

    def fit(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError

    def save_checkpoint(self, path, **kwargs):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            **kwargs,
        }
        torch.save(state, path)
