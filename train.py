from __future__ import annotations

from rtdetr_pytorch.config import load_train_config
from rtdetr_pytorch.engine import train_model


if __name__ == '__main__':
    config, config_path = load_train_config()
    train_model(config, config_path)
