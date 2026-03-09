from __future__ import annotations

from rtdetr_pytorch.config import load_eval_config
from rtdetr_pytorch.engine import evaluate_only


if __name__ == '__main__':
    config, config_path = load_eval_config()
    evaluate_only(config, config_path)
