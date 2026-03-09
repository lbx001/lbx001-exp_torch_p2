from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .utils import deep_update, load_yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / 'config.yaml'
DEFAULT_EVAL_CONFIG_PATH = REPO_ROOT / 'eval_config.yaml'


def load_train_config(config_path: str | Path | None = None) -> tuple[Dict[str, Any], Path]:
    config_file = Path(os.environ.get('RTDETR_CONFIG_PATH', config_path or DEFAULT_CONFIG_PATH)).resolve()
    cfg = load_yaml(config_file)
    return cfg, config_file



def load_eval_config(config_path: str | Path | None = None) -> tuple[Dict[str, Any], Path]:
    eval_file = Path(os.environ.get('RTDETR_EVAL_CONFIG_PATH', config_path or DEFAULT_EVAL_CONFIG_PATH)).resolve()
    eval_cfg = load_yaml(eval_file)
    base_path = Path(eval_cfg.get('base_config_path') or os.environ.get('RTDETR_CONFIG_PATH', DEFAULT_CONFIG_PATH)).resolve()
    base_cfg = load_yaml(base_path)
    merged = deep_update(base_cfg, eval_cfg.get('overrides', {}))
    merged['evaluation'] = deep_update(base_cfg.get('evaluation', {}), eval_cfg.get('evaluation', {}))
    merged['inference'] = deep_update(base_cfg.get('inference', {}), eval_cfg.get('inference', {}))
    merged['checkpoint'] = eval_cfg.get('checkpoint', {})
    return merged, eval_file
