from __future__ import annotations

import json
import logging
import random
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(data: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open('w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def create_run_dir(project_cfg: Dict[str, Any]) -> Path:
    work_root = ensure_dir(project_cfg['work_dir'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = work_root / f"{project_cfg['name']}_train_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def latest_run_dir(project_cfg: Dict[str, Any]) -> Path | None:
    work_root = Path(project_cfg['work_dir'])
    if not work_root.exists():
        return None
    candidates = [path for path in work_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def setup_logger(run_dir: str | Path, name: str = 'rtdetr') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = Path(run_dir) / 'train.log'
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_runtime_config(config: Dict[str, Any], source_config: str | Path, run_dir: str | Path) -> None:
    source = Path(source_config)
    copied_path = Path(run_dir) / source.name
    if source.exists():
        shutil.copy2(source, copied_path)
    dump_yaml(config, Path(run_dir) / 'resolved_config.yaml')


def to_serializable(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {key: to_serializable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_serializable(value) for value in data]
    return data


def dump_json(data: Any, path: str | Path) -> None:
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(to_serializable(data), handle, ensure_ascii=False, indent=2)
