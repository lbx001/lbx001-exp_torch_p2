"""YAMLConfig: loads a flat dict from yaml and exposes as attributes."""
import yaml
import copy
from .config import GLOBAL_CONFIG, create


class YAMLConfig:
    def __init__(self, cfg_path: str, **kwargs):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        self._cfg = cfg
        for k, v in kwargs.items():
            self._cfg[k] = v

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self._cfg.get(name)

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def __repr__(self):
        return f'YAMLConfig({self._cfg})'
