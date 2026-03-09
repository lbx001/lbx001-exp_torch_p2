"""YAML utilities for config loading and merging."""
import yaml
import copy


class BaseConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({attrs})'


def load_config(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def merge_config(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
