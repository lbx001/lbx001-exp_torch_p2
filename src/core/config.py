"""Register/create pattern for component registry."""

GLOBAL_CONFIG = {}

__all__ = ['GLOBAL_CONFIG', 'register', 'create']


def register(cls):
    """Class decorator that registers *cls* in GLOBAL_CONFIG by its name."""
    GLOBAL_CONFIG[cls.__name__] = cls
    return cls


def create(name: str, **kwargs):
    """Instantiate a registered class by name."""
    if name not in GLOBAL_CONFIG:
        raise KeyError(f"'{name}' is not registered. Available: {list(GLOBAL_CONFIG.keys())}")
    return GLOBAL_CONFIG[name](**kwargs)
