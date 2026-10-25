import os
import glob
import importlib

import torch.nn as nn

_LOSS_REGISTRY = {}


def register_loss(name):
    def decorator(cls):
        _LOSS_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def _auto_import():
    loss_dir = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(os.path.join(loss_dir, '*.py'))):
        module_name = os.path.basename(f)[:-3]
        if module_name != '__init__' and not module_name.startswith('_'):
            importlib.import_module(f'losses.{module_name}')


def get_loss(name: str) -> nn.Module:
    if not _LOSS_REGISTRY:
        _auto_import()
    key = name.lower()
    assert key in _LOSS_REGISTRY, (
        f"Loss '{name}' not found. Available: {list_losses()}"
    )
    return _LOSS_REGISTRY[key]()


def list_losses():
    if not _LOSS_REGISTRY:
        _auto_import()
    return list(_LOSS_REGISTRY.keys())
