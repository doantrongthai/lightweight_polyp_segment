import os
import glob
import importlib

_MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def _auto_import():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(glob.glob(os.path.join(model_dir, '*.py'))):
        module_name = os.path.basename(f)[:-3]
        if module_name != '__init__' and not module_name.startswith('_'):
            try:
                importlib.import_module(f'models.{module_name}')
            except ImportError as e:
                print(f"[models] Bỏ qua '{module_name}': thiếu dependency ({e})")


def get_model(name):
    if not _MODEL_REGISTRY:
        _auto_import()
    key = name.lower()
    assert key in _MODEL_REGISTRY, (
        f"Model '{name}' not found. Available: {list(_MODEL_REGISTRY.keys())}"
    )
    return _MODEL_REGISTRY[key]()


def list_models():
    if not _MODEL_REGISTRY:
        _auto_import()
    return list(_MODEL_REGISTRY.keys())
