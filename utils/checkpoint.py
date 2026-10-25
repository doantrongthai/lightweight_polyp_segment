from collections import OrderedDict


_THOP_BUFFER_NAMES = {"total_ops", "total_params"}


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def clean_state_dict(state_dict):
    clean = OrderedDict()
    stripped = 0
    for key, value in state_dict.items():
        if key.split(".")[-1] in _THOP_BUFFER_NAMES:
            stripped += 1
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        clean[key] = value
    return clean, stripped


def model_state_dict(model):
    state, _ = clean_state_dict(model.state_dict())
    return state


def load_model_weights(model, checkpoint, strict=True):
    state_dict = _extract_state_dict(checkpoint)
    clean, stripped = clean_state_dict(state_dict)
    result = model.load_state_dict(clean, strict=strict)
    if stripped:
        print(f"[checkpoint] Stripped {stripped} THOP profiling keys from weights.")
    return result
