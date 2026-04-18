"""Seed setting, device detection, and other small helpers."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops where possible (may slow things down slightly)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dict_to_namespace(d):
    """Recursively convert a nested dict into a SimpleNamespace for cfg.key.subkey access."""
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def namespace_to_dict(ns):
    """Recursively convert a SimpleNamespace back to a plain dict (for JSON serialisation)."""
    from types import SimpleNamespace
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns
