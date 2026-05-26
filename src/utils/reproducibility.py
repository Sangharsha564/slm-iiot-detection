"""
Reproducibility utilities — import at the top of every training script.
Usage:
    from src.utils.reproducibility import set_seed
    set_seed(42)
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.
    Must be called before any model initialisation or data splitting.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        # MPS does not have a separate seed function —
        # torch.manual_seed covers it
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[reproducibility] Seed set to {seed}")


def get_device() -> torch.device:
    """
    Returns the best available device: MPS > CUDA > CPU.
    Use this in every training script instead of hardcoding a device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
