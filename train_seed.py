"""
Deterministic seed helper.

This provides a single function `set_seed(seed)` which sets seeds for torch, numpy,
and Python's random, and configures cuDNN for deterministic behavior when using PyTorch.

Call `set_seed(...)` once before model initialization/training.

Note: This repository primarily uses TensorFlow. This helper is harmless to include
and can be extended to set TensorFlow determinism if needed.
"""
import random
try:
    import torch
except Exception:
    torch = None
import numpy as np


def set_seed(seed: int):
    """Set deterministic seeds for random, numpy, and (if available) torch.

    Args:
        seed: integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # cuDNN deterministic settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            # If torch is present but some calls fail, continue without crashing
            pass


if __name__ == "__main__":
    # Example: call once before training starts. Change the seed value as desired.
    SEED = 42
    set_seed(SEED)
    print(f"Deterministic seed set to {SEED}. torch available: {torch is not None}")
