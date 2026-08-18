"""Random seeding utilities for ALIVE benchmarks."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set the global random seed for reproducible benchmark runs.

    Covers Python ``random``, NumPy, and PyTorch (CPU and CUDA), and enables
    deterministic cuDNN kernels. Pandas has no global seed and inherits NumPy's.

    Args:
        seed: Integer seed applied to all supported random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN trades a little speed for fully reproducible results.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"🌱 Global seed set to: {seed} (Deterministic mode enabled)")
