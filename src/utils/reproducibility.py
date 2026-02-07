"""
Reproducibility utilities for consistent experiment results.

This module provides functions to configure the environment for reproducible
experiments by setting seeds across all random number generators.

Usage:
    from src.utils import configure_environment
    configure_environment(seed=42)  # ALWAYS call before data loading
"""

import os
import random
import warnings
from typing import Optional

import numpy as np

_current_seed: Optional[int] = None


def configure_environment(seed: int = 42, suppress_warnings: bool = True) -> None:
    """Configure the environment for reproducible experiments.

    Sets seeds for random, numpy, torch (if available), tensorflow (if available),
    and Python hash randomization. Should be called BEFORE any data loading or
    model initialization.

    Args:
        seed: Random seed for all generators.
        suppress_warnings: Whether to suppress FutureWarning and other warnings.

    Example:
        >>> configure_environment(seed=42)
        >>> # Now load data and train models...
    """
    global _current_seed
    _current_seed = seed

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Matplotlib config directory (avoid permission issues)
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "configs")

    # TensorFlow logging level (suppress verbose output)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed

    # TensorFlow (if available)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed

    # Suppress warnings if requested
    if suppress_warnings:
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=Warning)


def get_seed() -> Optional[int]:
    """Get the currently configured seed.

    Returns:
        The seed set by configure_environment, or None if not configured.
    """
    return _current_seed
