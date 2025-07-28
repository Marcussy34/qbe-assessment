"""
Utility functions for the neural network project.
Includes reproducibility framework and helper functions.
"""

import random
import numpy as np
import torch
import os


def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seeds set to {seed} for reproducibility")


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def print_system_info():
    """Print system information for reproducibility documentation."""
    print("=== System Information ===")
    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("===========================") 