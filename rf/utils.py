# rf/utils.py
import torch

def clamp_01(t):
    """Clamp t into [0, 1) as in your previous code."""
    return torch.clamp(t, min=0.0, max=1.0 - 1e-6)
