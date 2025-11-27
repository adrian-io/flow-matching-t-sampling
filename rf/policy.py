# rf/policy.py
"""
Policy network producing Beta parameters a,b  (paper notation).
π_φ(t | x0) = Beta(a(x0), b(x0))

The network outputs positive a,b via softplus.
Return shapes: a: (B,), b: (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, in_channels=1, hidden=128):
        """
        Small conv->pool->fc policy network.
        Produces two positive scalars per sample: a and b (paper notation).
        """
        super().__init__()
        # convolutional feature extractor (lightweight)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # global pooling to (B, hidden, 1, 1)
        )
        self.fc = nn.Linear(hidden, 2)  # outputs raw a,b

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns a, b : both (B,)
        """
        B = x.shape[0]
        h = self.conv(x).view(B, -1)  # (B, hidden)
        out = self.fc(h)              # (B, 2)
        # ensure positivity: softplus + small epsilon
        out = F.softplus(out) + 1e-4
        a = out[:, 0]
        b = out[:, 1]
        return a, b
