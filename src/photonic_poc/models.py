"""Demonstration models. Each is an ``nn.Sequential`` of layers the Q.ANT backend can execute."""

from __future__ import annotations

import math

import torch
from torch import nn

MODEL_NAMES = ("logreg", "mlp", "fourier")


class FourierKAN(nn.Module):
    """Learnable Fourier series per input-output connection (a KAN layer with cosine basis).

    ``y_j = b_j + sum_{i,l} a_jil * cos(k_l * x_i + phi_jil)`` with fixed integer frequencies
    ``k_l = 1..grid_size``. This is exactly the form of the toolkit's ``calc_kan_layer_fprop``,
    whose cosine the NPU evaluates optically, so the trained parameters map 1:1 onto the SDK call.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        shape = (out_features, in_features, grid_size)
        # Same initialisation as Q.ANT's QFourier example layer.
        std = math.sqrt(2.0 / ((in_features + out_features) * grid_size))
        self.amplitude = nn.Parameter(torch.randn(shape) * std)
        self.phase = nn.Parameter(torch.empty(shape).uniform_(-math.pi / 2, math.pi / 2))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("k", torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.get_buffer("k").to(x.dtype)
        angles = x[:, None, :, None] * k[None, None, None, :] + self.phase[None]
        return (self.amplitude[None] * torch.cos(angles)).sum(dim=(2, 3)) + self.bias

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, grid={self.grid_size}"


def build_model(name: str, n_features: int) -> nn.Sequential:
    """Return an untrained model. All output one logit per pair (logit > 0 means "match")."""
    if name == "logreg":
        return nn.Sequential(nn.Linear(n_features, 1))
    if name == "mlp":
        return nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    if name == "fourier":
        return nn.Sequential(FourierKAN(n_features, 4), FourierKAN(4, 1))
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
