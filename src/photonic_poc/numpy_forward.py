"""Plain NumPy float32 inference: a CPU baseline without deep-learning framework overhead.

PyTorch eager mode costs microseconds per layer call, which dominates at small batches and would
flatter any alternative pipeline. This baseline runs the same model as a few NumPy calls.
Fourier-KAN layers use the exact factorised form (Fourier basis, then one matrix product).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch import nn

from photonic_poc.kan_paths import factorized_mix_matrix
from photonic_poc.models import Cos, FourierKAN

Fn = Callable[[np.ndarray], np.ndarray]


def _layer(module: nn.Module) -> Fn | None:
    if isinstance(module, nn.Linear):
        w = module.weight.detach().numpy().T.copy()
        b = None if module.bias is None else module.bias.detach().numpy().copy()
        return (lambda x: x @ w) if b is None else (lambda x: x @ w + b)
    if isinstance(module, nn.ReLU):
        return lambda x: np.maximum(x, 0)
    if isinstance(module, Cos):
        return np.cos
    if isinstance(module, nn.Sigmoid):
        return lambda x: 1 / (1 + np.exp(-x))
    if isinstance(module, FourierKAN):
        k = module.get_buffer("k").numpy()
        mix = factorized_mix_matrix(
            module.phase.detach().numpy(),
            module.amplitude.detach().numpy(),
            module.bias.detach().numpy(),
        )
        w, b = mix[:, :-1].T.copy(), mix[:, -1].copy()

        def kan(x: np.ndarray) -> np.ndarray:
            kx = x[:, :, None] * k
            basis = np.stack([np.cos(kx), np.sin(kx)], axis=-1).reshape(x.shape[0], -1)
            return np.asarray(basis @ w + b)

        return kan
    if isinstance(module, nn.Identity | nn.Dropout):
        return None
    raise TypeError(f"no NumPy implementation for {type(module).__name__}")


def compile_numpy(model: nn.Sequential) -> Fn:
    """Return ``features (n, F) float32 -> logits (n,)``."""
    layers = [f for f in (_layer(m) for m in model) if f is not None]

    def forward(x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        for f in layers:
            h = f(h)
        return h.reshape(-1)

    return forward
