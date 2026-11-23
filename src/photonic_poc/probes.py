"""Probe inputs and assumed hardware perturbations for robustness studies.

These are tools for asking "what if"; none of them is a model of measured Q.ANT hardware.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from photonic_poc.models import Periodic


def uniform_inputs(n: int, n_features: int, seed: int = 0) -> np.ndarray:
    """Off-manifold inputs: every feature independently uniform on [0, 1]."""
    return np.random.default_rng(seed).uniform(0, 1, (n, n_features)).astype(np.float32)


def boundary_inputs(
    predict: Callable[[np.ndarray], np.ndarray],
    features: np.ndarray,
    n: int,
    seed: int = 0,
    jitter: float = 0.02,
    steps: int = 30,
) -> np.ndarray:
    """Points near the decision boundary of ``predict`` (logits), found by bisection.

    Each point lies on the segment between a predicted-positive and a predicted-negative input,
    at the crossing found by bisection plus a small uniform offset along the segment, so the
    sample is dense in small margins where numerical decision flips can happen.
    """
    rng = np.random.default_rng(seed)
    logits = predict(features)
    pos, neg = features[logits > 0], features[logits <= 0]
    a = pos[rng.integers(len(pos), size=n)]
    b = neg[rng.integers(len(neg), size=n)]
    lo, hi = np.zeros((n, 1), np.float32), np.ones((n, 1), np.float32)
    for _ in range(steps):
        mid = (lo + hi) / 2
        positive = predict(a + mid * (b - a)) > 0
        lo = np.where(positive[:, None], mid, lo)
        hi = np.where(positive[:, None], hi, mid)
    t = np.clip((lo + hi) / 2 + rng.uniform(-jitter, jitter, (n, 1)), 0, 1).astype(np.float32)
    return np.asarray(a + t * (b - a), dtype=np.float32)


def distorted_cos(epsilon: float) -> Periodic:
    """``(cos u + eps cos 2u) / (1 + eps)``: a 2π-periodic stand-in for a non-ideal cosine.

    Second-harmonic distortion is one plausible deviation of an optical cosine response; the
    true shape of the hardware ``tcos`` is not public.
    """

    def periodic(u: torch.Tensor) -> torch.Tensor:
        return (torch.cos(u) + epsilon * torch.cos(2 * u)) / (1 + epsilon)

    return periodic
