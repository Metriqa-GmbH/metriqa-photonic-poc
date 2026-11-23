"""Model zoo. Every model is an ``nn.Sequential`` of layers the Q.ANT backend can execute.

Architectures are named by a label (``ArchSpec``): ``logreg``, ``mlp-32x16``, ``cos-16``,
``kan-4-g4``. The original names ``logreg``, ``mlp`` and ``fourier`` remain valid aliases.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn

MODEL_NAMES = ("logreg", "mlp", "fourier")
FAMILIES = ("logreg", "mlp", "cos", "kan")
Periodic = Callable[[torch.Tensor], torch.Tensor]


class Cos(nn.Module):
    """Element-wise cosine activation: the toolkit's periodic nonlinearity with unit scale.

    ``periodic`` can be swapped (e.g. for a distorted cosine) to probe sensitivity to the shape
    of the hardware's cosine-like response; it is a plain attribute, not saved with weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.periodic: Periodic = torch.cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.periodic(x)


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
        self.periodic: Periodic = torch.cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.get_buffer("k").to(x.dtype)
        angles = x[:, None, :, None] * k[None, None, None, :] + self.phase[None]
        return (self.amplitude[None] * self.periodic(angles)).sum(dim=(2, 3)) + self.bias

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, grid={self.grid_size}"


@dataclass(frozen=True)
class ArchSpec:
    """An architecture: family, hidden widths and (for ``kan``) the number of frequencies."""

    family: str
    hidden: tuple[int, ...] = ()
    grid: int = 4

    @property
    def label(self) -> str:
        widths = "x".join(str(h) for h in self.hidden)
        if self.family == "logreg":
            return "logreg"
        if self.family == "kan":
            return f"kan-{widths}-g{self.grid}"
        return f"{self.family}-{widths}"

    @classmethod
    def parse(cls, label: str) -> ArchSpec:
        aliases = {"mlp": "mlp-32x16", "fourier": "kan-4-g4"}
        label = aliases.get(label, label)
        if label == "logreg":
            return cls("logreg")
        match = re.fullmatch(r"(mlp|cos|kan)-(\d+(?:x\d+)*)(?:-g(\d+))?", label)
        if match is None or (match[1] == "kan") != (match[3] is not None):
            raise ValueError(f"unknown architecture {label!r}")
        hidden = tuple(int(h) for h in match[2].split("x"))
        return cls(match[1], hidden, int(match[3]) if match[3] else 4)


def build_arch(spec: ArchSpec, n_features: int) -> nn.Sequential:
    widths = [n_features, *spec.hidden]
    layers: list[nn.Module] = []
    if spec.family == "logreg":
        return nn.Sequential(nn.Linear(n_features, 1))
    if spec.family == "kan":
        for a, b in zip(widths, [*spec.hidden, 1], strict=True):
            layers.append(FourierKAN(a, b, spec.grid))
        return nn.Sequential(*layers)
    activation = nn.ReLU if spec.family == "mlp" else Cos
    for a, b in pairwise(widths):
        layers += [nn.Linear(a, b), activation()]
    layers.append(nn.Linear(widths[-1], 1))
    return nn.Sequential(*layers)


def build_model(name: str, n_features: int) -> nn.Sequential:
    """Return an untrained model from a label or alias. Output: one logit per pair (> 0: match)."""
    return build_arch(ArchSpec.parse(name), n_features)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_periodic(model: nn.Module, periodic: Periodic) -> None:
    """Replace the cosine of every ``Cos``/``FourierKAN`` module (in place)."""
    for module in model.modules():
        if isinstance(module, Cos | FourierKAN):
            module.periodic = periodic
