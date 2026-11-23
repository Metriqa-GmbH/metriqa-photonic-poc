"""Backend-neutral inference interface and the PyTorch reference backends.

A backend wraps a trained ``nn.Sequential`` and maps float32 features ``(n, F)`` to float32
logits ``(n,)``. Benchmarks only talk to this protocol, so a private model or a new hardware
backend plugs in without changes to the evaluation code.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from photonic_poc.models import Cos, FourierKAN, Periodic, set_periodic

BACKEND_NAMES = ("torch-fp32", "torch-bf16", "qant")

# Layers that would run on the NPU's optical path; noise is injected at their outputs.
NPU_LAYER_TYPES: tuple[type[nn.Module], ...] = (nn.Linear, FourierKAN, Cos)


class InferenceBackend(Protocol):
    name: str

    def predict_logits(self, features: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


class TorchBackend:
    """Plain PyTorch inference in float32 (the reference) or bfloat16 (the precision control)."""

    def __init__(
        self, model: nn.Module, dtype: torch.dtype = torch.float32, periodic: Periodic | None = None
    ) -> None:
        self.dtype = dtype
        self.name = "torch-fp32" if dtype == torch.float32 else "torch-bf16"
        self.model = copy.deepcopy(model).to(dtype).eval()
        if periodic is not None:  # e.g. a distorted cosine standing in for the hardware tcos
            set_periodic(self.model, periodic)
            self.name += "-periodic"

    @torch.no_grad()
    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(self.dtype)
        out: torch.Tensor = self.model(x)
        return out.float().reshape(-1).numpy()

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "torch": torch.__version__, "dtype": str(self.dtype)}


class NoisyTorchBackend(TorchBackend):
    """float32 inference with additive Gaussian noise at every NPU-layer output.

    An *assumed* noise model for robustness studies; it is not derived from hardware.
    """

    def __init__(self, model: nn.Module, std: float, seed: int = 0) -> None:
        super().__init__(model, torch.float32)
        self.std = std
        self.seed = seed
        self.name = f"torch-noise-{std:g}"

    @torch.no_grad()
    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        generator = torch.Generator().manual_seed(self.seed)
        with output_noise(self.model, self.std, generator):
            return super().predict_logits(features)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "noise_std": self.std, "noise_model": "additive gaussian"}


@contextmanager
def output_noise(
    model: nn.Module, std: float, generator: torch.Generator | None = None
) -> Iterator[None]:
    """Add ``N(0, std^2)`` noise to the outputs of all NPU-layer modules while active."""

    def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(output.shape, generator=generator, dtype=output.dtype)
        return output + std * noise

    handles = [m.register_forward_hook(hook) for m in model.modules() if _is_npu_layer(m)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _is_npu_layer(module: nn.Module) -> bool:
    return isinstance(module, NPU_LAYER_TYPES)


def create_backend(name: str, model: nn.Module, **options: Any) -> InferenceBackend:
    """Build a backend by name. ``qant`` imports the toolkit lazily (optional dependency)."""
    factories: dict[str, Callable[[], InferenceBackend]] = {
        "torch-fp32": lambda: TorchBackend(model, torch.float32),
        "torch-bf16": lambda: TorchBackend(model, torch.bfloat16),
    }
    if name == "qant":
        from photonic_poc.qant_backend import QantBackend

        if not isinstance(model, nn.Sequential):
            raise TypeError("the qant backend needs an nn.Sequential model")
        return QantBackend(model, **options)
    if name not in factories:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKEND_NAMES}")
    return factories[name]()
