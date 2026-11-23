"""Run an ``nn.Sequential`` through the Q.ANT Native Computing Toolkit.

The same code targets the toolkit's ``cpu-backend`` simulator and physical NPUs; which one runs
is decided by the installed wheel and driver, not here. Where each call executes is taken from
the toolkit v2.5.0 source (see docs/sdk-capabilities.md): only matrix-vector products,
element-wise products and the periodic nonlinearity are dispatched to the NPU; bias, ReLU and
sigmoid are computed by the toolkit on the host in bfloat16.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from ml_dtypes import bfloat16
from torch import nn

from photonic_poc.models import FourierKAN

AUTO_DEVICE = -1  # toolkit sentinel: first NPU supporting the operation
Step = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LayerSupport:
    layer: str
    sdk_calls: str
    runs_on: str


LAYER_SUPPORT: dict[type[nn.Module], LayerSupport] = {
    nn.Linear: LayerSupport(
        "Linear", "ai.linear_fprop(_preloaded_weights) + ai.add_bias_fprop", "NPU matmul; host bias"
    ),
    nn.ReLU: LayerSupport("ReLU", "ai.relu_fprop", "host (toolkit, bf16)"),
    nn.Sigmoid: LayerSupport("Sigmoid", "ai.sigmoid_fprop", "host (toolkit, bf16)"),
    FourierKAN: LayerSupport(
        "FourierKAN",
        "ai.calc_kan_layer_fprop + ai.add_bias_fprop",
        "NPU k*x and cosine; host phase add, sum (bf16) and bias",
    ),
}
NOOP_TYPES: tuple[type[nn.Module], ...] = (nn.Identity, nn.Dropout)


class UnsupportedLayerError(ValueError):
    pass


def check_supported(model: nn.Module) -> list[str]:
    """List layers the backend cannot execute (empty means runnable). Needs no toolkit."""
    if not isinstance(model, nn.Sequential):
        return [f"{type(model).__name__}: the backend executes nn.Sequential models only"]
    return [
        f"layer {i}: {type(m).__name__}"
        for i, m in enumerate(model)
        if type(m) not in LAYER_SUPPORT and not isinstance(m, NOOP_TYPES)
    ]


def layer_report(model: nn.Sequential) -> list[dict[str, str]]:
    """Per-layer mapping onto toolkit calls, for reports."""
    return [asdict(LAYER_SUPPORT[type(m)]) for m in model if type(m) in LAYER_SUPPORT]


def import_toolkit() -> Any:
    try:
        import qant_native_computing_toolkit as qant
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "qant_native_computing_toolkit is not installed. "
            "Run `uv run python scripts/install_qant_sdk.py` (simulator) or install the "
            "hardware wheel and driver from Q.ANT."
        ) from exc
    return qant


def _bf16(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32).astype(bfloat16))


class QantBackend:
    """Executes a trained model layer by layer with toolkit calls in bfloat16."""

    name = "qant"

    def __init__(
        self,
        model: nn.Sequential,
        device_id: int = AUTO_DEVICE,
        preload_weights: bool = True,
        batch_size: int = 4096,
    ) -> None:
        problems = check_supported(model)
        if problems:
            raise UnsupportedLayerError("; ".join(problems))
        self._qant = import_toolkit()
        self.device_id = device_id
        self.batch_size = batch_size
        self.preload_weights = preload_weights and self._has_capability("MemHandling")
        self._steps = [self._compile(m) for m in model if not isinstance(m, NOOP_TYPES)]

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        outputs = []
        for start in range(0, len(features), self.batch_size):
            x = _bf16(features[start : start + self.batch_size])
            for step in self._steps:
                x = step(x)
            outputs.append(x.astype(np.float32).reshape(-1))
        return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        qant = self._qant
        try:
            driver = str(qant.info.get_driver_info(self.device_id))
        except Exception as exc:  # driver-specific; never fail a report on it
            driver = f"unavailable ({exc})"
        return {
            "backend": self.name,
            "toolkit_version": qant.__version__,
            "driver_info": driver,
            "simulated": driver.startswith("cpu-backend"),
            "device_id": self.device_id,
            "preloaded_weights": self.preload_weights,
            "available_npus": [list(n) for n in qant.generic.get_available_npus()],
        }

    def _has_capability(self, capability: str) -> bool:
        generic = self._qant.generic
        cap = getattr(generic.QantDeviceCapability, capability)
        if self.device_id == AUTO_DEVICE:
            return bool(generic.get_available_npus([cap]))
        return cap in self._qant.info.get_capabilities(self.device_id)

    def _compile(self, module: nn.Module) -> Step:
        ai = self._qant.ai
        device = self.device_id
        if isinstance(module, nn.Linear):
            weight = _bf16(module.weight.detach().numpy())
            bias = None if module.bias is None else _bf16(module.bias.detach().numpy())
            if self.preload_weights:
                generic = self._qant.generic
                handle = generic.upload(weight, generic.DataRole.Weights, device_id=device)

                def matmul(x: np.ndarray) -> np.ndarray:
                    return np.asarray(ai.linear_fprop_preloaded_weights(x, handle))
            else:

                def matmul(x: np.ndarray) -> np.ndarray:
                    return np.asarray(ai.linear_fprop(x, weight, device_id=device))

            if bias is None:
                return matmul
            return lambda x: np.asarray(ai.add_bias_fprop(matmul(x), bias, device_id=device))
        if isinstance(module, FourierKAN):
            phase = _bf16(module.phase.detach().numpy())
            amplitude = _bf16(module.amplitude.detach().numpy())
            ks = _bf16(module.get_buffer("k").numpy())
            kan_bias = _bf16(module.bias.detach().numpy())

            def kan(x: np.ndarray) -> np.ndarray:
                y = ai.calc_kan_layer_fprop(x, phase, amplitude, ks, device_id=device)
                return np.asarray(ai.add_bias_fprop(y, kan_bias, device_id=device))

            return kan
        if isinstance(module, nn.ReLU):
            return lambda x: np.asarray(ai.relu_fprop(x, device_id=device))
        if isinstance(module, nn.Sigmoid):
            return lambda x: np.asarray(ai.sigmoid_fprop(x, device_id=device))
        raise UnsupportedLayerError(type(module).__name__)  # guarded by check_supported
