"""Run an ``nn.Sequential`` through the Q.ANT Native Computing Toolkit.

The same code targets the toolkit's ``cpu-backend`` simulator and physical NPUs; which one runs
is decided by the installed wheel and driver, not here. Where each call executes is taken from
the toolkit v2.5.0 source (see docs/sdk-capabilities.md): only matrix-vector products,
element-wise products and the periodic nonlinearity are dispatched to the NPU; bias, ReLU and
sigmoid are computed by the toolkit on the host in bfloat16.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from torch import nn

from photonic_poc.hybrid import (
    AUTO_DEVICE,
    NOOP_TYPES,
    TOOLKIT_DEFAULT,
    HybridExecutor,
    HybridOptions,
)
from photonic_poc.models import Cos, FourierKAN


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
    Cos: LayerSupport("Cos", "native.calc_scaled_periodic_nl_fprop (scale 1)", "NPU nonlinearity"),
    FourierKAN: LayerSupport(
        "FourierKAN",
        "ai.calc_kan_layer_fprop + ai.add_bias_fprop",
        "NPU k*x and cosine; host phase add, sum (bf16) and bias",
    ),
}


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


class QantBackend:
    """Executes a trained model with toolkit calls in bfloat16, exactly as the toolkit would.

    A thin wrapper over ``HybridExecutor`` with the toolkit-default placement policy.
    """

    name = "qant"

    def __init__(
        self,
        model: nn.Sequential,
        device_id: int = AUTO_DEVICE,
        preload_weights: bool = True,
        batch_size: int = 4096,
        options: HybridOptions = TOOLKIT_DEFAULT,
    ) -> None:
        options = replace(options, preload_weights=preload_weights and options.preload_weights)
        self._executor = HybridExecutor(model, options, device_id, batch_size)
        self._qant = import_toolkit()
        self.device_id = device_id
        self.options = self._executor.options
        self.preload_weights = self.options.preload_weights

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        return self._executor.predict_logits(features)

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
            "placement_policy": self.options.label,
            "preloaded_weights": self.preload_weights,
            "available_npus": [list(n) for n in qant.generic.get_available_npus()],
        }
