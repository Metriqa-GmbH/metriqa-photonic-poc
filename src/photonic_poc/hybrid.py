"""Hybrid CPU/NPU execution of an ``nn.Sequential`` through the Q.ANT toolkit, stage by stage.

A model compiles into a list of stages, each placed on the NPU (a toolkit call the driver
executes) or on the host (toolkit host code in bf16, or NumPy in float32). ``HybridOptions``
selects the placement policy; the default reproduces exactly what the toolkit does on its own.
Dtype conversions between stages are explicit stages too, so their cost is visible.

Placement facts come from the toolkit v2.5.0 source (docs/sdk-capabilities.md).
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from ml_dtypes import bfloat16
from torch import nn

from photonic_poc.kan_paths import (
    append_ones,
    factorized_mix_matrix,
    kan_factorized_stages,
    kan_native_stages,
)
from photonic_poc.models import Cos, FourierKAN
from photonic_poc.stage import BF16, FP32, Stage, to_bf16, wrap_phase

AUTO_DEVICE = -1
NOOP_TYPES: tuple[type[nn.Module], ...] = (nn.Identity, nn.Dropout)


@dataclass(frozen=True)
class HybridOptions:
    """Placement policy. The defaults are the toolkit's own behaviour."""

    fold_bias: bool = False  # Linear bias as an extra weight column, so it runs on the NPU
    host_precision: str = BF16  # host-side bias/ReLU/sigmoid: toolkit bf16 or NumPy float32
    # KAN layers: "toolkit" = ai.calc_kan_layer_fprop; "native" = same math from NPU primitives,
    # summed in float32; "factorized" = fixed Fourier basis on the NPU + one preloaded matmul.
    kan_impl: str = "toolkit"
    wrap_phases: bool = False  # map KAN phases into [-pi, pi) before export (exact for cos)
    preload_weights: bool = True  # keep Linear weights in NPU memory when supported

    @property
    def label(self) -> str:
        parts = [
            "fold-bias" if self.fold_bias else "",
            "host-fp32" if self.host_precision == FP32 else "",
            f"kan-{self.kan_impl}" if self.kan_impl != "toolkit" else "",
            "wrap-phases" if self.wrap_phases else "",
            "" if self.preload_weights else "streamed",
        ]
        return "+".join(p for p in parts if p) or "toolkit"


TOOLKIT_DEFAULT = HybridOptions()


class HybridExecutor:
    """Runs a model with toolkit calls according to ``options``; can time every stage."""

    def __init__(
        self,
        model: nn.Sequential,
        options: HybridOptions = TOOLKIT_DEFAULT,
        device_id: int = AUTO_DEVICE,
        batch_size: int = 4096,
    ) -> None:
        from photonic_poc.qant_backend import UnsupportedLayerError, check_supported, import_toolkit

        problems = check_supported(model)
        if problems:
            raise UnsupportedLayerError("; ".join(problems))
        self._qant = import_toolkit()
        self.device_id = device_id
        self.batch_size = batch_size
        mem = self._has_capability("MemHandling")
        self.options = replace(options, preload_weights=options.preload_weights and mem)
        self._ones: dict[int, np.ndarray] = {}
        self.stages: list[Stage] = []
        for index, module in enumerate(model):
            if not isinstance(module, NOOP_TYPES):
                self.stages += self._compile(index, module)

    # --- execution -----------------------------------------------------------------------
    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        return self._execute(features, None)

    def timed_run(self, features: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Run once and return (logits, seconds per stage); conversions appear as ``convert``."""
        timings: dict[str, float] = defaultdict(float)
        logits = self._execute(features, timings)
        return logits, dict(timings)

    def _execute(self, features: np.ndarray, timings: dict[str, float] | None) -> np.ndarray:
        outputs = []
        for start in range(0, len(features), self.batch_size):
            x = np.ascontiguousarray(features[start : start + self.batch_size], dtype=np.float32)
            dtype = FP32
            for stage in self.stages:
                if stage.dtype_in != dtype:
                    x = self._timed(
                        timings, f"convert->{stage.dtype_in}", _converter(stage.dtype_in), x
                    )
                x = self._timed(timings, stage.name, stage.run, x)
                dtype = stage.dtype_out
            if dtype != FP32:
                x = self._timed(timings, "convert->fp32", _converter(FP32), x)
            outputs.append(x.reshape(-1))
        return np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)

    @staticmethod
    def _timed(
        timings: dict[str, float] | None,
        key: str,
        fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
    ) -> np.ndarray:
        if timings is None:
            return fn(x)
        start = time.perf_counter()
        out = fn(x)
        timings[key] += time.perf_counter() - start
        return out

    def _has_capability(self, capability: str) -> bool:
        generic = self._qant.generic
        cap = getattr(generic.QantDeviceCapability, capability)
        if self.device_id == AUTO_DEVICE:
            return bool(generic.get_available_npus([cap]))
        return cap in self._qant.info.get_capabilities(self.device_id)

    def _ones_like(self, n: int) -> np.ndarray:
        if n not in self._ones:
            self._ones[n] = np.ones(n, dtype=bfloat16)
        return self._ones[n]

    # --- compilation ---------------------------------------------------------------------
    def _compile(self, index: int, module: nn.Module) -> list[Stage]:
        if isinstance(module, nn.Linear):
            return self._linear(f"{index}:linear", module)
        if isinstance(module, FourierKAN):
            return self._kan(f"{index}:kan", module)
        if isinstance(module, Cos):
            return [self._cos(f"{index}:cos")]
        if isinstance(module, nn.ReLU):
            return [self._elementwise(f"{index}:relu", "relu")]
        if isinstance(module, nn.Sigmoid):
            return [self._elementwise(f"{index}:sigmoid", "sigmoid")]
        raise TypeError(type(module).__name__)  # guarded by check_supported

    def _matmul(self, name: str, weight: np.ndarray) -> Stage:
        ai, generic, device = self._qant.ai, self._qant.generic, self.device_id
        w = to_bf16(weight)
        if self.options.preload_weights:
            handle = generic.upload(w, generic.DataRole.Weights, device_id=device)

            def run(x: np.ndarray) -> np.ndarray:
                return np.asarray(ai.linear_fprop_preloaded_weights(x, handle))
        else:

            def run(x: np.ndarray) -> np.ndarray:
                return np.asarray(ai.linear_fprop(x, w, device_id=device))

        return Stage(f"{name}.matmul", "matmul", "npu", BF16, BF16, run)

    def _linear(self, name: str, module: nn.Linear) -> list[Stage]:
        weight = module.weight.detach().numpy()
        if module.bias is None:
            return [self._matmul(name, weight)]
        bias = module.bias.detach().numpy()
        if self.options.fold_bias:
            concat = Stage(f"{name}.concat", "concat", "host", BF16, BF16, append_ones)
            return [concat, self._matmul(name, np.concatenate([weight, bias[:, None]], axis=1))]
        return [self._matmul(name, weight), self._bias(name, bias)]

    def _bias(self, name: str, bias: np.ndarray) -> Stage:
        if self.options.host_precision == FP32:
            b32 = bias.astype(np.float32)
            return Stage(f"{name}.bias", "bias", "host", FP32, FP32, lambda x: x + b32)
        ai, device, b16 = self._qant.ai, self.device_id, to_bf16(bias)

        def run(x: np.ndarray) -> np.ndarray:
            return np.asarray(ai.add_bias_fprop(x, b16, device_id=device))

        return Stage(f"{name}.bias", "bias", "host", BF16, BF16, run)

    def _elementwise(self, name: str, op: str) -> Stage:
        if self.options.host_precision == FP32:
            fn32 = (
                (lambda x: np.maximum(x, 0)) if op == "relu" else (lambda x: 1 / (1 + np.exp(-x)))
            )
            return Stage(name, op, "host", FP32, FP32, fn32)
        ai, device = self._qant.ai, self.device_id
        toolkit_fn = ai.relu_fprop if op == "relu" else ai.sigmoid_fprop
        return Stage(
            name, op, "host", BF16, BF16, lambda x: np.asarray(toolkit_fn(x, device_id=device))
        )

    def _cos(self, name: str) -> Stage:
        native, device = self._qant.native, self.device_id

        def run(x: np.ndarray) -> np.ndarray:
            flat = np.ascontiguousarray(x.reshape(-1))
            out = native.calc_scaled_periodic_nl_fprop(
                flat, self._ones_like(flat.size), device_id=device
            )
            return np.asarray(out).reshape(x.shape)

        return Stage(name, "cos", "npu", BF16, BF16, run)

    def _kan(self, name: str, module: FourierKAN) -> list[Stage]:
        phase = module.phase.detach().numpy()
        if self.options.wrap_phases:
            phase = wrap_phase(phase)
        amplitude = module.amplitude.detach().numpy()
        ks = module.get_buffer("k").numpy()
        bias_stage = self._bias(name, module.bias.detach().numpy())
        if self.options.kan_impl == "factorized":
            mix = factorized_mix_matrix(phase, amplitude, module.bias.detach().numpy())
            return kan_factorized_stages(self._qant, self.device_id, name, ks, n_in=phase.shape[1],
                                         matmul=self._matmul(name, mix))  # fmt: skip
        if self.options.kan_impl == "toolkit":
            ai, device = self._qant.ai, self.device_id
            p16, a16, k16 = to_bf16(phase), to_bf16(amplitude), to_bf16(ks)

            def kan(x: np.ndarray) -> np.ndarray:
                return np.asarray(ai.calc_kan_layer_fprop(x, p16, a16, k16, device_id=device))

            return [Stage(f"{name}.kan", "kan", "mixed", BF16, BF16, kan), bias_stage]
        return [
            *kan_native_stages(self._qant, self.device_id, name, phase, amplitude, ks),
            bias_stage,
        ]


def _converter(dtype: str) -> Callable[[np.ndarray], np.ndarray]:
    if dtype == BF16:
        return to_bf16
    return lambda x: np.asarray(x, dtype=np.float32)


def describe_stages(executor: HybridExecutor) -> list[dict[str, Any]]:
    return [
        {"name": s.name, "op": s.op, "placement": s.placement, "dtype_in": s.dtype_in}
        for s in executor.stages
    ]
