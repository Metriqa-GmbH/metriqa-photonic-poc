"""Model x backend benchmark: matching quality, agreement with float32, size, host latency.

This is the integration point for private code: pass any ``nn.Sequential`` built from supported
layers plus a feature matrix and labels. Nothing here knows how features were computed, so
proprietary features, data and weights never have to enter this repository.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from photonic_poc.inference import InferenceBackend, NoisyTorchBackend, create_backend
from photonic_poc.metrics import classification_metrics, output_agreement
from photonic_poc.models import count_parameters

REFERENCE = "torch-fp32"


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    try:
        import qant_native_computing_toolkit as qant

        info["qant_toolkit"] = qant.__version__
        info["qant_driver"] = str(qant.info.get_driver_info())
    except Exception as exc:  # toolkit is optional
        info["qant_toolkit"] = f"unavailable ({type(exc).__name__})"
    return info


def time_backend(
    backend: InferenceBackend, features: np.ndarray, repeats: int = 5
) -> dict[str, float]:
    """Median host wall-clock time. For the simulator this measures CPU emulation, not an NPU."""
    single = features[:1]
    backend.predict_logits(single)  # warm-up (weight upload, lazy init)
    one = [_seconds(backend, single) for _ in range(max(repeats * 10, 20))]
    full = [_seconds(backend, features) for _ in range(repeats)]
    return {
        "batch1_ms": 1e3 * statistics.median(one),
        "full_batch_ms": 1e3 * statistics.median(full),
        "full_batch_size": float(len(features)),
        "us_per_pair": 1e6 * statistics.median(full) / max(len(features), 1),
    }


def _seconds(backend: InferenceBackend, features: np.ndarray) -> float:
    start = time.perf_counter()
    backend.predict_logits(features)
    return time.perf_counter() - start


def accuracy_by_kind(
    labels: np.ndarray, logits: np.ndarray, kinds: Sequence[str]
) -> dict[str, float]:
    kind_array = np.asarray(kinds)
    correct = (logits > 0) == labels.astype(bool)
    return {k: float(np.mean(correct[kind_array == k])) for k in sorted(set(kinds))}


def run_benchmark(
    models: Mapping[str, nn.Module],
    features: np.ndarray,
    labels: np.ndarray,
    backends: Sequence[str] = (REFERENCE, "torch-bf16", "qant"),
    kinds: Sequence[str] | None = None,
    latency_repeats: int = 5,
) -> dict[str, Any]:
    """Evaluate every model on every backend. Returns a JSON-serialisable report."""
    rows: list[dict[str, Any]] = []
    backend_info: dict[str, Any] = {}
    for model_name, model in models.items():
        reference = create_backend(REFERENCE, model).predict_logits(features)
        for backend_name in backends:
            backend = create_backend(backend_name, model)
            backend_info.setdefault(backend_name, backend.describe())
            logits = backend.predict_logits(features)
            row: dict[str, Any] = {
                "model": model_name,
                "backend": backend_name,
                "parameters": count_parameters(model),
                "metrics": classification_metrics(labels, logits),
                "agreement_vs_fp32": output_agreement(reference, logits),
                "latency": time_backend(backend, features, latency_repeats),
            }
            if kinds is not None:
                row["accuracy_by_kind"] = accuracy_by_kind(labels, logits, kinds)
            rows.append(row)
    return {"environment": environment_info(), "backends": backend_info, "rows": rows}


def noise_sweep(
    models: Mapping[str, nn.Module],
    features: np.ndarray,
    labels: np.ndarray,
    stds: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2),
    seeds: int = 5,
) -> list[dict[str, Any]]:
    """F1 under an assumed additive Gaussian noise at NPU-layer outputs (not hardware data)."""
    results = []
    for model_name, model in models.items():
        for std in stds:
            f1s = [
                classification_metrics(
                    labels, NoisyTorchBackend(model, std, seed).predict_logits(features)
                )["f1"]
                for seed in range(seeds if std > 0 else 1)
            ]
            results.append(
                {
                    "model": model_name,
                    "noise_std": std,
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "seeds": len(f1s),
                }
            )
    return results
