"""Hybrid CPU/NPU inference in five steps: train, profile, place, run, time.

    uv run python examples/hybrid_inference.py

Steps 1-2 need only PyTorch. Steps 3-5 need the Q.ANT toolkit (simulator or hardware); on the
simulator, NPU-stage timings are CPU emulation and say nothing about NPU speed.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from photonic_poc.data import GeneratorConfig
from photonic_poc.datasets import load_dataset
from photonic_poc.hybrid import HybridOptions
from photonic_poc.inference import TorchBackend
from photonic_poc.op_profile import profile_model
from photonic_poc.training import TrainConfig, train_model

POLICIES = {
    "toolkit": HybridOptions(),  # what the toolkit does on its own
    "fold-bias": HybridOptions(fold_bias=True),  # bias as an extra weight column, on the NPU
    "kan-factorized": HybridOptions(kan_impl="factorized"),  # Fourier basis + one matmul
}

# 1. A small synthetic dataset and two compact models.
data = load_dataset(GeneratorConfig(n_entities=1500, seed=3), cache_dir=None)
models = {
    arch: train_model(
        arch, data.x_train, data.y_train, data.x_val, data.y_val, TrainConfig(epochs=15)
    ).model
    for arch in ("cos-16", "kan-4-g4")
}

# 2. Static placement and cost per pair (no toolkit needed).
for arch, model in models.items():
    for name, options in POLICIES.items():
        p = profile_model(model, options)
        print(
            f"{arch:9} {name:15} NPU share {p['offload_fraction']:6.1%}"
            f"  NPU calls {p['npu_calls']}  elements over bus {p['transfer_elements']:5d}"
        )

if importlib.util.find_spec("qant_native_computing_toolkit") is None:
    raise SystemExit("toolkit not installed: run scripts/install_qant_sdk.py for steps 3-5")

from photonic_poc.hybrid import HybridExecutor  # noqa: E402  (needs the toolkit)

x = data.x_test[:256]
for arch, model in models.items():
    reference = TorchBackend(model).predict_logits(x)
    for name, options in POLICIES.items():
        # 3. Place every layer according to the policy; 4. run it through the toolkit.
        executor = HybridExecutor(model, options)
        logits, seconds = executor.timed_run(x)
        dev = np.sqrt(np.mean((logits - reference) ** 2) / np.mean(reference**2))
        # 5. Time per stage: host and conversion stages are real host costs, NPU stages emulated.
        placement = {s.name: s.placement for s in executor.stages}
        host = sum(
            t for k, t in seconds.items() if placement.get(k) == "host" or k.startswith("convert")
        )
        hidden = any(s.placement == "mixed" for s in executor.stages)
        note = " (+ host work hidden inside the toolkit's KAN call)" if hidden else ""
        print(
            f"{arch:9} {name:15} rel. RMS vs float32 {dev:6.2%}"
            f"  host work {1e6 * host / len(x):6.3f} µs/pair{note}"
        )
