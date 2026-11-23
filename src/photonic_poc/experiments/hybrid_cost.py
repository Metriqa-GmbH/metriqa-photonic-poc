"""Experiment C: which parts of an entity-matching pipeline could an NPU take over, and when would
that pay off end to end?

H_C1: for the compact matching models, host-side work that stays on the CPU in a hybrid
pipeline (toolkit host ops, dtype conversions, per-call overhead) already costs more than the
whole float32 CPU inference, so no NPU speed can yield an end-to-end gain.
H_C2: an advantage requires layers wide enough that the matrix work removed from the CPU
outweighs that host work; we locate this width analytically from measured host costs and
explicitly assumed NPU parameters.

Measured host costs are representative of a hardware run (the same host code runs); simulator
timings of NPU stages are CPU emulation and are reported only as such.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from photonic_poc.data import GeneratorConfig, generate_splits
from photonic_poc.experiments.config import dataset_from, write_json
from photonic_poc.features import featurize
from photonic_poc.hybrid import HybridExecutor, HybridOptions
from photonic_poc.inference import TorchBackend
from photonic_poc.models import FourierKAN, build_model
from photonic_poc.numpy_forward import compile_numpy
from photonic_poc.op_profile import profile_model
from photonic_poc.qant_backend import import_toolkit
from photonic_poc.stage import FP32, to_bf16
from photonic_poc.sweep import Job, run_jobs

POLICIES = {
    "toolkit": HybridOptions(),
    "fold-bias": HybridOptions(fold_bias=True),
    "host-fp32": HybridOptions(host_precision=FP32),
}
# The toolkit's KAN call hides its host work (phase add, sum) inside one call; the native
# decomposition exposes it, so only that policy gives a measured host cost for KAN models.
KAN_POLICIES = {
    "kan-native": HybridOptions(kan_impl="native", host_precision=FP32),
    "kan-factorized": HybridOptions(kan_impl="factorized"),
}


def _median_seconds(fn: Callable[[], object], repeats: int, min_time: float = 0.02) -> float:
    """Median over ``repeats`` of the mean time per call within a block of >= ``min_time``."""
    fn()
    samples = []
    for _ in range(repeats):
        n, start = 0, time.perf_counter()
        while True:
            fn()
            n += 1
            elapsed = time.perf_counter() - start
            if elapsed >= min_time:
                break
        samples.append(elapsed / n)
    return statistics.median(samples)


def host_microbenchmarks(repeats: int) -> dict[str, Any]:
    qant = import_toolkit()
    ai, native = qant.ai, qant.native
    out: dict[str, Any] = {"conversion_ns_per_element": {}, "toolkit_host_ns_per_element": {}}
    for n in (10_000, 1_000_000):
        x32 = np.random.default_rng(0).standard_normal(n).astype(np.float32)
        x16 = to_bf16(x32)
        out["conversion_ns_per_element"][str(n)] = {
            "fp32_to_bf16": 1e9 * _median_seconds(partial(to_bf16, x32), repeats) / n,
            "bf16_to_fp32": 1e9 * _median_seconds(partial(x16.astype, np.float32), repeats) / n,
        }
        rows = x16.reshape(-1, 100)
        bias = to_bf16(np.ones(100, np.float32))
        out["toolkit_host_ns_per_element"][str(n)] = {
            "relu_fprop": 1e9 * _median_seconds(partial(ai.relu_fprop, x16), repeats) / n,
            "add_bias_fprop": 1e9
            * _median_seconds(partial(ai.add_bias_fprop, rows, bias), repeats)
            / n,
            "numpy_fp32_relu": 1e9 * _median_seconds(partial(np.maximum, x32, 0), repeats) / n,
        }
    one, w = to_bf16(np.ones((1, 1), np.float32)), to_bf16(np.ones((1, 1), np.float32))
    v1 = to_bf16(np.ones(1, np.float32))
    handle = qant.generic.upload(w, qant.generic.DataRole.Weights)
    tiny = nn.Linear(1, 1).eval()
    t1 = torch.ones(1, 1)
    calls = {
        "ai.linear_fprop": lambda: ai.linear_fprop(one, w),
        "ai.linear_fprop_preloaded_weights": lambda: ai.linear_fprop_preloaded_weights(one, handle),
        "native.calc_scaled_periodic_nl_fprop": lambda: native.calc_scaled_periodic_nl_fprop(
            v1, v1
        ),
        "native.mul_elementwise": lambda: native.mul_elementwise(v1, v1),
        "ai.relu_fprop (host)": lambda: ai.relu_fprop(v1),
        "torch nn.Linear(1,1) forward": lambda: tiny(t1),
    }
    with torch.no_grad():
        out["call_overhead_us"] = {k: 1e6 * _median_seconds(f, repeats) for k, f in calls.items()}
    u = np.random.default_rng(1).uniform(-8, 8, 1_000_000).astype(np.float32)
    out["cpu_cos_ns_per_element_numpy"] = (
        1e9 * _median_seconds(partial(np.cos, u), repeats) / u.size
    )
    a = np.random.default_rng(2).standard_normal((256, 1024)).astype(np.float32)
    b = np.random.default_rng(3).standard_normal((1024, 1024)).astype(np.float32)
    seconds = _median_seconds(partial(np.matmul, a, b), repeats)
    out["cpu_matmul_gops_numpy"] = 2 * 256 * 1024 * 1024 / seconds / 1e9  # MAC = 2 ops
    return out


def _cpu_baselines(model: nn.Sequential, x: np.ndarray, repeats: int) -> dict[str, float]:
    """µs per pair: PyTorch (all threads / one thread) and plain NumPy; ``best`` = minimum."""
    forward = compile_numpy(model)
    rows = {
        "torch": _cpu_us_per_pair(model, x, repeats, None),
        "torch_1thread": _cpu_us_per_pair(model, x, repeats, 1),
        "numpy": 1e6 * _median_seconds(partial(forward, x), repeats) / len(x),
    }
    return rows | {"best": min(rows.values())}


def _cpu_us_per_pair(model: nn.Module, x: np.ndarray, repeats: int, threads: int | None) -> float:
    backend = TorchBackend(model)
    previous = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        return 1e6 * _median_seconds(lambda: backend.predict_logits(x), repeats) / len(x)
    finally:
        torch.set_num_threads(previous)


def _stage_breakdown(
    model: nn.Sequential, x: np.ndarray, options: HybridOptions, repeats: int
) -> dict[str, float]:
    executor = HybridExecutor(model, options, batch_size=max(len(x), 1))
    placement = {s.name: s.placement for s in executor.stages}
    samples: list[dict[str, float]] = []
    executor.timed_run(x)  # warm-up
    for _ in range(repeats):
        totals = {"host": 0.0, "convert": 0.0, "npu_emulated": 0.0, "mixed_emulated": 0.0}
        for name, seconds in executor.timed_run(x)[1].items():
            kind = "convert" if name.startswith("convert") else placement[name]
            key = {
                "host": "host",
                "convert": "convert",
                "npu": "npu_emulated",
                "mixed": "mixed_emulated",
            }[kind]
            totals[key] += seconds
        samples.append(totals)
    return {k: 1e6 * statistics.median(s[k] for s in samples) / len(x) for k in samples[0]}


def _project(host_us: float, profile: dict[str, Any], batch: int, sc: dict[str, float]) -> float:
    """Analytical µs per pair: measured host work + assumed call, transfer and NPU compute cost."""
    call = profile["npu_calls"] * sc["call_overhead_us"] / batch
    transfer = 1e6 * profile["transfer_elements"] * 2 / (sc["bandwidth_gbs"] * 1e9)
    compute = 1e6 * profile["npu_ops"] / sc["npu_ops_per_s"]
    return float(host_us + call + transfer + compute)


def _layer_study(widths: list[int], hcfg: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    """Two n x n Linear layers with ReLU (batch 256): measured CPU time vs measured host work."""
    qant, batch, rows = import_toolkit(), 256, []
    for n in widths:
        model = nn.Sequential(nn.Linear(n, n), nn.ReLU(), nn.Linear(n, n)).eval()
        x = np.random.default_rng(0).uniform(0, 1, (batch, n)).astype(np.float32)
        h16, b16 = to_bf16(x), to_bf16(np.ones(n, np.float32))
        host = (
            _median_seconds(partial(to_bf16, x), repeats)  # input conversion
            + 2 * _median_seconds(partial(qant.ai.add_bias_fprop, h16, b16), repeats)
            + _median_seconds(partial(qant.ai.relu_fprop, h16), repeats)
            + _median_seconds(partial(h16.astype, np.float32), repeats)
        )
        profile = profile_model(model)
        cpu = _cpu_baselines(model, x, repeats)["best"]
        host_us = 1e6 * host / batch
        row: dict[str, Any] = {
            "width": n,
            "cpu_fp32_us_per_pair": cpu,
            "host_us_per_pair": host_us,
            "npu_ops_per_pair": profile["npu_ops"],
            "max_speedup_bound": cpu / host_us,
        }
        for sc in hcfg["scenarios"]:
            row[f"speedup_{sc['name']}"] = cpu / _project(host_us, profile, batch, sc)
            free = _project(host_us, profile, batch, sc | {"npu_ops_per_s": float("inf")})
            row[f"required_npu_ops_per_s_{sc['name']}"] = (
                profile["npu_ops"] / ((cpu - free) * 1e-6) if cpu > free else None
            )
        rows.append(row)
    return rows


def run(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    hcfg, repeats = cfg["hybrid"], cfg["hybrid"]["repeats"]
    data = dataset_from(cfg)
    arch = json.loads((out / "architecture.json").read_text(encoding="utf-8"))
    lrs = arch["training"]["selected_learning_rates"]
    names = [
        n
        for n in dict.fromkeys(
            [
                "logreg",
                "mlp-32x16",
                "kan-4-g4",
                *(p for picks in arch["champions"].values() for p in picks.values()),
            ]
        )
        if n in lrs
    ]
    seed0, epochs = cfg["training"]["seeds"][0], cfg["training"]["epochs"]
    trained = run_jobs([Job(n, lrs[n], seed0, epochs) for n in names], data, 1)
    pairs = generate_splits(GeneratorConfig(n_entities=1000, seed=1))["test"][
        : hcfg["feature_pairs"]
    ]
    feature_us = 1e6 * _median_seconds(partial(featurize, pairs), 3, 0.0) / len(pairs)
    models: dict[str, Any] = {}
    for job, run_ in trained.items():
        model = run_.model
        record: dict[str, Any] = {}
        is_kan = any(isinstance(m, FourierKAN) for m in model.modules())
        policies = POLICIES | (KAN_POLICIES if is_kan else {})
        record["profiles"] = {
            p: {k: v for k, v in profile_model(model, o).items() if k != "stages"}
            for p, o in policies.items()
        }
        record["batches"] = []
        for batch in hcfg["batch_sizes"]:
            x = data.x_test[:batch]
            cpu = _cpu_baselines(model, x, repeats)
            row: dict[str, Any] = {"batch": batch, "cpu_us_per_pair": cpu}
            for p, o in policies.items():
                stages = _stage_breakdown(model, x, o, repeats)
                host_us = stages["host"] + stages["convert"]
                # A "mixed" toolkit call hides host work inside one call: host cost incomplete.
                complete = stages["mixed_emulated"] == 0
                row[p] = stages | {
                    "host_measured_us": host_us,
                    "host_measured_complete": complete,
                    "max_speedup_bound": cpu["best"] / host_us if complete else None,
                }
                for sc in hcfg["scenarios"]:
                    projected = _project(host_us, record["profiles"][p], batch, sc)
                    row[p][f"speedup_{sc['name']}"] = cpu["best"] / projected if complete else None
            record["batches"].append(row)
        models[job.arch] = record
    result = {
        "hypotheses": __doc__.split("\n\n")[1],
        "scenarios": hcfg["scenarios"],
        "host_microbenchmarks": host_microbenchmarks(repeats),
        "feature_extraction_us_per_pair": feature_us,
        "models": models,
        "layer_study": _layer_study(hcfg["layer_widths"], hcfg, repeats),
        "all_architectures": [
            {"arch": r["arch"], "family": r["family"], "fp32_f1": r["fp32_f1"]["mean"]}
            | {
                k: v
                for k, v in profile_model_for(r["arch"], data.n_features).items()
                if k != "stages"
            }
            for r in arch["records"]
        ],
    }
    write_json(out / "hybrid.json", result)
    return result


def profile_model_for(arch: str, n_features: int) -> dict[str, Any]:
    return profile_model(build_model(arch, n_features), HybridOptions(fold_bias=True))
