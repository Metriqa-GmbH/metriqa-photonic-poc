"""Experiment D: the matching models on a real electronic NPU (AMD XDNA 2, Windows ML).

H_D1: A16W8 execution on the NPU keeps test F1 within 0.002 of float32 for every model.
H_D2: per-call overhead dominates compact models, so the NPU is slower than the CPU at every
batch size for them; it can win only where per-pair work is large and batches are large.

A real-hardware check of Experiment C's cost structure, not a statement about photonic NPUs.
Reuses the models trained in Experiment A.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from photonic_poc import amd_npu
from photonic_poc.experiments.config import dataset_from, mean_std, train_seeds, write_json
from photonic_poc.inference import TorchBackend
from photonic_poc.metrics import classification_metrics
from photonic_poc.onnx_export import export_onnx

WORK = Path("artifacts/research/amd_npu")


def _rel_rms(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - ref) ** 2) / np.mean(ref**2)))


def _flips(a: np.ndarray, b: np.ndarray) -> int:
    return int(((a > 0) != (b > 0)).sum())


def _simulator_logits(model: Any, x: np.ndarray) -> np.ndarray | None:
    try:
        from photonic_poc.qant_backend import QantBackend

        return QantBackend(model).predict_logits(x)
    except ImportError:
        return None


def run(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    if sys.platform != "win32":
        raise SystemExit("Experiment D needs Windows 11 with Windows ML and an AMD Ryzen AI NPU")
    dcfg, data = cfg["amd_npu"], dataset_from(cfg)
    arch = json.loads((out / "architecture.json").read_text(encoding="utf-8"))
    runs = train_seeds(dcfg["models"], arch["training"]["selected_learning_rates"], cfg, data)
    batch, calib = dcfg["batch"], data.x_train[: dcfg["calibration_pairs"]]
    y, x = data.y_test, data.x_test
    models: dict[str, Any] = {}
    with amd_npu.windows_ml(dcfg.get("allow_download", False)) as devices:
        info = amd_npu.provider_info(devices)
        for name in dcfg["models"]:
            seeds = []
            for seed, run_ in enumerate(runs[name]):
                fp32_path = export_onnx(
                    run_.model, WORK / f"{name}_s{seed}_b{batch}.onnx", batch, amd_npu.PAD_OUTPUT
                )
                q_path = amd_npu.quantize_a16w8(
                    fp32_path, WORK / f"{name}_s{seed}_b{batch}_q.onnx", calib, batch
                )
                ref = TorchBackend(run_.model).predict_logits(x)
                onnx_fp32 = amd_npu.predict(amd_npu.cpu_session(fp32_path), x, batch)
                if np.abs(onnx_fp32 - ref).max() > 1e-4:
                    raise RuntimeError(f"ONNX export of {name} does not reproduce PyTorch")
                int_cpu = amd_npu.predict(amd_npu.cpu_session(q_path), x, batch)
                npu = amd_npu.predict(amd_npu.npu_session(q_path, devices), x, batch)
                row = {
                    "f1_fp32": classification_metrics(y, ref)["f1"],
                    "f1_int_cpu": classification_metrics(y, int_cpu)["f1"],
                    "f1_npu": classification_metrics(y, npu)["f1"],
                    "rel_rms_int_cpu": _rel_rms(int_cpu, ref),
                    "rel_rms_npu": _rel_rms(npu, ref),
                    "flips_int_cpu": _flips(int_cpu, ref),
                    "flips_npu": _flips(npu, ref),
                    "npu_vs_int_cpu_max_abs": float(np.abs(npu - int_cpu).max()),
                    "npu_vs_int_cpu_flips": _flips(npu, int_cpu),
                }
                sim = _simulator_logits(run_.model, x)
                if sim is not None:
                    row |= {
                        "f1_qant_sim": classification_metrics(y, sim)["f1"],
                        "rel_rms_qant_sim": _rel_rms(sim, ref),
                        "flips_qant_sim": _flips(sim, ref),
                    }
                seeds.append(row)
            record: dict[str, Any] = {k: mean_std([r[k] for r in seeds]) for k in seeds[0]}
            record["seeds"] = len(seeds)
            record["npu_matches_int_cpu"] = max(r["npu_vs_int_cpu_flips"] for r in seeds) == 0
            record["placement"] = amd_npu.placement(WORK / f"{name}_s0_b{batch}_q.onnx")
            record["latency"] = _latency(
                runs[name][0].model, x, calib, dcfg["latency_batches"], devices
            )
            models[name] = record
            f1_fp32, f1_npu = record["f1_fp32"]["mean"], record["f1_npu"]["mean"]
            print(f"{name}: F1 fp32 {f1_fp32:.4f}, AMD NPU {f1_npu:.4f}")
    result = {
        "hypotheses": __doc__.split("\n\n")[1],
        "environment": {"processor": platform.processor(), "platform": platform.platform()} | info,
        "quantization": "QDQ via onnxruntime.quantization: uint16 activations, symmetric int8 "
        "weights (A16W8)",
        "test_pairs": len(y),
        "models": models,
    }
    write_json(out / "amd_npu.json", result)
    return result


def _latency(
    model: Any, x: np.ndarray, calib: np.ndarray, batches: list[int], devices: list[Any]
) -> dict[str, Any]:
    rows = {}
    for b in batches:
        fp32 = export_onnx(model, WORK / f"latency_b{b}.onnx", b, amd_npu.PAD_OUTPUT)
        q = amd_npu.quantize_a16w8(fp32, WORK / f"latency_b{b}_q.onnx", calib, b)
        xb = np.ascontiguousarray(np.resize(x, (b, x.shape[1])), dtype=np.float32)
        rows[str(b)] = {
            "npu_us_per_call": amd_npu.median_call_us(amd_npu.npu_session(q, devices), xb),
            "cpu_fp32_us_per_call": amd_npu.median_call_us(amd_npu.cpu_session(fp32), xb),
            "cpu_int_us_per_call": amd_npu.median_call_us(amd_npu.cpu_session(q), xb),
        }
    return rows
