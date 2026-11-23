"""Run ONNX models on an AMD XDNA NPU (Ryzen AI) through Windows ML and the VitisAI EP.

This is an *electronic* NPU, used as a real-hardware reference for the cost model of the hybrid
experiment; it is unrelated to Q.ANT's photonic NPU. Needs Windows 11 and the optional
``amd-npu`` extra. The execution provider is downloaded by Windows ML only when
``allow_download=True`` (a system-wide package install), never implicitly.

Two provider issues found with VitisAI EP 2.1000.81 shape what this module does:
models are quantised (A16W8 QDQ) because on-the-fly float compilation fails from an unpackaged
Python, and single-column outputs at batch > 1 are wrong, so exports pad outputs to 8 columns.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

PROVIDER = "VitisAIExecutionProvider"
PAD_OUTPUT = 8
_REGISTERED: set[str] = set()


class NpuUnavailable(RuntimeError):
    pass


@contextmanager
def windows_ml(allow_download: bool = False) -> Iterator[list[Any]]:
    """Initialise Windows ML, make the VitisAI EP ready and registered; yield its ORT devices."""
    try:
        import onnxruntime as ort
        import winui3.microsoft.windows.ai.machinelearning as winml
        import winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap as bootstrap
    except ImportError as exc:
        raise NpuUnavailable("install the extra: uv sync --extra amd-npu (Windows only)") from exc
    with bootstrap.initialize(options=bootstrap.InitializeOptions.NONE):
        catalog = winml.ExecutionProviderCatalog.get_default()
        provider = next((p for p in catalog.find_all_providers() if p.name == PROVIDER), None)
        if provider is None:
            raise NpuUnavailable("Windows ML lists no VitisAI execution provider on this machine")
        if (
            provider.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT
            and not allow_download
        ):
            raise NpuUnavailable(
                "the VitisAI EP is not installed; rerun with allow_download=True to let Windows ML "
                "download and install it system-wide"
            )
        result = provider.ensure_ready_async().get()
        if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
            raise NpuUnavailable(f"VitisAI EP not ready: {result.status.name}")
        if PROVIDER not in _REGISTERED:
            ort.register_execution_provider_library(provider.name, provider.library_path)
            _REGISTERED.add(PROVIDER)
        devices = [d for d in ort.get_ep_devices() if d.ep_name == PROVIDER]
        if not devices:
            raise NpuUnavailable("VitisAI EP registered but exposes no NPU device")
        yield devices


def provider_info(devices: list[Any]) -> dict[str, Any]:
    import onnxruntime as ort

    device = devices[0]
    return {
        "onnxruntime": ort.__version__,
        "device": dict(device.device.metadata).get("Description", ""),
        "ep_library": dict(device.ep_metadata).get("library_path", ""),
    }


def quantize_a16w8(fp32: Path, out: Path, calibration: np.ndarray, batch: int) -> Path:
    """QDQ with uint16 activations and symmetric int8 weights, calibrated on real features."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    chunks = [calibration[i : i + batch] for i in range(0, len(calibration) - batch + 1, batch)]
    if not chunks:
        chunks = [np.resize(calibration, (batch, calibration.shape[1])).astype(np.float32)]

    class Reader(CalibrationDataReader):  # type: ignore[misc]
        def __init__(self) -> None:
            self._it = iter({"x": c} for c in chunks[:8])

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self._it, None)

    quantize_static(str(fp32), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt16, weight_type=QuantType.QInt8,
                    extra_options={"WeightSymmetric": True})  # fmt: skip
    return out


def npu_session(path: Path, devices: list[Any], log_level: int = 3) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = log_level
    options.add_provider_for_devices(devices, {})
    return ort.InferenceSession(str(path), sess_options=options)


def cpu_session(path: Path) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def predict(session: Any, features: np.ndarray, batch: int) -> np.ndarray:
    """Logits (first output column) for any number of rows, padding the last batch."""
    n = len(features)
    padded = np.zeros((-(-n // batch) * batch, features.shape[1]), np.float32)
    padded[:n] = features
    outs = [
        session.run(None, {"x": padded[i : i + batch]})[0][:, 0]
        for i in range(0, len(padded), batch)
    ]
    return np.concatenate(outs)[:n] if outs else np.zeros(0, np.float32)


def median_call_us(
    session: Any, x: np.ndarray, budget_s: float = 0.6, min_calls: int = 20
) -> float:
    session.run(None, {"x": x})
    times: list[float] = []
    start = time.perf_counter()
    while time.perf_counter() - start < budget_s or len(times) < min_calls:
        t = time.perf_counter()
        session.run(None, {"x": x})
        times.append(time.perf_counter() - t)
    return 1e6 * statistics.median(times)


def placement(path: Path) -> dict[str, Any]:
    """Operator counts per device from the EP's own log, read in a child process."""
    child = subprocess.run(
        [sys.executable, "-m", "photonic_poc.amd_npu", "placement", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    log = child.stdout + child.stderr
    counts: dict[str, Any] = {}
    block = re.search(r"No\. of Operators :(.*?)No\. of Subgraphs", log, re.S)
    for device, number in re.findall(r"(NPU|VITIS_EP_CPU|CPU)\s+(\d+)", block[1] if block else ""):
        counts[device] = int(number)
    cpu_ops = re.findall(r"type = ([a-z-]+)\} is partitioned to CPU", log)
    counts["cpu_op_types"] = sorted(set(cpu_ops))
    return counts


def _placement_main(path: str) -> None:
    with windows_ml() as devices:
        npu_session(Path(path), devices, log_level=1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "placement":
        _placement_main(sys.argv[2])
        print(json.dumps({"done": True}))
