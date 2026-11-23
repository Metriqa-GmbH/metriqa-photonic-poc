"""Operation-level numerics of the toolkit, independent of any trained model.

Errors are measured against a float64 reference computed on the *bf16-rounded* inputs, so they
isolate the arithmetic of the toolkit path from the unavoidable input quantisation. PyTorch
bfloat16 is the control: same input precision, PyTorch's own accumulation strategy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from ml_dtypes import bfloat16

from photonic_poc.qant_backend import import_toolkit


def _rounded(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (bf16 array for the SDK, the same values as float64 for references)."""
    b = np.ascontiguousarray(array.astype(np.float32).astype(bfloat16))
    return b, b.astype(np.float64)


def _errors(result: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = result.astype(np.float64) - reference
    rms_ref = float(np.sqrt(np.mean(reference**2)))
    return {
        "max_abs_err": float(np.max(np.abs(diff))),
        "rel_rms_err": float(np.sqrt(np.mean(diff**2)) / rms_ref) if rms_ref else 0.0,
    }


def linear_study(
    fan_ins: Sequence[int] = (16, 64, 256, 1024, 4096), n_out: int = 64, batch: int = 256
) -> list[dict[str, Any]]:
    qant = import_toolkit()
    rng = np.random.default_rng(0)
    rows = []
    for n in fan_ins:
        xb, x = _rounded(rng.uniform(-1, 1, (batch, n)))
        wb, w = _rounded(rng.normal(0, 1 / math.sqrt(n), (n_out, n)))
        exact = x @ w.T
        sdk = np.asarray(qant.ai.linear_fprop(xb, wb)).astype(np.float64)
        tb = torch.from_numpy(x).bfloat16() @ torch.from_numpy(w).bfloat16().T
        rows.append(
            {
                "op": "linear_fprop",
                "fan_in": n,
                "toolkit": _errors(sdk, exact),
                "torch_bf16": _errors(tb.double().numpy(), exact),
            }
        )
    return rows


def kan_study(
    fan_ins: Sequence[int] = (4, 16, 64, 256), n_out: int = 16, grid: int = 4, batch: int = 64
) -> list[dict[str, Any]]:
    qant = import_toolkit()
    rng = np.random.default_rng(1)
    rows = []
    for n in fan_ins:
        std = math.sqrt(2 / ((n + n_out) * grid))
        xb, x = _rounded(rng.uniform(0, 1, (batch, n)))
        pb, p = _rounded(rng.uniform(-math.pi / 2, math.pi / 2, (n_out, n, grid)))
        ab, a = _rounded(rng.normal(0, std, (n_out, n, grid)))
        kb, k = _rounded(np.arange(1, grid + 1, dtype=np.float64))

        def formula(x: Any, p: Any, a: Any, k: Any, lib: Any) -> Any:
            return (a[None] * lib.cos(x[:, None, :, None] * k + p[None])).sum(axis=(2, 3))

        exact = formula(x, p, a, k, np)
        sdk = np.asarray(qant.ai.calc_kan_layer_fprop(xb, pb, ab, kb)).astype(np.float64)
        t = [torch.from_numpy(v).bfloat16() for v in (x, p, a, k)]
        tb = (t[2][None] * torch.cos(t[0][:, None, :, None] * t[3] + t[1][None])).sum(dim=(2, 3))
        rows.append(
            {
                "op": "calc_kan_layer_fprop",
                "fan_in": n,
                "terms_per_output": n * grid,
                "toolkit": _errors(sdk, exact),
                "torch_bf16": _errors(tb.double().numpy(), exact),
                "emulated_bf16_terms_exact_sum": _errors(_kan_terms_exact_sum(x, p, a, k), exact),
            }
        )
    return rows


def _kan_terms_exact_sum(x: np.ndarray, p: np.ndarray, a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The toolkit's term pipeline (bf16 after k*x, +phi, cos*a) but summed in float64.

    Comparing this with the toolkit isolates the error of its bf16 accumulator.
    """

    def r(v: np.ndarray) -> np.ndarray:
        return v.astype(np.float32).astype(bfloat16).astype(np.float64)

    terms = r(np.cos(r(r(x[:, None, :, None] * k) + p[None])) * a[None])
    return terms.sum(axis=(2, 3))


def periodic_study(n: int = 4097) -> dict[str, Any]:
    """The simulator's periodic nonlinearity versus an exact cosine (hardware uses its own tcos)."""
    qant = import_toolkit()
    ub, u = _rounded(np.linspace(-4 * math.pi, 4 * math.pi, n))
    vb, v = _rounded(np.ones(n))
    sdk = np.asarray(qant.native.calc_scaled_periodic_nl_fprop(ub, vb)).astype(np.float64)
    return {"op": "calc_scaled_periodic_nl_fprop", "points": n, **_errors(sdk, np.cos(u) * v)}


def run_op_study() -> dict[str, Any]:
    return {"linear": linear_study(), "kan": kan_study(), "periodic": periodic_study()}
