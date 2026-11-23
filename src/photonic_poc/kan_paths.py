"""Alternative executions of a Fourier-KAN layer from toolkit primitives.

``native`` reproduces ``ai.calc_kan_layer_fprop`` step by step but sums in float32.
``factorized`` uses the exact identity

    a cos(k x + phi) = (a cos phi) cos(k x) - (a sin phi) sin(k x),   sin(u) = cos(u - pi/2),

so a KAN layer becomes: fixed Fourier basis [cos(k_l x_i), sin(k_l x_i)] on the NPU (2 n_in n_k
nonlinear evaluations, independent of the output width) followed by ONE preloaded matrix product
with float32 accumulation. Every arithmetic step then runs on the NPU, and no term tensor of size
n_out n_in n_k crosses the bus. On hardware the identity is exact only if tcos is a pure cosine.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from ml_dtypes import bfloat16

from photonic_poc.stage import BF16, FP32, Stage, to_bf16


def append_ones(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    )


def _mul_stage(qant: Any, name: str, ks: np.ndarray, n_in: int) -> Stage:
    """k_l * x_i for all inputs and frequencies, on the NPU (as the toolkit does)."""
    native, k16, n_k = qant.native, to_bf16(ks), len(ks)

    def mul(x: np.ndarray) -> np.ndarray:
        xs = np.ascontiguousarray(np.repeat(x, n_k, axis=1))
        kk = np.ascontiguousarray(np.tile(k16, x.shape[0] * n_in))
        return np.asarray(native.mul_elementwise(xs.reshape(-1), kk)).reshape(-1, n_in, n_k)

    return Stage(f"{name}.mul", "kan.mul", "npu", BF16, BF16, mul)


def kan_native_stages(
    qant: Any, device: int, name: str, phase: np.ndarray, amplitude: np.ndarray, ks: np.ndarray
) -> list[Stage]:
    native, n_in = qant.native, phase.shape[1]
    p32, a16 = phase.astype(np.float32), to_bf16(amplitude)

    def phase_add(kx: np.ndarray) -> np.ndarray:  # host, float32
        return kx[:, None] + p32[None]

    def nonlin(arg: np.ndarray) -> np.ndarray:  # a * tcos(arg) on the NPU
        flat = np.ascontiguousarray(arg.reshape(-1))
        amp = np.ascontiguousarray(np.broadcast_to(a16, arg.shape)).reshape(-1)
        return np.asarray(
            native.calc_scaled_periodic_nl_fprop(flat, amp, device_id=device)
        ).reshape(arg.shape)

    def total(terms: np.ndarray) -> np.ndarray:  # host, float32 accumulation
        return terms.sum(axis=(2, 3), dtype=np.float32)

    return [
        _mul_stage(qant, name, ks, n_in),
        Stage(f"{name}.phase", "kan.phase", "host", FP32, FP32, phase_add),
        Stage(f"{name}.nonlin", "kan.nonlin", "npu", BF16, BF16, nonlin),
        Stage(f"{name}.sum", "kan.sum", "host", FP32, FP32, total),
    ]


def factorized_mix_matrix(phase: np.ndarray, amplitude: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """(n_out, 2 n_in n_k + 1) weights; basis order (input, frequency, {cos, sin}), bias last."""
    alpha, beta = amplitude * np.cos(phase), -amplitude * np.sin(phase)
    mix = np.stack([alpha, beta], axis=-1).reshape(phase.shape[0], -1)
    return np.concatenate([mix, bias[:, None]], axis=1).astype(np.float32)


def kan_factorized_stages(
    qant: Any, device: int, name: str, ks: np.ndarray, n_in: int, matmul: Stage
) -> list[Stage]:
    native = qant.native
    ones: dict[int, np.ndarray] = {}

    def shift(kx: np.ndarray) -> np.ndarray:  # host: [k x, k x - pi/2] per input and frequency
        return np.stack([kx, kx - np.float32(math.pi / 2)], axis=-1).reshape(kx.shape[0], -1)

    def basis(u: np.ndarray) -> np.ndarray:  # cos and sin of every k x on the NPU
        flat = np.ascontiguousarray(u.reshape(-1))
        if flat.size not in ones:
            ones[flat.size] = np.ones(flat.size, dtype=bfloat16)
        out = native.calc_scaled_periodic_nl_fprop(flat, ones[flat.size], device_id=device)
        return np.asarray(out).reshape(u.shape)

    return [
        _mul_stage(qant, name, ks, n_in),
        Stage(f"{name}.shift", "kan.shift", "host", FP32, FP32, shift),
        Stage(f"{name}.basis", "kan.basis", "npu", BF16, BF16, basis),
        Stage(f"{name}.concat", "concat", "host", BF16, BF16, append_ones),
        matmul,
    ]
