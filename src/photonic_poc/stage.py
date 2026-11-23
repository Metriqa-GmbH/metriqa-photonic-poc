"""Execution stage type and bf16 helpers shared by the hybrid executor and the KAN paths."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

BF16, FP32 = "bf16", "fp32"


@dataclass(frozen=True)
class Stage:
    name: str  # e.g. "2:linear.matmul"
    op: str  # matmul, bias, concat, relu, sigmoid, cos, kan, kan.{mul,phase,nonlin,sum}
    placement: str  # "npu", "host" or "mixed" (one toolkit call that does both)
    dtype_in: str
    dtype_out: str
    run: Callable[[np.ndarray], np.ndarray]


def to_bf16(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32).astype(bfloat16))


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return ((phase + math.pi) % (2 * math.pi) - math.pi).astype(np.float32)
