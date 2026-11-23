"""Export a supported ``nn.Sequential`` to ONNX (Gemm, Relu, Sigmoid, Cos) with a static batch.

Fourier-KAN layers are written in their exact factorised form, Gemm(basis arguments) -> Cos ->
Gemm(mix), so every model becomes a chain of matrix products and element-wise functions that
generic NPU compilers handle. Requires the optional ``onnx`` package.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from torch import nn

from photonic_poc.kan_paths import factorized_mix_matrix
from photonic_poc.models import Cos, FourierKAN

Layer = tuple[str, np.ndarray | None, np.ndarray | None]  # (op, weight (in, out), bias)


def kan_basis_arguments(module: FourierKAN) -> tuple[np.ndarray, np.ndarray]:
    """Weights and bias mapping x to [k_l x_i, k_l x_i - pi/2] per input and frequency."""
    n_in, n_k = module.in_features, module.grid_size
    k = module.get_buffer("k").numpy()
    weight = np.zeros((n_in, 2 * n_in * n_k), np.float32)
    bias = np.zeros(2 * n_in * n_k, np.float32)
    for i in range(n_in):
        for col, kl in enumerate(k):
            j = 2 * (i * n_k + col)
            weight[i, j] = weight[i, j + 1] = kl
            bias[j + 1] = -math.pi / 2  # sin(u) = cos(u - pi/2)
    return weight, bias


def layer_list(model: nn.Sequential) -> list[Layer]:
    layers: list[Layer] = []
    for m in model:
        if isinstance(m, nn.Linear):
            bias = (
                np.zeros(m.out_features, np.float32) if m.bias is None else m.bias.detach().numpy()
            )
            layers.append(("Gemm", m.weight.detach().numpy().T.copy(), bias.copy()))
        elif isinstance(m, nn.ReLU):
            layers.append(("Relu", None, None))
        elif isinstance(m, nn.Sigmoid):
            layers.append(("Sigmoid", None, None))
        elif isinstance(m, Cos):
            layers.append(("Cos", None, None))
        elif isinstance(m, FourierKAN):
            mix = factorized_mix_matrix(
                m.phase.detach().numpy(), m.amplitude.detach().numpy(), m.bias.detach().numpy()
            )
            layers += [("Gemm", *kan_basis_arguments(m)), ("Cos", None, None)]
            layers.append(("Gemm", mix[:, :-1].T.copy(), mix[:, -1].copy()))
        elif not isinstance(m, nn.Identity | nn.Dropout):
            raise TypeError(f"no ONNX export for {type(m).__name__}")
    return layers


def export_onnx(model: nn.Sequential, path: Path, batch: int, pad_output_to: int = 1) -> Path:
    """Write ``x (batch, F) -> y (batch, max(out, pad_output_to))``; extra columns are zero."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    layers = layer_list(model)
    nodes: list[Any] = []
    inits: list[Any] = []
    previous, width, n_features = "x", 0, 0
    for index, (op, weight, bias) in enumerate(layers):
        out = "y" if index == len(layers) - 1 else f"t{index}"
        if op == "Gemm":
            assert weight is not None and bias is not None
            n_features = n_features or weight.shape[0]
            if index == len(layers) - 1 and weight.shape[1] < pad_output_to:
                extra = pad_output_to - weight.shape[1]
                weight, bias = np.pad(weight, ((0, 0), (0, extra))), np.pad(bias, (0, extra))
            inits += [numpy_helper.from_array(weight.astype(np.float32), f"w{index}"),
                      numpy_helper.from_array(bias.astype(np.float32), f"b{index}")]  # fmt: skip
            nodes.append(helper.make_node("Gemm", [previous, f"w{index}", f"b{index}"], [out]))
            width = weight.shape[1]
        else:
            nodes.append(helper.make_node(op, [previous], [out]))
        previous = out
    graph = helper.make_graph(
        nodes,
        "matcher",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, n_features])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, width])],
        inits,
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    proto.ir_version = 9
    onnx.checker.check_model(proto)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, path)
    return path
