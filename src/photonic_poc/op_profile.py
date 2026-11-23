"""Static cost model: per-sample operations, placement and host<->NPU traffic of a model.

Counting conventions (per sample, i.e. per record pair):

- a multiply-accumulate (MAC) counts as 2 operations;
- one element-wise function evaluation (cos, ReLU, sigmoid) counts as 1 operation;
- one element-wise multiply or add counts as 1 operation;
- ``transfer`` counts bf16 elements sent to and received from the NPU, excluding weights that
  were preloaded once. Every NPU call in toolkit v2.5.0 except the preloaded matrix product
  takes host arrays and returns host arrays, so each NPU stage is a round trip.

Stages mirror ``HybridExecutor`` (the toolkit's KAN call is split into its four steps).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch import nn

from photonic_poc.hybrid import NOOP_TYPES, TOOLKIT_DEFAULT, HybridOptions
from photonic_poc.models import Cos, FourierKAN, count_parameters


@dataclass(frozen=True)
class StageCost:
    layer: str
    op: str
    placement: str  # "npu" or "host"
    macs: int = 0
    nonlin: int = 0
    elementwise: int = 0
    transfer_in: int = 0
    transfer_out: int = 0
    weights: int = 0  # elements held in NPU memory (preloaded once)

    @property
    def ops(self) -> int:
        return 2 * self.macs + self.nonlin + self.elementwise


def _linear(name: str, m: nn.Linear, options: HybridOptions) -> list[StageCost]:
    n_in, n_out = m.in_features, m.out_features
    has_bias = m.bias is not None
    if has_bias and options.fold_bias:
        return [
            StageCost(name, "concat", "host"),
            StageCost(
                name,
                "matmul",
                "npu",
                macs=(n_in + 1) * n_out,
                transfer_in=n_in + 1,
                transfer_out=n_out,
                weights=(n_in + 1) * n_out,
            ),
        ]
    stages = [
        StageCost(
            name,
            "matmul",
            "npu",
            macs=n_in * n_out,
            transfer_in=n_in,
            transfer_out=n_out,
            weights=n_in * n_out,
        ),
    ]
    if has_bias:
        stages.append(StageCost(name, "bias", "host", elementwise=n_out))
    return stages


def _kan(name: str, m: FourierKAN, options: HybridOptions) -> list[StageCost]:
    n_in, n_out, g = m.in_features, m.out_features, m.grid_size
    terms = n_out * n_in * g
    if options.kan_impl == "factorized":
        basis = 2 * n_in * g
        return [
            StageCost(name, "kan.mul", "npu", elementwise=n_in * g,
                      transfer_in=2 * n_in * g, transfer_out=n_in * g),
            StageCost(name, "kan.shift", "host", elementwise=n_in * g),
            StageCost(name, "kan.basis", "npu", nonlin=basis, elementwise=basis,
                      transfer_in=2 * basis, transfer_out=basis),
            StageCost(name, "concat", "host"),
            StageCost(name, "matmul", "npu", macs=(basis + 1) * n_out,
                      transfer_in=basis + 1, transfer_out=n_out, weights=(basis + 1) * n_out),
        ]  # fmt: skip
    return [
        StageCost(
            name,
            "kan.mul",
            "npu",
            elementwise=n_in * g,
            transfer_in=2 * n_in * g,
            transfer_out=n_in * g,
        ),
        StageCost(name, "kan.phase", "host", elementwise=terms),
        StageCost(
            name,
            "kan.nonlin",
            "npu",
            nonlin=terms,
            elementwise=terms,
            transfer_in=2 * terms,
            transfer_out=terms,
        ),
        StageCost(name, "kan.sum", "host", elementwise=terms),
        StageCost(name, "bias", "host", elementwise=n_out),
    ]


def stage_costs(model: nn.Sequential, options: HybridOptions = TOOLKIT_DEFAULT) -> list[StageCost]:
    stages: list[StageCost] = []
    width = 0
    for index, m in enumerate(model):
        name = f"{index}:{type(m).__name__.lower()}"
        if isinstance(m, NOOP_TYPES):
            continue
        if isinstance(m, nn.Linear):
            stages += _linear(name, m, options)
            width = m.out_features
        elif isinstance(m, FourierKAN):
            stages += _kan(name, m, options)
            width = m.out_features
        elif isinstance(m, Cos):
            stages.append(
                StageCost(
                    name,
                    "cos",
                    "npu",
                    nonlin=width,
                    elementwise=width,
                    transfer_in=2 * width,
                    transfer_out=width,
                )
            )
        elif isinstance(m, nn.ReLU):
            stages.append(StageCost(name, "relu", "host", elementwise=width))
        elif isinstance(m, nn.Sigmoid):
            stages.append(StageCost(name, "sigmoid", "host", nonlin=width))
        else:
            raise TypeError(f"no cost model for {type(m).__name__}")
    return stages


def profile_model(model: nn.Sequential, options: HybridOptions = TOOLKIT_DEFAULT) -> dict[str, Any]:
    """Totals per sample plus the stage list. ``offload_fraction`` is NPU ops / all ops."""
    stages = stage_costs(model, options)
    npu = [s for s in stages if s.placement == "npu"]
    npu_ops = sum(s.ops for s in npu)
    total_ops = sum(s.ops for s in stages)
    transfer = sum(s.transfer_in + s.transfer_out for s in npu)
    return {
        "policy": options.label,
        "parameters": count_parameters(model),
        "macs": sum(s.macs for s in stages),
        "nonlin_evals": sum(s.nonlin for s in stages),
        "total_ops": total_ops,
        "npu_ops": npu_ops,
        "host_ops": total_ops - npu_ops,
        "offload_fraction": npu_ops / total_ops if total_ops else 0.0,
        "npu_calls": len(npu),
        "host_stages": len(stages) - len(npu),
        "transfer_elements": transfer,
        "arithmetic_intensity": npu_ops / transfer if transfer else 0.0,
        "preloaded_weight_elements": sum(s.weights for s in npu),
        "stages": [asdict(s) | {"ops": s.ops} for s in stages],
    }
