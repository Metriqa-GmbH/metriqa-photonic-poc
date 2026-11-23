"""Experiment D figure: measured per-pair latency on the AMD XDNA 2 NPU versus the CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from photonic_poc.figures.style import FAMILY, INK2, SURFACE, save_figure, style_axes

SHOWN = {"mlp-32": "mlp", "cos-16x8": "cos", "kan-8-g8": "kan"}


def amd_npu_latency(npu: dict[str, Any], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, family in SHOWN.items():
        if name not in npu["models"]:
            continue
        color, marker, _ = FAMILY[family]
        rows = sorted(npu["models"][name]["latency"].items(), key=lambda kv: int(kv[0]))
        batches = [int(b) for b, _ in rows]
        for key, style in (("npu_us_per_call", "-"), ("cpu_fp32_us_per_call", "--")):
            ax.plot(
                batches,
                [v[key] / b for b, (_, v) in zip(batches, rows, strict=True)],
                linestyle=style,
                marker=marker,
                color=color,
                markeredgecolor=SURFACE,
            )
        ax.annotate(
            name,
            (batches[-1], rows[-1][1]["npu_us_per_call"] / batches[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=7,
            color=INK2,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("µs per pair (measured)")
    ax.set_title("Real electronic NPU (AMD XDNA 2) vs CPU", loc="left")
    handles = [
        Line2D([], [], color=INK2, linestyle="-", label="AMD NPU, A16W8"),
        Line2D([], [], color=INK2, linestyle="--", label="CPU, float32 (ONNX Runtime)"),
    ]
    handles += [
        Line2D([], [], color=FAMILY[f][0], marker=FAMILY[f][1], linestyle="", label=n)
        for n, f in SHOWN.items()
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    style_axes(ax)
    return save_figure(fig, out / "amd_npu_latency.png")
