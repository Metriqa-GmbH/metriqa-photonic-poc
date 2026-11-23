"""Experiment C figures: time breakdown of the hybrid pipeline, speedup versus layer width."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from photonic_poc.figures.style import FAMILY, INK, INK2, SURFACE, save_figure, style_axes

SHOWN = ("mlp-8x8", "mlp-32", "cos-16", "cos-16x8", "kan-4-g2", "kan-8-g8")


def hybrid_breakdown(hyb: dict[str, Any], out: Path, batch: int = 256) -> Path:
    """Measured quantities only: best CPU-only inference vs host work left in a hybrid pipeline."""
    names = [n for n in SHOWN if n in hyb["models"]] or list(hyb["models"])
    sizes = [b["batch"] for b in hyb["models"][names[0]]["batches"]]
    batch = max((b for b in sizes if b <= batch), default=max(sizes))
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.62 * len(names)))
    ticks, labels = [], []
    for i, name in enumerate(names):
        row = next(b for b in hyb["models"][name]["batches"] if b["batch"] == batch)
        policy = "kan-factorized" if "kan-factorized" in row else "fold-bias"
        y = 2.6 * i
        ax.barh(y, row["cpu_us_per_pair"]["best"], color=FAMILY["mlp"][0], height=0.9)
        ax.barh(y + 1, row[policy]["host"], color="#eb6834", height=0.9)
        ax.barh(y + 1, row[policy]["convert"], left=row[policy]["host"], color="#eda100",
                height=0.9, edgecolor=SURFACE, linewidth=1)  # fmt: skip
        ticks += [y, y + 1]
        labels += [f"{name}: CPU only", f"{name}: hybrid host work ({policy})"]
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"µs per pair at batch {batch} (measured on the host)")
    handles = [
        Rectangle((0, 0), 1, 1, color=FAMILY["mlp"][0], label="CPU float32, best of PyTorch/NumPy"),
        Rectangle((0, 0), 1, 1, color="#eb6834", label="host ops that stay on the CPU"),
        Rectangle((0, 0), 1, 1, color="#eda100", label="fp32↔bf16 conversions"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_title(
        "Host work of a hybrid pipeline vs CPU-only inference (NPU time excluded)", loc="left"
    )
    style_axes(ax)
    return save_figure(fig, out / "hybrid_breakdown.png")


def hybrid_layers(hyb: dict[str, Any], out: Path) -> Path:
    rows = hyb["layer_study"]
    widths = [r["width"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(
        widths,
        [r["max_speedup_bound"] for r in rows],
        color=INK,
        marker="o",
        label="upper bound: NPU free and instant (measured host work only)",
    )
    for sc, color, marker in zip(
        hyb["scenarios"], ("#2a78d6", "#eb6834", "#1baf7a"), ("s", "^", "v"), strict=False
    ):
        ax.plot(
            widths,
            [r[f"speedup_{sc['name']}"] for r in rows],
            color=color,
            marker=marker,
            linestyle="--",
            label=f"scenario '{sc['name']}' (assumed)",
        )
    ax.axhline(1, color=INK2, linewidth=1)
    ax.text(widths[0], 1.08, "break-even", fontsize=7, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("layer width n (two n×n layers with ReLU, batch 256)")
    ax.set_ylabel("end-to-end speedup vs CPU float32")
    ax.set_title("When could offloading pay off?", loc="left")
    ax.legend(fontsize=7)
    style_axes(ax)
    return save_figure(fig, out / "hybrid_layer_width.png")
