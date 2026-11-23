"""Experiment A figures: quality versus size/compute, backend effect per architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from photonic_poc.figures.style import FAMILY, INK, INK2, SURFACE, save_figure, style_axes


def architecture_pareto(arch: dict[str, Any], out: Path) -> Path:
    records = arch["records"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    best = {picks["best"]: family for family, picks in arch["champions"].items()}
    offsets = {"logreg": (6, -4), "mlp": (-70, 16), "cos": (6, -16), "kan": (-40, -28)}
    for ax, cost, xlabel in zip(
        axes,
        ("parameters", "total_ops"),
        ("parameters (log scale)", "operations per pair (log scale)"),
        strict=True,
    ):
        for family, (color, marker, label) in FAMILY.items():
            rs = [r for r in records if r["family"] == family]
            ax.errorbar(
                [r[cost] for r in rs],
                [r["fp32_f1"]["mean"] for r in rs],
                yerr=[r["fp32_f1"]["std"] for r in rs],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=0,
                markeredgecolor=SURFACE,
                markeredgewidth=0.8,
                label=label,
            )
        front = [
            next(r for r in records if r["arch"] == a)
            for a in arch["pareto_fronts"][f"fp32:{cost}"]
        ]
        ax.step(
            [r[cost] for r in front],
            [r["fp32_f1"]["mean"] for r in front],
            where="post",
            color=INK2,
            linewidth=1,
            linestyle=":",
            label="Pareto front",
        )
        for r in records:
            if r["arch"] in best:
                ax.annotate(
                    f"{r['arch']} (best {best[r['arch']]})"
                    if r["family"] != "logreg"
                    else r["arch"],
                    (r[cost], r["fp32_f1"]["mean"]),
                    textcoords="offset points",
                    xytext=offsets[r["family"]],
                    fontsize=7,
                    color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.6},
                )
        ref = arch["reference"]["f1"]["mean"]
        ax.axhline(ref, color=INK2, linewidth=1, linestyle="--")
        ax.text(
            min(r[cost] for r in records),
            ref + 0.001,
            "gradient-boosted trees (reference)",
            fontsize=7,
            color=INK2,
        )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        style_axes(ax)
    axes[0].set_ylabel("test F1, float32 (mean ± sd over 5 seeds)")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_ylim(0.915, 0.962)
    clipped = [r["arch"] for r in records if r["fp32_f1"]["mean"] - r["fp32_f1"]["std"] < 0.915]
    fig.suptitle(
        "Matching quality versus model size and compute", fontweight="bold", x=0.08, ha="left"
    )
    if clipped:
        fig.text(0.08, -0.03, f"Axis clipped; unstable or weak configurations not fully shown: "
                 f"{', '.join(clipped)} (see table).", fontsize=7, color=INK2)  # fmt: skip
    return save_figure(fig, out / "architecture_pareto.png")


def architecture_backend(arch: dict[str, Any], out: Path) -> Path:
    order = {f: i for i, f in enumerate(FAMILY)}
    records = sorted(arch["records"], key=lambda r: (order[r["family"]], r["parameters"]))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, r in enumerate(records):
        color, marker, _ = FAMILY[r["family"]]
        ax.errorbar(
            i,
            1e3 * r["delta_f1"]["mean"],
            yerr=1e3 * r["delta_f1"]["std"],
            fmt=marker,
            color=color,
            elinewidth=1,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
        )
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(
        range(len(records)), [r["arch"] for r in records], rotation=60, ha="right", fontsize=7
    )
    ax.set_ylabel("ΔF1 simulator minus float32 (×10⁻³)")
    ax.set_title("Backend effect per architecture (mean ± sd over 5 seeds)", loc="left")
    handles = [
        Line2D([], [], marker=m, color=c, linestyle="", label=lab) for c, m, lab in FAMILY.values()
    ]
    ax.legend(handles=handles, loc="upper left", ncols=4, fontsize=8)
    style_axes(ax)
    return save_figure(fig, out / "architecture_backend_delta.png")
