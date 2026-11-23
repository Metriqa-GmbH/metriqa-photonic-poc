"""Experiment B figures: flips by margin, deviation by depth, assumed hardware effects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from photonic_poc.figures.style import FAMILY, INK2, SURFACE, save_figure, style_axes


def _family_models(rob: dict[str, Any], arch: dict[str, Any]) -> dict[str, str]:
    return {
        f: picks["best"]
        for f, picks in arch["champions"].items()
        if f != "logreg" and picks["best"] in rob["models"]
    }


def robustness_margin(rob: dict[str, Any], arch: dict[str, Any], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for family, name in _family_models(rob, arch).items():
        color, marker, label = FAMILY[family]
        prof = [
            p
            for p in rob["models"][name]["distributions"]["boundary"]["margin_profile"]
            if p["n"] > 0 and p["hi"] != float("inf")
        ]
        ax.plot(
            [p["hi"] for p in prof],
            [100 * p["rate"] for p in prof],
            marker=marker,
            color=color,
            label=f"{label} ({name})",
            markeredgecolor=SURFACE,
        )
    ax.set_xscale("log")
    ax.set_xlabel("|float32 logit| upper edge of band (log scale)")
    ax.set_ylabel("decisions flipped by the simulator (%)")
    ax.set_title("Flips happen only next to the decision boundary", loc="left")
    ax.legend(fontsize=8)
    style_axes(ax)
    return save_figure(fig, out / "robustness_margin.png")


def robustness_depth(rob: dict[str, Any], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, key, title in zip(
        axes,
        ("rel_rms_toolkit", "rel_rms_bf16"),
        ("Q.ANT toolkit simulator", "PyTorch bfloat16 (control)"),
        strict=True,
    ):
        for family in ("mlp", "cos", "kan"):
            rows = sorted(
                (r for r in rob["depth"] if r["family"] == family), key=lambda r: r["depth"]
            )
            color, marker, label = FAMILY[family]
            ax.errorbar(
                [r["depth"] for r in rows],
                [100 * r[key]["mean"] for r in rows],
                yerr=[100 * r[key]["std"] for r in rows],
                marker=marker,
                color=color,
                label=label,
                capsize=0,
                elinewidth=1,
                markeredgecolor=SURFACE,
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("layers (depth)")
        style_axes(ax)
    axes[0].set_ylabel("relative RMS logit deviation vs float32 (%)")
    axes[0].legend(fontsize=8)
    return save_figure(fig, out / "robustness_depth.png")


def robustness_hardware(rob: dict[str, Any], arch: dict[str, Any], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, key, axis, xlabel in zip(
        axes,
        ("noise", "tcos"),
        ("std", "epsilon"),
        (
            "assumed additive noise σ at NPU-layer outputs",
            "assumed 2nd-harmonic distortion ε of the cosine",
        ),
        strict=True,
    ):
        for family, name in _family_models(rob, arch).items():
            color, marker, label = FAMILY[family]
            rows = rob["models"][name][key]
            ax.plot(
                [r[axis] for r in rows],
                [100 * r["flip_rate"]["mean"] for r in rows],
                marker=marker,
                color=color,
                label=label,
                markeredgecolor=SURFACE,
            )
        ax.axhline(0.1, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel(xlabel)
        style_axes(ax)
    axes[0].set_ylabel("decisions flipped vs float32 (%)")
    axes[0].text(0, 0.13, "0.1% budget", fontsize=7, color=INK2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Assumed hardware effects (not measured)", fontweight="bold", x=0.08, ha="left")
    return save_figure(fig, out / "robustness_hardware_assumed.png")
