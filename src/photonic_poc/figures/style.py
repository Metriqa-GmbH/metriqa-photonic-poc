"""Shared look of all research figures: light surface, thin marks, recessive grid.

One hue per architecture family plus a marker shape, so identity is never colour alone; the three
family hues pass the colour-vision checks of the data-viz palette validator. Every figure has a
table with the same numbers in the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
FAMILY = {  # colour, marker, label
    "mlp": ("#2a78d6", "o", "ReLU MLP"),
    "cos": ("#eb6834", "s", "cosine MLP"),
    "kan": ("#1baf7a", "^", "Fourier-KAN"),
    "logreg": (INK2, "D", "logistic regression"),
}
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "legend.frameon": False,
    }
)


def style_axes(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_figure(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
