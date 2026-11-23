"""Render every research figure whose input JSON exists."""

from __future__ import annotations

from pathlib import Path

from photonic_poc.figures.architecture import architecture_backend, architecture_pareto
from photonic_poc.figures.hybrid import hybrid_breakdown, hybrid_layers
from photonic_poc.figures.robustness import robustness_depth, robustness_hardware, robustness_margin
from photonic_poc.figures.style import load_json


def render_figures(results: Path, out: Path) -> list[Path]:
    arch, rob, hyb = (
        load_json(results / f"{n}.json") for n in ("architecture", "robustness", "hybrid")
    )
    paths = []
    if arch:
        paths += [architecture_pareto(arch, out), architecture_backend(arch, out)]
    if arch and rob:
        paths += [
            robustness_margin(rob, arch, out),
            robustness_depth(rob, out),
            robustness_hardware(rob, arch, out),
        ]
    if hyb:
        paths += [hybrid_breakdown(hyb, out), hybrid_layers(hyb, out)]
    return paths
