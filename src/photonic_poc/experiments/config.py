"""Experiment configuration (TOML) and shared helpers for the experiment runners."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from photonic_poc.data import GeneratorConfig
from photonic_poc.datasets import Dataset, load_dataset
from photonic_poc.sweep import Job, Trained, run_jobs


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def dataset_from(cfg: dict[str, Any]) -> Dataset:
    return load_dataset(GeneratorConfig(**cfg["dataset"]))


def select_learning_rates(
    archs: list[str], cfg: dict[str, Any], data: Dataset
) -> tuple[dict[str, float], dict[Job, Trained]]:
    """Per architecture, the grid learning rate with the lowest validation loss (seed 0)."""
    tr = cfg["training"]
    seed0, epochs = tr["seeds"][0], tr["epochs"]
    jobs = [Job(a, lr, seed0, epochs) for a in archs for lr in tr["learning_rates"]]
    trained = run_jobs(jobs, data, tr["workers"])
    best = {
        a: min(
            tr["learning_rates"], key=lambda lr: (trained[Job(a, lr, seed0, epochs)].val_loss, lr)
        )
        for a in archs
    }
    return best, trained


def train_seeds(
    archs: list[str],
    lrs: dict[str, float],
    cfg: dict[str, Any],
    data: Dataset,
    noise_std: float = 0.0,
) -> dict[str, list[Trained]]:
    tr = cfg["training"]
    jobs = [Job(a, lrs[a], s, tr["epochs"], noise_std) for a in archs for s in tr["seeds"]]
    trained = run_jobs(jobs, data, tr["workers"])
    return {
        a: [trained[Job(a, lrs[a], s, tr["epochs"], noise_std)] for s in tr["seeds"]] for a in archs
    }


def mean_std(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
