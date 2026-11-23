"""Repeat training over seeds so model and backend differences can be told apart from seed noise."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from photonic_poc.inference import NoisyTorchBackend, create_backend
from photonic_poc.metrics import classification_metrics, output_agreement
from photonic_poc.training import TrainConfig, train_model

Split = tuple[np.ndarray, np.ndarray]


def run_seed_study(
    variants: Sequence[tuple[str, float]],
    train: Split,
    val: Split,
    test: Split,
    seeds: Sequence[int],
    epochs: int,
    eval_noise: float = 0.1,
    noise_seeds: int = 5,
    use_qant: bool = True,
) -> list[dict[str, Any]]:
    """For each (model, training noise) variant: test F1 on fp32 and qant, flips, noisy F1.

    ``eval_noise`` is the σ of the assumed Gaussian noise model used for the robustness column.
    """
    rows = []
    x_test, y_test = test
    for name, train_noise in variants:
        runs = []
        for seed in seeds:
            config = TrainConfig(epochs=epochs, seed=seed, noise_std=train_noise)
            model = train_model(name, *train, *val, config).model
            fp32 = create_backend("torch-fp32", model).predict_logits(x_test)
            run = {"f1_fp32": classification_metrics(y_test, fp32)["f1"]}
            if use_qant:
                qant = create_backend("qant", model).predict_logits(x_test)
                run["f1_qant"] = classification_metrics(y_test, qant)["f1"]
                run["flips"] = output_agreement(fp32, qant)["decision_flips"]
            noisy = [
                classification_metrics(
                    y_test, NoisyTorchBackend(model, eval_noise, s).predict_logits(x_test)
                )["f1"]
                for s in range(noise_seeds)
            ]
            run["f1_noisy"] = float(np.mean(noisy))
            runs.append(run)
        rows.append(_summarise(name, train_noise, eval_noise, runs))
    return rows


def _summarise(
    name: str, train_noise: float, eval_noise: float, runs: list[dict[str, float]]
) -> dict[str, Any]:
    def stats(key: str) -> dict[str, float]:
        values = np.array([r[key] for r in runs])
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
        }

    row: dict[str, Any] = {
        "model": name,
        "train_noise": train_noise,
        "eval_noise": eval_noise,
        "seeds": len(runs),
        "f1_fp32": stats("f1_fp32"),
        "f1_noisy": stats("f1_noisy"),
        "runs": runs,
    }
    if "f1_qant" in runs[0]:
        row["f1_qant"] = stats("f1_qant")
        row["flips"] = stats("flips")
        row["max_abs_f1_delta"] = max(abs(r["f1_qant"] - r["f1_fp32"]) for r in runs)
    return row
