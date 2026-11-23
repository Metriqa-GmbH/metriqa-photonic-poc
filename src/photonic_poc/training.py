"""Deterministic CPU training for the demonstration models."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from photonic_poc.inference import output_noise
from photonic_poc.metrics import classification_metrics
from photonic_poc.models import build_model

# A single linear layer needs larger steps to converge within the same epoch budget.
DEFAULT_LEARNING_RATE = {"logreg": 3e-2, "mlp": 3e-3, "fourier": 3e-3}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float | None = None  # None: DEFAULT_LEARNING_RATE for the model
    weight_decay: float = 1e-4
    seed: int = 0
    # Noise-aware training: Gaussian noise at NPU-layer outputs during training only.
    noise_std: float = 0.0


@dataclass
class TrainResult:
    model: nn.Sequential
    best_epoch: int
    val_metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


def train_model(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> TrainResult:
    """Train with Adam on BCE-with-logits; keep the epoch with the lowest validation loss."""
    torch.manual_seed(config.seed)
    model = build_model(name, x_train.shape[1])
    lr = config.learning_rate or DEFAULT_LEARNING_RATE[name]
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train)
    xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)
    shuffle = torch.Generator().manual_seed(config.seed)
    noise = torch.Generator().manual_seed(config.seed + 1)

    best_loss, best_state, best_epoch = float("inf"), copy.deepcopy(model.state_dict()), 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=shuffle)
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            optimiser.zero_grad()
            with (
                output_noise(model, config.noise_std, noise) if config.noise_std else nullcontext()
            ):
                loss = loss_fn(model(xt[idx]).reshape(-1), yt[idx])
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(xv).reshape(-1)
            val_loss = float(loss_fn(val_logits, yv))
        history.append({"epoch": epoch, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_metrics = classification_metrics(y_val, model(xv).reshape(-1).numpy())
    return TrainResult(model, best_epoch, val_metrics, history)


def save_checkpoint(path: Path, name: str, result: TrainResult, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "state_dict": result.model.state_dict(),
            "best_epoch": result.best_epoch,
            "val_metrics": result.val_metrics,
            **meta,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[str, nn.Sequential, dict[str, Any]]:
    """Return (model name, model in eval mode, checkpoint metadata)."""
    checkpoint: dict[str, Any] = torch.load(path, weights_only=True)
    name = str(checkpoint["model_name"])
    model = build_model(name, int(checkpoint["n_features"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return name, model, checkpoint
