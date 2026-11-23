"""Train many (architecture, learning rate, seed) jobs in parallel, with a resumable cache.

Each job trains single-threaded, so results do not depend on the number of workers. Trained
weights are cached under ``artifacts/`` keyed by dataset and job; re-running skips them.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from photonic_poc.datasets import Dataset
from photonic_poc.models import build_model
from photonic_poc.training import TrainConfig, train_model

DEFAULT_CACHE = Path("artifacts/research/models")


@dataclass(frozen=True)
class Job:
    arch: str
    lr: float
    seed: int
    epochs: int = 60
    noise_std: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.arch}__lr{self.lr:g}__s{self.seed}__e{self.epochs}__n{self.noise_std:g}"


@dataclass(frozen=True)
class Trained:
    job: Job
    model: nn.Sequential
    best_epoch: int
    val_loss: float
    val_f1: float
    train_seconds: float


_DATA: Dataset | None = None


def _init_worker(dataset: Dataset) -> None:
    global _DATA
    _DATA = dataset
    torch.set_num_threads(1)


def _train(job: Job) -> dict[str, Any]:
    data = _DATA
    assert data is not None
    config = TrainConfig(
        epochs=job.epochs, learning_rate=job.lr, seed=job.seed, noise_std=job.noise_std
    )
    start = time.perf_counter()
    result = train_model(job.arch, data.x_train, data.y_train, data.x_val, data.y_val, config)
    return {
        "state_dict": result.model.state_dict(),
        "best_epoch": result.best_epoch,
        "val_loss": result.best_val_loss,
        "val_f1": result.val_metrics["f1"],
        "train_seconds": time.perf_counter() - start,
    }


def _restore(job: Job, record: dict[str, Any], n_features: int) -> Trained:
    model = build_model(job.arch, n_features)
    model.load_state_dict(record["state_dict"])
    model.eval()
    fields = ("best_epoch", "val_loss", "val_f1", "train_seconds")
    return Trained(job, model, *(record[f] for f in fields))


def run_jobs(
    jobs: Sequence[Job],
    dataset: Dataset,
    workers: int = 1,
    cache_dir: Path | None = DEFAULT_CACHE,
) -> dict[Job, Trained]:
    """Train (or load from cache) every job; returns models in eval mode."""
    folder = cache_dir / dataset.key if cache_dir else None
    results: dict[Job, Trained] = {}
    todo = []
    for job in dict.fromkeys(jobs):
        path = folder / f"{job.key}.pt" if folder else None
        if path is not None and path.exists():
            results[job] = _restore(job, torch.load(path, weights_only=True), dataset.n_features)
        else:
            todo.append(job)

    def store(job: Job, record: dict[str, Any]) -> None:  # as soon as a job finishes
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(record, folder / f"{job.key}.pt")
        results[job] = _restore(job, record, dataset.n_features)

    if todo and workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(dataset,)) as pool:
            futures = {pool.submit(_train, job): job for job in todo}
            for future in as_completed(futures):
                store(futures[future], future.result())
    elif todo:
        threads = torch.get_num_threads()
        _init_worker(dataset)
        try:
            for job in todo:
                store(job, _train(job))
        finally:
            torch.set_num_threads(threads)
    return {job: results[job] for job in dict.fromkeys(jobs)}
