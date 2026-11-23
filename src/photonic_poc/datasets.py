"""Featurised train/validation/test splits with an on-disk cache.

The cache key covers the generator config and the source of the generator and feature code,
so a code change invalidates cached features instead of silently reusing them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from photonic_poc import features
from photonic_poc.data import GeneratorConfig, generate_splits, generator, perturb

DEFAULT_CACHE = Path("data/cache")


@dataclass(frozen=True)
class Dataset:
    key: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    kinds_val: np.ndarray
    kinds_test: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])

    def summary(self) -> dict[str, int]:
        return {
            "train_pairs": len(self.y_train),
            "val_pairs": len(self.y_val),
            "test_pairs": len(self.y_test),
            "test_matches": int(self.y_test.sum()),
            "features": self.n_features,
        }


def dataset_key(config: GeneratorConfig) -> str:
    code = "".join(inspect.getsource(m) for m in (generator, perturb, features))
    payload = json.dumps(asdict(config), sort_keys=True) + code
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_dataset(config: GeneratorConfig, cache_dir: Path | None = DEFAULT_CACHE) -> Dataset:
    key = dataset_key(config)
    path = cache_dir / f"{key}.npz" if cache_dir else None
    if path is not None and path.exists():
        with np.load(path) as stored:
            return Dataset(key, **{name: stored[name] for name in stored.files})
    splits = generate_splits(config)
    arrays: dict[str, np.ndarray] = {}
    for split, pairs in splits.items():
        x, y = features.featurize(pairs)
        arrays[f"x_{split}"], arrays[f"y_{split}"] = x, y
        if split != "train":
            arrays[f"kinds_{split}"] = np.asarray([p.kind for p in pairs])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    return Dataset(key, **arrays)
