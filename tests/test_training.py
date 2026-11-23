import numpy as np
import torch

from photonic_poc.training import TrainConfig, train_model


def _toy_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 6)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 1.0).astype(np.float32)
    return x, y


def test_training_learns_and_is_deterministic() -> None:
    x_train, y_train = _toy_data(1024, 0)
    x_val, y_val = _toy_data(256, 1)
    config = TrainConfig(epochs=15, seed=0)
    first = train_model("mlp", x_train, y_train, x_val, y_val, config)
    second = train_model("mlp", x_train, y_train, x_val, y_val, config)
    assert first.val_metrics["f1"] > 0.9
    for a, b in zip(
        first.model.state_dict().values(), second.model.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
