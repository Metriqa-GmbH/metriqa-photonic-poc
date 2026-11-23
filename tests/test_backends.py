import numpy as np
import pytest
import torch
from torch import nn

from photonic_poc.inference import NoisyTorchBackend, TorchBackend, create_backend
from photonic_poc.models import MODEL_NAMES, build_model
from photonic_poc.qant_backend import check_supported

N_FEATURES = 18


def _model(name: str) -> nn.Sequential:
    torch.manual_seed(0)
    return build_model(name, N_FEATURES).eval()


def _features(n: int = 512) -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 1, (n, N_FEATURES)).astype(np.float32)


def test_torch_backend_matches_module_forward() -> None:
    model, x = _model("mlp"), _features()
    expected = model(torch.from_numpy(x)).detach().numpy().reshape(-1)
    np.testing.assert_allclose(TorchBackend(model).predict_logits(x), expected, rtol=1e-6)


def test_noise_backend_is_seeded_and_zero_noise_is_identity() -> None:
    model, x = _model("mlp"), _features()
    reference = TorchBackend(model).predict_logits(x)
    np.testing.assert_allclose(NoisyTorchBackend(model, 0.0).predict_logits(x), reference)
    a = NoisyTorchBackend(model, 0.1, seed=3).predict_logits(x)
    np.testing.assert_array_equal(a, NoisyTorchBackend(model, 0.1, seed=3).predict_logits(x))
    assert not np.allclose(a, reference)


def test_check_supported_reports_unsupported_layers() -> None:
    assert all(check_supported(build_model(name, N_FEATURES)) == [] for name in MODEL_NAMES)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.LayerNorm(4), nn.Dropout(0.1))
    assert check_supported(model) == ["layer 1: Tanh", "layer 2: LayerNorm"]
    assert check_supported(nn.Linear(4, 1)) != []


# Relative RMS deviation from float32. A wrong weight mapping gives ~1.0. The Fourier bound is
# looser because the toolkit sums KAN terms in bf16 and the second layer's frequencies (k up to
# 4) amplify that error; see docs/sdk-capabilities.md.
REL_RMS_BOUND = {"logreg": 0.01, "mlp": 0.01, "fourier": 0.08}


@pytest.mark.qant
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_qant_backend_tracks_float32_reference(name: str) -> None:
    model, x = _model(name), _features()
    qant = create_backend("qant", model).predict_logits(x)
    fp32 = create_backend("torch-fp32", model).predict_logits(x)
    rel_rms = np.sqrt(np.mean((qant - fp32) ** 2) / np.mean(fp32**2))
    assert rel_rms < REL_RMS_BOUND[name]


@pytest.mark.qant
def test_qant_preloaded_and_streamed_weights_agree() -> None:
    from photonic_poc.qant_backend import QantBackend

    model, x = _model("mlp"), _features()
    preloaded = QantBackend(model, preload_weights=True)
    streamed = QantBackend(model, preload_weights=False)
    assert preloaded.preload_weights
    np.testing.assert_array_equal(preloaded.predict_logits(x), streamed.predict_logits(x))
    assert preloaded.describe()["simulated"] is True
