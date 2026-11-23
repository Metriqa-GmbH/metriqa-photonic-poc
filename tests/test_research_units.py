import numpy as np
import pytest
import torch

from photonic_poc.hybrid import HybridOptions
from photonic_poc.models import ArchSpec, build_model, count_parameters
from photonic_poc.op_profile import profile_model
from photonic_poc.probes import boundary_inputs
from photonic_poc.stability import calibration, kind_attribution, precision_threshold
from photonic_poc.statistics import paired_bootstrap_f1, pareto_front


@pytest.mark.parametrize("label", ["logreg", "mlp-32x16", "cos-16x8", "kan-4x4-g2"])
def test_arch_labels_round_trip(label: str) -> None:
    assert ArchSpec.parse(label).label == label
    assert ArchSpec.parse("fourier").label == "kan-4-g4"
    with pytest.raises(ValueError):
        ArchSpec.parse("kan-4")  # grid is mandatory for KAN


def test_profile_counts_match_hand_calculation() -> None:
    mlp = profile_model(build_model("mlp-8", 4))  # Linear(4,8) ReLU Linear(8,1)
    assert mlp["macs"] == 4 * 8 + 8
    assert mlp["host_ops"] == 8 + 8 + 1  # two biases and the ReLU
    assert mlp["transfer_elements"] == (4 + 8) + (8 + 1)
    folded = profile_model(build_model("cos-8", 4), HybridOptions(fold_bias=True))
    assert folded["offload_fraction"] == 1.0  # matmuls with folded bias and cosine all on the NPU
    kan = profile_model(build_model("kan-2-g3", 4))
    terms = 2 * 4 * 3 + 1 * 2 * 3
    assert kan["nonlin_evals"] == terms
    assert kan["parameters"] == count_parameters(build_model("kan-2-g3", 4))


def test_calibration_and_precision_threshold() -> None:
    labels = np.array([0, 0, 1, 1, 1])
    logits = np.array([-3.0, 0.5, 1.0, 2.0, 3.0])
    assert calibration(labels, np.array([-50.0, -50, 50, 50, 50]))["ece"] < 1e-9
    assert precision_threshold(labels, logits, 1.0) == 1.0  # 0.5 is the only negative above it
    assert precision_threshold(np.array([0, 0]), np.array([1.0, 2.0]), 0.9) is None


def test_kind_attribution_expected_flips_sum_to_observed() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cand = ref + rng.normal(0, 0.05, 5000)
    kinds = rng.choice(["a", "b"], 5000)
    rows = kind_attribution(kinds, ref, cand)
    observed = sum(r["observed_flips"] for r in rows)
    assert observed == int(((ref > 0) != (cand > 0)).sum())
    assert sum(r["expected_flips"] for r in rows) == pytest.approx(observed)


def test_pareto_front_and_bootstrap() -> None:
    assert pareto_front([1, 2, 3, 4], [0.5, 0.4, 0.7, 0.6]) == [0, 2]
    y = np.random.default_rng(0).integers(0, 2, 2000)
    preds = np.stack([y.astype(bool)] * 3)
    same = paired_bootstrap_f1(y, preds, preds, n_boot=200)
    assert same["delta_f1"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    worse = preds.copy()
    worse[:, :200] = ~worse[:, :200]
    stats = paired_bootstrap_f1(y, preds, worse, n_boot=200)
    assert stats["ci_low"] > 0


def test_boundary_inputs_concentrate_near_zero_logit() -> None:
    torch.manual_seed(0)
    model = build_model("mlp-8", 4).eval()

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(x)).reshape(-1).numpy()

    x = np.random.default_rng(0).uniform(-3, 3, (500, 4)).astype(np.float32)
    if not ((predict(x) > 0).any() and (predict(x) <= 0).any()):
        pytest.skip("random model has a one-sided decision")
    probe = boundary_inputs(predict, x, 300, jitter=0.0)
    assert np.median(np.abs(predict(probe))) < 1e-3


def test_factorized_kan_identity_is_exact() -> None:
    from photonic_poc.kan_paths import factorized_mix_matrix

    torch.manual_seed(0)
    layer = build_model("kan-3-g4", 5)[0]
    x = torch.rand(64, 5, dtype=torch.float64)
    reference = layer.double()(x).detach().numpy()
    k = layer.get_buffer("k").double().numpy()
    kx = x.numpy()[:, :, None] * k  # (batch, input, frequency)
    basis = np.stack([np.cos(kx), np.sin(kx)], axis=-1).reshape(64, -1)
    mix = factorized_mix_matrix(
        layer.phase.detach().numpy(), layer.amplitude.detach().numpy(), layer.bias.detach().numpy()
    ).astype(np.float64)
    np.testing.assert_allclose(basis @ mix[:, :-1].T + mix[:, -1], reference, atol=1e-5)


@pytest.mark.parametrize("label", ["mlp-8x4", "cos-8", "kan-3-g2"])
def test_numpy_baseline_matches_torch(label: str) -> None:
    from photonic_poc.inference import TorchBackend
    from photonic_poc.numpy_forward import compile_numpy

    torch.manual_seed(0)
    model = build_model(label, 5).eval()
    x = np.random.default_rng(0).uniform(0, 1, (50, 5)).astype(np.float32)
    np.testing.assert_allclose(
        compile_numpy(model)(x), TorchBackend(model).predict_logits(x), atol=1e-5
    )
