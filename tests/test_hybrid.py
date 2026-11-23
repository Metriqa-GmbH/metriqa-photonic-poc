import numpy as np
import pytest
import torch

from photonic_poc.inference import TorchBackend
from photonic_poc.models import build_model

pytestmark = pytest.mark.qant


def _setup(label: str) -> tuple[torch.nn.Sequential, np.ndarray, np.ndarray]:
    torch.manual_seed(0)
    model = build_model(label, 19).eval()
    x = np.random.default_rng(0).uniform(0, 1, (400, 19)).astype(np.float32)
    return model, x, TorchBackend(model).predict_logits(x)


def _rel_rms(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - ref) ** 2) / np.mean(ref**2)))


def test_toolkit_policy_is_the_qant_backend() -> None:
    from photonic_poc.hybrid import HybridExecutor
    from photonic_poc.qant_backend import QantBackend

    model, x, _ = _setup("mlp")
    np.testing.assert_array_equal(
        HybridExecutor(model).predict_logits(x), QantBackend(model).predict_logits(x)
    )


@pytest.mark.parametrize("label", ["mlp-16", "cos-16", "kan-4-g4"])
def test_all_policies_track_float32(label: str) -> None:
    from photonic_poc.hybrid import HybridExecutor, HybridOptions

    model, x, ref = _setup(label)
    policies = [
        HybridOptions(),
        HybridOptions(fold_bias=True),
        HybridOptions(host_precision="fp32"),
        HybridOptions(kan_impl="native", host_precision="fp32", wrap_phases=True),
        HybridOptions(kan_impl="factorized"),
    ]
    for options in policies:
        assert _rel_rms(HybridExecutor(model, options).predict_logits(x), ref) < 0.08, options.label


def test_native_kan_sums_more_accurately_than_toolkit() -> None:
    from photonic_poc.hybrid import HybridExecutor, HybridOptions

    model, x, ref = _setup("kan-8-g8")
    toolkit = _rel_rms(HybridExecutor(model).predict_logits(x), ref)
    native = _rel_rms(
        HybridExecutor(
            model, HybridOptions(kan_impl="native", host_precision="fp32")
        ).predict_logits(x),
        ref,
    )
    assert native < 0.7 * toolkit


def test_timed_run_reports_every_stage_and_conversions() -> None:
    from photonic_poc.hybrid import HybridExecutor

    model, x, _ = _setup("mlp-16")
    executor = HybridExecutor(model)
    logits, timings = executor.timed_run(x)
    assert {s.name for s in executor.stages} <= set(timings)
    assert "convert->bf16" in timings and "convert->fp32" in timings
    np.testing.assert_array_equal(logits, executor.predict_logits(x))
