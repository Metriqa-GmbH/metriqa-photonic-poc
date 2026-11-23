import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from photonic_poc import private_eval as pe


def factory() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 1)).eval()


def _arrays(n: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "x_test": rng.normal(size=(n, 6)).astype(np.float32),
        "y_test": rng.integers(0, 2, n),
        "x_val": rng.normal(size=(n, 6)).astype(np.float32),
        "y_val": rng.integers(0, 2, n),
        "kinds_test": np.array(["secret_kind_a", "secret_kind_b"] * (n // 2)),
    }


def _backends() -> list[str]:
    return ["torch-fp32", "torch-bf16"] + (["qant"] if pe.toolkit_available() else [])


def test_refuses_to_write_inside_repo() -> None:
    target = pe.REPO_ROOT / "private_out"
    with pytest.raises(PermissionError):
        pe.write_private_outputs({}, {}, target)
    assert not target.exists()


def test_end_to_end_summary_is_sanitised(tmp_path: Path) -> None:
    np.savez(tmp_path / "d.npz", **_arrays())
    loaded = pe.load_arrays(tmp_path / "d.npz")
    report = pe.evaluate_private(factory(), loaded, _backends())
    summary = pe.aggregate_summary(report)
    paths = pe.write_private_outputs(report, summary, tmp_path / "out")
    assert paths[0].name == "private_report.json"
    assert paths[2].read_text(encoding="utf-8").startswith("# NOT APPROVED FOR PUBLICATION")
    assert summary["approved_for_publication"] is False and summary["approval"] is None
    assert summary["n_test_approx"] == 300
    assert set(summary) <= pe.TOP_KEYS
    text = json.dumps(summary)
    assert "secret_kind" not in text and "layer" not in text and str(tmp_path) not in text
    assert all(not isinstance(v, list) or len(v) <= 5 for v in summary.values())
    pe._validate_summary(summary)


@pytest.mark.qant
def test_qant_comparison_present() -> None:
    summary = pe.aggregate_summary(pe.evaluate_private(factory(), _arrays(), ["qant"]))
    assert set(summary["qant_vs_fp32"]) == set(pe.COMPARISON_KEYS)


def test_unsupported_model_skips_toolkit() -> None:
    model = nn.Sequential(nn.Linear(6, 4), nn.Tanh(), nn.Linear(4, 1))
    report = pe.evaluate_private(model, _arrays(), ["torch-fp32", "qant"])
    assert "qant" in report["skipped_backends"] and "stability" not in report


def test_approve_adds_hash_and_rejects_extra_key(tmp_path: Path) -> None:
    summary = pe.aggregate_summary(pe.evaluate_private(factory(), _arrays(), _backends()))
    src = tmp_path / "s.json"
    src.write_text(json.dumps(summary))
    approved = pe.approve_summary(src, "A. Reviewer", tmp_path / "ok.json")
    assert approved["approved_for_publication"] is True
    assert len(approved["approval"]["sha256"]) == 64
    summary["feature_names"] = ["leak"]
    src.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="whitelist"):
        pe.approve_summary(src, "A. Reviewer", tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_load_model_and_array_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("fake_private_models")
    mod.make = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_private_models", mod)
    assert isinstance(pe.load_model("fake_private_models:make"), nn.Sequential)
    with pytest.raises(ValueError, match=r"package\.module"):
        pe.load_model("no_colon")
    bad = {"x_test": np.zeros((4, 3), np.float32), "y_test": np.array([0, 1, 2, 0])}
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="0 and 1"):
        pe.load_arrays(tmp_path / "bad.npz")
