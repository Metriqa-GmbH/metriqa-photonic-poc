import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
import torch

from photonic_poc.inference import TorchBackend
from photonic_poc.models import build_model

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("label", ["mlp-8x4", "cos-8", "kan-3-g2", "logreg"])
def test_onnx_export_matches_torch(label: str, tmp_path: Path) -> None:
    from photonic_poc.onnx_export import export_onnx

    torch.manual_seed(0)
    model = build_model(label, 5).eval()
    x = np.random.default_rng(0).uniform(0, 1, (16, 5)).astype(np.float32)
    path = export_onnx(model, tmp_path / "m.onnx", batch=16, pad_output_to=8)
    out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    assert out.shape == (16, 8) and np.all(out[:, 1:] == 0)
    np.testing.assert_allclose(out[:, 0], TorchBackend(model).predict_logits(x), atol=1e-5)


@pytest.mark.amd_npu
@pytest.mark.skipif(sys.platform != "win32", reason="Windows ML is Windows-only")
def test_quantised_model_runs_identically_on_npu_and_cpu(tmp_path: Path) -> None:
    from photonic_poc import amd_npu
    from photonic_poc.onnx_export import export_onnx

    torch.manual_seed(0)
    model = build_model("mlp-8", 19).eval()
    x = np.random.default_rng(0).uniform(0, 1, (64, 19)).astype(np.float32)
    fp32 = export_onnx(model, tmp_path / "m.onnx", batch=64, pad_output_to=amd_npu.PAD_OUTPUT)
    q = amd_npu.quantize_a16w8(fp32, tmp_path / "m_q.onnx", x, batch=64)
    with ExitStack() as stack:
        try:  # never downloads the execution provider in tests
            devices = stack.enter_context(amd_npu.windows_ml(allow_download=False))
        except amd_npu.NpuUnavailable as exc:
            pytest.skip(str(exc))
        npu = amd_npu.predict(amd_npu.npu_session(q, devices), x, 64)
    cpu = amd_npu.predict(amd_npu.cpu_session(q), x, 64)
    np.testing.assert_allclose(npu, cpu, atol=1e-3)
