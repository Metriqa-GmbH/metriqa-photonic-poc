import json
from pathlib import Path

import pytest

from photonic_poc.cli import main

pytestmark = pytest.mark.qant


def test_research_pipeline_quick_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)  # caches (data/, artifacts/) go to the temporary directory
    out = tmp_path / "research"
    main(["research", "--config", str(root / "configs" / "quick.toml"), "--out", str(out)])
    arch = json.loads((out / "architecture.json").read_text())
    assert {r["family"] for r in arch["records"]} == {"logreg", "mlp", "cos", "kan"}
    assert {"kan", "mlp"} <= set(arch["champions"])
    rob = json.loads((out / "robustness.json").read_text())
    assert rob["models"] and rob["depth"]
    hyb = json.loads((out / "hybrid.json").read_text())
    assert hyb["layer_study"] and hyb["host_microbenchmarks"]["call_overhead_us"]
    assert len(list((out / "figures").glob("*.png"))) == 7
