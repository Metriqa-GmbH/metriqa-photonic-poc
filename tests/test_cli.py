import json
from pathlib import Path

from conftest import HAS_QANT

from photonic_poc.cli import main


def test_reproduce_quick_end_to_end(tmp_path: Path) -> None:
    dirs = [
        "--data",
        str(tmp_path / "d"),
        "--artifacts",
        str(tmp_path / "a"),
        "--results",
        str(tmp_path / "r"),
    ]
    main(["reproduce", "--quick", *dirs, *([] if HAS_QANT else ["--no-qant"])])
    report = json.loads((tmp_path / "r" / "benchmark.json").read_text())
    backends = {row["backend"] for row in report["rows"]}
    assert backends == (
        {"torch-fp32", "torch-bf16", "qant"} if HAS_QANT else {"torch-fp32", "torch-bf16"}
    )
    assert len({row["model"] for row in report["rows"]}) == 5  # 3 plain + 2 noise-trained
    assert len(json.loads((tmp_path / "r" / "seed_study.json").read_text())) == 5
    assert "## Matching quality" in (tmp_path / "r" / "RESULTS.md").read_text(encoding="utf-8")
