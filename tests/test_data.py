import subprocess
import sys
from pathlib import Path

import pytest

from photonic_poc.data import (
    PAIR_KINDS,
    CompanyRecord,
    GeneratorConfig,
    LabeledPair,
    generate_splits,
    load_pairs,
    save_pairs,
)

CONFIG = GeneratorConfig(n_entities=600, seed=3)
POSITIVE = {"duplicate", "relocated", "legal_form_change"}


@pytest.fixture(scope="module")
def splits() -> dict[str, list[LabeledPair]]:
    return generate_splits(CONFIG)


def test_same_config_same_output_across_processes(splits: dict[str, list[LabeledPair]]) -> None:
    code = (
        "import hashlib, json, dataclasses\n"
        "from photonic_poc.data import GeneratorConfig, generate_splits\n"
        "s = generate_splits(GeneratorConfig(n_entities=600, seed=3))\n"
        "blob = json.dumps({k: [dataclasses.asdict(p) for p in v] for k, v in s.items()})\n"
        "print(hashlib.sha256(blob.encode()).hexdigest())\n"
    )
    digests = set()
    for hash_seed in ("1", "2"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={**__import__("os").environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1
    assert splits == generate_splits(CONFIG)


def test_labels_and_kinds_are_valid(splits: dict[str, list[LabeledPair]]) -> None:
    assert set(splits) == {"train", "val", "test"}
    for pairs in splits.values():
        assert pairs
        for p in pairs:
            assert p.kind in PAIR_KINDS
            assert p.label == int(p.kind in POSITIVE)
        positives = sum(p.label for p in pairs)
        assert 0.25 < positives / len(pairs) < 0.42
    with pytest.raises(ValueError):
        LabeledPair(splits["train"][0].left, splits["train"][0].right, 1, "random")


def test_splits_share_no_entities(splits: dict[str, list[LabeledPair]]) -> None:
    # Records carry no ids; identical records across splits would mean a shared entity.
    def records(pairs: list[LabeledPair]) -> set[CompanyRecord]:
        return {r for p in pairs for r in (p.left, p.right)}

    seen = {k: records(v) for k, v in splits.items()}
    assert not seen["train"] & seen["val"]
    assert not seen["train"] & seen["test"]
    assert not seen["val"] & seen["test"]


def test_save_load_roundtrip(tmp_path: Path, splits: dict[str, list[LabeledPair]]) -> None:
    path = tmp_path / "pairs.jsonl"
    save_pairs(path, splits["val"])
    assert load_pairs(path) == splits["val"]
