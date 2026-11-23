"""Record types and JSONL persistence for labelled company pairs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PAIR_KINDS: tuple[str, ...] = (
    "duplicate",
    "relocated",
    "legal_form_change",
    "random",
    "similar_name",
    "sister",
    "shared_address",
)
POSITIVE_KINDS: tuple[str, ...] = PAIR_KINDS[:3]
NEGATIVE_KINDS: tuple[str, ...] = PAIR_KINDS[3:]


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    """One company as it might appear in a business register or web listing."""

    name: str
    street: str
    house_number: str
    postcode: str
    city: str
    country: str
    phone: str | None
    website: str | None


@dataclass(frozen=True, slots=True)
class LabeledPair:
    """Two records plus the ground truth: label 1 = same company, 0 = different."""

    left: CompanyRecord
    right: CompanyRecord
    label: int
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in PAIR_KINDS:
            raise ValueError(f"unknown pair kind: {self.kind!r}")
        if self.label != int(self.kind in POSITIVE_KINDS):
            raise ValueError(f"label {self.label} contradicts kind {self.kind!r}")


def save_pairs(path: Path, pairs: Sequence[LabeledPair]) -> None:
    """Write one JSON object per line (UTF-8, umlauts kept readable)."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for pair in pairs:
            fh.write(json.dumps(asdict(pair), ensure_ascii=False) + "\n")


def load_pairs(path: Path) -> list[LabeledPair]:
    """Read pairs written by `save_pairs`; validates label and kind."""
    pairs: list[LabeledPair] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj: dict[str, Any] = json.loads(line)
            pairs.append(
                LabeledPair(
                    left=CompanyRecord(**obj["left"]),
                    right=CompanyRecord(**obj["right"]),
                    label=int(obj["label"]),
                    kind=str(obj["kind"]),
                )
            )
    return pairs
