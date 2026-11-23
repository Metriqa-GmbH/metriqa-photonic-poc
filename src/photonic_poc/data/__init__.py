"""Synthetic company records and labelled pairs for entity-matching experiments."""

from photonic_poc.data.generator import GeneratorConfig, generate_splits
from photonic_poc.data.records import (
    PAIR_KINDS,
    CompanyRecord,
    LabeledPair,
    load_pairs,
    save_pairs,
)

__all__ = [
    "PAIR_KINDS",
    "CompanyRecord",
    "GeneratorConfig",
    "LabeledPair",
    "generate_splits",
    "load_pairs",
    "save_pairs",
]
