import numpy as np

from photonic_poc.data import CompanyRecord, GeneratorConfig, generate_splits
from photonic_poc.features import FEATURE_NAMES, featurize, pair_features

REC = CompanyRecord(
    "Velto Logistik GmbH", "Kandorweg", "12a", "10115", "Berlin", "DE",
    "+49 30 1234567", "www.velto-logistik.de",
)  # fmt: skip


def test_identical_records_score_high() -> None:
    f = dict(zip(FEATURE_NAMES, pair_features(REC, REC), strict=True))
    for name in (
        "name_trigram_jaccard", "name_seq_ratio", "street_trigram_jaccard", "house_number_eq",
        "postcode_eq", "city_eq", "phone_eq", "domain_eq", "legal_form_eq",
    ):  # fmt: skip
        assert f[name] == 1.0, name


def test_normalisation_variants_match() -> None:
    other = CompanyRecord(
        "VELTO LOGISTIK G.m.b.H.", "Kandorstr.", "12A", "10115", "Berlin", "DE",
        "030/1234567", "https://velto-logistik.de/kontakt",
    )  # fmt: skip
    base = CompanyRecord(
        "Velto Logistik GmbH", "Kandorstrasse", "12a", "10115", "Berlin", "DE",
        "+49 30 1234567", "www.velto-logistik.de",
    )  # fmt: skip
    f = dict(zip(FEATURE_NAMES, pair_features(base, other), strict=True))
    assert f["name_token_jaccard"] == 1.0
    assert f["legal_form_eq"] == 1.0
    assert f["street_trigram_jaccard"] == 1.0
    assert f["phone_eq"] == 1.0 and f["domain_eq"] == 1.0


def test_missing_values_stay_finite_and_flagged() -> None:
    bare = CompanyRecord("", "", "", "", "", "DE", None, None)
    f = dict(zip(FEATURE_NAMES, pair_features(REC, bare), strict=True))
    assert np.isfinite(list(f.values())).all()
    assert f["phone_missing"] == 1.0 and f["domain_missing"] == 1.0
    assert f["legal_form_missing"] == 1.0


def test_featurize_shape_range_dtype() -> None:
    pairs = generate_splits(GeneratorConfig(n_entities=100, seed=1))["train"]
    x, y = featurize(pairs)
    assert x.shape == (len(pairs), len(FEATURE_NAMES)) and x.dtype == np.float32
    assert y.shape == (len(pairs),) and y.dtype == np.float32
    assert np.isfinite(x).all() and x.min() >= 0.0 and x.max() <= 1.0
