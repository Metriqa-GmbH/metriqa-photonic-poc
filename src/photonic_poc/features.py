"""Pair features: textbook string similarities, all in [0, 1], missing values give 0.

Features are deliberately generic (trigram Jaccard, SequenceMatcher, exact-match flags);
they are the input of the compact matching networks.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher

import numpy as np

from photonic_poc.data.records import CompanyRecord, LabeledPair

FEATURE_NAMES: tuple[str, ...] = (
    "name_trigram_jaccard",
    "name_token_jaccard",
    "name_seq_ratio",
    "name_prefix_ratio",
    "name_first_token_eq",
    "name_len_ratio",
    "legal_form_eq",
    "legal_form_missing",
    "street_trigram_jaccard",
    "house_number_eq",
    "postcode_eq",
    "postcode_prefix_eq",
    "city_eq",
    "country_eq",
    "phone_eq",
    "phone_missing",
    "domain_eq",
    "domain_missing",
    "domain_name_sim",
)

_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
# Order matters: the longest forms first. Input is folded, dot-free, "&" -> "und".
_LEGAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gmbh_co_kg", r"\bgmbh (?:und |u )?co (?:kg|k g)\b"),
    ("ug", r"\bug(?: haftungsbeschraenkt| haftungsbeschrankt| hb)?\b"),
    ("gmbh", r"\bgmbh\b"),
    ("ag", r"\bag\b"),
    ("kg", r"\bkg\b"),
    ("ohg", r"\bohg\b"),
    ("og", r"\bog\b"),
    ("ek", r"\be ?[ku]\b"),
    ("gbr", r"\bgbr\b"),
    ("sarl", r"\bsarl\b"),
    ("gen", r"\b(?:genossenschaft|gen)\b"),
)
_COMPILED = tuple((name, re.compile(pat)) for name, pat in _LEGAL_PATTERNS)
_ABBREV = (("handelsges", "handelsgesellschaft"), ("verw", "verwaltung"))


def _fold(s: str) -> str:
    """Lowercase, ASCII-fold, drop dots, turn other punctuation and "&" into spaces."""
    s = s.lower().translate(_FOLD).replace("&", " und ").replace(".", "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _split_name(name: str) -> tuple[str, str]:
    """(name without legal form, canonical legal form or "")."""
    text = _fold(name)
    form = ""
    for canon, pattern in _COMPILED:
        if pattern.search(text):
            form = canon
            text = " ".join(pattern.sub(" ", text).split())
            break
    for short, full in _ABBREV:
        text = re.sub(rf"\b{short}\b", full, text)
    return text, form


def _trigrams(s: str) -> set[str]:
    padded = f"  {s} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)} if s else set()


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _trigram_sim(a: str, b: str) -> float:
    return _jaccard(_trigrams(a), _trigrams(b))


def _street(s: str) -> str:
    s = _fold(s).replace("strasse", "str")
    return re.sub(r"\s+str\b", "str", s)


def _domain(website: str | None) -> str:
    if not website:
        return ""
    host = re.sub(r"^[a-z]+://", "", website.strip().lower())
    return re.sub(r"^www\.", "", host.split("/")[0])


def _city_eq(a: str, b: str) -> float:
    fa, fb = _fold(a).split(), _fold(b).split()
    if not fa or not fb:
        return 0.0
    return float(fa == fb or (fa[0] == fb[0] and min(len(fa), len(fb)) == 1))


def _prefix_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b))


def _digits(s: str | None) -> str:
    return re.sub(r"\D", "", s or "")


def pair_features(left: CompanyRecord, right: CompanyRecord) -> np.ndarray:
    """Feature vector in `FEATURE_NAMES` order; float32, every value in [0, 1]."""
    na, fa = _split_name(left.name)
    nb, fb = _split_name(right.name)
    ta, tb = na.split(), nb.split()
    da, db = _domain(left.website), _domain(right.website)
    pa, pb = _digits(left.phone)[-8:], _digits(right.phone)[-8:]
    pa, pb = (pa, pb) if len(pa) >= 6 and len(pb) >= 6 else ("", "")
    ha, hb = left.house_number.strip().lower(), right.house_number.strip().lower()
    za, zb = _digits(left.postcode), _digits(right.postcode)
    label_a, label_b = (d.rsplit(".", 1)[0].replace("-", " ") for d in (da, db))
    values = (
        _trigram_sim(na, nb),
        _jaccard(set(ta), set(tb)),
        SequenceMatcher(None, na, nb).ratio() if na and nb else 0.0,
        _prefix_ratio(na, nb),
        float(bool(ta) and bool(tb) and ta[0] == tb[0]),
        min(len(na), len(nb)) / max(len(na), len(nb), 1),
        float(bool(fa) and fa == fb),
        float(not fa or not fb),
        _trigram_sim(_street(left.street), _street(right.street)),
        float(bool(ha) and ha == hb),
        float(bool(za) and za == zb),
        float(len(za) >= 2 and za[:2] == zb[:2]),
        _city_eq(left.city, right.city),
        float(left.country == right.country),
        float(bool(pa) and pa == pb),
        float(not pa or not pb),
        float(bool(da) and da == db),
        float(not da or not db),
        max(_trigram_sim(label_a, nb), _trigram_sim(label_b, na)) if da or db else 0.0,
    )
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def featurize(pairs: Sequence[LabeledPair]) -> tuple[np.ndarray, np.ndarray]:
    """Stack features and labels: X float32 (n, F), y float32 (n,)."""
    x = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
    for i, pair in enumerate(pairs):
        x[i] = pair_features(pair.left, pair.right)
    y = np.asarray([p.label for p in pairs], dtype=np.float32)
    return x, y
