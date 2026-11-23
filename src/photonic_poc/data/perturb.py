"""Perturbations that turn a clean record into a realistic second observation.

Every function takes an explicit `random.Random`, so output depends only on the seed.
"""

from __future__ import annotations

import dataclasses
import random
import re

from photonic_poc.data.records import CompanyRecord
from photonic_poc.data.vocabulary import CALLING_CODES, LEGAL_FORM_VARIANTS

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_ABBREVIATIONS = (("Handelsgesellschaft", "Handelsges."), ("Verwaltung", "Verw."))
# Longest first, so "GmbH & Co. KG" is recognised before "GmbH".
_FORMS_LONGEST_FIRST = sorted(LEGAL_FORM_VARIANTS, key=lambda f: (-len(f), f))


def typo(rng: random.Random, s: str) -> str:
    """Swap, delete, insert or substitute one character."""
    if len(s) < 3:
        return s
    op = rng.randrange(4)
    i = rng.randrange(len(s) - 1 if op == 0 else len(s))
    if op == 0:
        return s[:i] + s[i + 1] + s[i] + s[i + 2 :]
    if op == 1:
        return s[:i] + s[i + 1 :]
    letter = rng.choice(_LETTERS)
    return s[:i] + letter + (s[i:] if op == 2 else s[i + 1 :])


def abbreviate_name(rng: random.Random, name: str) -> str:
    """Abbreviate "Handelsgesellschaft"/"Verwaltung" or swap "&" and "und"."""
    for long, short in _ABBREVIATIONS:
        if long in name:
            return name.replace(long, short)
    if " & " in name:
        return name.replace(" & ", " und ", 1)
    if " und " in name:
        return name.replace(" und ", " & ", 1)
    return name


def abbreviate_street(rng: random.Random, street: str) -> str:
    """ "Hauptstraße" -> "Hauptstr." / "Hauptstrasse" / "Haupt Str."."""
    if not street.endswith("straße"):
        return street
    stem = street[: -len("straße")]
    return rng.choice((stem + "str.", stem + "strasse", stem + " Str.", stem + "str"))


def vary_legal_form(rng: random.Random, name: str) -> str:
    """Rewrite the trailing legal form ("GmbH" -> "G.m.b.H.") or drop it."""
    for form in _FORMS_LONGEST_FIRST:
        if name.endswith(" " + form):
            head = name[: -len(form) - 1]
            return (head + " " + rng.choice(LEGAL_FORM_VARIANTS[form])).strip()
    return name


def noise_text(rng: random.Random, s: str) -> str:
    """Case, punctuation or whitespace noise."""
    op = rng.randrange(5)
    if op == 0:
        return s.lower()
    if op == 1:
        return s.upper()
    if op == 2:
        return s.replace(" ", "  ", 1)
    if op == 3:
        return s.replace(" ", ", ", 1)
    return s.replace("-", " ") + " "


def reorder_tokens(rng: random.Random, name: str) -> str:
    """Swap two neighbouring words ("Velto Logistik GmbH" -> "Logistik Velto GmbH")."""
    tokens = name.split()
    if len(tokens) < 3:
        return name
    i = rng.randrange(len(tokens) - 2)
    tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    return " ".join(tokens)


def drop_word(rng: random.Random, name: str) -> str:
    """Drop one word after the first (typically the industry word)."""
    tokens = name.split()
    if len(tokens) < 3:
        return name
    del tokens[rng.randrange(1, len(tokens) - 1)]
    return " ".join(tokens)


def format_phone(cc: str, national: str, area_len: int, style: int) -> str:
    """Render national digits (area code + subscriber, no leading zero) in one of 6 styles."""
    area, sub = national[:area_len], national[area_len:]
    return (
        f"+{cc} {area} {sub}",
        f"0{area} {sub}",
        f"0{area}/{sub}",
        f"00{cc}{area}{sub}",
        f"(0{area}) {sub}",
        f"+{cc}-{area}-{sub}",
    )[style % 6]


def reformat_phone(rng: random.Random, phone: str, country: str) -> str:
    """Same number, different notation. The area-code split is a guess; digits are kept."""
    digits = re.sub(r"\D", "", phone)
    cc = CALLING_CODES[country]
    for prefix in ("00" + cc, cc, "0"):
        if digits.startswith(prefix):
            digits = digits[len(prefix) :]
            break
    return format_phone(cc, digits, min(3, len(digits) - 1), rng.randrange(6))


def vary_website(rng: random.Random, website: str) -> str:
    """Add or remove scheme, "www." and a path; occasionally swap the TLD."""
    host = re.sub(r"^https?://", "", website)
    host = re.sub(r"^www\.", "", host).split("/")[0]
    if rng.random() < 0.05:
        host = host.rsplit(".", 1)[0] + ".com"
    scheme = rng.choice(("", "http://", "https://"))
    www = rng.choice(("", "www."))
    path = rng.choice(("", "", "/", "/kontakt", "/impressum"))
    return f"{scheme}{www}{host}{path}"


def perturb_record(rng: random.Random, rec: CompanyRecord, strength: float) -> CompanyRecord:
    """Apply independent perturbations; `strength` scales every probability (1.0 = typical)."""

    def hit(p: float) -> bool:
        return rng.random() < p * strength

    name, street, number = rec.name, rec.street, rec.house_number
    postcode, city, phone, website = rec.postcode, rec.city, rec.phone, rec.website
    if hit(0.5):
        for _ in range(rng.choice((1, 1, 2))):
            name = typo(rng, name)
    if hit(0.25):
        name = abbreviate_name(rng, name)
    if hit(0.3):
        name = vary_legal_form(rng, name)
    if hit(0.35):
        name = noise_text(rng, name)
    if hit(0.12):
        name = reorder_tokens(rng, name)
    if hit(0.12):
        name = drop_word(rng, name)
    if hit(0.4):
        street = abbreviate_street(rng, street)
    if hit(0.15):
        street = typo(rng, street)
    if hit(0.05):
        number = number.rstrip("abc") if number[-1:] in "abc" else number + "a"
    if hit(0.06):  # stale or mistyped number
        number = str(max(1, int(number.rstrip("abc")) + rng.choice((-2, -1, 1, 2))))
    if hit(0.06):
        postcode = typo(rng, postcode)
    if hit(0.08):
        city = city.split()[0]
    if hit(0.05):
        city = typo(rng, city)
    if phone is not None:
        phone = (
            None if hit(0.12) else (reformat_phone(rng, phone, rec.country) if hit(0.3) else phone)
        )
    if website is not None:
        website = None if hit(0.15) else (vary_website(rng, website) if hit(0.4) else website)
    return dataclasses.replace(
        rec,
        name=name,
        street=street,
        house_number=number,
        postcode=postcode,
        city=city,
        phone=phone,
        website=website,
    )
