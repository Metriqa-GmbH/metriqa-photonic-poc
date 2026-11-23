"""Synthetic company entities, families and labelled pairs.

Each split is generated on its own (own families, own business centres), so no entity or
family can cross a split boundary. Everything draws from one seeded `random.Random`; no set
is iterated and `hash()` is never used, so output is reproducible across processes.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass

from photonic_poc.data import perturb
from photonic_poc.data.records import CompanyRecord, LabeledPair
from photonic_poc.data.vocabulary import (
    CALLING_CODES,
    CITIES,
    COUNTRY_WEIGHTS,
    INDUSTRY_WORDS,
    LEGAL_FORMS,
    SISTER_WORDS,
    STREET_PREFIXES,
    STREET_SUFFIXES,
    SYLLABLES,
    TLDS,
)

_POSITIVE_MIX = (("duplicate", 0.75), ("relocated", 0.15), ("legal_form_change", 0.10))
_NEGATIVE_MIX = (
    ("random", 0.35),
    ("similar_name", 0.25),
    ("sister", 0.20),
    ("shared_address", 0.20),
)
_LIGHT = 0.5  # noise on both sides of every pair, so neither side is a pristine reference
_City = tuple[str, str, str, str]
_Address = tuple[str, str, str, _City]  # street, house number, postcode, city
_SLUG = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", " ": "-", "&": ""})


@dataclass(frozen=True)
class GeneratorConfig:
    """Size, seed and split fractions (test = remainder)."""

    n_entities: int = 6000
    seed: int = 7
    train_fraction: float = 0.70
    val_fraction: float = 0.15


@dataclass(frozen=True, slots=True)
class _Entity:
    family: int
    stem: str
    industry: str
    form: str
    record: CompanyRecord


class _World:
    """The entities of one split plus the lookups the pair makers need."""

    def __init__(self, rng: random.Random, used_stems: set[str], next_family: int) -> None:
        self.rng, self.used_stems, self.next_family = rng, used_stems, next_family
        self.entities: list[_Entity] = []
        self.families: dict[int, list[_Entity]] = {}
        self.twins: list[tuple[_Entity, _Entity]] = []
        self.centres: list[list[_Entity]] = []

    # --- building blocks -------------------------------------------------------------
    def city(self, country: str | None = None) -> _City:
        if country is None:
            countries, weights = [c for c, _ in COUNTRY_WEIGHTS], [w for _, w in COUNTRY_WEIGHTS]
            country = self.rng.choices(countries, weights)[0]
        return self.rng.choice([c for c in CITIES if c[1] == country])

    def syllables(self, n: int) -> str:
        return "".join(self.rng.choice(SYLLABLES) for _ in range(n))

    def address(self, city: _City) -> _Address:
        rng = self.rng
        if rng.random() < 0.5:
            street = rng.choice(STREET_PREFIXES) + rng.choice(STREET_SUFFIXES)
        else:
            street = self.syllables(2).capitalize() + rng.choice(STREET_SUFFIXES)
        number = str(rng.randint(1, 150)) + (rng.choice("abc") if rng.random() < 0.15 else "")
        tail = "".join(str(rng.randrange(10)) for _ in range(3 if city[1] == "DE" else 2))
        return street, number, city[2] + tail, city

    def stem(self, like: str | None = None) -> str:
        """A fresh unused stem; with `like`, one 1-2 character edits away from it."""
        while True:
            if like is None:
                stem = self.syllables(self.rng.choice((2, 2, 3)))
            else:
                stem = like.lower()
                for _ in range(self.rng.choice((1, 1, 2))):
                    stem = perturb.typo(self.rng, stem)
            if len(stem) >= 4 and stem.isalpha() and stem not in self.used_stems:
                self.used_stems.add(stem)
                return stem.capitalize()

    def legal_form(self, country: str, avoid: tuple[str, ...] = ()) -> str:
        forms = [(f, w) for f, w in LEGAL_FORMS[country] if f not in avoid] or LEGAL_FORMS[country]
        return self.rng.choices([f for f, _ in forms], [w for _, w in forms])[0]

    def phone(self, city: _City) -> str | None:
        if self.rng.random() < 0.15:
            return None
        sub = "".join(str(self.rng.randrange(10)) for _ in range(self.rng.randint(5, 8)))
        style = self.rng.randrange(6)
        return perturb.format_phone(CALLING_CODES[city[1]], city[3] + sub, len(city[3]), style)

    def website(self, stem: str, industry: str, country: str) -> str | None:
        if self.rng.random() < 0.25:
            return None
        label = stem.lower()
        if industry and self.rng.random() < 0.7:
            label += "-" + industry.lower().translate(_SLUG).replace("--", "-")
        scheme = self.rng.choice(("", "", "http://", "https://"))
        return f"{scheme}{self.rng.choice(('www.', '', 'www.'))}{label}.{TLDS[country]}"

    def industry(self) -> str:
        first = self.rng.choice(INDUSTRY_WORDS)
        if self.rng.random() < 0.1:
            return f"{first} & {self.rng.choice(INDUSTRY_WORDS)}"
        return first

    # --- families --------------------------------------------------------------------
    def add_family(
        self, centre: _Address | None = None, twin_of: _Entity | None = None
    ) -> list[_Entity]:
        """Add 1-3 entities sharing a brand stem; a twin is a single look-alike entity."""
        rng = self.rng
        fid, self.next_family = self.next_family, self.next_family + 1
        if twin_of is not None:
            old = twin_of.record
            home = next(c for c in CITIES if (c[0], c[1]) == (old.city, old.country))
            city = home if rng.random() < 0.6 else self.city(old.country)
            stem, size = self.stem(like=twin_of.stem), 1
        else:
            city = centre[3] if centre else self.city()
            stem, size = self.stem(), rng.choices((1, 2, 3), (0.6, 0.25, 0.15))[0]
        members: list[_Entity] = []
        for k in range(size):
            first = members[0] if members else None
            if twin_of is not None:
                industry = twin_of.industry
            elif first is None:
                industry = "" if rng.random() < 0.12 else self.industry()
            else:
                pool = SISTER_WORDS if rng.random() < 0.5 else INDUSTRY_WORDS
                industry = first.industry
                if rng.random() < 0.7:  # else: same line of business, told apart by form only
                    industry = rng.choice([w for w in pool if w != first.industry])
            form = self.legal_form(city[1], tuple(m.form for m in members))
            if first is not None and rng.random() < 0.5:  # sisters at one address
                r0 = first.record
                addr: _Address = (r0.street, r0.house_number, r0.postcode, city)
                phone = r0.phone if rng.random() < 0.3 else self.phone(city)
            elif centre is not None:
                addr, phone = centre, self.phone(city)
            else:
                where = city if (k == 0 or rng.random() < 0.8) else self.city()
                addr, phone = self.address(where), self.phone(where)
            if first is not None and first.record.website and rng.random() < 0.5:
                site: str | None = first.record.website  # sisters sharing a domain
            else:
                site = self.website(stem, industry, addr[3][1])
            name = " ".join(w for w in (stem, industry, form) if w)
            street, number, postcode, c = addr
            rec = CompanyRecord(name, street, number, postcode, c[0], c[1], phone, site)
            members.append(_Entity(fid, stem, industry, form, rec))
        self.entities.extend(members)
        self.families[fid] = members
        return members


def _build_world(rng: random.Random, used: set[str], next_family: int, target: int) -> _World:
    """Families, look-alike twins and business centres until `target` entities exist."""
    world = _World(rng, used, next_family)
    centres = [world.address(world.city()) for _ in range(max(2, target // 80))]
    groups: list[list[_Entity]] = [[] for _ in centres]
    while len(world.entities) < target:
        r = rng.random()
        if r < 0.12 and world.entities:
            base = rng.choice(world.entities)
            world.twins.append((base, world.add_family(twin_of=base)[0]))
        elif r < 0.22:
            i = rng.randrange(len(centres))
            groups[i].extend(world.add_family(centre=centres[i]))
        else:
            world.add_family()
    world.centres = [g for g in groups if len({e.family for e in g}) >= 2]
    return world


def _relocate(world: _World, rec: CompanyRecord) -> CompanyRecord:
    """New address (maybe new city and phone); the website usually stays."""
    rng = world.rng
    home = next(c for c in CITIES if (c[0], c[1]) == (rec.city, rec.country))
    new_city = world.city(rec.country) if rng.random() < 0.5 else home
    street, number, postcode, city = world.address(new_city)
    phone = rec.phone
    if phone is not None and rng.random() < (0.8 if city[0] != rec.city else 0.3):
        phone = world.phone(city)
    site = rec.website if rng.random() < 0.85 else None
    return dataclasses.replace(
        rec, street=street, house_number=number, postcode=postcode, city=city[0],
        phone=phone, website=site,
    )  # fmt: skip


def _positive(world: _World, kind: str, ent: _Entity) -> tuple[CompanyRecord, CompanyRecord]:
    rng, rec = world.rng, ent.record
    if kind == "duplicate":
        return rec, perturb.perturb_record(rng, rec, 0.5 + 2.0 * rng.random())
    if kind == "relocated":
        new = _relocate(world, rec)
        return rec, perturb.perturb_record(rng, new, 0.8) if rng.random() < 0.4 else new
    form = world.legal_form(rec.country, (ent.form,))
    name = " ".join(w for w in (ent.stem, ent.industry, form) if w)
    return rec, perturb.perturb_record(rng, dataclasses.replace(rec, name=name), 0.5)


def _negative(world: _World, kind: str) -> tuple[CompanyRecord, CompanyRecord] | None:
    """One negative pair of `kind`, or None when this split cannot supply it."""
    rng = world.rng
    if kind == "similar_name" and world.twins:
        a, b = rng.choice(world.twins)
    elif kind == "sister":
        multi = [m for m in world.families.values() if len(m) > 1]
        if not multi:
            return None
        a, b = rng.sample(rng.choice(multi), 2)
    elif kind == "shared_address" and world.centres:
        a, b = rng.sample(rng.choice(world.centres), 2)
    elif kind == "random":
        a, b = rng.sample(world.entities, 2)
    else:
        return None
    return (a.record, b.record) if a.family != b.family or kind == "sister" else None


def _counts(total: int, mix: tuple[tuple[str, float], ...]) -> list[tuple[str, int]]:
    counts = [(k, round(total * f)) for k, f in mix]
    counts[0] = (counts[0][0], total - sum(c for _, c in counts[1:]))
    return counts


def _make_pairs(world: _World) -> list[LabeledPair]:
    """One positive per entity and two negatives per positive, in the configured mix."""
    rng = world.rng
    order = list(world.entities)
    rng.shuffle(order)
    pairs: list[LabeledPair] = []
    todo = [(k, 1, n) for k, n in _counts(len(order), _POSITIVE_MIX)]
    todo += [(k, 0, n) for k, n in _counts(2 * len(order), _NEGATIVE_MIX)]
    entities = iter(order)
    for kind, label, n in todo:
        for _ in range(n):
            final = kind
            if label:
                left, right = _positive(world, kind, next(entities))
            else:
                found, used = None, kind
                while found is None:
                    found = _negative(world, used)
                    if found is None:
                        used = "random"
                left, right = found
                final = used
            left = perturb.perturb_record(rng, left, _LIGHT)
            right = perturb.perturb_record(rng, right, _LIGHT)
            if rng.random() < 0.5:
                left, right = right, left
            pairs.append(LabeledPair(left, right, label, final))
    rng.shuffle(pairs)
    return pairs


def generate_splits(config: GeneratorConfig) -> dict[str, list[LabeledPair]]:
    """Return {"train", "val", "test"}; splits share no entity, family or brand stem."""
    rng = random.Random(config.seed)
    n = config.n_entities
    n_train, n_val = round(n * config.train_fraction), round(n * config.val_fraction)
    sizes = {"train": n_train, "val": n_val, "test": n - n_train - n_val}
    used: set[str] = set()  # membership tests only, never iterated
    next_family = 0
    out: dict[str, list[LabeledPair]] = {}
    for name, size in sizes.items():
        world = _build_world(rng, used, next_family, size)
        next_family = world.next_family
        out[name] = _make_pairs(world)
    return out
