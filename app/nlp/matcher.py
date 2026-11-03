"""Dictionary-based clinical term matcher.

All synonyms of all entity types are matched, then overlaps are resolved globally:
longest span wins, ties broken by type priority. This makes "rapid strep test" a procedure,
"hay fever" a diagnosis, and "fever" a symptom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from app.knowledge import terms
from app.knowledge.drugs import drug_aliases
from app.models.schemas import EntityType

_PRIORITY = {
    EntityType.PROCEDURE: 0,
    EntityType.DIAGNOSIS: 1,
    EntityType.FINDING: 2,
    EntityType.MEDICATION: 3,
    EntityType.SYMPTOM: 4,
}


@dataclass(frozen=True)
class TermMatch:
    start: int
    end: int
    type: EntityType
    canonical: str
    surface: str


def _key(surface: str) -> str:
    return re.sub(r"[\s\-]+", "", surface.lower())


def _pattern(surface: str) -> str:
    out = []
    for ch in surface.lower():
        if ch == " ":
            out.append(r"[\s\-]+")
        elif ch == "-":
            out.append(r"[\s\-]?")
        else:
            out.append(re.escape(ch))
    return "".join(out)


class LexiconMatcher:
    def __init__(self, vocab: dict[EntityType, dict[str, tuple[str, ...]]]):
        self._lookup: dict[EntityType, dict[str, str]] = {}
        self._regex: dict[EntityType, re.Pattern[str]] = {}
        for etype, entries in vocab.items():
            lookup: dict[str, str] = {}
            surfaces: list[str] = []
            for canonical, synonyms in entries.items():
                for surface in (canonical, *synonyms):
                    lookup.setdefault(_key(surface), canonical)
                    surfaces.append(surface)
            surfaces = sorted(set(surfaces), key=len, reverse=True)
            alternation = "|".join(_pattern(s) for s in surfaces)
            self._lookup[etype] = lookup
            self._regex[etype] = re.compile(rf"(?<![A-Za-z0-9])(?:{alternation})(?![A-Za-z0-9])", re.IGNORECASE)

    def find(self, text: str, types: set[EntityType] | None = None) -> list[TermMatch]:
        candidates: list[TermMatch] = []
        for etype, regex in self._regex.items():
            if types and etype not in types:
                continue
            for m in regex.finditer(text):
                canonical = self._lookup[etype].get(_key(m.group()))
                if canonical:
                    candidates.append(TermMatch(m.start(), m.end(), etype, canonical, m.group()))

        candidates.sort(key=lambda c: (-(c.end - c.start), _PRIORITY.get(c.type, 9), c.start))
        chosen: list[TermMatch] = []
        for cand in candidates:
            if all(cand.end <= c.start or cand.start >= c.end for c in chosen):
                chosen.append(cand)
        return sorted(chosen, key=lambda c: c.start)

    def canonical(self, etype: EntityType, surface: str) -> str | None:
        return self._lookup.get(etype, {}).get(_key(surface))


def _medication_vocab() -> dict[str, tuple[str, ...]]:
    vocab: dict[str, list[str]] = {}
    for surface, canonical in drug_aliases().items():
        vocab.setdefault(canonical, []).append(surface)
    return {k: tuple(v) for k, v in vocab.items()}


@lru_cache(maxsize=1)
def clinical_matcher() -> LexiconMatcher:
    return LexiconMatcher({
        EntityType.SYMPTOM: terms.SYMPTOMS,
        EntityType.FINDING: terms.FINDINGS,
        EntityType.DIAGNOSIS: terms.DIAGNOSES,
        EntityType.PROCEDURE: terms.PROCEDURES,
        EntityType.MEDICATION: _medication_vocab(),
    })
