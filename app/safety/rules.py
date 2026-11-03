"""Base class and shared context for deterministic safety rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models.schemas import (
    ACTIVE_ALLERGY_STATUSES,
    ACTIVE_MEDICATION_STATUSES,
    AFFIRMED_STATUSES,
    ClinicalEntity,
    Conflict,
    ConflictStatus,
    EntitySet,
    EntityStatus,
    EntityType,
    Evidence,
    Transcript,
)

NKDA = "no known drug allergies"


@dataclass
class SafetyContext:
    entities: EntitySet
    transcript: Transcript
    low_asr_confidence: float = 0.6

    def active_medications(self) -> list[ClinicalEntity]:
        return [e for e in self.entities.of_type(EntityType.MEDICATION) if e.status in ACTIVE_MEDICATION_STATUSES]

    def candidate_medications(self) -> list[ClinicalEntity]:
        """Active medications plus clinician mentions with unclear intent (reviewed, never ignored)."""
        return [e for e in self.entities.of_type(EntityType.MEDICATION)
                if e.status in ACTIVE_MEDICATION_STATUSES or e.status == EntityStatus.UNKNOWN]

    def active_allergies(self) -> list[ClinicalEntity]:
        return [e for e in self.entities.of_type(EntityType.ALLERGY)
                if e.status in ACTIVE_ALLERGY_STATUSES and e.normalized != NKDA]

    def sex(self) -> tuple[str, str, ClinicalEntity] | None:
        """(value, basis, entity) preferring explicitly stated sex over form of address."""
        candidates = [e for e in self.entities.of_type(EntityType.DEMOGRAPHIC) if e.normalized == "sex" and e.value]
        if not candidates:
            return None
        candidates.sort(key=lambda e: (e.attributes.get("basis") != "stated", -e.confidence))
        best = candidates[0]
        return best.value, best.attributes.get("basis", "stated"), best

    def age(self) -> tuple[int, ClinicalEntity] | None:
        ages = [e for e in self.entities.of_type(EntityType.DEMOGRAPHIC) if e.normalized == "age" and e.value]
        if not ages:
            return None
        best = max(ages, key=lambda e: e.confidence)
        return int(best.value), best

    def pregnancy(self) -> ClinicalEntity | None:
        return next((e for e in self.entities.of_type(EntityType.DIAGNOSIS)
                     if e.normalized == "pregnancy" and e.status in AFFIRMED_STATUSES), None)


class SafetyRule(ABC):
    rule_id: str = "BASE"
    description: str = ""

    @abstractmethod
    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        """Return the conflicts this rule detects."""


def evidence_of(*entities: ClinicalEntity) -> list[Evidence]:
    seen, out = set(), []
    for entity in entities:
        for ev in entity.evidence:
            if (ev.segment_id, ev.quote) not in seen:
                seen.add((ev.segment_id, ev.quote))
                out.append(ev)
    return sorted(out, key=lambda e: e.start_time)


def min_confidence(*entities: ClinicalEntity) -> float:
    return round(min((e.confidence for e in entities), default=1.0), 3)


def status_for(med: ClinicalEntity, block: bool) -> ConflictStatus:
    """Block only when the rule says so AND the medication is actively in use / being prescribed."""
    if block and med.status in ACTIVE_MEDICATION_STATUSES:
        return ConflictStatus.BLOCKED
    return ConflictStatus.REVIEW_REQUIRED
