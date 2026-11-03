"""Deterministic safety engine.

Fail-closed: if a rule raises, the consultation gets a REVIEW_REQUIRED conflict instead of
silently passing. Output is deterministic (stable ordering and ids) for auditability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.models.schemas import Conflict, ConflictStatus, ConflictType, EntitySet, Severity, Transcript
from app.safety.allergy_rules import AllergyCrossReactivityRule, AllergyMedicationRule
from app.safety.clinical_rules import InconsistentHistoryRule, TranscriptionQualityRule, VitalSignRule
from app.safety.demographic_rules import AgeContraindicationRule, PregnancyContraindicationRule, SexSpecificRule
from app.safety.medication_rules import DoseRangeRule, DrugInteractionRule, DuplicateTherapyRule
from app.safety.rules import SafetyContext, SafetyRule

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
_STATUS_ORDER = {ConflictStatus.BLOCKED: 0, ConflictStatus.REVIEW_REQUIRED: 1}


def default_rules() -> list[SafetyRule]:
    return [
        AllergyMedicationRule(),
        AllergyCrossReactivityRule(),
        DrugInteractionRule(),
        DuplicateTherapyRule(),
        DoseRangeRule(),
        SexSpecificRule(),
        PregnancyContraindicationRule(),
        AgeContraindicationRule(),
        VitalSignRule(),
        InconsistentHistoryRule(),
        TranscriptionQualityRule(),
    ]


@dataclass
class SafetyResult:
    conflicts: list[Conflict] = field(default_factory=list)
    blocked_entity_ids: set[str] = field(default_factory=set)

    @property
    def has_blocking(self) -> bool:
        return any(c.status == ConflictStatus.BLOCKED for c in self.conflicts)


class SafetyEngine:
    def __init__(self, rules: list[SafetyRule] | None = None, low_asr_confidence: float = 0.6):
        self.rules = rules if rules is not None else default_rules()
        self.low_asr_confidence = low_asr_confidence

    def run(self, entities: EntitySet, transcript: Transcript) -> SafetyResult:
        ctx = SafetyContext(entities, transcript, self.low_asr_confidence)
        conflicts: list[Conflict] = []
        for rule in self.rules:
            try:
                conflicts.extend(rule.evaluate(ctx))
            except Exception as exc:  # noqa: BLE001 - fail closed on any rule bug
                logger.error("Safety rule %s failed: %s", rule.rule_id, type(exc).__name__)
                conflicts.append(Conflict(
                    rule_id=rule.rule_id,
                    type=ConflictType.RULE_ENGINE_ERROR,
                    severity=Severity.HIGH,
                    status=ConflictStatus.REVIEW_REQUIRED,
                    message=f"Safety rule {rule.rule_id} could not be evaluated; manual safety review required.",
                ))

        unique: dict[tuple, Conflict] = {}
        for c in conflicts:
            key = (c.rule_id, c.type, tuple(sorted(c.entity_ids)), c.message)
            unique.setdefault(key, c)
        ordered = sorted(unique.values(), key=lambda c: (_STATUS_ORDER[c.status], _SEVERITY_ORDER[c.severity],
                                                          c.rule_id, c.message))
        for i, conflict in enumerate(ordered, start=1):
            conflict.conflict_id = f"K{i:03d}"

        blocked = {eid for c in ordered if c.status == ConflictStatus.BLOCKED for eid in c.blocked_entity_ids}
        return SafetyResult(ordered, blocked)
