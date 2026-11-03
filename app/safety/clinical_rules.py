"""Clinical-data integrity rules: vital signs, inconsistent history, transcription quality."""

from __future__ import annotations

from app.models.schemas import (
    AFFIRMED_STATUSES,
    ClinicalEntity,
    Conflict,
    ConflictStatus,
    ConflictType,
    EntityStatus,
    EntityType,
    Evidence,
    Severity,
    Speaker,
)
from app.safety.rules import NKDA, SafetyContext, SafetyRule, evidence_of, min_confidence

# name -> ((plausible low, plausible high), is_critical(value))
_TEMP_F = ((90.0, 110.0), lambda v: v >= 104 or v < 95)
_TEMP_C = ((32.0, 43.5), lambda v: v >= 40 or v < 35)
_SIMPLE_VITALS = {
    "heart rate": ((20, 250), lambda v: v > 130 or v < 40),
    "respiratory rate": ((4, 60), lambda v: v > 30 or v < 8),
    "oxygen saturation": ((50, 100), lambda v: v < 90),
}


def _vital_assessment(vital: ClinicalEntity) -> tuple[str, str] | None:
    """Return (kind, reason) where kind is 'implausible' or 'critical'."""
    try:
        if vital.normalized == "blood pressure":
            systolic, diastolic = (int(x) for x in vital.value.split("/"))
            if not (50 <= systolic <= 260 and 20 <= diastolic <= 160) or diastolic >= systolic:
                return "implausible", f"blood pressure {vital.value} is physiologically implausible"
            if systolic >= 180 or diastolic >= 120:
                return "critical", f"blood pressure {vital.value} mmHg is in the hypertensive-crisis range"
            if systolic < 90:
                return "critical", f"blood pressure {vital.value} mmHg indicates hypotension"
            return None
        value = float(vital.value)
    except (TypeError, ValueError, AttributeError):
        return "implausible", f"{vital.normalized} value {vital.value!r} could not be parsed"

    if vital.normalized == "temperature":
        (low, high), critical = _TEMP_C if vital.unit == "°C" else _TEMP_F
    elif vital.normalized in _SIMPLE_VITALS:
        (low, high), critical = _SIMPLE_VITALS[vital.normalized]
    else:
        return None
    unit = vital.unit or ""
    if not low <= value <= high:
        return "implausible", f"{vital.normalized} {vital.value}{unit} is outside the plausible range"
    if critical(value):
        return "critical", f"{vital.normalized} {vital.value}{unit} is a critical value"
    return None


class VitalSignRule(SafetyRule):
    rule_id = "VITAL_SIGNS"
    description = "Implausible (likely transcription error), critical, or conflicting vital-sign readings."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        vitals = ctx.entities.of_type(EntityType.VITAL)
        for vital in vitals:
            assessment = _vital_assessment(vital)
            if assessment is None:
                continue
            kind, reason = assessment
            conflicts.append(Conflict(
                rule_id=self.rule_id,
                type=ConflictType.VITAL_SIGN_ALERT,
                severity=Severity.HIGH if kind == "critical" else Severity.MEDIUM,
                status=ConflictStatus.REVIEW_REQUIRED,
                message=reason[:1].upper() + reason[1:] + ("." if kind == "critical" else " - possible transcription error."),
                involved_terms=[vital.normalized],
                entity_ids=[vital.entity_id],
                evidence=evidence_of(vital),
                recommendation="Re-measure or verify against the device reading." if kind == "implausible"
                else "Confirm the reading and assess urgently if accurate.",
                confidence=min_confidence(vital),
            ))

        by_name: dict[str, list[ClinicalEntity]] = {}
        for vital in vitals:
            by_name.setdefault(vital.normalized, []).append(vital)
        for name, readings in by_name.items():
            if len({r.value for r in readings}) > 1:
                values = ", ".join(f"{r.value} {r.unit}" if r.unit else str(r.value) for r in readings)
                conflicts.append(Conflict(
                    rule_id=self.rule_id,
                    type=ConflictType.VITAL_SIGN_ALERT,
                    severity=Severity.MEDIUM,
                    status=ConflictStatus.REVIEW_REQUIRED,
                    message=f"Conflicting {name} readings: {values}.",
                    involved_terms=[name],
                    entity_ids=[r.entity_id for r in readings],
                    evidence=evidence_of(*readings),
                    recommendation="Confirm which reading should be documented.",
                    confidence=min_confidence(*readings),
                ))
        return conflicts


class InconsistentHistoryRule(SafetyRule):
    rule_id = "INCONSISTENT_HISTORY"
    description = "Patient statements that contradict each other within the consultation."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        allergies = ctx.entities.of_type(EntityType.ALLERGY)
        nkda = next((a for a in allergies if a.normalized == NKDA), None)
        active = ctx.active_allergies()
        if nkda and active:
            conflicts.append(self._conflict(
                Severity.MEDIUM, f"Patient reported no known drug allergies but also an allergy to "
                f"{', '.join(a.normalized for a in active)}.", [nkda, *active]))

        for entity in ctx.entities.of_type(EntityType.ALLERGY, EntityType.SYMPTOM, EntityType.DIAGNOSIS):
            history = set(entity.status_history)
            if history & AFFIRMED_STATUSES and EntityStatus.ABSENT in history:
                severity = Severity.MEDIUM if entity.type == EntityType.ALLERGY else Severity.LOW
                conflicts.append(self._conflict(
                    severity, f"{entity.display_name} was both affirmed and denied during the consultation.", [entity]))

        for field_name, severity in (("sex", Severity.MEDIUM), ("age", Severity.LOW)):
            values = [e for e in ctx.entities.of_type(EntityType.DEMOGRAPHIC) if e.normalized == field_name]
            if len({e.value for e in values}) > 1:
                conflicts.append(self._conflict(
                    severity, f"Conflicting {field_name} values: {', '.join(sorted({str(e.value) for e in values}))}.",
                    values))
        return conflicts

    def _conflict(self, severity: Severity, message: str, entities: list[ClinicalEntity]) -> Conflict:
        return Conflict(
            rule_id=self.rule_id,
            type=ConflictType.INCONSISTENT_HISTORY,
            severity=severity,
            status=ConflictStatus.REVIEW_REQUIRED,
            message=message,
            involved_terms=[e.normalized for e in entities],
            entity_ids=[e.entity_id for e in entities],
            evidence=evidence_of(*entities),
            recommendation="Clarify with the patient before finalising the note.",
            confidence=min_confidence(*entities),
        )


_SAFETY_CRITICAL_TYPES = {EntityType.ALLERGY, EntityType.MEDICATION, EntityType.VITAL, EntityType.DIAGNOSIS}


class TranscriptionQualityRule(SafetyRule):
    rule_id = "TRANSCRIPTION_QUALITY"
    description = "Safety-critical facts that rest on low-confidence ASR or an unidentified speaker."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        for segment in ctx.transcript.segments:
            dependents = [e for e in ctx.entities.items if segment.segment_id in e.segment_ids]
            critical = [e for e in dependents if e.type in _SAFETY_CRITICAL_TYPES]
            evidence = [Evidence.from_segment(segment)]
            if critical and segment.confidence is not None and segment.confidence < ctx.low_asr_confidence:
                conflicts.append(Conflict(
                    rule_id=self.rule_id,
                    type=ConflictType.TRANSCRIPTION_QUALITY,
                    severity=Severity.MEDIUM,
                    status=ConflictStatus.REVIEW_REQUIRED,
                    message=f"Low transcription confidence ({segment.confidence:.2f}) on segment {segment.segment_id} "
                            f"containing {', '.join(sorted({e.normalized for e in critical}))}.",
                    involved_terms=sorted({e.normalized for e in critical}),
                    entity_ids=[e.entity_id for e in critical],
                    evidence=evidence,
                    recommendation="Replay the audio at this timestamp and confirm the wording.",
                    confidence=round(segment.confidence, 3),
                ))
            if dependents and segment.speaker == Speaker.UNKNOWN:
                conflicts.append(Conflict(
                    rule_id=self.rule_id,
                    type=ConflictType.TRANSCRIPTION_QUALITY,
                    severity=Severity.LOW,
                    status=ConflictStatus.REVIEW_REQUIRED,
                    message=f"Speaker role unknown for segment {segment.segment_id}; clinical statements could not be "
                            f"attributed to doctor or patient.",
                    involved_terms=sorted({e.normalized for e in dependents}),
                    entity_ids=[e.entity_id for e in dependents],
                    evidence=evidence,
                    recommendation="Confirm who made this statement (diarization).",
                    confidence=0.5,
                ))
        return conflicts
