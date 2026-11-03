"""Medication rules: drug-drug interactions, duplicate therapy, dose range."""

from __future__ import annotations

from itertools import combinations

from app.knowledge.drugs import CLASS_LABELS, DRUGS, DUPLICATE_THERAPY_CLASSES, INTERACTIONS, classes_of, matches
from app.models.schemas import NEW_MEDICATION_STATUSES, Conflict, ConflictStatus, ConflictType, Severity
from app.safety.rules import SafetyContext, SafetyRule, evidence_of, min_confidence

_SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


class DrugInteractionRule(SafetyRule):
    rule_id = "DRUG_INTERACTION"
    description = "Clinically significant interaction between two active medications."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        for m1, m2 in combinations(ctx.active_medications(), 2):
            if m1.normalized == m2.normalized:
                continue
            hits = [i for i in INTERACTIONS
                    if (matches(m1.normalized, i.a) and matches(m2.normalized, i.b))
                    or (matches(m2.normalized, i.a) and matches(m1.normalized, i.b))]
            if not hits:
                continue
            worst = max(hits, key=lambda i: (i.block, _SEVERITY_RANK[i.severity]))
            new_meds = [m for m in (m1, m2) if m.status in NEW_MEDICATION_STATUSES]
            # Only a newly started drug is blocked; an existing regimen is left for physician review.
            blocked = worst.block and bool(new_meds)
            conflicts.append(Conflict(
                rule_id=self.rule_id,
                type=ConflictType.DRUG_INTERACTION,
                severity=Severity(worst.severity),
                status=ConflictStatus.BLOCKED if blocked else ConflictStatus.REVIEW_REQUIRED,
                message=f"{m1.display_name} + {m2.display_name}: {worst.effect}.",
                medication=(new_meds[0] if new_meds else m2).display_name,
                involved_terms=[m1.normalized, m2.normalized],
                entity_ids=[m1.entity_id, m2.entity_id],
                blocked_entity_ids=[m.entity_id for m in new_meds] if blocked else [],
                evidence=evidence_of(m1, m2),
                recommendation=worst.recommendation,
                confidence=min_confidence(m1, m2),
            ))
        return conflicts


class DuplicateTherapyRule(SafetyRule):
    rule_id = "DUPLICATE_THERAPY"
    description = "Two active medications from the same therapeutic class."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        meds = ctx.active_medications()
        for cls in sorted(DUPLICATE_THERAPY_CLASSES):
            members = [m for m in meds if cls in classes_of(m.normalized)]
            names = list(dict.fromkeys(m.normalized for m in members))
            if len(names) < 2:
                continue
            conflicts.append(Conflict(
                rule_id=self.rule_id,
                type=ConflictType.DUPLICATE_THERAPY,
                severity=Severity.MEDIUM,
                status=ConflictStatus.REVIEW_REQUIRED,
                message=f"Duplicate {CLASS_LABELS.get(cls, cls)}: {', '.join(m.display_name for m in members)}.",
                medication=members[-1].display_name,
                involved_terms=names,
                entity_ids=[m.entity_id for m in members],
                evidence=evidence_of(*members),
                recommendation="Confirm that concurrent use is intended; otherwise discontinue one agent.",
                confidence=min_confidence(*members),
            ))
        return conflicts


class DoseRangeRule(SafetyRule):
    rule_id = "DOSE_RANGE"
    description = "Stated dose exceeds the usual adult maximum (often an ASR error, e.g. 'fifty' vs 'five hundred')."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        for med in ctx.active_medications():
            drug = DRUGS.get(med.normalized)
            if drug is None or "dose_mg" not in med.attributes:
                continue
            dose = float(med.attributes["dose_mg"])
            per_day = int(med.attributes.get("doses_per_day", 0) or 0)
            problems = []
            if drug.max_single_dose_mg and dose > drug.max_single_dose_mg:
                problems.append(f"single dose {dose:g} mg exceeds usual maximum of {drug.max_single_dose_mg:g} mg")
            if drug.max_daily_dose_mg and per_day and dose * per_day > drug.max_daily_dose_mg:
                problems.append(f"daily total {dose * per_day:g} mg exceeds usual maximum of {drug.max_daily_dose_mg:g} mg/day")
            if not problems:
                continue
            conflicts.append(Conflict(
                rule_id=self.rule_id,
                type=ConflictType.DOSE_RANGE,
                severity=Severity.HIGH,
                status=ConflictStatus.REVIEW_REQUIRED,
                message=f"{med.display_name}: {'; '.join(problems)}.",
                medication=med.display_name,
                involved_terms=[med.normalized, med.attributes.get("dose", "")],
                entity_ids=[med.entity_id],
                evidence=evidence_of(med),
                recommendation="Verify the dose against the audio; correct before signing.",
                confidence=min_confidence(med),
            ))
        return conflicts
