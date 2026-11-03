"""Population rules: sex-specific conditions, pregnancy and age contraindications."""

from __future__ import annotations

from app.knowledge import terms
from app.knowledge.drugs import AGE_RULES, CLASS_LABELS, PREGNANCY_CONTRAINDICATIONS, matches
from app.models.schemas import (
    AFFIRMED_STATUSES,
    Conflict,
    ConflictStatus,
    ConflictType,
    EntityStatus,
    EntityType,
    Severity,
)
from app.safety.rules import SafetyContext, SafetyRule, evidence_of, min_confidence, status_for

_ORDERED = {EntityStatus.PRESENT, EntityStatus.RECOMMENDED}


class SexSpecificRule(SafetyRule):
    rule_id = "SEX_SPECIFIC"
    description = "Condition or procedure incompatible with the patient's documented sex."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        sex = ctx.sex()
        if sex is None:
            return []
        value, basis, sex_entity = sex
        if value == "male":
            diagnoses, procedures, symptoms = terms.FEMALE_ONLY_DIAGNOSES, terms.FEMALE_ONLY_PROCEDURES, terms.FEMALE_ONLY_SYMPTOMS
        elif value == "female":
            diagnoses, procedures, symptoms = terms.MALE_ONLY_DIAGNOSES, terms.MALE_ONLY_PROCEDURES, frozenset()
        else:
            return []

        stated = basis == "stated"
        basis_note = "" if stated else " (sex inferred from form of address)"
        conflicts = []
        checks = [
            (EntityType.DIAGNOSIS, diagnoses, AFFIRMED_STATUSES, stated, Severity.HIGH),
            (EntityType.PROCEDURE, procedures, _ORDERED, False, Severity.HIGH),
            (EntityType.SYMPTOM, symptoms, AFFIRMED_STATUSES, False, Severity.MEDIUM),
        ]
        for etype, vocabulary, statuses, block, severity in checks:
            for entity in ctx.entities.of_type(etype):
                if entity.normalized not in vocabulary or entity.status not in statuses:
                    continue
                status = ConflictStatus.BLOCKED if block else ConflictStatus.REVIEW_REQUIRED
                conflicts.append(Conflict(
                    rule_id=self.rule_id,
                    type=ConflictType.DEMOGRAPHIC_CONFLICT,
                    severity=severity,
                    status=status,
                    message=f"{entity.display_name} recorded for a patient documented as {value}{basis_note}.",
                    involved_terms=[f"sex: {value}", entity.normalized],
                    entity_ids=[sex_entity.entity_id, entity.entity_id],
                    blocked_entity_ids=[entity.entity_id] if status == ConflictStatus.BLOCKED else [],
                    evidence=evidence_of(sex_entity, entity),
                    recommendation="Verify the patient's sex / anatomical inventory and the source statement "
                                   "(possible speaker-diarization or transcription error).",
                    confidence=min_confidence(sex_entity, entity),
                ))
        return conflicts


class PregnancyContraindicationRule(SafetyRule):
    rule_id = "PREGNANCY_CONTRAINDICATION"
    description = "Medication contraindicated or cautioned in pregnancy."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        pregnancy = ctx.pregnancy()
        if pregnancy is None:
            return []
        confirmed = pregnancy.status == EntityStatus.PRESENT
        conflicts = []
        for med in ctx.candidate_medications():
            rule = next((r for r in PREGNANCY_CONTRAINDICATIONS if matches(med.normalized, r.term)), None)
            if rule is None:
                continue
            status = status_for(med, block=rule.block and confirmed)
            label = CLASS_LABELS.get(rule.term, rule.term)
            qualifier = "" if confirmed else " (pregnancy possible)"
            conflicts.append(Conflict(
                rule_id=self.rule_id,
                type=ConflictType.PREGNANCY_CONTRAINDICATION,
                severity=Severity(rule.severity),
                status=status,
                message=f"{med.display_name} ({label}) in a pregnant patient{qualifier}: {rule.reason}.",
                medication=med.display_name,
                involved_terms=["pregnancy", med.normalized],
                entity_ids=[pregnancy.entity_id, med.entity_id],
                blocked_entity_ids=[med.entity_id] if status == ConflictStatus.BLOCKED else [],
                evidence=evidence_of(pregnancy, med),
                recommendation="Choose a pregnancy-compatible alternative; confirm pregnancy status if uncertain.",
                confidence=min_confidence(pregnancy, med),
            ))
        return conflicts


class AgeContraindicationRule(SafetyRule):
    rule_id = "AGE_CONTRAINDICATION"
    description = "Medication contraindicated or cautioned for the patient's age group."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        age_info = ctx.age()
        if age_info is None:
            return []
        age, age_entity = age_info
        conflicts = []
        for med in ctx.candidate_medications():
            for rule in AGE_RULES:
                applies = (rule.max_age is not None and age < rule.max_age) or (rule.min_age is not None and age >= rule.min_age)
                if not applies or not matches(med.normalized, rule.term):
                    continue
                status = status_for(med, block=rule.block)
                article = "an" if str(age)[0] == "8" or age in (11, 18) else "a"
                conflicts.append(Conflict(
                    rule_id=self.rule_id,
                    type=ConflictType.AGE_CONTRAINDICATION,
                    severity=Severity(rule.severity),
                    status=status,
                    message=f"{med.display_name} for {article} {age}-year-old patient: {rule.reason}.",
                    medication=med.display_name,
                    involved_terms=[f"age: {age}", med.normalized],
                    entity_ids=[age_entity.entity_id, med.entity_id],
                    blocked_entity_ids=[med.entity_id] if status == ConflictStatus.BLOCKED else [],
                    evidence=evidence_of(age_entity, med),
                    recommendation="Choose an age-appropriate alternative or document the justification.",
                    confidence=min_confidence(age_entity, med),
                ))
                break
        return conflicts
