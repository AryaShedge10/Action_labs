"""Allergy safety rules: direct / same-class conflicts and known cross-reactivity."""

from __future__ import annotations

from app.knowledge.drugs import ALLERGY_RELEVANT_CLASSES, CLASS_LABELS, DRUGS, classes_of
from app.models.schemas import ClinicalEntity, Conflict, ConflictStatus, ConflictType, Severity
from app.safety.rules import SafetyContext, SafetyRule, evidence_of, min_confidence, status_for


def allergen_classes(allergy: ClinicalEntity) -> frozenset[str]:
    if allergy.attributes.get("allergen_class"):
        return frozenset({allergy.attributes["allergen_class"]})
    if allergy.normalized in DRUGS:
        return DRUGS[allergy.normalized].classes & ALLERGY_RELEVANT_CLASSES
    return classes_of(allergy.normalized) & ALLERGY_RELEVANT_CLASSES


class AllergyMedicationRule(SafetyRule):
    rule_id = "ALLERGY_MEDICATION"
    description = "Medication is the documented allergen or belongs to the same drug class."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        for allergy in ctx.active_allergies():
            a_classes = allergen_classes(allergy)
            for med in ctx.candidate_medications():
                shared = classes_of(med.normalized) & a_classes
                if med.normalized != allergy.normalized and not shared:
                    continue
                status = status_for(med, block=True)
                class_allergen = allergy.attributes.get("allergen_kind") == "drug_class"
                why = ""
                if shared and med.normalized != allergy.normalized and not class_allergen:
                    cls = sorted(shared)[0]
                    why = f" ({CLASS_LABELS.get(cls, cls)})"
                intent = "" if status == ConflictStatus.BLOCKED else " Medication was mentioned without a clear prescribing intent."
                conflicts.append(Conflict(
                    rule_id=self.rule_id,
                    type=ConflictType.MEDICATION_ALLERGY,
                    severity=Severity.HIGH,
                    status=status,
                    message=f"{med.display_name} conflicts with documented {allergy.normalized} allergy{why}.{intent}",
                    allergy=allergy.display_name,
                    medication=med.display_name,
                    involved_terms=[allergy.normalized, med.normalized],
                    entity_ids=[allergy.entity_id, med.entity_id],
                    blocked_entity_ids=[med.entity_id] if status == ConflictStatus.BLOCKED else [],
                    evidence=evidence_of(allergy, med),
                    recommendation="Do not prescribe. Select an agent outside the allergen's drug class and confirm "
                                   "the allergy history and reaction severity with the patient.",
                    confidence=min_confidence(allergy, med),
                ))
        return conflicts


_CROSS_REACTIVITY = [
    ("penicillins", "cephalosporins", Severity.MEDIUM,
     "low but real cross-reactivity (higher with first-generation cephalosporins)"),
    ("penicillins", "carbapenems", Severity.LOW, "cross-reactivity under 1%"),
    ("cephalosporins", "penicillins", Severity.MEDIUM, "possible beta-lactam cross-reactivity"),
]


class AllergyCrossReactivityRule(SafetyRule):
    rule_id = "ALLERGY_CROSS_REACTIVITY"
    description = "Medication from a class with known cross-reactivity to the documented allergy."

    def evaluate(self, ctx: SafetyContext) -> list[Conflict]:
        conflicts = []
        for allergy in ctx.active_allergies():
            a_classes = allergen_classes(allergy)
            severe_reaction = allergy.attributes.get("reaction") == "anaphylaxis"
            for med in ctx.candidate_medications():
                m_classes = classes_of(med.normalized)
                if m_classes & a_classes or med.normalized == allergy.normalized:
                    continue  # handled by AllergyMedicationRule
                for allergy_cls, med_cls, severity, note in _CROSS_REACTIVITY:
                    if allergy_cls in a_classes and med_cls in m_classes:
                        conflicts.append(Conflict(
                            rule_id=self.rule_id,
                            type=ConflictType.ALLERGY_CROSS_REACTIVITY,
                            severity=Severity.HIGH if severe_reaction else severity,
                            status=ConflictStatus.REVIEW_REQUIRED,
                            message=f"{med.display_name} ({CLASS_LABELS[med_cls]}) with documented {allergy.normalized} "
                                    f"allergy: {note}.",
                            allergy=allergy.display_name,
                            medication=med.display_name,
                            involved_terms=[allergy.normalized, med.normalized],
                            entity_ids=[allergy.entity_id, med.entity_id],
                            evidence=evidence_of(allergy, med),
                            recommendation="Confirm the type and severity of the original reaction before prescribing.",
                            confidence=min_confidence(allergy, med),
                        ))
                        break
        return conflicts
