"""Deterministic SOAP note generator.

Every FACT claim is built from verified entities and cites their segments, so this generator
cannot introduce facts that are not in the transcript. It is the offline default and the
fallback whenever the LLM is unavailable or returns unusable output.
"""

from __future__ import annotations

from app.models.schemas import (
    CLINICIAN_SPEAKERS,
    Claim,
    ClaimKind,
    ClinicalEntity,
    Conflict,
    ConflictStatus,
    DraftPrescription,
    EntitySet,
    EntityStatus,
    EntityType,
    PrescriptionStatus,
    SOAPNote,
    SOAPSection,
)
from app.nlp.text import oxford_join

NKDA = "no known drug allergies"


def _claim(section: SOAPSection, text: str, entities: list[ClinicalEntity] = (), kind: ClaimKind = ClaimKind.FACT,
           conflicts: list[Conflict] = ()) -> Claim:
    segment_ids = list(dict.fromkeys(
        [sid for e in entities for sid in e.segment_ids] + [ev.segment_id for c in conflicts for ev in c.evidence]))
    return Claim(section=section, text=text, kind=kind, entity_ids=[e.entity_id for e in entities],
                 segment_ids=segment_ids, conflict_ids=[c.conflict_id for c in conflicts])


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def _symptom_phrase(e: ClinicalEntity) -> str:
    severity = e.attributes.get("severity")
    return f"{severity} {e.normalized}" if severity else e.normalized


def _is_clinician_stated(e: ClinicalEntity) -> bool:
    if e.attributes.get("stated_by"):
        return e.attributes["stated_by"] in {s.value for s in CLINICIAN_SPEAKERS}
    return any(ev.speaker in CLINICIAN_SPEAKERS for ev in e.evidence)


def vital_text(e: ClinicalEntity) -> str:
    unit = e.unit or ""
    value = f"{e.value}{'' if unit == '%' else ' '}{unit}".strip()
    return f"{e.display_name} {value}"


def medication_text(e: ClinicalEntity) -> str:
    parts = [e.normalized]
    for key in ("dose", "route", "frequency"):
        if e.attributes.get(key):
            parts.append(e.attributes[key])
    if e.attributes.get("duration"):
        parts.append(f"for {e.attributes['duration']}")
    return " ".join(parts)


def safety_claims(conflicts: list[Conflict]) -> list[Claim]:
    claims = []
    for c in conflicts:
        if c.status == ConflictStatus.BLOCKED:
            text = f"SAFETY BLOCK: {c.message} Excluded from this draft; physician review required."
        else:
            text = f"SAFETY REVIEW: {c.message}"
        claims.append(_claim(SOAPSection.PLAN, text, kind=ClaimKind.SAFETY, conflicts=[c]))
    return claims


class TemplateSOAPGenerator:
    def generate(self, entities: EntitySet, conflicts: list[Conflict], blocked_ids: set[str],
                 prescriptions: list[DraftPrescription]) -> SOAPNote:
        usable = EntitySet(items=[e for e in entities.items if e.entity_id not in blocked_ids])
        return SOAPNote(
            subjective=self._subjective(usable),
            objective=self._objective(usable),
            assessment=self._assessment(usable),
            plan=self._plan(usable, conflicts, prescriptions, blocked_ids),
        )

    # ------------------------------------------------------------------ S

    def _subjective(self, es: EntitySet) -> list[Claim]:
        S = SOAPSection.SUBJECTIVE
        claims: list[Claim] = []

        sex = next((e for e in es.of_type(EntityType.DEMOGRAPHIC) if e.normalized == "sex"), None)
        age = next((e for e in es.of_type(EntityType.DEMOGRAPHIC) if e.normalized == "age"), None)
        if sex and age:
            claims.append(_claim(S, f"Patient is a {age.value}-year-old {sex.value}.", [age, sex]))
        elif age:
            claims.append(_claim(S, f"Patient is {age.value} years old.", [age]))
        elif sex:
            basis = " (from form of address)" if sex.attributes.get("basis") == "form_of_address" else ""
            claims.append(_claim(S, f"Patient sex recorded as {sex.value}{basis}.", [sex]))

        symptoms = es.of_type(EntityType.SYMPTOM)
        groups: dict[str, list[ClinicalEntity]] = {}
        for e in symptoms:
            if e.status == EntityStatus.PRESENT:
                groups.setdefault(e.segment_ids[-1] if e.segment_ids else "", []).append(e)
        for group in groups.values():
            durations = {e.attributes.get("duration") for e in group if e.attributes.get("duration")}
            text = "Patient reports " + oxford_join([_symptom_phrase(e) for e in group])
            if len(durations) == 1:
                text += f" for {durations.pop()}"
            if any(e.attributes.get("course") == "worsening" for e in group):
                text += ", worsening"
            claims.append(_claim(S, _sentence(text), group))

        possible = [e for e in symptoms if e.status == EntityStatus.POSSIBLE]
        if possible:
            claims.append(_claim(S, f"Possible {oxford_join([e.normalized for e in possible])} (uncertain report).", possible))
        absent = [e for e in symptoms if e.status == EntityStatus.ABSENT]
        if absent:
            claims.append(_claim(S, f"Patient denies {oxford_join([e.normalized for e in absent])}.", absent))
        resolved = [e for e in symptoms if e.status == EntityStatus.HISTORICAL]
        if resolved:
            claims.append(_claim(S, f"Previous {oxford_join([e.normalized for e in resolved])}, since resolved.", resolved))

        history = [e for e in es.of_type(EntityType.DIAGNOSIS)
                   if not _is_clinician_stated(e) and e.status in (EntityStatus.PRESENT, EntityStatus.HISTORICAL)]
        if history:
            claims.append(_claim(S, f"Past medical history: {oxford_join([e.normalized for e in history])}.", history))

        allergies = [e for e in es.of_type(EntityType.ALLERGY)
                     if e.normalized != NKDA and e.status not in (EntityStatus.ABSENT, EntityStatus.FAMILY_MEMBER)]
        nkda = next((e for e in es.of_type(EntityType.ALLERGY) if e.normalized == NKDA), None)
        if allergies:
            items = [f"{e.normalized} (reaction: {e.attributes['reaction']})" if e.attributes.get("reaction") else e.normalized
                     for e in allergies]
            if nkda:
                claims.append(_claim(S, f"Allergies: {oxford_join(items)} (patient initially reported no known drug "
                                        f"allergies).", allergies + [nkda]))
            else:
                claims.append(_claim(S, f"Allergies: {oxford_join(items)}.", allergies))
        elif nkda:
            claims.append(_claim(S, "No known drug allergies.", [nkda]))

        meds = es.of_type(EntityType.MEDICATION)
        current = [e for e in meds if e.status == EntityStatus.CURRENT]
        if current:
            claims.append(_claim(S, f"Current medications: {oxford_join([medication_text(e) for e in current])}.", current))
        past = [e for e in meds if e.status in (EntityStatus.STOPPED, EntityStatus.HISTORICAL)]
        if past:
            claims.append(_claim(S, f"Previously took {oxford_join([e.normalized for e in past])}; no longer taking.", past))

        family = [e for e in es.items if e.status == EntityStatus.FAMILY_MEMBER]
        if family:
            items = [f"{e.normalized} ({e.attributes.get('relation', 'family member')})" for e in family]
            claims.append(_claim(S, f"Family history: {oxford_join(items)}.", family))
        return claims

    # ------------------------------------------------------------------ O

    def _objective(self, es: EntitySet) -> list[Claim]:
        O = SOAPSection.OBJECTIVE
        claims: list[Claim] = []
        for vital in es.of_type(EntityType.VITAL):
            claims.append(_claim(O, _sentence(vital_text(vital)), [vital]))

        findings = es.of_type(EntityType.FINDING)
        present = [e for e in findings if e.status in (EntityStatus.PRESENT, EntityStatus.POSSIBLE)]
        if present:
            claims.append(_claim(O, f"Examination: {oxford_join([e.normalized for e in present])}.", present))
        absent = [e for e in findings if e.status == EntityStatus.ABSENT]
        if absent:
            claims.append(_claim(O, f"No {oxford_join([e.normalized for e in absent])} noted.", absent))

        for proc in es.of_type(EntityType.PROCEDURE):
            if proc.status != EntityStatus.PRESENT:
                continue
            if proc.attributes.get("result"):
                claims.append(_claim(O, f"{proc.display_name}: {proc.attributes['result']}.", [proc]))
            else:
                claims.append(_claim(O, f"{proc.display_name} performed.", [proc]))

        if not claims:
            claims.append(_claim(O, "No vital signs or examination findings were documented in the transcript.",
                                 kind=ClaimKind.META))
        return claims

    # ------------------------------------------------------------------ A

    def _assessment(self, es: EntitySet) -> list[Claim]:
        A = SOAPSection.ASSESSMENT
        claims: list[Claim] = []
        diagnoses = [e for e in es.of_type(EntityType.DIAGNOSIS) if _is_clinician_stated(e)]
        confirmed = [e for e in diagnoses if e.status == EntityStatus.PRESENT]
        possible = [e for e in diagnoses if e.status == EntityStatus.POSSIBLE]
        ruled_out = [e for e in diagnoses if e.status == EntityStatus.ABSENT]
        if confirmed:
            claims.append(_claim(A, f"Clinician assessment: {oxford_join([e.normalized for e in confirmed])}.", confirmed))
        if possible:
            claims.append(_claim(A, f"Possible {oxford_join([e.normalized for e in possible])} "
                                    f"(clinician impression, not confirmed).", possible))
        if ruled_out:
            claims.append(_claim(A, f"Clinician considers {oxford_join([e.normalized for e in ruled_out])} unlikely.", ruled_out))
        if not claims:
            claims.append(_claim(A, "No diagnosis was stated during the consultation; assessment deferred to the "
                                    "reviewing physician.", kind=ClaimKind.META))
        return claims

    # ------------------------------------------------------------------ P

    def _plan(self, es: EntitySet, conflicts: list[Conflict], prescriptions: list[DraftPrescription],
              blocked_ids: set[str]) -> list[Claim]:
        P = SOAPSection.PLAN
        claims: list[Claim] = []
        for rx in prescriptions:
            if rx.status == PrescriptionStatus.BLOCKED or rx.entity_id in blocked_ids:
                continue
            med = es.get(rx.entity_id)
            if med is None:
                continue
            verb = "Prescribe" if rx.intent == "PRESCRIBED" else "Recommend"
            suffix = " (pending safety review)" if rx.status == PrescriptionStatus.NEEDS_REVIEW else ""
            claims.append(_claim(P, f"{verb} {medication_text(med)}{suffix}.", [med]))

        avoided = [e for e in es.of_type(EntityType.MEDICATION)
                   if e.status == EntityStatus.ABSENT and e.attributes.get("intent") == "avoid"]
        if avoided:
            allergy_names = {a.normalized for a in es.of_type(EntityType.ALLERGY) if a.status != EntityStatus.ABSENT}
            related = [a for a in es.of_type(EntityType.ALLERGY) if a.normalized in {e.normalized for e in avoided}]
            reason = " (documented allergy)" if {e.normalized for e in avoided} & allergy_names else ""
            claims.append(_claim(P, f"Avoid {oxford_join([e.normalized for e in avoided])}{reason}.", avoided + related))

        planned = [e for e in es.of_type(EntityType.PROCEDURE) if e.status == EntityStatus.RECOMMENDED]
        if planned:
            claims.append(_claim(P, f"Planned: {oxford_join([e.normalized for e in planned])}.", planned))

        has_blocked_plan = any(rx.status == PrescriptionStatus.BLOCKED for rx in prescriptions)
        if not claims and not has_blocked_plan:
            claims.append(_claim(P, "No treatment plan was stated during the consultation.", kind=ClaimKind.META))
        claims.extend(safety_claims(conflicts))
        return claims
