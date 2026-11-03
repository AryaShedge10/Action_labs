"""SOAP generation facade: template or LLM, followed by the deterministic blocked-item scrub."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.generation.llm_soap import LLMSOAPGenerator
from app.generation.template import TemplateSOAPGenerator, safety_claims
from app.knowledge.drugs import DRUGS
from app.models.schemas import (
    ClaimKind,
    Conflict,
    DraftPrescription,
    EntitySet,
    RemovedClaim,
    SOAPNote,
    SOAPSection,
)
from app.utils.llm import LLMError

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    note: SOAPNote
    mode_used: str
    removed_claims: list[RemovedClaim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SOAPGenerator:
    def __init__(self, llm_generator: LLMSOAPGenerator | None = None):
        self.template = TemplateSOAPGenerator()
        self.llm = llm_generator

    def generate(self, entities: EntitySet, conflicts: list[Conflict], blocked_ids: set[str],
                 prescriptions: list[DraftPrescription], mode: str = "auto") -> GenerationResult:
        template_note = self.template.generate(entities, conflicts, blocked_ids, prescriptions)
        if mode == "template":
            return GenerationResult(template_note, "template")

        if self.llm is None or not self.llm.client.is_available():
            if mode == "llm":
                raise LLMError("LLM generation requested but the local LLM is not available")
            return GenerationResult(template_note, "template", warnings=["Local LLM unavailable - used template SOAP generator"])

        try:
            llm_note = self.llm.generate(entities, blocked_ids)
        except LLMError as exc:
            if mode == "llm":
                raise
            logger.warning("LLM SOAP generation failed (%s); using template", type(exc).__name__)
            return GenerationResult(template_note, "template",
                                    warnings=[f"LLM SOAP generation failed ({type(exc).__name__}) - used template"])

        if not llm_note.all_claims():
            return GenerationResult(template_note, "template", warnings=["LLM returned an empty note - used template"])

        backfilled = backfill_omissions(llm_note, template_note)
        llm_note.plan.extend(safety_claims(conflicts))
        removed = scrub_blocked(llm_note, entities, blocked_ids)
        warnings = [f"{len(removed)} generated claim(s) removed for mentioning blocked items"] if removed else []
        if backfilled:
            warnings.append(f"{backfilled} deterministic claim(s) added for facts the LLM omitted")
        return GenerationResult(llm_note, "llm", removed, warnings)


def backfill_omissions(llm_note: SOAPNote, template_note: SOAPNote) -> int:
    """Omission is a safety issue too (e.g. dropped pertinent negatives): add template claims for uncovered facts."""
    covered = {eid for c in llm_note.all_claims() for eid in c.entity_ids}
    added = 0
    for section in SOAPSection:
        target = llm_note.section(section)
        for claim in template_note.section(section):
            if claim.kind == ClaimKind.SAFETY:
                continue
            if claim.kind == ClaimKind.META:
                if not target:
                    target.append(claim.model_copy())
                continue
            if set(claim.entity_ids) - covered:
                target.append(claim.model_copy())
                covered |= set(claim.entity_ids)
                added += 1
    return added


def _blocked_terms(entities: EntitySet, blocked_ids: set[str]) -> list[str]:
    terms: set[str] = set()
    for eid in blocked_ids:
        entity = entities.get(eid)
        if entity is None:
            continue
        terms.update({entity.normalized, entity.text.lower()})
        drug = DRUGS.get(entity.normalized)
        if drug:
            terms.update(a.lower() for a in drug.aliases)
    return sorted(t for t in terms if len(t) > 2)


def scrub_blocked(note: SOAPNote, entities: EntitySet, blocked_ids: set[str]) -> list[RemovedClaim]:
    """Remove every non-safety claim that mentions a blocked medication / condition / procedure."""
    terms = _blocked_terms(entities, blocked_ids)
    if not terms:
        return []
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    removed: list[RemovedClaim] = []
    for section in SOAPSection:
        kept = []
        for claim in note.section(section):
            if claim.kind == ClaimKind.SAFETY:
                kept.append(claim)
                continue
            cites_blocked = any(eid in blocked_ids for eid in claim.entity_ids)
            hit = pattern.search(claim.text)
            # Documenting the allergy itself ("Allergic to amoxicillin") is required information, not a leak.
            if hit and not cites_blocked and _ALLERGY_STATEMENT.search(claim.text):
                hit = None
            if hit or cites_blocked:
                what = hit.group() if hit else "cited entity"
                removed.append(RemovedClaim(section=section, text=claim.text, reason=f"mentions blocked item '{what}'"))
            else:
                kept.append(claim)
        setattr(note, section.value, kept)
    return removed


_ALLERGY_STATEMENT = re.compile(r"\ballerg", re.IGNORECASE)
