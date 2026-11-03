"""Extraction facade: rule-based, LLM-only, or hybrid (rule-based + verified LLM additions).

Merge policy for hybrid mode (deterministic wins on disagreement):
* same (type, normalized) found by both  -> keep rule entity, add LLM evidence, small confidence boost
                                            if statuses agree; penalty + `llm_status` attribute if not.
* LLM-only entity                        -> added (already quote-verified), source="llm".
* LLM "medication" that is a known allergen of this patient -> dropped (classic LLM confusion).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.extraction.llm_extractor import LLMExtractor
from app.extraction.rule_based import RuleBasedExtractor, assign_ids
from app.models.schemas import ClinicalEntity, EntitySet, EntityType, Transcript
from app.utils.llm import LLMError

logger = logging.getLogger(__name__)

_KEY_WITH_VALUE = {EntityType.VITAL, EntityType.DEMOGRAPHIC}


@dataclass
class ExtractionResult:
    entities: EntitySet
    mode_used: str
    warnings: list[str] = field(default_factory=list)
    llm_proposed: int = 0
    llm_rejected: int = 0


class ClinicalExtractor:
    def __init__(self, rule_extractor: RuleBasedExtractor, llm_extractor: LLMExtractor | None = None):
        self.rule = rule_extractor
        self.llm = llm_extractor

    def extract(self, transcript: Transcript, mode: str = "auto") -> ExtractionResult:
        if mode == "rule":
            return ExtractionResult(self.rule.extract(transcript), "rule")

        llm_ready = self.llm is not None and self.llm.client.is_available()
        if not llm_ready:
            if mode in ("llm", "hybrid"):
                raise LLMError("LLM extraction requested but the local LLM is not available")
            return ExtractionResult(self.rule.extract(transcript), "rule",
                                    ["Local LLM unavailable - used deterministic extraction only"])

        try:
            llm_result = self.llm.extract(transcript)
        except LLMError as exc:
            if mode in ("llm", "hybrid"):
                raise
            logger.warning("LLM extraction failed (%s); falling back to rules", type(exc).__name__)
            return ExtractionResult(self.rule.extract(transcript), "rule",
                                    [f"LLM extraction failed ({type(exc).__name__}) - used deterministic extraction only"])

        warnings = []
        if llm_result.rejected:
            warnings.append(f"{len(llm_result.rejected)} of {llm_result.proposed} LLM-proposed entities rejected "
                            f"by evidence verification")
        if mode == "llm":
            entities = EntitySet(items=assign_ids(_dedupe(llm_result.entities)))
            return ExtractionResult(entities, "llm", warnings, llm_result.proposed, len(llm_result.rejected))

        merged = merge(self.rule.extract(transcript), llm_result.entities)
        return ExtractionResult(merged, "hybrid", warnings, llm_result.proposed, len(llm_result.rejected))


def _key(e: ClinicalEntity) -> tuple:
    return (e.type, e.normalized, e.value if e.type in _KEY_WITH_VALUE else None)


def _dedupe(entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
    by_key: dict[tuple, ClinicalEntity] = {}
    for e in entities:
        if _key(e) in by_key:
            existing = by_key[_key(e)]
            existing.evidence.extend(ev for ev in e.evidence if ev not in existing.evidence)
            existing.status_history.extend(e.status_history)
            existing.status = e.status  # latest mention wins
        else:
            by_key[_key(e)] = e.model_copy(deep=True)
    return list(by_key.values())


def merge(rule_set: EntitySet, llm_entities: list[ClinicalEntity]) -> EntitySet:
    merged = {_key(e): e.model_copy(deep=True) for e in rule_set.items}
    allergens = {e.normalized for e in rule_set.items if e.type == EntityType.ALLERGY}
    reactions = {r.strip() for e in rule_set.items if e.type == EntityType.ALLERGY
                 for r in e.attributes.get("reaction", "").split(",") if r.strip()}

    for llm_entity in _dedupe(llm_entities):
        key = _key(llm_entity)
        if key not in merged:
            if llm_entity.type == EntityType.MEDICATION and llm_entity.normalized in allergens:
                continue  # allergen mis-typed as a medication
            if llm_entity.type == EntityType.SYMPTOM and llm_entity.normalized in reactions:
                continue  # past allergic reaction ("hives last time"), not a current symptom
        if key in merged:
            base = merged[key]
            base.source = "rule+llm"
            for ev in llm_entity.evidence:
                if all(ev.segment_id != existing.segment_id for existing in base.evidence):
                    base.evidence.append(ev)
            if llm_entity.status == base.status:
                base.confidence = round(min(1.0, base.confidence + 0.05), 3)
            else:
                base.attributes["llm_status"] = llm_entity.status.value
                base.confidence = round(max(0.0, base.confidence - 0.15), 3)
            for k, v in llm_entity.attributes.items():
                base.attributes.setdefault(k, v)
        else:
            merged[key] = llm_entity
    return EntitySet(items=assign_ids(list(merged.values())))
