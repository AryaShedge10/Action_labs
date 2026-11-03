"""Claim-level grounding verification ("citation grounding" / hallucination scoring).

Each FACT claim is checked independently of how it was generated:
  1. citations     - cited segment ids must exist in the transcript
  2. clinical terms - every symptom / diagnosis / drug / finding / procedure named in the claim must
                      exist in the transcript (else: hallucination -> UNSUPPORTED)
  3. polarity      - asserting something the patient denied, a family member's condition, or a blocked
                      item -> CONTRADICTED
  4. numbers       - every number / dose in the claim must appear in the cited segments
  5. lexical support - content-word overlap with the cited segments
"""

from __future__ import annotations

from app.models.schemas import (
    Citation,
    Claim,
    ClaimKind,
    ClinicalEntity,
    EntitySet,
    EntityStatus,
    EntityType,
    GroundingReport,
    GroundingStatus,
    RemovedClaim,
    SOAPNote,
    Transcript,
)
from app.nlp import context as ctxlib
from app.nlp.matcher import LexiconMatcher, TermMatch, clinical_matcher
from app.nlp.text import content_stems, numbers_in

_POSITIVE = {EntityStatus.PRESENT, EntityStatus.CURRENT, EntityStatus.PRESCRIBED, EntityStatus.RECOMMENDED}
_NEGATING_MED_CUES = {"NEGATED", "AVOID", "AVOID_NEAR"}


class GroundingVerifier:
    def __init__(self, matcher: LexiconMatcher | None = None, support_threshold: float = 0.5):
        self.matcher = matcher or clinical_matcher()
        self.support_threshold = support_threshold

    def verify(self, note: SOAPNote, transcript: Transcript, entities: EntitySet, blocked_ids: set[str],
               removed_claims: list[RemovedClaim] | None = None) -> GroundingReport:
        transcript_terms = {m.canonical for s in transcript.segments for m in self.matcher.find(s.text)}
        attribute_terms = {v.lower() for e in entities.items for k, v in e.attributes.items()
                           if k in ("reaction", "severity", "relation")}
        requested = verified = 0

        for i, claim in enumerate(note.all_claims(), start=1):
            claim.claim_id = f"C{i:03d}"
            wanted = list(dict.fromkeys(
                claim.segment_ids + [sid for eid in claim.entity_ids if entities.get(eid) for sid in entities.get(eid).segment_ids]))
            claim.citations, invalid = self._citations(wanted, transcript)
            if claim.kind != ClaimKind.META:
                requested += len(wanted)
                verified += len(wanted) - len(invalid)
            claim.issues = [f"cited segment {sid} does not exist" for sid in invalid]

            if claim.kind == ClaimKind.META:
                claim.grounding_status, claim.support_score, claim.confidence = GroundingStatus.NOT_APPLICABLE, 1.0, 1.0
            elif claim.kind == ClaimKind.SAFETY:
                claim.grounding_status = GroundingStatus.SUPPORTED if claim.citations else GroundingStatus.NOT_APPLICABLE
                claim.support_score = 1.0
                claim.confidence = _asr_mean(claim.citations)
            else:
                self._verify_fact(claim, entities, blocked_ids, transcript_terms, attribute_terms)

        facts = [c for c in note.all_claims() if c.kind == ClaimKind.FACT]
        counts = {s: sum(c.grounding_status == s for c in facts) for s in GroundingStatus}
        documentable = {e.entity_id for e in entities.items if e.entity_id not in blocked_ids}
        referenced = {eid for c in note.all_claims() for eid in c.entity_ids} & documentable
        n = len(facts)
        rate = (lambda k: round(k / n, 4) if n else 0.0)
        return GroundingReport(
            total_claims=len(note.all_claims()),
            fact_claims=n,
            supported=counts[GroundingStatus.SUPPORTED],
            partially_supported=counts[GroundingStatus.PARTIALLY_SUPPORTED],
            unsupported=counts[GroundingStatus.UNSUPPORTED],
            contradicted=counts[GroundingStatus.CONTRADICTED],
            grounding_score=rate(counts[GroundingStatus.SUPPORTED] + 0.5 * counts[GroundingStatus.PARTIALLY_SUPPORTED]),
            unsupported_claim_rate=rate(counts[GroundingStatus.UNSUPPORTED]),
            contradiction_rate=rate(counts[GroundingStatus.CONTRADICTED]),
            hallucination_score=rate(counts[GroundingStatus.UNSUPPORTED] + counts[GroundingStatus.CONTRADICTED]),
            citation_verification_rate=round(verified / requested, 4) if requested else 1.0,
            entity_coverage=round(len(referenced) / len(documentable), 4) if documentable else 1.0,
            removed_claims=list(removed_claims or []),
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _citations(segment_ids: list[str], transcript: Transcript) -> tuple[list[Citation], list[str]]:
        citations, invalid = [], []
        for sid in segment_ids:
            seg = transcript.segment(sid)
            if seg is None:
                invalid.append(sid)
                continue
            citations.append(Citation(segment_id=seg.segment_id, speaker=seg.speaker, start_time=seg.start_time,
                                      end_time=seg.end_time, text=seg.text, segment_sha256=seg.sha256,
                                      asr_confidence=seg.confidence))
        return citations, invalid

    def _verify_fact(self, claim: Claim, entities: EntitySet, blocked_ids: set[str],
                     transcript_terms: set[str], attribute_terms: set[str]) -> None:
        cited_ids = {c.segment_id for c in claim.citations}
        cited_text = " ".join(c.text for c in claim.citations)
        contradictions: list[str] = []
        hallucinated: list[str] = []
        partial: list[str] = []
        supported_terms = 0
        support_stems = content_stems(cited_text)
        support_numbers = numbers_in(cited_text)
        for eid in claim.entity_ids:
            cited_entity = entities.get(eid)
            if cited_entity is not None:  # verified entity values (e.g. sex "male" from "Mr.") count as support
                support_stems |= content_stems(" ".join([cited_entity.normalized, cited_entity.value or "",
                                                         *cited_entity.attributes.values()]))

        terms = self.matcher.find(claim.text)
        for term in terms:
            entity = self._entity_for(term, claim.text, entities)
            if entity is None:
                if term.canonical in attribute_terms:
                    supported_terms += 1
                elif term.canonical in transcript_terms:
                    partial.append(f"'{term.surface}' is in the transcript but was not extracted as a fact")
                else:
                    hallucinated.append(f"'{term.surface}' does not appear anywhere in the transcript")
                continue

            support_stems |= content_stems(entity.normalized) | content_stems(entity.text)
            support_numbers |= numbers_in(" ".join([entity.value or "", *entity.attributes.values()]))
            problem = self._polarity_problem(term, claim.text, entity, blocked_ids)
            if problem:
                contradictions.append(problem)
                continue
            if not cited_ids & set(entity.segment_ids):
                partial.append(f"'{term.surface}' is supported by {', '.join(entity.segment_ids)} which is not cited")
            elif entity.status == EntityStatus.POSSIBLE and not ctxlib.analyze(claim.text, term.start, term.end).uncertain:
                partial.append(f"'{term.surface}' is stated with more certainty than in the transcript")
            else:
                supported_terms += 1

        missing_numbers = sorted(numbers_in(claim.text) - support_numbers)
        for num in missing_numbers:
            hallucinated.append(f"value '{num}' does not appear in the cited segments")

        claim_stems = content_stems(claim.text, drop_scaffold=True)
        overlap = len(claim_stems & support_stems) / len(claim_stems) if claim_stems else 1.0
        term_score = supported_terms / len(terms) if terms else 1.0
        claim.support_score = round(0.6 * term_score + 0.4 * overlap if terms else overlap, 3)
        claim.issues += contradictions + hallucinated + partial

        if contradictions:
            claim.grounding_status = GroundingStatus.CONTRADICTED
        elif not claim.citations or hallucinated:
            if not claim.citations:
                claim.issues.append("claim has no valid citation")
            claim.grounding_status = GroundingStatus.UNSUPPORTED
        elif not partial and (overlap >= self.support_threshold or (terms and term_score == 1.0)):
            claim.grounding_status = GroundingStatus.SUPPORTED
        else:
            if overlap < self.support_threshold and not partial:
                claim.issues.append(f"low lexical overlap with cited segments ({overlap:.2f})")
            claim.grounding_status = GroundingStatus.PARTIALLY_SUPPORTED

        if claim.grounding_status in (GroundingStatus.CONTRADICTED, GroundingStatus.UNSUPPORTED):
            claim.confidence = 0.0
        else:
            claim.confidence = round(claim.support_score * _asr_mean(claim.citations), 3)

    @staticmethod
    def _entity_for(term: TermMatch, claim_text: str, entities: EntitySet) -> ClinicalEntity | None:
        candidates = [e for e in entities.items if e.normalized == term.canonical]
        if not candidates:
            return None
        allergy_context = "allerg" in claim_text[:term.start].lower()

        def rank(e: ClinicalEntity) -> tuple[int, int]:
            if allergy_context:
                return (0 if e.type == EntityType.ALLERGY else 1, 0)
            return (0 if e.type == term.type else 1, 1 if e.type == EntityType.ALLERGY else 0)
        return sorted(candidates, key=rank)[0]

    @staticmethod
    def _polarity_problem(term: TermMatch, claim_text: str, entity: ClinicalEntity, blocked_ids: set[str]) -> str | None:
        if entity.type == EntityType.ALLERGY:
            return None  # allergy statements are informational; allergy polarity is checked by safety rules
        ctx = ctxlib.analyze(claim_text, term.start, term.end)
        negated = ctx.negated or (entity.type == EntityType.MEDICATION and ctx.med_cue in _NEGATING_MED_CUES)

        if entity.entity_id in blocked_ids and not negated:
            return f"'{term.surface}' was BLOCKED by the safety engine but is asserted in the note"
        if entity.status == EntityStatus.FAMILY_MEMBER and not ctx.family_relation:
            return f"'{term.surface}' refers to a family member ({entity.attributes.get('relation', 'relative')}), not the patient"
        if entity.status == EntityStatus.ABSENT and not negated:
            return f"transcript states '{term.surface}' is absent / denied / avoided"
        if entity.status in _POSITIVE and negated:
            return f"claim negates '{term.surface}' but the transcript affirms it"
        return None


def _asr_mean(citations: list[Citation]) -> float:
    values = [c.asr_confidence for c in citations if c.asr_confidence is not None]
    return round(sum(values) / len(values), 3) if values else 1.0
