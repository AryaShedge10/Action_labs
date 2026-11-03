"""End-to-end orchestration:

  Team A transcript -> normalise -> extract -> deterministic safety -> draft prescriptions
  -> SOAP generation -> blocked-item scrub -> claim-level grounding -> status + confidence
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config import Settings
from app.extraction.extractor import ClinicalExtractor
from app.extraction.llm_extractor import LLMExtractor
from app.extraction.rule_based import RuleBasedExtractor
from app.generation.llm_soap import LLMSOAPGenerator
from app.generation.prescriptions import build_prescriptions
from app.generation.soap_generator import SOAPGenerator
from app.grounding.verifier import GroundingVerifier
from app.ingestion.adapter import to_transcript
from app.models.contracts import LegacyTranscriptIn, TeamATranscriptIn
from app.models.schemas import (
    AnalysisMetadata,
    ClaimKind,
    ConfidenceSummary,
    ConflictStatus,
    ConsultationAnalysis,
    Demographics,
    EntitySet,
    EntityStatus,
    EntityType,
    GroundingReport,
    PatientInfo,
    SafetyStatus,
    Transcript,
)
from app.safety.engine import SafetyEngine, SafetyResult
from app.safety.rules import SafetyContext
from app.utils.llm import OllamaClient, PrivacyViolationError

logger = logging.getLogger(__name__)

TranscriptPayload = TeamATranscriptIn | LegacyTranscriptIn | Transcript | dict[str, Any]


class ClinicalPipeline:
    def __init__(self, settings: Settings, llm_client: OllamaClient | None = None, use_llm: bool = True):
        self.settings = settings
        self.init_warnings: list[str] = []
        if llm_client is None and use_llm:
            try:
                llm_client = OllamaClient(settings.ollama_base_url, settings.llm_model, settings.llm_timeout_s,
                                          allow_remote=settings.allow_remote_llm)
            except PrivacyViolationError as exc:
                self.init_warnings.append(str(exc))
                llm_client = None
        self.llm_client = llm_client

        rule = RuleBasedExtractor()
        llm_extractor = LLMExtractor(llm_client, redact_phi=settings.redact_phi_for_llm) if llm_client else None
        self.extractor = ClinicalExtractor(rule, llm_extractor)
        self.safety = SafetyEngine(low_asr_confidence=settings.low_asr_confidence)
        self.generator = SOAPGenerator(LLMSOAPGenerator(llm_client) if llm_client else None)
        self.verifier = GroundingVerifier()

    def llm_status(self) -> dict[str, Any]:
        if self.llm_client is None:
            return {"available": False, "model": self.settings.llm_model, "reason": "; ".join(self.init_warnings) or "disabled"}
        return {"available": self.llm_client.is_available(), "model": self.llm_client.model,
                "endpoint_is_local": self.llm_client.is_local}

    def analyze(self, payload: TranscriptPayload, extraction_mode: str | None = None,
                generation_mode: str | None = None, safety_enabled: bool = True) -> ConsultationAnalysis:
        extraction_mode = extraction_mode or self.settings.extraction_mode
        generation_mode = generation_mode or self.settings.generation_mode
        timings: dict[str, float] = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round((now - clock) * 1000, 1)
            clock = now

        transcript = payload if isinstance(payload, Transcript) else to_transcript(payload)
        lap("ingest")
        extraction = self.extractor.extract(transcript, extraction_mode)
        entities = extraction.entities
        lap("extraction")
        safety = self.safety.run(entities, transcript) if safety_enabled else SafetyResult()
        lap("safety")
        prescriptions = build_prescriptions(entities, safety)
        generation = self.generator.generate(entities, safety.conflicts, safety.blocked_entity_ids, prescriptions,
                                             generation_mode)
        lap("generation")
        grounding = self.verifier.verify(generation.note, transcript, entities, safety.blocked_entity_ids,
                                         generation.removed_claims)
        lap("grounding")

        llm_used = extraction.mode_used in ("llm", "hybrid") or generation.mode_used == "llm"
        warnings = self.init_warnings + extraction.warnings + generation.warnings
        if not safety_enabled:
            warnings.append("Safety engine DISABLED (evaluation ablation only)")
        timings["total"] = round(sum(timings.values()), 1)

        result = ConsultationAnalysis(
            consultation_id=transcript.consultation_id,
            safety_status=decide_status(safety, grounding),
            patient_info=build_patient_info(entities, transcript),
            soap_note=generation.note.render(),
            soap_claims=generation.note,
            conflicts=safety.conflicts,
            draft_prescriptions=prescriptions,
            entities=entities.items,
            grounding=grounding,
            confidence=build_confidence(transcript, entities, generation.note),
            transcript=transcript.segments,
            metadata=AnalysisMetadata(
                source_format=transcript.source_format,
                extraction_mode=extraction.mode_used,
                generation_mode=generation.mode_used,
                model=self.settings.llm_model if llm_used else None,
                phi_redacted_for_llm=llm_used and self.settings.redact_phi_for_llm,
                transcript_sha256=transcript.sha256,
                processing_ms=timings,
                warnings=warnings,
            ),
        )
        logger.info("Analysed consultation %s: status=%s conflicts=%d claims=%d (%.0f ms)",
                    transcript.consultation_id, result.safety_status.value, len(result.conflicts),
                    grounding.total_claims, timings["total"])
        return result


def decide_status(safety: SafetyResult, grounding: GroundingReport) -> SafetyStatus:
    if any(c.status == ConflictStatus.BLOCKED for c in safety.conflicts):
        return SafetyStatus.CONFLICT_DETECTED
    if (any(c.status == ConflictStatus.REVIEW_REQUIRED for c in safety.conflicts)
            or grounding.unsupported or grounding.contradicted or grounding.removed_claims):
        return SafetyStatus.REVIEW_REQUIRED
    return SafetyStatus.SAFE


def build_patient_info(entities: EntitySet, transcript: Transcript) -> PatientInfo:
    ctx = SafetyContext(entities, transcript)
    sex = ctx.sex()
    age = ctx.age()
    pregnancy = next((e for e in entities.of_type(EntityType.DIAGNOSIS) if e.normalized == "pregnancy"
                      and e.status != EntityStatus.FAMILY_MEMBER), None)
    pregnancy_status = None
    if pregnancy is not None:
        pregnancy_status = {EntityStatus.PRESENT: "pregnant", EntityStatus.POSSIBLE: "possibly pregnant",
                            EntityStatus.ABSENT: "not pregnant"}.get(pregnancy.status)
    return PatientInfo(
        demographics=Demographics(sex=sex[0] if sex else None, age=age[0] if age else None,
                                  pregnancy_status=pregnancy_status),
        allergies=[a.display_name for a in ctx.active_allergies()],
        medications=[m.display_name for m in entities.of_type(EntityType.MEDICATION) if m.status == EntityStatus.CURRENT],
    )


def build_confidence(transcript: Transcript, entities: EntitySet, note) -> ConfidenceSummary:
    extraction = (round(sum(e.confidence for e in entities.items) / len(entities.items), 3)
                  if entities.items else None)
    facts = [c for c in note.all_claims() if c.kind == ClaimKind.FACT]
    grounding = round(sum(c.confidence for c in facts) / len(facts), 3) if facts else None
    transcription = transcript.mean_asr_confidence
    parts = [(v, w) for v, w in ((transcription, 0.3), (extraction, 0.3), (grounding, 0.4)) if v is not None]
    overall = round(sum(v * w for v, w in parts) / sum(w for _, w in parts), 3) if parts else None
    return ConfidenceSummary(overall=overall, transcription=transcription, extraction=extraction, grounding=grounding)
