"""Domain models used inside the pipeline and in the JSON contract served to Team C.

Design notes
------------
* The LLM never produces timestamps or hashes. Every `Evidence` / `Citation` is built by
  deterministic code from a `Segment` we received from Team A, so provenance cannot be
  hallucinated.
* `Segment.sha256` lets Team C (or an auditor) re-hash the transcript and verify that a
  citation really points at the text that was spoken.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from app import PIPELINE_VERSION, RULE_VERSION, SCHEMA_VERSION


# --------------------------------------------------------------------------- transcript

class Speaker(str, Enum):
    DOCTOR = "DOCTOR"
    NURSE = "NURSE"
    PATIENT = "PATIENT"
    UNKNOWN = "UNKNOWN"


CLINICIAN_SPEAKERS = {Speaker.DOCTOR, Speaker.NURSE}


class Segment(BaseModel):
    segment_id: str                      # "S001" — stable id used by citations
    index: int
    speaker: Speaker
    start_time: float                    # seconds from start of recording
    end_time: float
    text: str
    confidence: float | None = None      # ASR confidence from Team A (0..1)
    sha256: str


class Transcript(BaseModel):
    consultation_id: str
    language: str = "en-US"
    source_format: str
    segments: list[Segment]
    sha256: str                          # hash over all segment hashes

    _by_id: dict[str, Segment] = PrivateAttr(default_factory=dict)

    def segment(self, segment_id: str) -> Segment | None:
        if not self._by_id:
            self._by_id = {s.segment_id: s for s in self.segments}
        return self._by_id.get(segment_id)

    @property
    def mean_asr_confidence(self) -> float | None:
        values = [s.confidence for s in self.segments if s.confidence is not None]
        return round(sum(values) / len(values), 4) if values else None


class Evidence(BaseModel):
    segment_id: str
    speaker: Speaker
    start_time: float
    end_time: float
    quote: str                           # exact substring of the segment text
    segment_sha256: str

    @classmethod
    def from_segment(cls, segment: Segment, quote: str | None = None) -> "Evidence":
        return cls(
            segment_id=segment.segment_id,
            speaker=segment.speaker,
            start_time=segment.start_time,
            end_time=segment.end_time,
            quote=quote if quote else segment.text,
            segment_sha256=segment.sha256,
        )


# --------------------------------------------------------------------------- entities

class EntityType(str, Enum):
    SYMPTOM = "SYMPTOM"
    FINDING = "FINDING"          # clinician-observed examination finding
    ALLERGY = "ALLERGY"
    MEDICATION = "MEDICATION"
    VITAL = "VITAL"
    DIAGNOSIS = "DIAGNOSIS"
    PROCEDURE = "PROCEDURE"      # tests, exams, orders
    DEMOGRAPHIC = "DEMOGRAPHIC"


class EntityStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"            # negated, denied, or (for medications) explicitly avoided
    POSSIBLE = "POSSIBLE"
    HISTORICAL = "HISTORICAL"
    CURRENT = "CURRENT"
    STOPPED = "STOPPED"
    RECOMMENDED = "RECOMMENDED"
    PRESCRIBED = "PRESCRIBED"
    FAMILY_MEMBER = "FAMILY_MEMBER"
    UNKNOWN = "UNKNOWN"


ACTIVE_MEDICATION_STATUSES = {EntityStatus.CURRENT, EntityStatus.PRESCRIBED, EntityStatus.RECOMMENDED}
NEW_MEDICATION_STATUSES = {EntityStatus.PRESCRIBED, EntityStatus.RECOMMENDED}
ACTIVE_ALLERGY_STATUSES = {EntityStatus.PRESENT, EntityStatus.HISTORICAL, EntityStatus.POSSIBLE, EntityStatus.UNKNOWN}
AFFIRMED_STATUSES = {EntityStatus.PRESENT, EntityStatus.POSSIBLE, EntityStatus.CURRENT}


class ClinicalEntity(BaseModel):
    entity_id: str = ""
    type: EntityType
    text: str                            # surface form as spoken
    normalized: str                      # canonical lower-case term (e.g. generic drug name)
    status: EntityStatus = EntityStatus.UNKNOWN
    value: str | None = None             # vitals value, demographic value
    unit: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["rule", "llm", "rule+llm"] = "rule"
    # Status of every explicit mention, in time order (used by consistency checks).
    status_history: list[EntityStatus] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.normalized[:1].upper() + self.normalized[1:]

    @property
    def segment_ids(self) -> list[str]:
        return list(dict.fromkeys(e.segment_id for e in self.evidence))

    @property
    def first_time(self) -> float:
        return min((e.start_time for e in self.evidence), default=0.0)


class EntitySet(BaseModel):
    items: list[ClinicalEntity] = Field(default_factory=list)

    def of_type(self, *types: EntityType) -> list[ClinicalEntity]:
        return [e for e in self.items if e.type in types]

    def get(self, entity_id: str) -> ClinicalEntity | None:
        return next((e for e in self.items if e.entity_id == entity_id), None)

    def find(self, normalized: str, type_: EntityType | None = None) -> list[ClinicalEntity]:
        return [e for e in self.items if e.normalized == normalized and (type_ is None or e.type == type_)]


# --------------------------------------------------------------------------- safety

class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ConflictStatus(str, Enum):
    BLOCKED = "BLOCKED"                  # must not appear in the note / prescription
    REVIEW_REQUIRED = "REVIEW_REQUIRED"  # physician must look at it before sign-off


class ConflictType(str, Enum):
    MEDICATION_ALLERGY = "MEDICATION_ALLERGY"
    ALLERGY_CROSS_REACTIVITY = "ALLERGY_CROSS_REACTIVITY"
    DRUG_INTERACTION = "DRUG_INTERACTION"
    DUPLICATE_THERAPY = "DUPLICATE_THERAPY"
    DOSE_RANGE = "DOSE_RANGE"
    DEMOGRAPHIC_CONFLICT = "DEMOGRAPHIC_CONFLICT"
    PREGNANCY_CONTRAINDICATION = "PREGNANCY_CONTRAINDICATION"
    AGE_CONTRAINDICATION = "AGE_CONTRAINDICATION"
    VITAL_SIGN_ALERT = "VITAL_SIGN_ALERT"
    INCONSISTENT_HISTORY = "INCONSISTENT_HISTORY"
    TRANSCRIPTION_QUALITY = "TRANSCRIPTION_QUALITY"
    RULE_ENGINE_ERROR = "RULE_ENGINE_ERROR"


class Conflict(BaseModel):
    conflict_id: str = ""
    rule_id: str
    type: ConflictType
    severity: Severity
    status: ConflictStatus
    message: str
    allergy: str | None = None           # kept flat for Team C's agreed contract
    medication: str | None = None
    involved_terms: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    blocked_entity_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    recommendation: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- SOAP + grounding

class SOAPSection(str, Enum):
    SUBJECTIVE = "subjective"
    OBJECTIVE = "objective"
    ASSESSMENT = "assessment"
    PLAN = "plan"


class ClaimKind(str, Enum):
    FACT = "FACT"          # clinical assertion — must be grounded in the transcript
    SAFETY = "SAFETY"      # deterministic safety notice injected by the pipeline
    META = "META"          # e.g. "no diagnosis stated" — not a clinical assertion


class GroundingStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Citation(BaseModel):
    segment_id: str
    speaker: Speaker
    start_time: float
    end_time: float
    text: str
    segment_sha256: str
    asr_confidence: float | None = None


class Claim(BaseModel):
    claim_id: str = ""
    section: SOAPSection
    text: str
    kind: ClaimKind = ClaimKind.FACT
    entity_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    grounding_status: GroundingStatus | None = None
    support_score: float = 0.0
    confidence: float = 0.0
    issues: list[str] = Field(default_factory=list)


class SOAPNote(BaseModel):
    subjective: list[Claim] = Field(default_factory=list)
    objective: list[Claim] = Field(default_factory=list)
    assessment: list[Claim] = Field(default_factory=list)
    plan: list[Claim] = Field(default_factory=list)

    def all_claims(self) -> list[Claim]:
        return self.subjective + self.objective + self.assessment + self.plan

    def section(self, section: SOAPSection) -> list[Claim]:
        return getattr(self, section.value)

    def render(self) -> "SOAPText":
        def join(claims: list[Claim]) -> str:
            return " ".join(c.text.strip() for c in claims if c.text.strip())
        return SOAPText(
            subjective=join(self.subjective),
            objective=join(self.objective),
            assessment=join(self.assessment),
            plan=join(self.plan),
        )


class SOAPText(BaseModel):
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""


class RemovedClaim(BaseModel):
    section: SOAPSection
    text: str
    reason: str


class GroundingReport(BaseModel):
    total_claims: int = 0
    fact_claims: int = 0
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    grounding_score: float = 0.0             # (supported + 0.5 * partial) / fact_claims
    unsupported_claim_rate: float = 0.0
    contradiction_rate: float = 0.0
    hallucination_score: float = 0.0         # (unsupported + contradicted) / fact_claims
    citation_verification_rate: float = 0.0  # cited segments that exist in the transcript
    entity_coverage: float = 0.0             # extracted facts referenced by at least one claim (completeness)
    removed_claims: list[RemovedClaim] = Field(default_factory=list)


# --------------------------------------------------------------------------- prescriptions

class PrescriptionStatus(str, Enum):
    DRAFT = "DRAFT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"


class DraftPrescription(BaseModel):
    prescription_id: str
    entity_id: str
    medication: str
    intent: Literal["PRESCRIBED", "RECOMMENDED"]
    dose: str | None = None
    frequency: str | None = None
    route: str | None = None
    duration: str | None = None
    status: PrescriptionStatus
    missing_fields: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    requires_signature: bool = True


# --------------------------------------------------------------------------- response (Team C contract)

class SafetyStatus(str, Enum):
    SAFE = "SAFE"                            # no conflicts or grounding issues detected
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CONFLICT_DETECTED = "CONFLICT_DETECTED"  # at least one BLOCKED conflict


class Demographics(BaseModel):
    sex: str | None = None
    age: int | None = None
    pregnancy_status: str | None = None


class PatientInfo(BaseModel):
    demographics: Demographics = Field(default_factory=Demographics)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)   # medications the patient currently takes


class ConfidenceSummary(BaseModel):
    overall: float | None = None
    transcription: float | None = None   # mean ASR confidence from Team A
    extraction: float | None = None      # mean entity confidence
    grounding: float | None = None       # mean claim confidence (FACT claims)


class AnalysisMetadata(BaseModel):
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    rule_version: str = RULE_VERSION
    source_format: str
    extraction_mode: str
    generation_mode: str
    model: str | None = None
    phi_redacted_for_llm: bool = False
    transcript_sha256: str
    processing_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ConsultationAnalysis(BaseModel):
    schema_version: str = SCHEMA_VERSION
    consultation_id: str
    generated_at: str = Field(default_factory=_utc_now)
    safety_status: SafetyStatus
    requires_physician_signoff: bool = True
    patient_info: PatientInfo
    soap_note: SOAPText
    soap_claims: SOAPNote
    conflicts: list[Conflict]
    draft_prescriptions: list[DraftPrescription]
    entities: list[ClinicalEntity]
    grounding: GroundingReport
    confidence: ConfidenceSummary
    transcript: list[Segment]
    metadata: AnalysisMetadata
