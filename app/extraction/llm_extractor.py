"""LLM-based extraction whose every output is verified against the transcript.

The model cites a segment id and a verbatim quote. We keep an entity only if the quote is
really in that segment, it is not taken from a question, and (for diagnoses) the term itself
was actually said. Timestamps and hashes are attached by our code, never by the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.knowledge.drugs import ALLERGEN_CLASS_TERMS
from app.models.schemas import ClinicalEntity, EntityStatus, EntityType, Evidence, Transcript
from app.nlp.matcher import LexiconMatcher, clinical_matcher
from app.nlp.text import split_sentences, tokens
from app.privacy.phi import PHIRedactor
from app.utils.llm import OllamaClient

EXTRACTION_PROMPT = """You are a clinical information extraction engine. Extract clinical facts from the doctor-patient transcript below.

Rules:
1. Extract ONLY facts explicitly stated. Never infer a diagnosis the clinician did not say.
2. Every entity cites exactly ONE segment_id and a verbatim "quote" (3-15 words) copied exactly from that segment.
3. Questions asked by the doctor are not facts; use the patient's answer instead.
4. Negated or denied items have status ABSENT. Items about family members have status FAMILY_MEMBER.
5. An allergy is type ALLERGY with the allergen as name. Never list an allergen as a MEDICATION.
6. Medication status: CURRENT (patient already takes it), PRESCRIBED (clinician prescribes it now),
   RECOMMENDED (advised, e.g. over the counter), STOPPED, HISTORICAL, ABSENT (explicitly avoided or not taken).
7. For VITAL put the number in "value" and the unit in "unit". For MEDICATION put dose, frequency, duration,
   route in "attributes". For DEMOGRAPHIC use name "sex" or "age" with the value in "value".
8. Bracketed tokens such as [NAME_1] are redacted personal data; never output them.

Allowed types: SYMPTOM, FINDING, ALLERGY, MEDICATION, VITAL, DIAGNOSIS, PROCEDURE, DEMOGRAPHIC
Allowed statuses: PRESENT, ABSENT, POSSIBLE, HISTORICAL, CURRENT, STOPPED, RECOMMENDED, PRESCRIBED, FAMILY_MEMBER, UNKNOWN

Return ONLY JSON in this shape:
{{"entities": [{{"type": "SYMPTOM", "name": "sore throat", "status": "PRESENT", "segment_id": "S002",
  "quote": "I've had a severe sore throat", "value": null, "unit": null, "attributes": {{}}}}]}}

Transcript:
{transcript}
"""

_VITAL_NAMES = {
    "temp": "temperature", "temperature": "temperature", "body temperature": "temperature",
    "bp": "blood pressure", "blood pressure": "blood pressure", "hr": "heart rate", "pulse": "heart rate",
    "heart rate": "heart rate", "rr": "respiratory rate", "respiratory rate": "respiratory rate",
    "spo2": "oxygen saturation", "oxygen saturation": "oxygen saturation", "o2 saturation": "oxygen saturation",
    "weight": "weight",
}


@dataclass
class LLMExtractionResult:
    entities: list[ClinicalEntity] = field(default_factory=list)
    proposed: int = 0
    rejected: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("’", "'")).strip()


class LLMExtractor:
    def __init__(self, client: OllamaClient, matcher: LexiconMatcher | None = None, redact_phi: bool = True):
        self.client = client
        self.matcher = matcher or clinical_matcher()
        self.redact_phi = redact_phi

    def render_transcript(self, transcript: Transcript) -> str:
        redactor = PHIRedactor()
        lines = []
        for s in transcript.segments:
            text = redactor.redact(s.text) if self.redact_phi else s.text
            lines.append(f"[{s.segment_id}] {s.speaker.value} ({s.start_time:.1f}s): {text}")
        return "\n".join(lines)

    def extract(self, transcript: Transcript) -> LLMExtractionResult:
        prompt = EXTRACTION_PROMPT.format(transcript=self.render_transcript(transcript))
        data = self.client.generate_json(prompt)
        items = data.get("entities", [])
        result = LLMExtractionResult(proposed=len(items) if isinstance(items, list) else 0)
        if not isinstance(items, list):
            result.rejected.append("response had no 'entities' list")
            return result
        for item in items:
            entity, reason = self._verify(item, transcript)
            if entity:
                result.entities.append(entity)
            else:
                result.rejected.append(reason)
        return result

    # ------------------------------------------------------------------ verification

    def _verify(self, item: object, transcript: Transcript) -> tuple[ClinicalEntity | None, str]:
        if not isinstance(item, dict):
            return None, "non-object entity"
        try:
            etype = EntityType(str(item.get("type", "")).upper())
        except ValueError:
            return None, f"unknown type {item.get('type')!r}"
        name = str(item.get("name") or "").strip()
        if not name or "[" in name:
            return None, "missing or redacted name"
        segment = transcript.segment(str(item.get("segment_id", "")).strip())
        if segment is None:
            return None, f"invalid segment_id {item.get('segment_id')!r} for {name!r}"

        quote = self._locate_quote(str(item.get("quote") or ""), segment.text)
        if quote is None:
            return None, f"quote for {name!r} not found in {segment.segment_id}"
        sentence = next((s for s in split_sentences(segment.text) if quote in s.text or s.text in quote), None)
        if sentence is not None and sentence.is_question:
            return None, f"{name!r} cited from a question"

        normalized, value = self._normalize(etype, name, item.get("value"))
        if normalized is None:
            return None, f"could not normalise {name!r}"
        mentioned = self._term_in_text(etype, normalized, name, segment.text)
        if etype == EntityType.DIAGNOSIS and not mentioned:
            return None, f"diagnosis {name!r} was not stated in {segment.segment_id}"

        try:
            status = EntityStatus(str(item.get("status", "UNKNOWN")).upper())
        except ValueError:
            status = EntityStatus.UNKNOWN
        attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        attrs = {str(k): str(v) for k, v in attrs.items() if v not in (None, "", [], {})}
        asr = segment.confidence if segment.confidence is not None else 1.0
        conf = (0.75 if mentioned else 0.55) * asr
        unit = item.get("unit")
        return ClinicalEntity(
            type=etype,
            text=name,
            normalized=normalized,
            status=status,
            value=value,
            unit=str(unit) if unit else None,
            attributes=attrs,
            evidence=[Evidence.from_segment(segment, quote)],
            confidence=round(conf, 3),
            source="llm",
            status_history=[status],
        ), ""

    @staticmethod
    def _locate_quote(quote: str, segment_text: str) -> str | None:
        q, t = _norm(quote), _norm(segment_text)
        if not q:
            return None
        idx = t.find(q)
        if idx != -1:
            return segment_text[idx:idx + len(q)] if _norm(segment_text[idx:idx + len(q)]) == q else segment_text
        q_tokens = tokens(q)
        if q_tokens and sum(tok in set(tokens(t)) for tok in q_tokens) / len(q_tokens) >= 0.8:
            return segment_text
        return None

    def _normalize(self, etype: EntityType, name: str, value: object) -> tuple[str | None, str | None]:
        lowered = _norm(name)
        value_str = str(value).strip() if value not in (None, "") else None
        if etype == EntityType.DEMOGRAPHIC:
            if lowered in ("male", "female", "man", "woman"):
                return "sex", "male" if lowered in ("male", "man") else "female"
            if lowered == "sex" and value_str:
                v = value_str.lower()
                return "sex", "male" if v in ("male", "m", "man") else "female" if v in ("female", "f", "woman") else v
            if "age" in lowered and value_str and re.match(r"^\d{1,3}", value_str):
                return "age", re.match(r"^\d{1,3}", value_str).group()
            return None, None
        if etype == EntityType.VITAL:
            return _VITAL_NAMES.get(lowered, lowered), value_str
        if etype == EntityType.ALLERGY:
            if lowered in ALLERGEN_CLASS_TERMS:
                return lowered, value_str
            canonical = self.matcher.canonical(EntityType.MEDICATION, lowered)
            return canonical or lowered, value_str
        canonical = self.matcher.canonical(etype, lowered)
        if canonical:
            return canonical, value_str
        found = [m for m in self.matcher.find(name) if m.type == etype]
        return (found[0].canonical if len(found) == 1 else lowered), value_str

    def _term_in_text(self, etype: EntityType, normalized: str, name: str, text: str) -> bool:
        if _norm(name) in _norm(text) or normalized in _norm(text):
            return True
        if etype in (EntityType.VITAL, EntityType.DEMOGRAPHIC):
            return True
        lookup_type = EntityType.MEDICATION if etype == EntityType.ALLERGY else etype
        return any(m.canonical == normalized for m in self.matcher.find(text, {lookup_type}))
