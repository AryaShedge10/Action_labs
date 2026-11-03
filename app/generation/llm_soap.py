"""LLM SOAP writer constrained to verified entities.

The model sees only non-blocked entities (by id) and must cite entity ids for every claim.
It never writes safety notices: those are appended deterministically afterwards.
"""

from __future__ import annotations

import json
import re

from app.models.schemas import Claim, ClaimKind, EntitySet, SOAPNote, SOAPSection
from app.utils.llm import OllamaClient

SOAP_PROMPT = """You are a clinical scribe writing a SOAP note from VERIFIED FACTS only.

Hard rules:
- Use ONLY the facts listed below. Do not add any symptom, diagnosis, medication, dose, number or finding
  that is not in the facts. If a section has no facts, return an empty list for it.
- Respect each fact's status: ABSENT = patient denies / avoided; FAMILY_MEMBER = about a relative, not the patient;
  POSSIBLE = uncertain (say "possible"); HISTORICAL/STOPPED = past only.
- Only DIAGNOSIS facts may appear in the assessment. Never invent a diagnosis.
- Refer to the patient as "the patient"; never use names.
- Do NOT write safety warnings or conflict notices - the system adds them.
- Write concise clinical prose, e.g. "Severe sore throat and low-grade fever for three days." or
  "Denies cough and shortness of breath." Group related facts; include pertinent negatives.
- Every claim is one short sentence and lists ALL entity_ids it is based on.

Return ONLY JSON:
{{"subjective": [{{"text": "...", "entity_ids": ["E001"]}}], "objective": [], "assessment": [], "plan": []}}

Facts (JSON):
{facts}
"""

_PHI_TOKEN = re.compile(r"\[(?:NAME|PHONE|EMAIL|ID|DATE|ADDRESS)_\d+\]")


class LLMSOAPGenerator:
    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(self, entities: EntitySet, blocked_ids: set[str]) -> SOAPNote:
        facts = [
            {
                "entity_id": e.entity_id,
                "type": e.type.value,
                "name": e.normalized,
                "status": e.status.value,
                **({"value": f"{e.value} {e.unit or ''}".strip()} if e.value else {}),
                **({"details": e.attributes} if e.attributes else {}),
                "said_by": sorted({ev.speaker.value for ev in e.evidence}),
            }
            for e in entities.items if e.entity_id not in blocked_ids
        ]
        data = self.client.generate_json(SOAP_PROMPT.format(facts=json.dumps(facts, indent=1)), temperature=0.1)

        valid_ids = {e.entity_id for e in entities.items}
        note = SOAPNote()
        for section in SOAPSection:
            items = data.get(section.value, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    item = {"text": item, "entity_ids": []}
                if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                    continue
                text = _PHI_TOKEN.sub("the patient", str(item["text"]).strip())
                ids = [str(i) for i in item.get("entity_ids", []) if str(i) in valid_ids]
                segment_ids = list(dict.fromkeys(sid for i in ids for sid in entities.get(i).segment_ids))
                note.section(section).append(Claim(section=section, text=text, kind=ClaimKind.FACT,
                                                   entity_ids=ids, segment_ids=segment_ids))
        return note
