"""LLM integration layer, tested with a fake client (no network, deterministic)."""

import pytest

from app.config import Settings
from app.extraction.llm_extractor import LLMExtractor
from app.generation.soap_generator import scrub_blocked
from app.models.schemas import Claim, EntityStatus, SOAPNote, SOAPSection
from app.pipeline import ClinicalPipeline
from app.privacy.phi import PHIRedactor
from app.utils.llm import LLMError, OllamaClient, PrivacyViolationError, is_local_endpoint, parse_json_object
from tests.conftest import FakeLLMClient, make_payload, make_transcript


# --------------------------------------------------------------------------- privacy

def test_phi_redaction_is_consistent_and_keeps_clinical_text():
    r = PHIRedactor()
    out = r.redact("Good morning Mr. Davis, call 555-123-4567 or davis@mail.com. My name is John Smith. BP 120/80.")
    assert "Davis" not in out and "555-123-4567" not in out and "davis@mail.com" not in out and "John Smith" not in out
    assert "Mr. [NAME_1]" in out and "BP 120/80" in out
    assert r.redact("Mr. Davis again") == "Mr. [NAME_1] again"


def test_remote_llm_endpoint_refused_by_default():
    assert is_local_endpoint("http://localhost:11434") and is_local_endpoint("http://192.168.1.20:11434")
    assert is_local_endpoint("http://host.docker.internal:11434")
    assert not is_local_endpoint("https://api.example.com")
    with pytest.raises(PrivacyViolationError):
        OllamaClient("https://api.example.com", "m")
    OllamaClient("https://api.example.com", "m", allow_remote=True)  # explicit opt-in works


def test_pipeline_degrades_gracefully_with_remote_llm_configured(team_a_payload):
    p = ClinicalPipeline(Settings(ollama_base_url="https://api.example.com", store_results=False))
    result = p.analyze(team_a_payload)
    assert result.metadata.extraction_mode == "rule"
    assert any("non-local" in w for w in result.metadata.warnings)


@pytest.mark.parametrize("raw", [
    '{"a": 1}', '```json\n{"a": 1}\n```', '<think>reasoning</think>{"a": 1}', 'Here you go: {"a": 1} hope it helps',
])
def test_parse_json_object_is_robust(raw):
    assert parse_json_object(raw) == {"a": 1}


def test_parse_json_object_rejects_garbage():
    with pytest.raises(LLMError):
        parse_json_object("no json here")


# --------------------------------------------------------------------------- verified LLM extraction

TRANSCRIPT_TURNS = [("DOCTOR", "Good morning Mr. Davis. Any cough?"), ("PATIENT", "No cough, but a bad sore throat.")]


def test_llm_entities_are_verified_against_transcript():
    fake = FakeLLMClient({"entities": [
        {"type": "SYMPTOM", "name": "sore throat", "status": "PRESENT", "segment_id": "S002", "quote": "a bad sore throat"},
        {"type": "SYMPTOM", "name": "cough", "status": "ABSENT", "segment_id": "S002", "quote": "No cough"},
        {"type": "DIAGNOSIS", "name": "strep throat", "status": "PRESENT", "segment_id": "S002", "quote": "a bad sore throat"},
        {"type": "SYMPTOM", "name": "fever", "status": "PRESENT", "segment_id": "S002", "quote": "high fever"},
        {"type": "SYMPTOM", "name": "cough", "status": "PRESENT", "segment_id": "S001", "quote": "Any cough?"},
        {"type": "SYMPTOM", "name": "nausea", "status": "PRESENT", "segment_id": "S042", "quote": "nausea"},
    ]})
    result = LLMExtractor(fake).extract(make_transcript(*TRANSCRIPT_TURNS))
    kept = {(e.normalized, e.status) for e in result.entities}
    assert kept == {("sore throat", EntityStatus.PRESENT), ("cough", EntityStatus.ABSENT)}
    assert len(result.rejected) == 4   # inferred diagnosis, fake quote, question, bad segment id
    assert "Davis" not in fake.prompts[0] and "[NAME_1]" in fake.prompts[0]   # PHI redacted before the LLM
    assert result.entities[0].evidence[0].segment_sha256   # provenance attached by code, not by the LLM


def test_hybrid_merge_keeps_rule_status_on_disagreement():
    fake = FakeLLMClient(
        {"entities": [{"type": "SYMPTOM", "name": "cough", "status": "PRESENT", "segment_id": "S002", "quote": "No cough"},
                      {"type": "SYMPTOM", "name": "throat pain", "status": "PRESENT", "segment_id": "S002",
                       "quote": "bad sore throat"}]},
    )
    p = ClinicalPipeline(Settings(store_results=False), llm_client=fake)
    result = p.analyze(make_payload(*TRANSCRIPT_TURNS), extraction_mode="hybrid", generation_mode="template")
    cough = next(e for e in result.entities if e.normalized == "cough")
    assert cough.status == EntityStatus.ABSENT and cough.attributes["llm_status"] == "PRESENT"
    assert result.metadata.extraction_mode == "hybrid"


def test_llm_soap_is_scrubbed_backfilled_and_safety_injected():
    llm_note = {
        "subjective": [{"text": "The patient reports a bad sore throat.", "entity_ids": ["E003"]}],
        "objective": [], "assessment": [],
        "plan": [{"text": "Start amoxicillin.", "entity_ids": []}],
    }
    turns = [("PATIENT", "I'm allergic to penicillin. No fever."), ("PATIENT", "I have a bad sore throat."),
             ("DOCTOR", "I'll prescribe amoxicillin.")]
    fake = FakeLLMClient(llm_note)
    p = ClinicalPipeline(Settings(store_results=False), llm_client=fake)
    result = p.analyze(make_payload(*turns), extraction_mode="rule", generation_mode="llm")
    texts = [c.text for c in result.soap_claims.all_claims()]
    assert "Start amoxicillin." not in texts                                  # blocked drug scrubbed
    assert result.grounding.removed_claims[0].text == "Start amoxicillin."
    assert any(t.startswith("SAFETY BLOCK") for t in texts)                   # safety notice injected by code
    assert any("denies fever" in t for t in texts)                            # omitted negative backfilled
    assert result.safety_status.value == "CONFLICT_DETECTED"


def test_scrub_keeps_allergy_documentation():
    t = make_transcript(("PATIENT", "I'm allergic to amoxicillin."), ("DOCTOR", "I'll prescribe amoxicillin."))
    from app.extraction.rule_based import RuleBasedExtractor
    entities = RuleBasedExtractor().extract(t)
    med = next(e for e in entities.items if e.type.value == "MEDICATION")
    note = SOAPNote(subjective=[Claim(section=SOAPSection.SUBJECTIVE, text="Allergic to amoxicillin.")],
                    plan=[Claim(section=SOAPSection.PLAN, text="Prescribe amoxicillin.")])
    removed = scrub_blocked(note, entities, {med.entity_id})
    assert [r.text for r in removed] == ["Prescribe amoxicillin."]
    assert note.subjective[0].text == "Allergic to amoxicillin."


def test_llm_failure_falls_back_to_template(team_a_payload):
    class Failing(FakeLLMClient):
        def generate_json(self, prompt, temperature=0.0):
            raise LLMError("boom")

    p = ClinicalPipeline(Settings(store_results=False), llm_client=Failing())
    result = p.analyze(team_a_payload)  # auto modes
    assert result.metadata.extraction_mode == "rule" and result.metadata.generation_mode == "template"
    assert len(result.metadata.warnings) == 2


def test_explicit_llm_mode_errors_when_unavailable(team_a_payload):
    p = ClinicalPipeline(Settings(store_results=False), llm_client=FakeLLMClient(available=False))
    with pytest.raises(LLMError):
        p.analyze(team_a_payload, extraction_mode="llm")
