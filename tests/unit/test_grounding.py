import pytest

from app.extraction.rule_based import RuleBasedExtractor
from app.grounding.verifier import GroundingVerifier
from app.models.schemas import Claim, ClaimKind, GroundingStatus as G, SOAPNote, SOAPSection
from tests.conftest import make_transcript

TURNS = [
    ("PATIENT", "I've had a sore throat and fever for three days."),
    ("PATIENT", "No cough. My father takes lisinopril."),
    ("DOCTOR", "Your temperature is 101.2 degrees."),
]


@pytest.fixture
def ctx():
    t = make_transcript(*TURNS)
    return t, RuleBasedExtractor().extract(t)


def verify(ctx, text, section=SOAPSection.SUBJECTIVE, segment_ids=("S001",), entity_ids=(), blocked=frozenset(),
           kind=ClaimKind.FACT):
    transcript, entities = ctx
    claim = Claim(section=section, text=text, segment_ids=list(segment_ids), entity_ids=list(entity_ids), kind=kind)
    note = SOAPNote()
    note.section(section).append(claim)
    report = GroundingVerifier().verify(note, transcript, entities, set(blocked))
    return claim, report


def test_supported_claim(ctx):
    claim, report = verify(ctx, "Patient reports sore throat and fever for three days.")
    assert claim.grounding_status == G.SUPPORTED and report.grounding_score == 1.0
    assert claim.citations[0].segment_sha256 and claim.citations[0].start_time == 0.0


def test_hallucinated_diagnosis_is_unsupported(ctx):
    claim, report = verify(ctx, "Clinician assessment: streptococcal pharyngitis.", SOAPSection.ASSESSMENT)
    assert claim.grounding_status == G.UNSUPPORTED and report.hallucination_score == 1.0
    assert "does not appear anywhere in the transcript" in claim.issues[0]


def test_hallucinated_number_is_unsupported(ctx):
    claim, _ = verify(ctx, "Patient reports sore throat for 10 days.")
    assert claim.grounding_status == G.UNSUPPORTED


def test_negation_flip_is_contradicted(ctx):
    claim, report = verify(ctx, "Patient reports a cough.", segment_ids=["S002"])
    assert claim.grounding_status == G.CONTRADICTED and report.contradiction_rate == 1.0


def test_pertinent_negative_is_supported(ctx):
    claim, _ = verify(ctx, "Patient denies cough.", segment_ids=["S002"])
    assert claim.grounding_status == G.SUPPORTED


def test_family_misattribution_is_contradicted(ctx):
    claim, _ = verify(ctx, "Patient takes lisinopril.", segment_ids=["S002"])
    assert claim.grounding_status == G.CONTRADICTED


def test_family_history_claim_is_supported(ctx):
    claim, _ = verify(ctx, "Family history: lisinopril (father).", segment_ids=["S002"])
    assert claim.grounding_status == G.SUPPORTED


def test_invalid_citation_is_unsupported(ctx):
    claim, report = verify(ctx, "Patient reports sore throat.", segment_ids=["S999"])
    assert claim.grounding_status == G.UNSUPPORTED and report.citation_verification_rate == 0.0


def test_uncited_but_present_term_is_partial(ctx):
    claim, _ = verify(ctx, "Temperature 101.2 °F.", SOAPSection.OBJECTIVE, segment_ids=["S001"])
    assert claim.grounding_status in (G.PARTIALLY_SUPPORTED, G.UNSUPPORTED)  # number not in cited segment
    claim, _ = verify(ctx, "Temperature 101.2 °F.", SOAPSection.OBJECTIVE, segment_ids=["S003"])
    assert claim.grounding_status == G.SUPPORTED


def test_blocked_item_asserted_is_contradicted():
    t = make_transcript(("PATIENT", "I'm allergic to penicillin."), ("DOCTOR", "I'll prescribe amoxicillin."))
    entities = RuleBasedExtractor().extract(t)
    amox = next(e for e in entities.items if e.normalized == "amoxicillin")
    claim = Claim(section=SOAPSection.PLAN, text="Prescribe amoxicillin.", entity_ids=[amox.entity_id])
    note = SOAPNote(plan=[claim])
    GroundingVerifier().verify(note, t, entities, {amox.entity_id})
    assert claim.grounding_status == G.CONTRADICTED


def test_meta_claims_excluded_from_metrics(ctx):
    _, report = verify(ctx, "No diagnosis was stated.", SOAPSection.ASSESSMENT, segment_ids=[], kind=ClaimKind.META)
    assert report.fact_claims == 0 and report.hallucination_score == 0.0


def test_template_notes_are_fully_grounded(pipeline, team_a_payload):
    result = pipeline.analyze(team_a_payload, "rule", "template")
    assert result.grounding.unsupported == 0 and result.grounding.contradicted == 0
    assert result.grounding.entity_coverage == 1.0
