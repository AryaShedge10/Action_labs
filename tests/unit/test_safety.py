import pytest

from app.extraction.rule_based import RuleBasedExtractor
from app.models.schemas import ConflictStatus as CS
from app.models.schemas import ConflictType as CT
from app.safety.engine import SafetyEngine
from app.safety.rules import SafetyRule
from tests.conftest import make_transcript


def run(*turns, engine=None):
    t = make_transcript(*turns)
    return (engine or SafetyEngine()).run(RuleBasedExtractor().extract(t), t)


def kinds(result):
    return {(c.type, c.status) for c in result.conflicts}


def test_penicillin_allergy_blocks_amoxicillin():
    r = run(("PATIENT", "I am allergic to penicillin."), ("DOCTOR", "I will prescribe amoxicillin."))
    [c] = r.conflicts
    assert (c.type, c.status) == (CT.MEDICATION_ALLERGY, CS.BLOCKED)
    assert c.allergy == "Penicillin" and c.medication == "Amoxicillin"   # Team C contract fields
    assert c.evidence and r.blocked_entity_ids


def test_avoided_allergen_is_not_a_conflict(team_a_payload):
    from app.extraction.rule_based import RuleBasedExtractor
    from app.ingestion.adapter import to_transcript
    t = to_transcript(team_a_payload)
    assert SafetyEngine().run(RuleBasedExtractor().extract(t), t).conflicts == []


def test_unclear_intent_mention_is_reviewed_not_ignored():
    r = run(("PATIENT", "I'm allergic to penicillin."), ("DOCTOR", "Amoxicillin is usually first line for this."))
    assert kinds(r) == {(CT.MEDICATION_ALLERGY, CS.REVIEW_REQUIRED)}


def test_class_level_allergy():
    r = run(("PATIENT", "I have a sulfa allergy."), ("DOCTOR", "I'll prescribe Bactrim."))
    assert (CT.MEDICATION_ALLERGY, CS.BLOCKED) in kinds(r)


def test_food_allergy_does_not_conflict_with_drugs():
    assert run(("PATIENT", "I'm allergic to peanuts."), ("DOCTOR", "Take ibuprofen 400 mg.")).conflicts == []


def test_cross_reactivity_is_review():
    r = run(("PATIENT", "I'm allergic to penicillin."), ("DOCTOR", "I'll prescribe cephalexin 500 mg."))
    assert kinds(r) == {(CT.ALLERGY_CROSS_REACTIVITY, CS.REVIEW_REQUIRED)}


def test_anaphylaxis_raises_cross_reactivity_severity():
    r = run(("PATIENT", "I'm allergic to penicillin, I had anaphylaxis."), ("DOCTOR", "I'll prescribe cephalexin."))
    assert r.conflicts[0].severity.value == "HIGH"


def test_male_pregnancy_blocked():
    r = run(("PATIENT", "I'm a 34 year old man."), ("DOCTOR", "Since you're pregnant, let's be careful."))
    assert (CT.DEMOGRAPHIC_CONFLICT, CS.BLOCKED) in kinds(r)


def test_sex_from_form_of_address_only_triggers_review():
    r = run(("DOCTOR", "Hello Mr. Patel."), ("DOCTOR", "You are pregnant."))
    assert kinds(r) == {(CT.DEMOGRAPHIC_CONFLICT, CS.REVIEW_REQUIRED)}


def test_female_psa_is_review():
    r = run(("PATIENT", "I'm a 40 year old woman."), ("DOCTOR", "Let's order a PSA test."))
    assert kinds(r) == {(CT.DEMOGRAPHIC_CONFLICT, CS.REVIEW_REQUIRED)}


def test_pregnancy_contraindication_blocks_ace_inhibitor():
    r = run(("PATIENT", "I'm 12 weeks pregnant."), ("DOCTOR", "I'll start you on lisinopril 10 mg daily."))
    assert (CT.PREGNANCY_CONTRAINDICATION, CS.BLOCKED) in kinds(r)


def test_contraindicated_interaction_blocks_new_drug():
    r = run(("PATIENT", "I'm on nitroglycerin."), ("DOCTOR", "I'll prescribe sildenafil."))
    [c] = r.conflicts
    assert (c.type, c.status, c.medication) == (CT.DRUG_INTERACTION, CS.BLOCKED, "Sildenafil")


def test_major_interaction_is_review():
    r = run(("PATIENT", "I take warfarin."), ("DOCTOR", "Take ibuprofen 400 mg every 6 hours."))
    assert kinds(r) == {(CT.DRUG_INTERACTION, CS.REVIEW_REQUIRED)}


def test_existing_regimen_interaction_is_review_not_block():
    r = run(("PATIENT", "I take sertraline and phenelzine."))
    assert kinds(r) == {(CT.DRUG_INTERACTION, CS.REVIEW_REQUIRED)}


def test_duplicate_therapy():
    r = run(("PATIENT", "I take ibuprofen daily."), ("DOCTOR", "I'll also prescribe naproxen 500 mg twice daily."))
    assert (CT.DUPLICATE_THERAPY, CS.REVIEW_REQUIRED) in kinds(r)


@pytest.mark.parametrize("text", [
    "Start acetaminophen 5000 mg once.",          # single dose
    "Take acetaminophen 1000 mg every 4 hours.",  # daily total 6000 mg
])
def test_dose_range(text):
    assert (CT.DOSE_RANGE, CS.REVIEW_REQUIRED) in kinds(run(("DOCTOR", text)))


def test_aspirin_in_child_and_codeine_under_12():
    assert (CT.AGE_CONTRAINDICATION, CS.REVIEW_REQUIRED) in kinds(
        run(("DOCTOR", "She is a 6-year-old girl."), ("DOCTOR", "Give her aspirin.")))
    assert (CT.AGE_CONTRAINDICATION, CS.BLOCKED) in kinds(
        run(("DOCTOR", "He is a 10-year-old boy."), ("DOCTOR", "I'll prescribe codeine.")))


@pytest.mark.parametrize("text", ["Temperature is 201 degrees.", "Blood pressure is 80 over 120.", "Oxygen saturation is 85 percent."])
def test_vital_alerts(text):
    assert (CT.VITAL_SIGN_ALERT, CS.REVIEW_REQUIRED) in kinds(run(("NURSE", text)))


def test_normal_vitals_no_alert():
    assert run(("NURSE", "Temperature is 98.6, blood pressure 118 over 76, pulse 72.")).conflicts == []


def test_inconsistent_allergy_history():
    r = run(("PATIENT", "I don't have any allergies."), ("PATIENT", "Actually I'm allergic to penicillin."))
    assert (CT.INCONSISTENT_HISTORY, CS.REVIEW_REQUIRED) in kinds(r)


def test_low_asr_confidence_on_safety_critical_segment():
    r = run(("PATIENT", "I'm allergic to penicillin.", 0.4))
    assert (CT.TRANSCRIPTION_QUALITY, CS.REVIEW_REQUIRED) in kinds(r)


def test_unknown_speaker_flagged():
    r = run(("SPEAKER_1", "I have chest pain."))
    assert (CT.TRANSCRIPTION_QUALITY, CS.REVIEW_REQUIRED) in kinds(r)


def test_engine_fails_closed_when_a_rule_crashes():
    class Broken(SafetyRule):
        rule_id = "BROKEN"

        def evaluate(self, ctx):
            raise RuntimeError("bug")

    r = run(("PATIENT", "I have a cough."), engine=SafetyEngine(rules=[Broken()]))
    assert kinds(r) == {(CT.RULE_ENGINE_ERROR, CS.REVIEW_REQUIRED)}


def test_conflicts_sorted_blocked_first_with_stable_ids():
    r = run(("PATIENT", "I'm allergic to penicillin.", 0.4), ("DOCTOR", "I'll prescribe amoxicillin."))
    assert r.conflicts[0].status == CS.BLOCKED
    assert [c.conflict_id for c in r.conflicts] == [f"K{i:03d}" for i in range(1, len(r.conflicts) + 1)]
