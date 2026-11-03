import pytest

from app.extraction.rule_based import RuleBasedExtractor
from app.models.schemas import EntityStatus as St
from app.models.schemas import EntityType as T
from tests.conftest import make_transcript

extractor = RuleBasedExtractor()


def extract(*turns):
    return {(e.type, e.normalized): e for e in extractor.extract(make_transcript(*turns)).items}


def status_of(entities, etype, name):
    return entities[(etype, name)].status


def test_team_a_sample_entities(team_a_payload):
    from app.ingestion.adapter import to_transcript
    es = {(e.type, e.normalized): e for e in extractor.extract(to_transcript(team_a_payload)).items}
    assert status_of(es, T.ALLERGY, "amoxicillin") == St.PRESENT
    assert es[(T.ALLERGY, "penicillin")].attributes["reaction"] == "hives"
    assert (T.MEDICATION, "amoxicillin") not in es          # allergen is never a prescription
    assert status_of(es, T.MEDICATION, "penicillin") == St.ABSENT  # "we will avoid the penicillin family"
    assert es[(T.SYMPTOM, "sore throat")].attributes == {"severity": "severe", "duration": "three days"}
    assert status_of(es, T.SYMPTOM, "cough") == St.ABSENT
    assert status_of(es, T.SYMPTOM, "rash") == St.ABSENT    # inferred from question + "No ..., just ..."
    assert status_of(es, T.SYMPTOM, "body aches") == St.PRESENT
    assert (T.SYMPTOM, "hives") not in es                    # past reaction, not a current symptom


@pytest.mark.parametrize("text,name,expected", [
    ("I do not have a fever or chest pain.", "chest pain", St.ABSENT),
    ("No cough or shortness of breath, just fatigue.", "fatigue", St.PRESENT),
    ("I deny any nausea.", "nausea", St.ABSENT),
    ("I had a headache yesterday but it's gone now.", "headache", St.HISTORICAL),
    ("I think I might have a fever.", "fever", St.POSSIBLE),
    ("My wife has a bad cough.", "cough", St.FAMILY_MEMBER),
    ("I'm not sure if it's a rash.", "rash", St.POSSIBLE),
])
def test_symptom_context(text, name, expected):
    assert status_of(extract(("PATIENT", text)), T.SYMPTOM, name) == expected


@pytest.mark.parametrize("speaker,text,name,expected", [
    ("PATIENT", "I take lisinopril 10 mg daily.", "lisinopril", St.CURRENT),
    ("PATIENT", "I used to take metformin last year, but I stopped.", "metformin", St.HISTORICAL),
    ("PATIENT", "My father takes lisinopril for his heart.", "lisinopril", St.FAMILY_MEMBER),
    ("DOCTOR", "I'll prescribe amoxicillin 500 mg three times a day.", "amoxicillin", St.PRESCRIBED),
    ("DOCTOR", "You can take ibuprofen as needed.", "ibuprofen", St.RECOMMENDED),
    ("DOCTOR", "We will avoid ibuprofen.", "ibuprofen", St.ABSENT),
    ("DOCTOR", "Let's avoid amoxicillin and use azithromycin.", "azithromycin", St.PRESCRIBED),
    ("DOCTOR", "Instead of amoxicillin, I'll prescribe azithromycin.", "amoxicillin", St.ABSENT),
    ("DOCTOR", "Amoxicillin is often used for this.", "amoxicillin", St.UNKNOWN),
    ("PATIENT", "I took some Tylenol this morning.", "acetaminophen", St.CURRENT),
])
def test_medication_status(speaker, text, name, expected):
    assert status_of(extract((speaker, text)), T.MEDICATION, name) == expected


def test_brand_names_normalise_to_generic():
    es = extract(("PATIENT", "I took Advil and Augmentin."))
    assert (T.MEDICATION, "ibuprofen") in es and (T.MEDICATION, "amoxicillin-clavulanate") in es


def test_dosing_attributes():
    med = extract(("DOCTOR", "Take amoxicillin 875 mg by mouth twice a day for 10 days."))[(T.MEDICATION, "amoxicillin")]
    assert med.attributes["dose_mg"] == "875"
    assert med.attributes["frequency"] == "twice a day" and med.attributes["doses_per_day"] == "2"
    assert med.attributes["duration"] == "10 days" and med.attributes["route"] == "by mouth"


def test_allergy_forms():
    es = extract(("PATIENT", "I have a sulfa allergy and I'm allergic to peanuts. Codeine gives me hives."))
    assert es[(T.ALLERGY, "sulfa drugs")].attributes["allergen_class"] == "sulfonamide_antibiotics"
    assert es[(T.ALLERGY, "peanuts")].attributes["allergen_kind"] == "non_drug"
    assert es[(T.ALLERGY, "codeine")].attributes["reaction"] == "hives"
    assert (T.MEDICATION, "codeine") not in es


def test_negated_and_family_allergy():
    es = extract(("PATIENT", "I'm not allergic to penicillin. My mother is allergic to sulfa."))
    assert status_of(es, T.ALLERGY, "penicillin") == St.ABSENT
    assert status_of(es, T.ALLERGY, "sulfa drugs") == St.FAMILY_MEMBER


def test_no_known_allergies():
    es = extract(("PATIENT", "No known drug allergies."))
    assert status_of(es, T.ALLERGY, "no known drug allergies") == St.ABSENT


def test_dialogue_inference_yes_no_and_allergy_answer():
    es = extract(("DOCTOR", "Do you have a cough?"), ("PATIENT", "Yes, for a week."),
                 ("DOCTOR", "Any fever?"), ("PATIENT", "No."),
                 ("DOCTOR", "Any allergies?"), ("PATIENT", "Penicillin."))
    assert status_of(es, T.SYMPTOM, "cough") == St.PRESENT
    assert status_of(es, T.SYMPTOM, "fever") == St.ABSENT
    assert status_of(es, T.ALLERGY, "penicillin") == St.PRESENT
    assert (T.MEDICATION, "penicillin") not in es


def test_doctor_questions_are_not_facts():
    es = extract(("DOCTOR", "Have you had any chest pain or vomiting?"))
    assert not es


def test_tag_question_is_still_a_statement():
    es = extract(("DOCTOR", "I'm going to prescribe azithromycin, okay?"))
    assert status_of(es, T.MEDICATION, "azithromycin") == St.PRESCRIBED


def test_vitals_and_units():
    es = extract(("NURSE", "Temperature is 38.5, blood pressure 120 over 80, pulse 88, and oxygen saturation is 97 percent."))
    temp = es[(T.VITAL, "temperature")]
    assert temp.value == "38.5" and temp.unit == "°C"
    assert es[(T.VITAL, "blood pressure")].value == "120/80"
    assert es[(T.VITAL, "heart rate")].value == "88"
    assert es[(T.VITAL, "oxygen saturation")].value == "97"


def test_demographics():
    es = extractor.extract(make_transcript(("PATIENT", "I'm a 45 year old woman."))).items
    values = {(e.normalized, e.value) for e in es}
    assert ("age", "45") in values and ("sex", "female") in values


def test_honorific_gives_low_confidence_sex():
    es = extractor.extract(make_transcript(("DOCTOR", "Good morning, Mr. Davis."))).items
    sex = next(e for e in es if e.normalized == "sex")
    assert sex.value == "male" and sex.attributes["basis"] == "form_of_address" and sex.confidence < 0.8


def test_flu_like_is_not_influenza():
    assert (T.DIAGNOSIS, "influenza") not in extract(("PATIENT", "I have flu-like symptoms."))


def test_longest_match_wins():
    es = extract(("DOCTOR", "The rapid strep test came back positive, so this is strep throat."))
    assert es[(T.PROCEDURE, "rapid strep test")].attributes["result"] == "positive"
    assert status_of(es, T.DIAGNOSIS, "streptococcal pharyngitis") == St.PRESENT


def test_asr_confidence_scales_entity_confidence():
    low = extract(("PATIENT", "I have a cough.", 0.5))[(T.SYMPTOM, "cough")]
    high = extract(("PATIENT", "I have a cough.", 1.0))[(T.SYMPTOM, "cough")]
    assert low.confidence < high.confidence
