"""Build data/evaluation/cases.jsonl - the labelled Team B evaluation set (Team A transcript format).

Labels: expected safety_status, expected conflicts (type/status/medication) and expected entities
as [type, key, status] where key is the normalized term ("sex:male" / "age:34" for demographics).
Run:  python scripts/build_eval_cases.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "cases.jsonl"


def seg(turns, conf=0.95):
    out, t = [], 0.0
    for turn in turns:
        speaker, text = turn[0], turn[1]
        c = turn[2] if len(turn) > 2 else conf
        dur = round(max(1.5, len(text.split()) / 2.6), 2)
        out.append({"speaker": speaker, "start_time": round(t, 2), "end_time": round(t + dur, 2), "text": text, "confidence": c})
        t += dur + 0.3
    return out


def case(case_id, scenario, source, turns, status, conflicts=(), entities=()):
    return {
        "case_id": case_id, "scenario": scenario, "source": source,
        "expected": {"safety_status": status,
                     "conflicts": [dict(zip(("type", "status", "medication"), c)) for c in conflicts],
                     "entities": [list(e) for e in entities]},
        "consultation_id": case_id, "segments": seg(turns),
    }


team_a = json.loads((OUT.parents[1] / "samples" / "team_a_20261031_143022.json").read_text())

CASES = [
    {**case("TEAMA-001", "Team A sample transcript (penicillin family avoided)", "team_a", [], "SAFE", entities=[
        ("DEMOGRAPHIC", "sex:male", "PRESENT"), ("SYMPTOM", "sore throat", "PRESENT"), ("SYMPTOM", "fever", "PRESENT"),
        ("SYMPTOM", "difficulty swallowing", "PRESENT"), ("SYMPTOM", "cough", "ABSENT"),
        ("SYMPTOM", "shortness of breath", "ABSENT"), ("SYMPTOM", "rash", "ABSENT"), ("SYMPTOM", "body aches", "PRESENT"),
        ("SYMPTOM", "fatigue", "PRESENT"), ("ALLERGY", "amoxicillin", "PRESENT"), ("ALLERGY", "penicillin", "PRESENT"),
        ("MEDICATION", "penicillin", "ABSENT"), ("PROCEDURE", "throat examination", "RECOMMENDED")]),
     "segments": team_a["segments"]},
    case("EMAIL-001", "Strep throat with penicillin allergy, safe alternative prescribed", "team_b_email", [
        ("DOCTOR", "Good morning. What brings you in today?"),
        ("PATIENT", "I've had a sore throat and fever for about three days now. It's getting worse."),
        ("DOCTOR", "Any other symptoms, cough, body aches?"),
        ("PATIENT", "Some body aches, no cough really."),
        ("DOCTOR", "Any allergies I should know about?"),
        ("PATIENT", "Yes, I'm allergic to penicillin. I broke out in hives last time I took it."),
        ("DOCTOR", "Your temperature is 101.2 degrees. Your throat looks red and your tonsils are swollen. Let's do a rapid strep test."),
        ("DOCTOR", "The rapid strep came back positive. This is strep throat. Since you're allergic to penicillin, I'll prescribe azithromycin 500 mg once daily for five days. You can take acetaminophen as needed for the fever."),
    ], "SAFE", entities=[
        ("SYMPTOM", "sore throat", "PRESENT"), ("SYMPTOM", "fever", "PRESENT"), ("SYMPTOM", "body aches", "PRESENT"),
        ("SYMPTOM", "cough", "ABSENT"), ("ALLERGY", "penicillin", "PRESENT"), ("VITAL", "temperature", "PRESENT"),
        ("FINDING", "pharyngeal erythema", "PRESENT"), ("FINDING", "tonsillar swelling", "PRESENT"),
        ("PROCEDURE", "rapid strep test", "PRESENT"), ("DIAGNOSIS", "streptococcal pharyngitis", "PRESENT"),
        ("MEDICATION", "azithromycin", "PRESCRIBED"), ("MEDICATION", "acetaminophen", "RECOMMENDED")]),
    case("ADV-001", "Penicillin allergy -> amoxicillin prescribed", "legacy_adversarial", [
        ("PATIENT", "I am allergic to penicillin."), ("DOCTOR", "I will prescribe amoxicillin.")],
        "CONFLICT_DETECTED", [("MEDICATION_ALLERGY", "BLOCKED", "amoxicillin")],
        [("ALLERGY", "penicillin", "PRESENT"), ("MEDICATION", "amoxicillin", "PRESCRIBED")]),
    case("ADV-002", "Negated symptoms", "legacy_adversarial", [("PATIENT", "I do not have a fever or chest pain.")],
         "SAFE", entities=[("SYMPTOM", "fever", "ABSENT"), ("SYMPTOM", "chest pain", "ABSENT")]),
    case("ADV-003", "Historical medication", "legacy_adversarial",
         [("PATIENT", "I used to take metformin last year, but I stopped.")], "SAFE",
         entities=[("MEDICATION", "metformin", "HISTORICAL")]),
    case("ADV-004", "Family-member medication", "legacy_adversarial",
         [("PATIENT", "My father takes lisinopril for his heart.")], "SAFE",
         entities=[("MEDICATION", "lisinopril", "FAMILY_MEMBER")]),
    case("ADV-005", "Symptom only - no diagnosis may be invented", "legacy_adversarial",
         [("PATIENT", "My throat hurts.")], "SAFE", entities=[("SYMPTOM", "sore throat", "PRESENT")]),
    case("ADV-006", "Prescription without dose - no dose may be invented", "legacy_adversarial",
         [("DOCTOR", "I'll start you on ibuprofen.")], "SAFE", entities=[("MEDICATION", "ibuprofen", "PRESCRIBED")]),
    case("ADV-007", "Conflicting blood pressure readings", "legacy_adversarial", [
        ("NURSE", "Blood pressure is 120 over 80."), ("DOCTOR", "Oh wait, this other machine says BP is 180 over 100.")],
        "REVIEW_REQUIRED", [("VITAL_SIGN_ALERT", "REVIEW_REQUIRED", None), ("VITAL_SIGN_ALERT", "REVIEW_REQUIRED", None)],
        [("VITAL", "blood pressure", "PRESENT")]),
    case("ADV-008", "Resolved symptom", "legacy_adversarial",
         [("PATIENT", "I had a headache yesterday but it's gone now.")], "SAFE",
         entities=[("SYMPTOM", "headache", "HISTORICAL")]),
    case("ADV-009", "Third-party instruction", "legacy_adversarial",
         [("PATIENT", "The nurse told me to take aspirin.")], "SAFE", entities=[("MEDICATION", "aspirin", "RECOMMENDED")]),
    case("ADV-010", "Single symptom", "legacy_adversarial", [("PATIENT", "I have a rash.")], "SAFE",
         entities=[("SYMPTOM", "rash", "PRESENT")]),
    case("GOLD-001", "Simple symptom with duration", "legacy_gold",
         [("PATIENT", "I have had a sore throat for three days.")], "SAFE", entities=[("SYMPTOM", "sore throat", "PRESENT")]),
    case("GOLD-002", "No clinical content", "legacy_gold",
         [("PATIENT", "I feel fine today, just here for a checkup.")], "SAFE"),
    case("DEMO-001", "Stated male + pregnancy", "authored", [
        ("PATIENT", "I'm a 34 year old man."), ("DOCTOR", "Since you're pregnant, we will avoid ibuprofen.")],
        "CONFLICT_DETECTED", [("DEMOGRAPHIC_CONFLICT", "BLOCKED", None)],
        [("DEMOGRAPHIC", "age:34", "PRESENT"), ("DEMOGRAPHIC", "sex:male", "PRESENT"),
         ("DIAGNOSIS", "pregnancy", "PRESENT"), ("MEDICATION", "ibuprofen", "ABSENT")]),
    case("DEMO-002", "Male by form of address + positive pregnancy test", "authored", [
        ("DOCTOR", "Good morning Mr. Patel."), ("DOCTOR", "The pregnancy test is positive, so you are pregnant.")],
        "REVIEW_REQUIRED", [("DEMOGRAPHIC_CONFLICT", "REVIEW_REQUIRED", None), ("DEMOGRAPHIC_CONFLICT", "REVIEW_REQUIRED", None)],
        [("DEMOGRAPHIC", "sex:male", "PRESENT"), ("PROCEDURE", "pregnancy test", "PRESENT"),
         ("DIAGNOSIS", "pregnancy", "PRESENT")]),
    case("DEMO-003", "Female patient + PSA test", "authored", [
        ("PATIENT", "I'm a 40 year old woman with some fatigue."), ("DOCTOR", "Let's order a PSA test and a blood count.")],
        "REVIEW_REQUIRED", [("DEMOGRAPHIC_CONFLICT", "REVIEW_REQUIRED", None)],
        [("DEMOGRAPHIC", "age:40", "PRESENT"), ("DEMOGRAPHIC", "sex:female", "PRESENT"), ("SYMPTOM", "fatigue", "PRESENT"),
         ("PROCEDURE", "psa test", "RECOMMENDED"), ("PROCEDURE", "complete blood count", "RECOMMENDED")]),
    case("PREG-001", "Pregnant patient + ACE inhibitor", "authored", [
        ("PATIENT", "I'm 28 and I'm 12 weeks pregnant."),
        ("DOCTOR", "Your blood pressure is 150 over 95. I'll start you on lisinopril 10 mg daily.")],
        "CONFLICT_DETECTED", [("PREGNANCY_CONTRAINDICATION", "BLOCKED", "lisinopril")],
        [("DEMOGRAPHIC", "age:28", "PRESENT"), ("DIAGNOSIS", "pregnancy", "PRESENT"), ("VITAL", "blood pressure", "PRESENT"),
         ("MEDICATION", "lisinopril", "PRESCRIBED")]),
    case("DDI-001", "Warfarin + NSAID", "authored", [
        ("PATIENT", "I take warfarin for my afib."), ("DOCTOR", "Take ibuprofen 400 mg every 6 hours as needed.")],
        "REVIEW_REQUIRED", [("DRUG_INTERACTION", "REVIEW_REQUIRED", "ibuprofen")],
        [("MEDICATION", "warfarin", "CURRENT"), ("DIAGNOSIS", "atrial fibrillation", "PRESENT"),
         ("MEDICATION", "ibuprofen", "PRESCRIBED")]),
    case("DDI-002", "Nitrate + PDE5 inhibitor (contraindicated)", "authored", [
        ("PATIENT", "I'm on nitroglycerin for angina."), ("DOCTOR", "I'll prescribe sildenafil 50 mg as needed.")],
        "CONFLICT_DETECTED", [("DRUG_INTERACTION", "BLOCKED", "sildenafil")],
        [("MEDICATION", "nitroglycerin", "CURRENT"), ("DIAGNOSIS", "angina", "PRESENT"),
         ("MEDICATION", "sildenafil", "PRESCRIBED")]),
    case("DDI-003", "ACE inhibitor + potassium-sparing diuretic", "authored", [
        ("PATIENT", "I take lisinopril 20 mg daily for high blood pressure."),
        ("DOCTOR", "I'll add spironolactone 25 mg daily.")],
        "REVIEW_REQUIRED", [("DRUG_INTERACTION", "REVIEW_REQUIRED", "spironolactone")],
        [("MEDICATION", "lisinopril", "CURRENT"), ("DIAGNOSIS", "hypertension", "PRESENT"),
         ("MEDICATION", "spironolactone", "PRESCRIBED")]),
    case("XREACT-001", "Penicillin allergy + cephalosporin", "authored", [
        ("PATIENT", "I'm allergic to penicillin, it gives me a rash."),
        ("DOCTOR", "I'll prescribe cephalexin 500 mg four times a day for 7 days.")],
        "REVIEW_REQUIRED", [("ALLERGY_CROSS_REACTIVITY", "REVIEW_REQUIRED", "cephalexin")],
        [("ALLERGY", "penicillin", "PRESENT"), ("MEDICATION", "cephalexin", "PRESCRIBED")]),
    case("SULFA-001", "Sulfa allergy from Q&A + Bactrim", "authored", [
        ("PATIENT", "It burns when I pee."), ("DOCTOR", "Any allergies?"),
        ("PATIENT", "Sulfa drugs, they give me a rash."), ("DOCTOR", "You have a UTI. I'll prescribe Bactrim twice daily for three days.")],
        "CONFLICT_DETECTED", [("MEDICATION_ALLERGY", "BLOCKED", "sulfamethoxazole-trimethoprim")],
        [("SYMPTOM", "painful urination", "PRESENT"), ("ALLERGY", "sulfa drugs", "PRESENT"),
         ("DIAGNOSIS", "urinary tract infection", "PRESENT"), ("MEDICATION", "sulfamethoxazole-trimethoprim", "PRESCRIBED")]),
    case("NSAID-001", "Aspirin allergy + naproxen on low-confidence ASR", "authored", [
        ("PATIENT", "I am allergic to aspirin.", 0.48), ("DOCTOR", "Take naproxen 500 mg twice a day.")],
        "CONFLICT_DETECTED", [("MEDICATION_ALLERGY", "BLOCKED", "naproxen"), ("TRANSCRIPTION_QUALITY", "REVIEW_REQUIRED", None)],
        [("ALLERGY", "aspirin", "PRESENT"), ("MEDICATION", "naproxen", "PRESCRIBED")]),
    case("NKDA-001", "No known allergies + NSAID", "authored", [
        ("PATIENT", "I've had lower back pain for a week."), ("DOCTOR", "Do you have any allergies?"),
        ("PATIENT", "No known drug allergies."), ("DOCTOR", "Take ibuprofen 400 mg every 8 hours as needed.")],
        "SAFE", entities=[("SYMPTOM", "back pain", "PRESENT"), ("ALLERGY", "no known drug allergies", "ABSENT"),
                          ("MEDICATION", "ibuprofen", "PRESCRIBED")]),
    case("PEANUT-001", "Food allergy is not a drug conflict", "authored", [
        ("PATIENT", "I'm allergic to peanuts and I have a headache."), ("DOCTOR", "You can take ibuprofen for the headache.")],
        "SAFE", entities=[("ALLERGY", "peanuts", "PRESENT"), ("SYMPTOM", "headache", "PRESENT"),
                          ("MEDICATION", "ibuprofen", "RECOMMENDED")]),
    case("PEDS-001", "Aspirin for a child", "authored", [
        ("DOCTOR", "She is a 6-year-old girl with a fever of 102."), ("DOCTOR", "Give her aspirin for the fever.")],
        "REVIEW_REQUIRED", [("AGE_CONTRAINDICATION", "REVIEW_REQUIRED", "aspirin")],
        [("DEMOGRAPHIC", "age:6", "PRESENT"), ("DEMOGRAPHIC", "sex:female", "PRESENT"), ("SYMPTOM", "fever", "PRESENT"),
         ("VITAL", "temperature", "PRESENT"), ("MEDICATION", "aspirin", "RECOMMENDED")]),
    case("VITAL-001", "Implausible temperature (ASR error) and low SpO2", "authored", [
        ("NURSE", "Temperature is 201 degrees and oxygen saturation is 89 percent.")],
        "REVIEW_REQUIRED", [("VITAL_SIGN_ALERT", "REVIEW_REQUIRED", None), ("VITAL_SIGN_ALERT", "REVIEW_REQUIRED", None)],
        [("VITAL", "temperature", "PRESENT"), ("VITAL", "oxygen saturation", "PRESENT")]),
    case("DUP-001", "Duplicate NSAIDs", "authored", [
        ("PATIENT", "I take ibuprofen every day for my knees."), ("DOCTOR", "I'll also prescribe naproxen 500 mg twice daily.")],
        "REVIEW_REQUIRED", [("DUPLICATE_THERAPY", "REVIEW_REQUIRED", None)],
        [("MEDICATION", "ibuprofen", "CURRENT"), ("MEDICATION", "naproxen", "PRESCRIBED")]),
    case("DOSE-001", "Ten-fold amoxicillin dose (ASR error)", "authored", [
        ("PATIENT", "My ear hurts."), ("DOCTOR", "You have an ear infection. I'm prescribing amoxicillin 5000 mg three times a day for 10 days.")],
        "REVIEW_REQUIRED", [("DOSE_RANGE", "REVIEW_REQUIRED", "amoxicillin")],
        [("SYMPTOM", "ear pain", "PRESENT"), ("DIAGNOSIS", "otitis media", "PRESENT"), ("MEDICATION", "amoxicillin", "PRESCRIBED")]),
    case("INCONS-001", "Denies allergies then reports one", "authored", [
        ("PATIENT", "I don't have any allergies."), ("PATIENT", "Oh wait, I'm actually allergic to penicillin.")],
        "REVIEW_REQUIRED", [("INCONSISTENT_HISTORY", "REVIEW_REQUIRED", None)],
        [("ALLERGY", "no known drug allergies", "ABSENT"), ("ALLERGY", "penicillin", "PRESENT")]),
    case("AVOID-001", "Allergic drug explicitly replaced", "authored", [
        ("PATIENT", "I'm allergic to penicillin."),
        ("DOCTOR", "Instead of amoxicillin, I'll prescribe azithromycin 500 mg once daily for three days.")],
        "SAFE", entities=[("ALLERGY", "penicillin", "PRESENT"), ("MEDICATION", "amoxicillin", "ABSENT"),
                          ("MEDICATION", "azithromycin", "PRESCRIBED")]),
    case("SPEAKER-001", "Undiarized speakers", "authored", [
        ("SPEAKER_1", "I've been having chest pain since this morning."), ("SPEAKER_2", "Let's get an ECG right away.")],
        "REVIEW_REQUIRED", [("TRANSCRIPTION_QUALITY", "REVIEW_REQUIRED", None), ("TRANSCRIPTION_QUALITY", "REVIEW_REQUIRED", None)],
        [("SYMPTOM", "chest pain", "PRESENT"), ("PROCEDURE", "electrocardiogram", "RECOMMENDED")]),
    case("QA-001", "Yes/no answers to symptom questions", "authored", [
        ("DOCTOR", "Do you have a cough?"), ("PATIENT", "Yes, for about a week."),
        ("DOCTOR", "Any fever?"), ("PATIENT", "No.")],
        "SAFE", entities=[("SYMPTOM", "cough", "PRESENT"), ("SYMPTOM", "fever", "ABSENT")]),
]

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for c in CASES:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(CASES)} cases to {OUT}")
