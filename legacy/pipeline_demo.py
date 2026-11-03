"""
Team B — Clinical NLP & Safety Logic
Live-demo pipeline: Transcript -> Stage 1 (extraction) -> Stage 2 (deterministic
conflict check) -> Stage 3 (constrained SOAP generation)

Requires a local Ollama server running qwen3:8b:
    ollama pull qwen3:8b
    ollama serve          (usually already running as a background service)

Run:
    pip install requests
    python pipeline_demo.py
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """Single call to the local model. Low temperature = more deterministic output,
    which matters when we're about to parse the result as JSON."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# ---------------------------------------------------------------------------
# STAGE 1 — Grounded entity extraction
# ---------------------------------------------------------------------------
STAGE1_PROMPT_TEMPLATE = """You are a clinical entity extractor. Extract ONLY facts
explicitly stated in the transcript below. Do not infer, summarize, or add anything
that isn't directly said.

Return ONLY valid JSON, no markdown, no commentary, in exactly this shape:
{{
  "symptoms": [string],
  "allergies": [string],
  "medications": [string],
  "demographics": {{"age": string or null, "sex": string or null}},
  "vitals": [string],
  "orders": [string]
}}

Transcript:
{transcript}

JSON:"""


def stage1_extract(transcript: str) -> dict:
    raw = call_llm(STAGE1_PROMPT_TEMPLATE.format(transcript=transcript))
    # Models sometimes wrap JSON in ```json fences even when told not to — strip defensively.
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Stage 1 did not return valid JSON:\n{raw}") from e


# ---------------------------------------------------------------------------
# STAGE 2 — Deterministic conflict check (plain code, NOT an LLM call)
# ---------------------------------------------------------------------------
# Demo-scope lookup table. Production version swaps this for RxNorm / openFDA.
ALLERGY_CLASS_MAP = {
    "penicillin": ["amoxicillin", "ampicillin", "penicillin", "augmentin"],
    "sulfa": ["sulfamethoxazole", "bactrim", "sulfasalazine"],
    "nsaid": ["ibuprofen", "naproxen", "aspirin"],
}
FEMALE_ONLY_ORDERS = ["pelvic ultrasound", "hcg test", "pap smear"]


def normalize(text: str) -> str:
    return text.lower().strip()


def stage2_check_conflicts(entities: dict) -> list[dict]:
    """Returns a list of conflict records. Each blocked medication is also marked
    so Stage 3 knows never to mention it."""
    conflicts = []
    blocked_medications = set()

    # Allergy vs. medication
    for allergy in entities.get("allergies", []):
        allergy_key = normalize(allergy)
        for known_allergen, drug_list in ALLERGY_CLASS_MAP.items():
            if known_allergen in allergy_key:
                for med in entities.get("medications", []):
                    if normalize(med) in drug_list:
                        conflicts.append({
                            "type": "allergy_conflict",
                            "detail": f"{med} conflicts with stated {allergy} allergy",
                        })
                        blocked_medications.add(normalize(med))

    # Demographics vs. orders
    sex = normalize(entities.get("demographics", {}).get("sex") or "")
    if sex == "male":
        for order in entities.get("orders", []):
            if normalize(order) in FEMALE_ONLY_ORDERS:
                conflicts.append({
                    "type": "demographic_conflict",
                    "detail": f"{order} ordered but patient is male",
                })

    return conflicts, blocked_medications


# ---------------------------------------------------------------------------
# STAGE 3 — Constrained SOAP generation (only sees the VERIFIED entity list)
# ---------------------------------------------------------------------------
STAGE3_PROMPT_TEMPLATE = """You are a clinical scribe. Using ONLY the verified facts
and conflicts below, write a SOAP note (Subjective, Objective, Assessment, Plan).

Hard rules:
- Do not include any medication listed under "blocked_medications" anywhere in the note.
- Do not suggest, recommend, or reference any alternative medication. That is a
  physician decision, not yours.
- If a conflict is listed, the Plan must say "CONFLICT: <detail> — flagged for
  physician review" and must NOT include a prescription for that item.
- Do not add any fact not present in the verified data below.

Verified facts (JSON):
{entities}

Conflicts found (JSON):
{conflicts}

Blocked medications (must not appear anywhere in your output):
{blocked}

SOAP Note:"""


def stage3_generate(entities: dict, conflicts: list, blocked_medications: set) -> str:
    prompt = STAGE3_PROMPT_TEMPLATE.format(
        entities=json.dumps(entities, indent=2),
        conflicts=json.dumps(conflicts, indent=2),
        blocked=json.dumps(sorted(blocked_medications)),
    )
    return call_llm(prompt, temperature=0.2)


# ---------------------------------------------------------------------------
# STAGE 2B — Post-generation scrub (safety net in case Stage 3 ignores the rules)
# ---------------------------------------------------------------------------
def scrub_output(soap_text: str, blocked_medications: set) -> tuple[str, bool]:
    """Returns (possibly-flagged text, was_violation_found)."""
    lowered = soap_text.lower()
    for drug in blocked_medications:
        if drug in lowered:
            banner = (
                f"\n\n*** SAFETY GATE: blocked medication '{drug}' was found in "
                f"generated text and would have been removed/rejected before "
                f"reaching a physician. ***"
            )
            return soap_text + banner, True
    return soap_text, False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(transcript: str) -> None:
    print("=" * 70)
    print("STAGE 1 — Extraction")
    print("=" * 70)
    entities = stage1_extract(transcript)
    print(json.dumps(entities, indent=2))

    print("\n" + "=" * 70)
    print("STAGE 2 — Deterministic Conflict Check")
    print("=" * 70)
    conflicts, blocked = stage2_check_conflicts(entities)
    print(json.dumps(conflicts, indent=2) if conflicts else "No conflicts found.")
    print(f"Blocked medications: {sorted(blocked) if blocked else 'none'}")

    print("\n" + "=" * 70)
    print("STAGE 3 — Constrained SOAP Generation")
    print("=" * 70)
    soap = stage3_generate(entities, conflicts, blocked)
    final_text, violation = scrub_output(soap, blocked)
    print(final_text)
    if violation:
        print("\n[!] Stage 3 violated the constraint — post-gen scrub caught it.")
    else:
        print("\n[OK] No blocked medication reached the final note.")


# ---------------------------------------------------------------------------
# Demo run — Transcript 1 (allergy conflict)
# ---------------------------------------------------------------------------
TRANSCRIPT_1 = """
Doctor: Good morning. What brings you in today?
Patient: I've had a sore throat and fever for about three days now. It's getting worse.
Doctor: Any other symptoms — cough, body aches?
Patient: Some body aches, no cough really.
Doctor: Okay. Any allergies I should know about?
Patient: Yes, I'm allergic to penicillin. I broke out in hives last time I took it.
Doctor: Noted. Let's check your throat... it does look inflamed, looks like strep. Temperature is 101.2F.
Doctor: I'm going to prescribe you amoxicillin, 500mg, three times a day for 10 days.
"""

if __name__ == "__main__":
    run_pipeline(TRANSCRIPT_1)