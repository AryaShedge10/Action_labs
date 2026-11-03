# Current State of Team B Prototype

> **Historical snapshot (before v1.0).** This file describes the original single-script prototype
> (`legacy/pipeline_demo.py`) and is kept for comparison. For the current system, see
> [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md), section 14 "What changed from the original prototype".

## Existing Architecture
The current prototype is a single monolithic Python script (`legacy/pipeline_demo.py`) that executes a linear 3-stage pipeline:
1. **Stage 1 (Extraction):** Sends the raw transcript to an LLM (Qwen3 8B via Ollama) asking for a JSON response containing symptoms, allergies, medications, demographics, vitals, and orders.
2. **Stage 2 (Deterministic Conflict Check):** Uses a hardcoded Python dictionary (`ALLERGY_CLASS_MAP`) to check for basic medication-allergy conflicts. Also contains a basic demographic check (`FEMALE_ONLY_ORDERS`).
3. **Stage 3 (Constrained SOAP Generation):** Sends the extracted entities and conflict list to the LLM to generate a SOAP note, instructing it not to include blocked medications.
4. **Stage 2B (Post-Generation Scrub):** A final string-matching check that appends a warning banner if any blocked medication is found in the final SOAP text.

## Existing Files
- `legacy/pipeline_demo.py` (Main demo script)
- `myvenv/` (Python virtual environment)

## Current Model
- `qwen3:8b` running locally via Ollama.

## Current Pipeline
Transcript -> `stage1_extract` (LLM) -> `stage2_check_conflicts` (Python code) -> `stage3_generate` (LLM) -> `scrub_output` (String match)

## Known Limitations
- No Pydantic schemas or validation; uses raw dictionaries and `json.loads`.
- No evidence/provenance layer; facts are extracted without linking back to the transcript turn or timestamps.
- Conflict rules are trivial and hardcoded.
- Lacks robust Claim-Level Grounding Verification.
- No testing framework, unit tests, or evaluation datasets (no `pytest`).
- Fails completely if Ollama is not running on localhost:11434 (Connection Refused).
- Prompts are hardcoded string templates without dynamic constraint handling.
- No REST API; only a terminal script execution.
- Doesn't extract temporal data, status of conditions, or negate symptoms properly.

## Current Test Results
- No tests exist in the repository.
- Running the demo script failed with `ConnectionRefusedError: [Errno 61] Connection refused` because the local Ollama server is not running on port 11434. When Ollama is available, it would process the hardcoded `TRANSCRIPT_1` and output the SOAP note to the console.
