# Team 2B — Clinical NLP & Safety Logic

Part of the **Privacy-Preserving Clinical Scribe and Diagnostic Copilot**. This service turns Team 2A's speaker-labelled transcripts into a **safety-checked, claim-grounded SOAP note** and **draft prescriptions** for Team 2C's physician dashboard. It runs fully offline.

```
Team A transcript → verified entities → deterministic safety rules → SOAP + prescriptions → claim-level grounding → Team C JSON
```

- **Deterministic first.** Rule-based extraction and safety logic decide everything safety-relevant. A local LLM (Qwen3-8B via Ollama) is optional and every LLM output is verified against the transcript.
- **11 fail-closed safety rules:**
  - allergies (including drug classes) and cross-reactivity;
  - drug interactions and duplicate therapy;
  - dose range;
  - sex, pregnancy and age contraindications;
  - vital signs;
  - inconsistent history;
  - transcription quality.
- **Every claim cites transcript segments**, with audio timestamps and SHA-256 hashes. Hallucinated terms, negation flips, invented numbers and blocked items are flagged.
- **Privacy:** local LLM only by default, PHI pseudonymised before LLM calls, and no PHI in logs.

📖 **Full documentation (architecture, every feature, API contracts, rule catalogue, evaluation):** [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)

## Quick start

```bash
python -m venv myvenv && source myvenv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.main:app --port 8001          # API → http://localhost:8001/docs
streamlit run streamlit_app.py            # developer console → http://localhost:8501
pytest                                    # 117 tests, offline, <1 s
python scripts/evaluate.py                # writes reports/evaluation_report.md
```

Optional local LLM: `ollama pull qwen3:8b && ollama serve`. Without it the pipeline runs in deterministic mode.

Docker: `docker compose up --build` (API on port 8001; reaches Ollama and Team A on the host).

## API at a glance

| Endpoint | Used by | Purpose |
|---|---|---|
| `POST /v1/analyze` | Team A | Analyse a transcript in Team A's JSON format |
| `POST /v1/team-a/consultations/{id}/analyze` | Team B | Pull from Team A's `GET /api/v1/live/team-b/consultation/{id}` and analyse |
| `GET /v1/consultations/{id}` | Team C | Structured note, conflicts, prescriptions, grounding and confidence |
| `GET /v1/consultations` | Team C | Recent results |
| `GET /v1/health`, `GET /v1/version` | Ops | Status and versions |

Example responses: [`docs/samples/`](docs/samples/).

> ⚠️ Academic prototype — not for clinical use. Generated notes are drafts that require physician sign-off.
