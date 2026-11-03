"""Team 2B REST API.

Inbound (Team A):  POST /v1/analyze                       - push a transcript (Team A JSON format)
                   POST /v1/team-a/consultations/{id}/analyze - pull from Team A's API, then analyse
Outbound (Team C): GET  /v1/consultations/{id}            - structured SOAP + conflicts + grounding
                   GET  /v1/consultations                 - recent results
"""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app import PIPELINE_VERSION, RULE_VERSION, SCHEMA_VERSION
from app.config import Settings, get_settings
from app.ingestion.adapter import TranscriptFormatError
from app.ingestion.team_a_client import TeamAClient, TeamAError, TeamANotFoundError
from app.models.contracts import CONSULTATION_ID_PATTERN, LegacyTranscriptIn, TeamATranscriptIn
from app.models.schemas import ConsultationAnalysis
from app.pipeline import ClinicalPipeline
from app.storage.repository import ConsultationStore
from app.utils.llm import LLMError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("teamb.api")

ExtractionMode = Literal["auto", "rule", "hybrid", "llm"]
GenerationMode = Literal["auto", "template", "llm"]


def create_app(settings: Settings | None = None, pipeline: ClinicalPipeline | None = None,
               team_a_client: TeamAClient | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.pipeline = pipeline or ClinicalPipeline(settings)
        app.state.store = ConsultationStore(settings.store_path) if settings.store_results else None
        app.state.team_a = team_a_client or TeamAClient(settings.team_a_base_url, settings.team_a_timeout_s)
        yield

    app = FastAPI(
        title="Team 2B - Clinical NLP & Safety Logic",
        description="Privacy-preserving clinical scribe: transcript -> verified entities -> deterministic safety "
                    "checks -> grounded SOAP note with claim-level citations.",
        version=PIPELINE_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origin_regex=settings.cors_origin_regex,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def run(request: Request, payload, extraction_mode, generation_mode) -> ConsultationAnalysis:
        try:
            result = request.app.state.pipeline.analyze(payload, extraction_mode, generation_mode)
        except TranscriptFormatError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            error_id = uuid.uuid4().hex[:12]
            # Do not log the exception message: it may contain transcript text (PHI).
            logger.error("Pipeline failure error_id=%s type=%s", error_id, type(exc).__name__)
            raise HTTPException(status_code=500, detail=f"Pipeline error (error_id={error_id})") from exc
        store = request.app.state.store
        if store is not None:
            store.save(result)
        return result

    @app.get("/v1/health", tags=["ops"])
    def health(request: Request):
        state = request.app.state
        return {
            "status": "ok",
            "llm": state.pipeline.llm_status(),
            "store_enabled": state.store is not None,
            "team_a_base_url": state.settings.team_a_base_url,
        }

    @app.get("/v1/version", tags=["ops"])
    def version(request: Request):
        return {"pipeline_version": PIPELINE_VERSION, "rule_version": RULE_VERSION,
                "schema_version": SCHEMA_VERSION, "model": request.app.state.settings.llm_model}

    @app.post("/v1/analyze", response_model=ConsultationAnalysis, tags=["pipeline"])
    def analyze(
        request: Request,
        payload: TeamATranscriptIn | LegacyTranscriptIn = Body(...),
        extraction_mode: ExtractionMode | None = Query(None, description="Override EXTRACTION_MODE"),
        generation_mode: GenerationMode | None = Query(None, description="Override GENERATION_MODE"),
    ) -> ConsultationAnalysis:
        """Analyse a transcript pushed in Team A's JSON format."""
        return run(request, payload, extraction_mode, generation_mode)

    @app.post("/v1/team-a/consultations/{consultation_id}/analyze", response_model=ConsultationAnalysis,
              tags=["pipeline"])
    def analyze_from_team_a(
        request: Request,
        consultation_id: str,
        extraction_mode: ExtractionMode | None = Query(None),
        generation_mode: GenerationMode | None = Query(None),
    ) -> ConsultationAnalysis:
        """Fetch the transcript from Team A (GET /api/v1/live/team-b/consultation/{id}) and analyse it."""
        _check_id(consultation_id)
        try:
            raw = request.app.state.team_a.fetch_transcript(consultation_id)
        except TeamANotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except TeamAError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        raw.setdefault("consultation_id", consultation_id)
        return run(request, raw, extraction_mode, generation_mode)

    @app.get("/v1/consultations/{consultation_id}", response_model=ConsultationAnalysis, tags=["team-c"])
    def get_consultation(request: Request, consultation_id: str) -> ConsultationAnalysis:
        """Latest analysis for a consultation (consumed by Team C's dashboard)."""
        _check_id(consultation_id)
        store = request.app.state.store
        result = store.get(consultation_id) if store else None
        if result is None:
            raise HTTPException(status_code=404, detail=f"No analysis stored for consultation '{consultation_id}'")
        return result

    @app.get("/v1/consultations", tags=["team-c"])
    def list_consultations(request: Request, limit: int = Query(50, ge=1, le=500)):
        store = request.app.state.store
        return {"items": store.list(limit) if store else []}

    return app


def _check_id(consultation_id: str) -> None:
    if not re.match(CONSULTATION_ID_PATTERN, consultation_id):
        raise HTTPException(status_code=422, detail="Invalid consultation_id")


app = create_app()
