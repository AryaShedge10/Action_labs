"""Runtime configuration, read from environment variables.

Everything has a safe local-first default so the service runs with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache

EXTRACTION_MODES = ("auto", "rule", "hybrid", "llm")
GENERATION_MODES = ("auto", "template", "llm")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _ollama_base_url() -> str:
    # Backwards compatible with the old OLLAMA_URL=http://host:11434/api/generate variable.
    url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_URL") or "http://localhost:11434"
    return url.rstrip("/").removesuffix("/api/generate").removesuffix("/api/chat")


@dataclass(frozen=True)
class Settings:
    # Local LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    llm_timeout_s: float = 180.0
    # Privacy: refuse to send transcripts to a non-local LLM endpoint unless explicitly allowed.
    allow_remote_llm: bool = False
    redact_phi_for_llm: bool = True

    # Pipeline behaviour
    extraction_mode: str = "auto"   # auto = rule-based + verified LLM if reachable
    generation_mode: str = "auto"   # auto = LLM if reachable, else deterministic template
    low_asr_confidence: float = 0.60

    # Team A integration
    team_a_base_url: str = "http://localhost:8000"
    team_a_timeout_s: float = 15.0

    # Persistence of results for Team C to pull
    store_results: bool = True
    store_path: str = "data/store/consultations.db"

    # CORS for Team C's dashboard (browser app on another port)
    cors_origin_regex: str = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_base_url=_ollama_base_url(),
            llm_model=os.getenv("LLM_MODEL") or os.getenv("MODEL") or cls.llm_model,
            llm_timeout_s=_env_float("LLM_TIMEOUT_S", cls.llm_timeout_s),
            allow_remote_llm=_env_bool("ALLOW_REMOTE_LLM", cls.allow_remote_llm),
            redact_phi_for_llm=_env_bool("REDACT_PHI_FOR_LLM", cls.redact_phi_for_llm),
            extraction_mode=_choice("EXTRACTION_MODE", EXTRACTION_MODES, cls.extraction_mode),
            generation_mode=_choice("GENERATION_MODE", GENERATION_MODES, cls.generation_mode),
            low_asr_confidence=_env_float("LOW_ASR_CONFIDENCE", cls.low_asr_confidence),
            team_a_base_url=(os.getenv("TEAM_A_BASE_URL") or cls.team_a_base_url).rstrip("/"),
            team_a_timeout_s=_env_float("TEAM_A_TIMEOUT_S", cls.team_a_timeout_s),
            store_results=_env_bool("STORE_RESULTS", cls.store_results),
            store_path=os.getenv("STORE_PATH") or cls.store_path,
            cors_origin_regex=os.getenv("CORS_ORIGIN_REGEX") or cls.cors_origin_regex,
        )

    def with_overrides(self, **overrides) -> "Settings":
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean) if clean else self


def _choice(name: str, allowed: tuple[str, ...], default: str) -> str:
    value = (os.getenv(name) or default).strip().lower()
    return value if value in allowed else default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
