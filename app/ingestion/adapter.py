"""Normalise any supported inbound transcript into the internal `Transcript` model."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import ValidationError

from app.models.contracts import LegacyTranscriptIn, TeamATranscriptIn
from app.models.schemas import Segment, Speaker, Transcript

_SPEAKER_ALIASES = {
    Speaker.DOCTOR: {"DOCTOR", "DR", "PHYSICIAN", "CLINICIAN", "PROVIDER", "PRACTITIONER"},
    Speaker.NURSE: {"NURSE", "RN", "MEDICAL_ASSISTANT", "MA"},
    Speaker.PATIENT: {"PATIENT", "PT"},
}


class TranscriptFormatError(ValueError):
    """Raised when a payload is not a recognised transcript format."""


def normalize_speaker(raw: str) -> Speaker:
    key = re.sub(r"[\s\-]+", "_", (raw or "").strip().upper()).rstrip(".:")
    for speaker, aliases in _SPEAKER_ALIASES.items():
        if key in aliases:
            return speaker
    # Generic diarization labels (SPEAKER_1, SPEAKER_00, ...) cannot be trusted for role-based logic.
    return Speaker.UNKNOWN


def _clean_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def _segment_hash(consultation_id: str, index: int, speaker: Speaker, start: float, end: float, text: str) -> str:
    payload = f"{consultation_id}|{index}|{speaker.value}|{start:.3f}|{end:.3f}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build(consultation_id: str, language: str, source_format: str,
           rows: list[tuple[str, float, float, str, float | None]]) -> Transcript:
    rows = [(spk, start, end, _clean_text(text), conf) for spk, start, end, text, conf in rows]
    rows = [r for r in rows if r[3]]
    if not rows:
        raise TranscriptFormatError("Transcript contains no non-empty segments")
    rows.sort(key=lambda r: (r[1], r[2]))

    segments: list[Segment] = []
    for index, (raw_speaker, start, end, text, conf) in enumerate(rows):
        speaker = normalize_speaker(raw_speaker)
        segments.append(Segment(
            segment_id=f"S{index + 1:03d}",
            index=index,
            speaker=speaker,
            start_time=round(start, 3),
            end_time=round(end, 3),
            text=text,
            confidence=conf,
            sha256=_segment_hash(consultation_id, index, speaker, start, end, text),
        ))
    transcript_hash = hashlib.sha256("".join(s.sha256 for s in segments).encode()).hexdigest()
    return Transcript(
        consultation_id=consultation_id,
        language=language,
        source_format=source_format,
        segments=segments,
        sha256=transcript_hash,
    )


def from_team_a(payload: TeamATranscriptIn) -> Transcript:
    return _build(
        payload.consultation_id, payload.language, "team_a_v1",
        [(s.speaker, s.start_time, s.end_time, s.text, s.confidence) for s in payload.segments],
    )


def from_legacy(payload: LegacyTranscriptIn) -> Transcript:
    return _build(
        payload.consultation_id, payload.language, "legacy_turns",
        [(t.speaker, t.start_ms / 1000, t.end_ms / 1000, t.text, None) for t in payload.transcript],
    )


def to_transcript(payload: TeamATranscriptIn | LegacyTranscriptIn | dict[str, Any]) -> Transcript:
    """Accept a validated contract model or a raw dict (e.g. straight from Team A's API)."""
    if isinstance(payload, TeamATranscriptIn):
        return from_team_a(payload)
    if isinstance(payload, LegacyTranscriptIn):
        return from_legacy(payload)
    if not isinstance(payload, dict):
        raise TranscriptFormatError("Transcript payload must be a JSON object")

    payload = _unwrap(payload)
    try:
        if "segments" in payload:
            return from_team_a(TeamATranscriptIn.model_validate(payload))
        if "transcript" in payload:
            return from_legacy(LegacyTranscriptIn.model_validate(payload))
    except ValidationError as exc:
        raise TranscriptFormatError(f"Invalid transcript payload: {exc.errors(include_url=False)}") from exc
    raise TranscriptFormatError("Unrecognised transcript format: expected a 'segments' list (Team A format)")


def _unwrap(payload: dict[str, Any]) -> dict[str, Any]:
    """Tolerate a response envelope such as {"data": {...}} around Team A's transcript."""
    for key in ("data", "result", "transcript"):
        inner = payload.get(key)
        if isinstance(inner, dict) and "segments" in inner:
            if "consultation_id" not in inner and "consultation_id" in payload:
                inner = {**inner, "consultation_id": payload["consultation_id"]}
            return inner
    return payload
