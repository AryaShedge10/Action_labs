"""Inbound payload contracts.

`TeamATranscriptIn` is the format agreed with Team 2A (email thread, Sep–Oct 2026).
`LegacyTranscriptIn` is the earlier turn-based format, kept so old fixtures keep working.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONSULTATION_ID_PATTERN = r"^[A-Za-z0-9_.:\-]{1,128}$"


class TeamASegmentIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: str = Field(description="DOCTOR or PATIENT (other labels are mapped to UNKNOWN)")
    start_time: float = Field(ge=0, description="Seconds from start of recording")
    end_time: float = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1, description="ASR confidence 0..1")

    @model_validator(mode="after")
    def _check_times(self) -> "TeamASegmentIn":
        if self.end_time < self.start_time:
            raise ValueError(f"end_time ({self.end_time}) is before start_time ({self.start_time})")
        return self


class TeamATranscriptIn(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "consultation_id": "20261031_143022",
                "segments": [
                    {"speaker": "DOCTOR", "start_time": 0.0, "end_time": 3.45,
                     "text": "Good afternoon. What brings you in today?", "confidence": 0.98},
                    {"speaker": "PATIENT", "start_time": 3.82, "end_time": 9.15,
                     "text": "I've had a sore throat and fever for three days.", "confidence": 0.96},
                ],
            }
        },
    )

    consultation_id: str = Field(pattern=CONSULTATION_ID_PATTERN)
    language: str = "en-US"
    segments: list[TeamASegmentIn] = Field(min_length=1)


class LegacyTurnIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    turn_id: int
    speaker: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str


class LegacyTranscriptIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    consultation_id: str = Field(pattern=CONSULTATION_ID_PATTERN)
    language: str = "en-US"
    transcript: list[LegacyTurnIn] = Field(min_length=1)
