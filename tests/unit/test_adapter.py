import pytest

from app.ingestion.adapter import TranscriptFormatError, normalize_speaker, to_transcript
from app.models.schemas import Speaker


def test_team_a_format_is_normalised(team_a_payload):
    t = to_transcript(team_a_payload)
    assert t.source_format == "team_a_v1"
    assert t.consultation_id == "20261031_143022"
    assert [s.segment_id for s in t.segments][:3] == ["S001", "S002", "S003"]
    assert t.segments[1].speaker == Speaker.PATIENT
    assert t.segments[1].start_time == 3.82 and t.segments[1].confidence == 0.96
    assert len(t.sha256) == 64 and all(len(s.sha256) == 64 for s in t.segments)


def test_hashes_are_deterministic_and_content_sensitive(team_a_payload):
    a, b = to_transcript(team_a_payload), to_transcript(team_a_payload)
    assert a.sha256 == b.sha256
    team_a_payload["segments"][0]["text"] += " Extra words."
    assert to_transcript(team_a_payload).sha256 != a.sha256


def test_legacy_turn_format_still_supported():
    t = to_transcript({"consultation_id": "legacy_1", "transcript": [
        {"turn_id": 1, "speaker": "PATIENT", "start_ms": 1500, "end_ms": 4000, "text": "I have a cough."}]})
    assert t.source_format == "legacy_turns"
    assert t.segments[0].start_time == 1.5 and t.segments[0].end_time == 4.0


def test_envelope_is_unwrapped(team_a_payload):
    t = to_transcript({"data": team_a_payload})
    assert len(t.segments) == 7


def test_segments_sorted_and_empty_dropped():
    t = to_transcript({"consultation_id": "c1", "segments": [
        {"speaker": "DOCTOR", "start_time": 5, "end_time": 6, "text": "Second."},
        {"speaker": "PATIENT", "start_time": 0, "end_time": 1, "text": "First."},
        {"speaker": "PATIENT", "start_time": 2, "end_time": 3, "text": "   "},
    ]})
    assert [s.text for s in t.segments] == ["First.", "Second."]


@pytest.mark.parametrize("raw,expected", [
    ("DOCTOR", Speaker.DOCTOR), ("doctor", Speaker.DOCTOR), ("Dr", Speaker.DOCTOR), ("PATIENT", Speaker.PATIENT),
    ("Nurse", Speaker.NURSE), ("SPEAKER_1", Speaker.UNKNOWN), ("", Speaker.UNKNOWN),
])
def test_speaker_normalisation(raw, expected):
    assert normalize_speaker(raw) == expected


@pytest.mark.parametrize("payload", [
    {"consultation_id": "c1", "segments": []},
    {"consultation_id": "c1", "segments": [{"speaker": "DOCTOR", "start_time": 5, "end_time": 1, "text": "x"}]},
    {"consultation_id": "c1", "segments": [{"speaker": "DOCTOR", "start_time": 0, "end_time": 1, "text": "x", "confidence": 1.5}]},
    {"consultation_id": "../etc/passwd", "segments": [{"speaker": "DOCTOR", "start_time": 0, "end_time": 1, "text": "x"}]},
    {"something": "else"},
])
def test_invalid_payloads_rejected(payload):
    with pytest.raises(TranscriptFormatError):
        to_transcript(payload)
