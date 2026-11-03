import json
from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.adapter import to_transcript
from app.pipeline import ClinicalPipeline

ROOT = Path(__file__).resolve().parents[1]
TEAM_A_SAMPLE = ROOT / "data" / "samples" / "team_a_20261031_143022.json"


def make_payload(*turns, consultation_id="test_001"):
    """turns: (speaker, text) or (speaker, text, asr_confidence)."""
    segments, t = [], 0.0
    for turn in turns:
        speaker, text = turn[0], turn[1]
        conf = turn[2] if len(turn) > 2 else 0.95
        segments.append({"speaker": speaker, "start_time": t, "end_time": t + 3.0, "text": text, "confidence": conf})
        t += 3.5
    return {"consultation_id": consultation_id, "segments": segments}


def make_transcript(*turns):
    return to_transcript(make_payload(*turns))


class FakeLLMClient:
    """Stands in for OllamaClient: returns queued JSON responses, never touches the network."""

    model = "fake-llm"
    is_local = True

    def __init__(self, *responses, available=True):
        self.responses = list(responses)
        self.available = available
        self.prompts = []

    def is_available(self, cache_s=30.0):
        return self.available

    def generate_json(self, prompt, temperature=0.0):
        self.prompts.append(prompt)
        return self.responses.pop(0)


@pytest.fixture(scope="session")
def pipeline():
    return ClinicalPipeline(Settings(store_results=False), use_llm=False)


@pytest.fixture
def team_a_payload():
    return json.loads(TEAM_A_SAMPLE.read_text())
