import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.team_a_client import TeamAError, TeamANotFoundError
from app.main import create_app
from app.pipeline import ClinicalPipeline


class FakeTeamA:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error
        self.requested = []

    def fetch_transcript(self, consultation_id):
        self.requested.append(consultation_id)
        if self.error:
            raise self.error
        return dict(self.payload)


@pytest.fixture
def make_client(tmp_path):
    def _make(team_a=None):
        settings = Settings(store_path=str(tmp_path / "test.db"), extraction_mode="rule", generation_mode="template")
        app = create_app(settings, pipeline=ClinicalPipeline(settings, use_llm=False), team_a_client=team_a or FakeTeamA())
        return TestClient(app)
    return _make


def test_health_and_version(make_client):
    with make_client() as client:
        assert client.get("/v1/health").json()["status"] == "ok"
        assert client.get("/v1/version").json()["schema_version"] == "1.0"


def test_analyze_team_a_payload_matches_team_c_contract(make_client, team_a_payload):
    with make_client() as client:
        resp = client.post("/v1/analyze", json=team_a_payload)
        assert resp.status_code == 200
        body = resp.json()
        # Fields promised to Team C (email, Sep 15) plus confidence scores (requested Sep 18).
        for key in ("patient_info", "soap_note", "conflicts", "safety_status", "confidence"):
            assert key in body
        assert set(body["soap_note"]) == {"subjective", "objective", "assessment", "plan"}
        assert body["patient_info"]["allergies"] == ["Amoxicillin", "Penicillin"]
        assert body["safety_status"] == "SAFE"
        assert body["requires_physician_signoff"] is True
        claim = body["soap_claims"]["subjective"][1]
        assert claim["citations"][0]["start_time"] == 3.82 and len(claim["citations"][0]["segment_sha256"]) == 64


def test_conflict_payload_shape(make_client):
    payload = {"consultation_id": "c_conflict", "segments": [
        {"speaker": "PATIENT", "start_time": 0, "end_time": 2, "text": "I'm allergic to penicillin.", "confidence": 0.9},
        {"speaker": "DOCTOR", "start_time": 2, "end_time": 4, "text": "I'll prescribe amoxicillin.", "confidence": 0.9}]}
    with make_client() as client:
        body = client.post("/v1/analyze", json=payload).json()
        assert body["safety_status"] == "CONFLICT_DETECTED"
        c = body["conflicts"][0]
        assert {k: c[k] for k in ("type", "allergy", "medication", "severity", "status")} == {
            "type": "MEDICATION_ALLERGY", "allergy": "Penicillin", "medication": "Amoxicillin",
            "severity": "HIGH", "status": "BLOCKED"}
        assert c["message"] and 0 <= c["confidence"] <= 1
        assert body["draft_prescriptions"][0]["status"] == "BLOCKED"
        assert "amoxicillin" not in body["soap_note"]["plan"].lower().replace("safety block: amoxicillin", "")


def test_results_are_stored_for_team_c(make_client, team_a_payload):
    with make_client() as client:
        client.post("/v1/analyze", json=team_a_payload)
        client.post("/v1/analyze", json=team_a_payload)
        got = client.get(f"/v1/consultations/{team_a_payload['consultation_id']}")
        assert got.status_code == 200 and got.json()["consultation_id"] == "20261031_143022"
        items = client.get("/v1/consultations").json()["items"]
        assert items[0]["revision"] == 2
        assert client.get("/v1/consultations/unknown_id").status_code == 404


def test_pull_from_team_a(make_client, team_a_payload):
    team_a = FakeTeamA(payload=team_a_payload)
    with make_client(team_a) as client:
        resp = client.post("/v1/team-a/consultations/20261031_143022/analyze")
        assert resp.status_code == 200 and team_a.requested == ["20261031_143022"]


@pytest.mark.parametrize("error,status", [(TeamANotFoundError("nope"), 404), (TeamAError("down"), 502)])
def test_team_a_errors_are_mapped(make_client, error, status):
    with make_client(FakeTeamA(error=error)) as client:
        assert client.post("/v1/team-a/consultations/abc/analyze").status_code == status


def test_validation_errors(make_client):
    with make_client() as client:
        assert client.post("/v1/analyze", json={"consultation_id": "x", "segments": []}).status_code == 422
        assert client.post("/v1/analyze", json={"consultation_id": "x", "segments": [
            {"speaker": "DOCTOR", "start_time": 3, "end_time": 1, "text": "hi"}]}).status_code == 422
        assert client.post("/v1/analyze?extraction_mode=bogus", json={"consultation_id": "x", "segments": [
            {"speaker": "DOCTOR", "start_time": 0, "end_time": 1, "text": "hi"}]}).status_code == 422


def test_cors_allows_local_dashboard(make_client):
    with make_client() as client:
        resp = client.options("/v1/consultations", headers={"Origin": "http://localhost:3000",
                                                            "Access-Control-Request-Method": "GET"})
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
