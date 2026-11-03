"""Client for Team 2A's transcript endpoint:

    GET {TEAM_A_BASE_URL}/api/v1/live/team-b/consultation/{consultation_id}
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

TEAM_A_TRANSCRIPT_PATH = "/api/v1/live/team-b/consultation/{consultation_id}"


class TeamAError(RuntimeError):
    """Team A's service could not provide the transcript."""


class TeamANotFoundError(TeamAError):
    """Team A has no transcript for this consultation (yet)."""


class TeamAClient:
    def __init__(self, base_url: str, timeout_s: float = 15.0, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def transcript_url(self, consultation_id: str) -> str:
        return self.base_url + TEAM_A_TRANSCRIPT_PATH.format(consultation_id=quote(consultation_id, safe=""))

    def fetch_transcript(self, consultation_id: str) -> dict[str, Any]:
        url = self.transcript_url(consultation_id)
        try:
            resp = self.session.get(url, timeout=self.timeout_s, headers={"Accept": "application/json"})
        except requests.RequestException as exc:
            raise TeamAError(f"Team A service unreachable at {self.base_url}: {type(exc).__name__}") from exc

        if resp.status_code == 404:
            raise TeamANotFoundError(f"Team A has no transcript for consultation '{consultation_id}'")
        if resp.status_code >= 400:
            raise TeamAError(f"Team A returned HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise TeamAError("Team A returned a non-JSON response") from exc
        if not isinstance(data, dict):
            raise TeamAError("Team A returned an unexpected JSON shape")
        logger.info("Fetched transcript from Team A", extra={"consultation_id": consultation_id})
        return data
