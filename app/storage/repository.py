"""SQLite persistence of analysis results so Team C can pull them by consultation id.

Results contain PHI: the database file lives on the local machine / container volume only.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from app.models.schemas import ConsultationAnalysis

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultations (
    consultation_id TEXT PRIMARY KEY,
    generated_at    TEXT NOT NULL,
    safety_status   TEXT NOT NULL,
    conflict_count  INTEGER NOT NULL,
    revision        INTEGER NOT NULL DEFAULT 1,
    payload         TEXT NOT NULL
)
"""


class ConsultationStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def save(self, analysis: ConsultationAnalysis) -> int:
        with closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT revision FROM consultations WHERE consultation_id = ?",
                               (analysis.consultation_id,)).fetchone()
            revision = (row[0] + 1) if row else 1
            conn.execute(
                "INSERT OR REPLACE INTO consultations VALUES (?, ?, ?, ?, ?, ?)",
                (analysis.consultation_id, analysis.generated_at, analysis.safety_status.value,
                 len(analysis.conflicts), revision, analysis.model_dump_json()),
            )
        return revision

    def get(self, consultation_id: str) -> ConsultationAnalysis | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload FROM consultations WHERE consultation_id = ?",
                               (consultation_id,)).fetchone()
        return ConsultationAnalysis.model_validate_json(row[0]) if row else None

    def list(self, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT consultation_id, generated_at, safety_status, conflict_count, revision FROM consultations "
                "ORDER BY generated_at DESC LIMIT ?", (limit,)).fetchall()
        keys = ("consultation_id", "generated_at", "safety_status", "conflict_count", "revision")
        return [dict(zip(keys, r)) for r in rows]
