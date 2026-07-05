"""
Audit log storage (Section 3a: `storage/audit_log.py`).

"Append-only structured log" — every /submit decision (content_id,
signals, scores, label) gets written here. Backed by SQLite (stdlib
`sqlite3`, no ORM), per the Section 3 tech stack table.

Note on schema: the requested example log shape included `creator_id`
and `llm_score` fields. Neither is populated here:
    - `creator_id` would require an auth layer, which doesn't exist in
      this system (Section 3b explicitly excludes auth endpoints for
      v1).
    - `llm_score` belongs to Signal 2 (Groq LLM classifier), which
      hasn't been built yet — that's M4 work per the AI Tool Plan.
Both can be added as real columns once those pieces exist, rather than
filled with placeholder/fake data now.

`status` defaults to "classified" and will later flip to "under_review"
once the appeals flow (`storage/appeals.py`, M5) is wired up.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "audit_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    content_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    signals TEXT NOT NULL,
    raw_score REAL NOT NULL,
    confidence REAL NOT NULL,
    attribution_result TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'classified'
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_entry(
    content_id: str,
    signals: dict,
    raw_score: float,
    confidence: float,
    attribution_result: str,
    label: str,
    status: str = "classified",
) -> None:
    """
    Write one audit log entry. Called from api/routes.py right before
    the /submit response goes out, per the Architecture diagram's
    "content_id, signals, scores, label all written together" note.

    `INSERT OR REPLACE` on content_id: since content_id is derived from
    a hash of the normalized text (see api/routes.py), re-submitting
    identical content overwrites rather than duplicates. Revisit if
    duplicate submissions of the same text should instead be tracked as
    separate events.
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO audit_log
                (content_id, timestamp, signals, raw_score, confidence,
                 attribution_result, label, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(signals),
                raw_score,
                confidence,
                attribution_result,
                label,
                status,
            ),
        )


def get_log(limit: int = 3) -> list[dict]:
    """
    Return the most recent `limit` audit log entries, newest first.

    Section 3b also calls for an optional `?content_id=` filter on
    `GET /log` — not implemented here since the current ask is just
    "most recent N"; add a `content_id` kwarg here (and a query-param
    passthrough in the route) when that's needed.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [
        {
            "content_id": row["content_id"],
            "timestamp": row["timestamp"],
            "signals": json.loads(row["signals"]),
            "raw_score": row["raw_score"],
            "confidence": row["confidence"],
            "attribution_result": row["attribution_result"],
            "label": row["label"],
            "status": row["status"],
        }
        for row in rows
    ]
