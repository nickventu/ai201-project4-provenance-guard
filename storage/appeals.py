"""
Appeals storage (Section 3a: storage/appeals.py).

Appeal records, linked to the original audit log entry by content_id.
Lives in the same SQLite file as audit_log.py (not a separate database)
so GET /appeals can join an appeal against its original decision in one
place, matching Section 3b: "a read view over the same audit log data...
not a separate store."

Per Section 3b's worked appeal flow: no automated re-scoring happens
here. This module only records the creator's reasoning and flips status
to "under_review" -- a human reviewer decides manually via
GET /appeals?status=under_review.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from storage.audit_log import DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS appeals (
    content_id   TEXT PRIMARY KEY,
    reasoning    TEXT NOT NULL,
    status       TEXT NOT NULL,   -- "under_review" (only status in scope for now)
    submitted_at TEXT NOT NULL    -- ISO 8601 UTC
);
"""


@contextmanager
def _connect(db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_appeal(content_id: str, reasoning: str, db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Record an appeal against `content_id` and set status to
    "under_review". Does NOT check whether content_id exists in the
    audit log -- that check belongs to the caller (api/routes.py),
    which is the layer that knows how to turn "not found" into a 404.

    Re-appealing the same content_id overwrites the previous appeal
    record (INSERT OR REPLACE), same "latest write wins" convention as
    audit_log.log_result -- a second appeal on the same content replaces
    the first rather than silently failing or duplicating.
    """
    entry = {
        "content_id": content_id,
        "reasoning": reasoning,
        "status": "under_review",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO appeals (content_id, reasoning, status, submitted_at)
            VALUES (?, ?, ?, ?)
            """,
            (entry["content_id"], entry["reasoning"], entry["status"], entry["submitted_at"]),
        )

    return entry


def get_appeal(content_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[dict]:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM appeals WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def get_appeals(status: Optional[str] = "under_review", db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    List appeals, most recent first. Defaults to status="under_review"
    per Section 3b's GET /appeals?status= (defaults to under_review).
    Pass status=None to list all appeals regardless of status.
    """
    with _connect(db_path) as conn:
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM appeals WHERE status = ? ORDER BY submitted_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM appeals ORDER BY submitted_at DESC"
            ).fetchall()
    return [dict(row) for row in rows]
