"""
Audit log (Section 3a: storage/audit_log.py).

Append-only structured log, keyed by content_id (the content hash), per
Section 3a: "Append-only structured log ... Hash = audit log key".

Uses SQLite (stdlib sqlite3, no ORM) -- one of the two options explicitly
allowed by Section 3 tech stack ("SQLite (built-in) or structured JSON").
SQLite is picked here over flat JSON because Section 3b's GET /log
(optional ?content_id= filter) and GET /appeals (join against the
original decision) both want simple, indexed lookups rather than
scanning a JSON file on every request.

Each row captures, per the checkpoint requirement: the individual signal
scores, the combined raw_score/confidence, and (once label.py exists in
M5) the label text. label is nullable for now since M4 doesn't generate
labels yet -- that's Section 5/M5 work, not this module's job.

This module does NOT compute content_id or run the signals -- it only
persists what it's given. Keeping hashing/scoring out of this file
matches the separation of concerns already established elsewhere in
the pipeline (aggregate.py doesn't know about storage; storage doesn't
know about signals).
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

DEFAULT_DB_PATH = "audit_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    content_id  TEXT PRIMARY KEY,
    signals     TEXT NOT NULL,   -- JSON: {"stylometric": 0.62, "llm_classifier": 0.80}
    raw_score   REAL NOT NULL,
    confidence  REAL NOT NULL,
    label       TEXT,              -- nullable until label.py (M5) exists
    created_at  TEXT NOT NULL      -- ISO 8601 UTC
);
"""


def compute_content_id(text: str) -> str:
    """
    Hash the (already-normalized) text into a short content_id, matching
    the "f3a9c1"-style ids used throughout planning.md's examples.
    NOTE: real normalization belongs to pipeline/preprocess.py (not yet
    built) -- this just hashes whatever string it's given.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]


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


def log_result(
    content_id: str,
    signals: dict,
    raw_score: float,
    confidence: float,
    label: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """
    Write one audit entry. Append-only: re-logging the same content_id
    overwrites that row (a resubmission of identical content should
    reflect the latest scoring run, not silently fail or duplicate) --
    this is the one deliberate exception to "append-only" and is
    explicitly INSERT OR REPLACE, not an UPDATE-by-mistake.

    Returns the entry as a dict, exactly as it was stored.
    """
    entry = {
        "content_id": content_id,
        "signals": signals,
        "raw_score": raw_score,
        "confidence": confidence,
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO audit_log
                (content_id, signals, raw_score, confidence, label, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry["content_id"],
                json.dumps(entry["signals"]),
                entry["raw_score"],
                entry["confidence"],
                entry["label"],
                entry["created_at"],
            ),
        )

    return entry


def get_entry(content_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[dict]:
    """Fetch one entry by content_id, or None if it doesn't exist."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE content_id = ?", (content_id,)
        ).fetchone()

    if row is None:
        return None
    return _row_to_dict(row)


def get_all(content_id: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """
    Fetch all entries, most recent first, matching GET /log's
    "?content_id=" optional filter (Section 3b).
    """
    with _connect(db_path) as conn:
        if content_id is not None:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE content_id = ? ORDER BY created_at DESC",
                (content_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY created_at DESC"
            ).fetchall()

    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "content_id": row["content_id"],
        "signals": json.loads(row["signals"]),
        "raw_score": row["raw_score"],
        "confidence": row["confidence"],
        "label": row["label"],
        "created_at": row["created_at"],
    }
