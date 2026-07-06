"""
API layer (Section 3a: api/routes.py).

Four endpoints, matching Section 3b exactly:
    POST /submit                    -- run both signals, return the
                                        attribution result + confidence
                                        + transparency label
    POST /appeal/<content_id>       -- record a creator's appeal against
                                        an existing decision
    GET  /log                       -- audit log, optional ?content_id=
    GET  /appeals                   -- reviewer queue, optional ?status=
                                        (defaults to under_review)

Wires together everything built in M3/M4/M5:
    pipeline.aggregate.aggregate()   -- both signals + reliability-aware
                                         combination (pipeline/aggregate.py)
    pipeline.label.label_for()       -- the three verbatim transparency
                                         labels (pipeline/label.py)
    storage.audit_log                -- persists every decision
    storage.appeals                  -- persists appeals, linked by
                                         content_id
    middleware.rate_limit            -- protects the one endpoint that
                                         calls a paid external API
"""

from __future__ import annotations

import re

from flask import Flask, Blueprint, request, jsonify

from pipeline.aggregate import aggregate, AggregationError
from pipeline.label import label_for
from storage import audit_log, appeals
from middleware.rate_limit import limiter, SUBMIT_RATE_LIMIT

# Blind spot noted in Section 4 / edge case #2: both signals are
# unreliable under ~50-100 words. No minimum-length gate exists yet
# (open question, Section 6) — this is just a basic non-empty check,
# not a reliability check.
MIN_CONTENT_LENGTH = 1

bp = Blueprint("api", __name__)


def _normalize(text: str) -> str:
    """Minimal placeholder normalization (whitespace collapse). Real
    preprocessing (case folding decisions, unicode normalization, etc.)
    belongs in pipeline/preprocess.py, which doesn't exist yet."""
    return re.sub(r"\s+", " ", text.strip())


@bp.route("/submit", methods=["POST"])
@limiter.limit(SUBMIT_RATE_LIMIT)
def submit():
    body = request.get_json(silent=True) or {}
    content = body.get("content")

    if not isinstance(content, str) or len(content.strip()) < MIN_CONTENT_LENGTH:
        return jsonify({"error": "missing or too-short 'content'"}), 400

    cleaned = _normalize(content)
    content_id = audit_log.compute_content_id(cleaned)

    try:
        result = aggregate(cleaned)
    except AggregationError as exc:
        # A signal outage (e.g. Groq API down) should surface as a real
        # error, not a silently degraded/fabricated score.
        return jsonify({"error": f"detection pipeline failed: {exc}"}), 500

    labeled = label_for(result["raw_score"], result["confidence"])

    audit_log.log_result(
        content_id=content_id,
        signals=result["signals"],
        raw_score=result["raw_score"],
        confidence=result["confidence"],
        label=labeled.label,
    )

    return jsonify({
        "content_id": content_id,
        "attribution_result": labeled.attribution_result,
        "confidence_score": round(result["confidence"], 2),
        "label": labeled.label,
    }), 200


@bp.route("/appeal/<content_id>", methods=["POST"])
def submit_appeal(content_id: str):
    body = request.get_json(silent=True) or {}
    reasoning = body.get("reasoning")

    if not isinstance(reasoning, str) or not reasoning.strip():
        return jsonify({"error": "missing 'reasoning'"}), 400

    original_entry = audit_log.get_entry(content_id)
    if original_entry is None:
        return jsonify({"error": f"unknown content_id '{content_id}'"}), 404

    appeal = appeals.create_appeal(content_id=content_id, reasoning=reasoning)

    return jsonify({
        "content_id": content_id,
        "status": appeal["status"],
        "submitted_at": appeal["submitted_at"],
    }), 200


@bp.route("/log", methods=["GET"])
def get_log_route():
    content_id = request.args.get("content_id")
    entries = audit_log.get_all(content_id=content_id)

    # Enrich each entry with its appeal, if one exists -- null if not.
    # This is what makes appeals actually "captured in the structured
    # audit log" (not just visible via the separate GET /appeals reviewer
    # view): joined at read-time here, same pattern as GET /appeals uses
    # in reverse, still no separate appeals store per planning.md.
    for entry in entries:
        appeal = appeals.get_appeal(entry["content_id"])
        entry["appeal"] = (
            {
                "reasoning": appeal["reasoning"],
                "status": appeal["status"],
                "submitted_at": appeal["submitted_at"],
            }
            if appeal is not None
            else None
        )

    return jsonify(entries), 200


@bp.route("/appeals", methods=["GET"])
def get_appeals_route():
    status = request.args.get("status", "under_review")
    appeal_records = appeals.get_appeals(status=status)

    # Reviewer queue view (Section 3b): each open appeal, joined against
    # its original decision. This is a read-time join over two tables,
    # not a separate store, per planning.md's explicit instruction.
    queue = []
    for appeal in appeal_records:
        original = audit_log.get_entry(appeal["content_id"])
        queue.append({
            "content_id": appeal["content_id"],
            "original_decision": {
                "signals": original["signals"],
                "raw_score": original["raw_score"],
                "confidence": original["confidence"],
                "label": original["label"],
            } if original else None,
            "appeal": {
                "reasoning": appeal["reasoning"],
                "submitted_at": appeal["submitted_at"],
            },
            "status": appeal["status"],
        })

    return jsonify(queue), 200


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(bp)
    limiter.init_app(app)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
