"""
API layer (Section 3a: `api/routes.py` -> `POST /submit`, plus the other
endpoints listed for later milestones).

This is the M3 skeleton: only the submission flow is wired up, and only
against Signal 1 (stylometric). Per the Architecture diagram, `/submit`
should eventually run stylometric + LLM classifier in parallel, feed
both into a calibrated Score Aggregator, then a Label Generator, then
log the result before responding. Signal 2 (`pipeline/signals/
llm_classifier.py`), the aggregator (`pipeline/aggregate.py`), and the
label generator (`pipeline/label.py`) are M4/M5 work — see planning.md's
"AI Tool Plan". Everywhere this skeleton stands in for that later logic,
it's marked TODO below.

`/appeal/{content_id}`, `GET /log`, and `GET /appeals` are listed in
Section 3a/3b but intentionally not stubbed here — they depend on the
audit log (`storage/audit_log.py`) which doesn't exist yet either.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from flask import Flask, Blueprint, request, jsonify

from pipeline.signals import stylometric
from storage.audit_log import log_entry, get_log

# Blind spot noted in Section 4 / edge case #2: both signals are
# unreliable under ~50-100 words. No minimum-length gate exists yet
# (open question, Section 6) — this is just a basic non-empty check,
# not a reliability check.
MIN_CONTENT_LENGTH = 1

bp = Blueprint("api", __name__)


def _make_content_id(cleaned_text: str) -> str:
    """
    Deterministic content_id from a hash of the normalized text. This
    stands in for `pipeline/preprocess.py` ("Clean text, compute content
    hash" per Section 3a) until that module exists. Truncated to match
    the short hex ids shown in planning.md's examples (e.g. "f3a9c1").
    """
    digest = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
    return digest[:6]


def _normalize(text: str) -> str:
    """Minimal placeholder normalization (whitespace collapse). Real
    preprocessing (case folding decisions, unicode normalization, etc.)
    belongs in pipeline/preprocess.py."""
    return re.sub(r"\s+", " ", text.strip())


def _label_for(raw_score: float, confidence: float) -> str:
    """
    TODO(M5): replace with pipeline/label.py, which will hold the
    verbatim label strings from planning.md's "Transparency labels"
    table. Duplicated here only so /submit returns a realistic-shaped
    response before that module exists.
    """
    if confidence >= 0.7 and raw_score > 0.5:
        return "This content appears to be AI-generated (confidence: high)."
    if confidence >= 0.7 and raw_score < 0.5:
        return "This content appears to be human-created (confidence: high)."
    return (
        "We could not confidently determine whether this content is "
        "AI-generated or human-created. Treat this result as inconclusive."
    )


@bp.route("/submit", methods=["POST"])
def submit():
    body = request.get_json(silent=True) or {}
    content = body.get("content")

    if not isinstance(content, str) or len(content.strip()) < MIN_CONTENT_LENGTH:
        return jsonify({"error": "missing or too-short 'content'"}), 400

    cleaned = _normalize(content)
    content_id = _make_content_id(cleaned)

    # --- Signal 1: stylometric (this milestone) ---
    signal_1_score = stylometric.score(cleaned)

    # TODO(M4): Signal 2 (LLM classifier via Groq) runs in parallel with
    # Signal 1 per the Architecture diagram. For now, signal_1 alone
    # stands in for both branches feeding the aggregator.
    #
    # TODO(M4): replace this placeholder combination with the real
    # Score Aggregator (pipeline/aggregate.py), which fits a calibration
    # model (logistic regression / Platt scaling, Section 5) over both
    # signal outputs instead of just relaying one signal's raw score.
    raw_score = signal_1_score
    confidence = 2 * abs(raw_score - 0.5)

    label = _label_for(raw_score, confidence)
    attribution_result = (
        "ai" if raw_score > 0.5 and confidence >= 0.7
        else "human" if raw_score < 0.5 and confidence >= 0.7
        else "uncertain"
    )

    # TODO(M5): pipeline/label.py should replace the inline _label_for()
    # helper above, and storage/appeals.py will add the appeal-linked
    # status flips. Basic audit logging (content_id, signals, scores,
    # label) is already wired in below.

    # Audit log: written right before the response goes out, per the
    # Architecture diagram's Audit Logger step.
    log_entry(
        content_id=content_id,
        signals={"stylometric": round(signal_1_score, 3)},
        raw_score=raw_score,
        confidence=confidence,
        attribution_result=attribution_result,
        label=label,
    )

    return jsonify({
        "content_id": content_id,
        "attribution_result": attribution_result,
        "confidence_score": round(confidence, 2),
        "label": label,
    }), 200


@bp.route("/log", methods=["GET"])
def get_log_route():
    return jsonify({"entries": get_log()})


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
