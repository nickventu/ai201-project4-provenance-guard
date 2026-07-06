"""
Label generator (Section 3a: pipeline/label.py).

Maps (raw_score, confidence) -> one of exactly three fixed label strings,
per planning.md's "Transparency labels (verbatim)" table. No fourth
option, no freeform text -- these strings are shown directly to a
non-technical reader and must match exactly.

Thresholds (Section 5, "Thresholds (tunable, documented in README)"):
    confidence >= 0.7 and raw_score > 0.5  -> high-confidence AI
    confidence >= 0.7 and raw_score < 0.5  -> high-confidence human
    everything else                         -> uncertain

This is the real version of the placeholder `_label_for()` that used to
live inline in api/routes.py -- pulled into its own module so it has a
single source of truth and can be unit-tested independently of Flask.
"""

from dataclasses import dataclass

CONFIDENCE_THRESHOLD = 0.7

LABEL_AI = "This content appears to be AI-generated (confidence: high)."
LABEL_HUMAN = "This content appears to be human-created (confidence: high)."
LABEL_UNCERTAIN = (
    "We could not confidently determine whether this content is "
    "AI-generated or human-created. Treat this result as inconclusive."
)

ATTRIBUTION_AI = "ai"
ATTRIBUTION_HUMAN = "human"
ATTRIBUTION_UNCERTAIN = "uncertain"


@dataclass
class LabelResult:
    attribution_result: str  # "ai" | "human" | "uncertain"
    label: str                # one of the three verbatim strings above


def label_for(raw_score: float, confidence: float) -> LabelResult:
    """
    Apply the Section 5 thresholds to produce the attribution_result and
    the exact label text shown to the reader.
    """
    if confidence >= CONFIDENCE_THRESHOLD and raw_score > 0.5:
        return LabelResult(attribution_result=ATTRIBUTION_AI, label=LABEL_AI)
    if confidence >= CONFIDENCE_THRESHOLD and raw_score < 0.5:
        return LabelResult(attribution_result=ATTRIBUTION_HUMAN, label=LABEL_HUMAN)
    return LabelResult(attribution_result=ATTRIBUTION_UNCERTAIN, label=LABEL_UNCERTAIN)
