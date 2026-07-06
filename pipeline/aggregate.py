"""
Score aggregator (Section 5, planning.md).

Combines the two detection signals into a single calibrated raw_score
and a confidence value, per the spec:

  raw_score  = P(AI-generated) in [0, 1], produced by a calibration
               model fit on labeled (signal_1, signal_2) -> ground_truth
               pairs -- NOT a fixed 50/50 average of the raw signals.
  confidence = 2 * |raw_score - 0.5|   (distance from a coin flip, in [0, 1])

Per Section 6 (open questions), the labeled calibration set doesn't exist
yet ("Pick actual signal weights (start 50/50, tune against labeled
set)"). So this module ships with:
  - A CalibrationModel that CAN be fit against labeled data once it
    exists (logistic regression via plain-Python gradient descent --
    Platt scaling -- no extra ML dependency, consistent with the rest
    of the stack).
  - An explicit, clearly-labeled fallback average used only when no
    fitted calibration model is available, so it's never confused with
    a real calibrated score.

The fallback average is reliability-aware, not a flat 50/50: when
stylometric.score_text() flags low_reliability (under ~50 tokens), that
signal is down-weighted rather than trusted equally. This isn't a new
invented threshold -- it's using a flag the stylometric signal already
computes and documents. Testing surfaced concrete cases (short/casual
and short/jargon-heavy samples) where stylometric's sub-metrics
disagreed with ground truth while the LLM classifier ranked correctly,
so leaning on the LLM signal in exactly the regime stylometric already
admits it's unreliable in is a direct, simple response to that evidence
-- not a threshold hand-tuned to make 6 examples look right. Once real
calibration data exists (Section 5), a fitted CalibrationModel should
learn this kind of reweighting itself and this manual override becomes
unnecessary.
"""

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple

from pipeline.signals import llm_classifier, stylometric

DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "calibration_weights.json"
)

# Fallback-only weighting when stylometric.low_reliability is True.
# Provisional, same status as the 50/50 default: a documented stopgap,
# not a calibrated value.
_LOW_RELIABILITY_STYLOMETRIC_WEIGHT = 0.2
_LOW_RELIABILITY_LLM_WEIGHT = 0.8


class AggregationError(Exception):
    """Raised when one or both signals fail and no score can be produced."""


def _sigmoid(x: float) -> float:
    # Guard against overflow on extreme inputs.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class CalibrationModel:
    """Platt-scaling logistic model: raw_score = sigmoid(w1*s1 + w2*s2 + b)."""

    weight_1: float
    weight_2: float
    bias: float
    fitted: bool = False

    def predict(self, signal_1_score: float, signal_2_score: float) -> float:
        return _sigmoid(
            self.weight_1 * signal_1_score + self.weight_2 * signal_2_score + self.bias
        )

    @classmethod
    def unfitted(cls) -> "CalibrationModel":
        return cls(weight_1=0.5, weight_2=0.5, bias=0.0, fitted=False)

    @classmethod
    def fit(
        cls,
        pairs: Iterable[Tuple[float, float]],
        labels: Iterable[int],
        lr: float = 0.1,
        epochs: int = 2000,
    ) -> "CalibrationModel":
        """
        Fit weight_1, weight_2, bias via gradient descent on logistic loss,
        using labeled (signal_1_score, signal_2_score) -> is_ai pairs.
        Step 1-3 of Section 5's "raw signal outputs -> calibrated score".
        """
        pairs = list(pairs)
        labels = list(labels)
        if len(pairs) != len(labels):
            raise ValueError("pairs and labels must be the same length")
        if len(pairs) < 10:
            raise ValueError(
                "need a meaningfully sized labeled set to calibrate, not just a few points"
            )

        w1, w2, b = 0.0, 0.0, 0.0
        n = len(pairs)

        for _ in range(epochs):
            grad_w1 = grad_w2 = grad_b = 0.0
            for (s1, s2), y in zip(pairs, labels):
                pred = _sigmoid(w1 * s1 + w2 * s2 + b)
                error = pred - y
                grad_w1 += error * s1
                grad_w2 += error * s2
                grad_b += error
            w1 -= lr * grad_w1 / n
            w2 -= lr * grad_w2 / n
            b -= lr * grad_b / n

        return cls(weight_1=w1, weight_2=w2, bias=b, fitted=True)

    def save(self, path: str = DEFAULT_CALIBRATION_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str = DEFAULT_CALIBRATION_PATH) -> "CalibrationModel":
        if not os.path.exists(path):
            return cls.unfitted()
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compute_confidence(raw_score: float) -> float:
    """confidence = 2 * |raw_score - 0.5|, per Section 5."""
    return 2 * abs(raw_score - 0.5)


def aggregate(text: str, calibration: CalibrationModel = None) -> dict:
    """
    Run both detection signals on `text` and combine them into a
    calibrated raw_score + confidence.

    Returns:
        {
          "signals": {"stylometric": float, "llm_classifier": float},
          "raw_score": float,
          "confidence": float,
          "calibrated": bool,          # False if this used the fallback average
          "stylometric_low_reliability": bool,
        }

    Raises AggregationError if either signal fails -- a signal outage
    should surface as an error, not silently degrade into a fabricated
    score built from only one input.
    """
    try:
        stylometric_result = stylometric.score_text(text)
        signal_1_score = stylometric_result.score
        low_reliability = stylometric_result.low_reliability
    except Exception as exc:
        raise AggregationError(f"stylometric signal failed: {exc}") from exc

    try:
        signal_2_score = llm_classifier.score(text)
    except llm_classifier.LLMClassifierError as exc:
        raise AggregationError(f"llm_classifier signal failed: {exc}") from exc

    if calibration is None:
        calibration = CalibrationModel.load()

    if calibration.fitted:
        raw_score = calibration.predict(signal_1_score, signal_2_score)
        calibrated = True
    elif low_reliability:
        # Stylometric already flagged itself unreliable at this length --
        # don't average it in as an equal partner. See module docstring.
        raw_score = (
            _LOW_RELIABILITY_STYLOMETRIC_WEIGHT * signal_1_score
            + _LOW_RELIABILITY_LLM_WEIGHT * signal_2_score
        )
        calibrated = False
    else:
        # Explicit placeholder fallback (Section 5 / Section 6 TODO) --
        # a plain average, not a calibrated probability. Kept separate
        # from the calibrated path so it's never mistaken for one.
        raw_score = 0.5 * signal_1_score + 0.5 * signal_2_score
        calibrated = False

    confidence = compute_confidence(raw_score)

    return {
        "signals": {
            "stylometric": signal_1_score,
            "llm_classifier": signal_2_score,
        },
        "raw_score": raw_score,
        "confidence": confidence,
        "calibrated": calibrated,
        "stylometric_low_reliability": low_reliability,
    }

