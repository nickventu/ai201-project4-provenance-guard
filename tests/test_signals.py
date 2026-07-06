"""
Test suite for M4: pipeline/signals/llm_classifier.py + pipeline/aggregate.py
(exercised alongside the already-written pipeline/signals/stylometric.py).

Layers, cheapest/safest first:

  1. Unit tests for stylometric.py    -- pure functions, no mocking needed
  2. Unit tests for llm_classifier.py -- Groq client is MOCKED, no network/cost
  3. Unit tests for aggregate.py      -- both signals MOCKED, tests the math only
  4. Integration test (marked, skipped by default) -- hits the REAL Groq API,
     costs a few tokens, only runs if GROQ_API_KEY is set and you pass
     --run-integration

Run:
    pip install pytest --break-system-packages
    pytest tests/test_signals.py -v                      # layers 1-3, free, no network
    pytest tests/test_signals.py -v --run-integration     # + layer 4, needs GROQ_API_KEY
"""

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from pipeline.signals import stylometric, llm_classifier
from pipeline import aggregate
from pipeline.aggregate import CalibrationModel, compute_confidence, AggregationError


# ---------------------------------------------------------------------------
# Fixtures: known-ish AI-flavored vs human-flavored text, reused across layers
# ---------------------------------------------------------------------------

AI_TEXT = (
    "In today's fast-paced world, it is important to prioritize "
    "self-care and personal growth. By taking small, consistent "
    "steps, individuals can achieve their goals and improve their "
    "overall well-being. It is essential to remember that progress "
    "takes time, and every step forward is a step in the right "
    "direction."
)

HUMAN_TEXT = (
    "Honestly? I almost didn't submit this one. It's messy, the "
    "middle stanza doesn't quite land, and I rewrote the ending "
    "four times at 2am fueled by cold coffee and stubbornness. "
    "But there's a line in there about my grandmother's kitchen "
    "that I still can't read out loud without my voice cracking, "
    "so I'm sending it anyway, warts and all."
)

SHORT_TEXT = "This is a very short sample text."


# ---------------------------------------------------------------------------
# Layer 1 -- stylometric.py (pure functions, no mocking)
# ---------------------------------------------------------------------------

class TestStylometric:
    def test_score_in_unit_interval(self):
        for text in (AI_TEXT, HUMAN_TEXT, SHORT_TEXT):
            s = stylometric.score(text)
            assert 0.0 <= s <= 1.0

    def test_ai_text_scores_higher_than_human_text(self):
        # This is the M3 "Verify" check made automatic: direction, not
        # an exact value, since the reference model is a placeholder.
        ai_score = stylometric.score(AI_TEXT)
        human_score = stylometric.score(HUMAN_TEXT)
        assert ai_score > human_score

    def test_flags_low_reliability_on_short_text(self):
        result = stylometric.score_text(SHORT_TEXT)
        assert result.low_reliability is True
        assert result.token_count < stylometric.MIN_RELIABLE_TOKENS

    def test_does_not_flag_low_reliability_on_long_text(self):
        result = stylometric.score_text(AI_TEXT)
        assert result.low_reliability is False

    def test_empty_text_does_not_crash(self):
        # Note: this does NOT come out to a neutral 0.5. With zero tokens,
        # _compute_perplexity returns 0.0, which maps to a *maximal*
        # AI-likelihood sub-score (1.0), while burstiness defaults to a
        # neutral 0.0 -> sub-score 0.6. Combined: 0.8. Documented here so
        # a future refactor doesn't accidentally "fix" this into a silent
        # behavior change.
        result = stylometric.score_text("")
        assert result.token_count == 0
        assert result.score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Layer 2 -- llm_classifier.py (Groq client mocked, no real network call)
# ---------------------------------------------------------------------------

def _mock_groq_response(p_ai: float, rationale: str = "test"):
    """Build a fake Groq SDK response shaped like the real one."""
    message = MagicMock()
    message.content = json.dumps({"p_ai": p_ai, "rationale": rationale})
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMClassifier:
    def test_score_parses_valid_response(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
        with patch.object(llm_classifier, "Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _mock_groq_response(0.87)
            )
            result = llm_classifier.score("some text")
        assert result == 0.87

    def test_raises_on_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(llm_classifier.LLMClassifierError, match="GROQ_API_KEY"):
            llm_classifier.score("some text")

    def test_raises_on_unparseable_response(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
        with patch.object(llm_classifier, "Groq") as MockGroq:
            bad_response = _mock_groq_response(0.5)
            bad_response.choices[0].message.content = "not json at all"
            MockGroq.return_value.chat.completions.create.return_value = bad_response
            with pytest.raises(llm_classifier.LLMClassifierError, match="could not parse"):
                llm_classifier.score("some text")

    def test_raises_on_out_of_range_score(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
        with patch.object(llm_classifier, "Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                _mock_groq_response(1.5)  # invalid: outside [0, 1]
            )
            with pytest.raises(llm_classifier.LLMClassifierError, match="out of range"):
                llm_classifier.score("some text")

    def test_raises_on_empty_text(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
        with pytest.raises(llm_classifier.LLMClassifierError, match="empty"):
            llm_classifier.score("")

    def test_raises_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
        with patch.object(llm_classifier, "Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = RuntimeError("boom")
            with pytest.raises(llm_classifier.LLMClassifierError, match="Groq request failed"):
                llm_classifier.score("some text")


# ---------------------------------------------------------------------------
# Layer 3 -- aggregate.py (both signals mocked, tests only the combination math)
# ---------------------------------------------------------------------------

class TestCalibrationModel:
    def test_unfitted_flag(self):
        m = CalibrationModel.unfitted()
        assert m.fitted is False

    def test_fit_recovers_a_clear_separation(self):
        # Synthetic, perfectly-separable-ish data: label = 1 iff s1+s2 > 1.0
        import random
        random.seed(0)
        pairs = [(random.random(), random.random()) for _ in range(200)]
        labels = [1 if s1 + s2 > 1.0 else 0 for s1, s2 in pairs]
        model = CalibrationModel.fit(pairs, labels, lr=0.5, epochs=500)
        assert model.fitted is True
        # High/high should score clearly higher than low/low post-fit.
        assert model.predict(0.9, 0.9) > model.predict(0.1, 0.1)

    def test_fit_rejects_too_few_examples(self):
        with pytest.raises(ValueError, match="meaningfully sized"):
            CalibrationModel.fit([(0.5, 0.5)] * 3, [1, 0, 1])

    def test_fit_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            CalibrationModel.fit([(0.5, 0.5)] * 10, [1, 0])


class TestComputeConfidence:
    @pytest.mark.parametrize("raw_score,expected", [
        (0.5, 0.0),
        (0.6, 0.2),
        (0.9, 0.8),
        (0.1, 0.8),
        (1.0, 1.0),
        (0.0, 1.0),
    ])
    def test_matches_spec_formula(self, raw_score, expected):
        assert compute_confidence(raw_score) == pytest.approx(expected)


class TestAggregate:
    def test_uses_fallback_average_when_uncalibrated(self, monkeypatch):
        monkeypatch.setattr(stylometric, "score", lambda text: 0.8)
        monkeypatch.setattr(llm_classifier, "score", lambda text: 0.6)
        result = aggregate.aggregate("irrelevant text", calibration=CalibrationModel.unfitted())
        assert result["calibrated"] is False
        assert result["raw_score"] == pytest.approx(0.7)  # plain 50/50 average
        assert result["confidence"] == pytest.approx(compute_confidence(0.7))

    def test_uses_calibration_model_when_fitted(self, monkeypatch):
        monkeypatch.setattr(stylometric, "score", lambda text: 0.8)
        monkeypatch.setattr(llm_classifier, "score", lambda text: 0.6)
        fitted_model = CalibrationModel(weight_1=1.0, weight_2=1.0, bias=0.0, fitted=True)
        result = aggregate.aggregate("irrelevant text", calibration=fitted_model)
        assert result["calibrated"] is True
        expected_raw = aggregate._sigmoid(1.0 * 0.8 + 1.0 * 0.6 + 0.0)
        assert result["raw_score"] == pytest.approx(expected_raw)
        assert result["raw_score"] != pytest.approx(0.7)  # must differ from naive average

    def test_combined_scores_separate_ai_and_human_groups(self, monkeypatch):
        # This is the M4 "Verify" step from planning.md, automated: confirm the
        # combined score differs meaningfully between groups, not just in the
        # same direction as one signal alone.
        def fake_llm(text):
            return 0.9 if text == AI_TEXT else 0.15

        monkeypatch.setattr(llm_classifier, "score", fake_llm)
        # Real stylometric.score() runs unmocked here on purpose.

        ai_result = aggregate.aggregate(AI_TEXT, calibration=CalibrationModel.unfitted())
        human_result = aggregate.aggregate(HUMAN_TEXT, calibration=CalibrationModel.unfitted())

        assert ai_result["raw_score"] > human_result["raw_score"]
        assert (ai_result["raw_score"] - human_result["raw_score"]) > 0.2

    def test_raises_when_stylometric_signal_fails(self, monkeypatch):
        def broken(text):
            raise RuntimeError("stylometric exploded")
        monkeypatch.setattr(stylometric, "score", broken)
        with pytest.raises(AggregationError, match="stylometric signal failed"):
            aggregate.aggregate("text")

    def test_raises_when_llm_signal_fails(self, monkeypatch):
        monkeypatch.setattr(stylometric, "score", lambda text: 0.5)
        def broken(text):
            raise llm_classifier.LLMClassifierError("groq down")
        monkeypatch.setattr(llm_classifier, "score", broken)
        with pytest.raises(AggregationError, match="llm_classifier signal failed"):
            aggregate.aggregate("text")


# ---------------------------------------------------------------------------
# Layer 4 -- real integration test against the actual Groq API.
# Skipped unless you explicitly opt in, since it costs tokens and needs a key.
# ---------------------------------------------------------------------------

@pytest.fixture
def run_integration(request):
    if not request.config.getoption("--run-integration"):
        pytest.skip("use --run-integration to run real-API tests")


class TestIntegrationRealGroq:
    def test_llm_classifier_direction_on_real_api(self, run_integration):
        import os
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")
        ai_score = llm_classifier.score(AI_TEXT)
        human_score = llm_classifier.score(HUMAN_TEXT)
        print(f"\nreal llm_classifier scores: ai={ai_score:.2f} human={human_score:.2f}")
        assert 0.0 <= ai_score <= 1.0
        assert 0.0 <= human_score <= 1.0
        # Soft check only -- a single LLM call is noisy, this just catches
        # a badly broken prompt/parse, not fine-grained accuracy.
        assert ai_score > human_score

    def test_full_aggregate_on_real_api(self, run_integration):
        import os
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")
        ai_result = aggregate.aggregate(AI_TEXT)
        human_result = aggregate.aggregate(HUMAN_TEXT)
        print(f"\nreal aggregate: ai={ai_result} human={human_result}")
        assert ai_result["raw_score"] > human_result["raw_score"]
