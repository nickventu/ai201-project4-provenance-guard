"""
Signal 2: LLM classifier signal (Section 4, planning.md).

Judges tone, cliche density, coherence, and "voice" holistically via
Groq (llama-3.3-70b-versatile) and returns its own P(AI) estimate in [0, 1].

Blind spots (per spec, kept here as a comment, not solved by this module):
subjective, can be overconfident/inconsistent run-to-run; biased toward
flagging clean/edited writing; gameable by inserting deliberate "quirky"
errors; only sees typicality, not real provenance; uneven reliability
across genres.

Interface mirrors pipeline/signals/stylometric.py: a single `score(text)`
function returning a float in [0, 1], so aggregate.py can call both
signals the same way.
"""

import json
import os

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

# Kept low so the model reports its judgment rather than hedging/rambling.
TEMPERATURE = 0.0

SYSTEM_PROMPT = """You are a text-provenance classifier. Given a piece of \
writing, judge holistically whether it reads as AI-generated or \
human-written.

Weigh these dimensions:
- Tone: safe, generic, hedged phrasing vs. idiosyncratic voice
- Cliche density: stock phrases and cliches vs. concrete, personal detail
- Coherence: locally smooth but occasionally shallow reasoning vs. \
messier, more particular reasoning
- Voice: absence of a distinct personality vs. a consistent, quirky \
authorial voice

Respond with ONLY a JSON object, no other text, no markdown fences:
{"p_ai": <float between 0.0 and 1.0>, "rationale": "<one short sentence>"}

p_ai is your probability estimate that the text is AI-generated. Use the \
full range of the scale -- avoid defaulting to 0.5 unless the text is \
genuinely ambiguous."""


class LLMClassifierError(Exception):
    """Raised when the Groq call fails or returns an unparseable response."""


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMClassifierError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def score(text: str) -> float:
    """
    Return P(AI-generated) in [0, 1] for `text`, per the LLM classifier
    signal in Section 4.

    Raises LLMClassifierError on API failure or an unparseable response,
    so callers (aggregate.py) can decide how to handle a signal outage
    (e.g. fall back to the other signal alone) rather than silently
    treating a failure as a real 0.5 reading.
    """
    if not text or not text.strip():
        raise LLMClassifierError("empty text")

    client = _client()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
    except Exception as exc:  # network/auth/rate-limit errors from Groq SDK
        raise LLMClassifierError(f"Groq request failed: {exc}") from exc

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        p_ai = float(parsed["p_ai"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise LLMClassifierError(f"could not parse model output: {raw!r}") from exc

    if not 0.0 <= p_ai <= 1.0:
        raise LLMClassifierError(f"p_ai out of range: {p_ai}")

    return p_ai
