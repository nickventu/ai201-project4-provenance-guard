"""
Stylometric signal (Signal 1).

Measures vocabulary diversity (Guiraud's Index) and burstiness (variance
in sentence length and structure). AI generation tends toward safe,
repetitive phrasing -- less unique vocabulary relative to text length
and more uniform sentence structure; human writing tends to draw on a
wider vocabulary and vary sentence length/structure more.

Vocabulary diversity is measured as Guiraud's Index (V / sqrt(N), where
V = unique word count, N = total word count) rather than raw type-token
ratio (V / N). Two iterations led here:

  1. Originally, this measured pseudo-perplexity against a hardcoded
     table of ~260 common English words. Dropped because any text using
     vocabulary outside that table (technical/business jargon, formal
     language) scored as "surprising" regardless of whether it was AI-
     or human-written -- confirmed directly: a formulaic AI corporate-
     speak sample scored as MORE human-like than a human personal
     narrative, purely because of word choice unrelated to authorship.

  2. Raw TTR (V / N) has no reference-vocabulary dependency, but is
     severely length-sensitive: for short texts (well under ~100 words),
     nearly any writer -- AI or human -- hasn't had the chance to repeat
     many words yet, so TTR clusters near its ceiling for everyone,
     contributing no real signal. Confirmed directly: six test samples
     of 39-74 words all produced TTR between 0.86-0.90, regardless of
     label.

  Guiraud's Index (V / sqrt(N)) grows more slowly with text length than
  raw TTR, which spreads scores out more usefully at short lengths --
  but IMPORTANT, documented limitation: it does NOT eliminate the length
  confound, only reduces it. A longer AI sample can still out-score a
  shorter human sample on this metric for reasons of length alone, not
  word choice. Confirmed directly: a 68-word formulaic AI sample scored
  as more vocabulary-diverse (less AI-like) than several shorter, clearly
  human samples. A proper fix requires a length-normalized/windowed
  measure (e.g. MATTR) -- deliberately not implemented here to keep this
  module dependency-free and simple; see pipeline/aggregate.py, which
  compensates by down-weighting this signal when `low_reliability` is
  True rather than trying to force a short-text-unreliable metric to be
  falsely precise via threshold-tuning.

Known blind spots (worth keeping visible in code, not just docs, since
they explain *why* this signal can be confidently wrong):
    - unreliable under ~50-100 words (flagged via low_reliability)
    - vocabulary diversity is length-confounded even after the Guiraud
      correction above -- comparing texts of very different lengths on
      this sub-signal alone is not apples-to-apples
    - naturally formulaic human writing (legal boilerplate, fixed
      poetic forms, ESL sentence patterns) reads as falsely "smooth"
    - heavily human-edited AI drafts can pick up enough irregularity to
      hide
    - no semantic understanding at all
    - burstiness's naive period/question-mark sentence splitter can
      misread informal punctuation (e.g. short, casual sentences ending
      in "?") as low sentence-length variance, understating burstiness
      on casual human text

No external libraries (per Section 3 tech stack: "Pure Python, no
external libraries needed").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

# Tunable thresholds — deliberately not hidden constants, so they can be
# adjusted alongside the aggregation weights in Section 5 without hunting
# through the function body. THESE ARE PROVISIONAL, fit by hand against a
# handful of manual examples, not a labeled dataset -- calibrate properly
# per Section 5 step 1-3 before trusting them in production.
_VOCAB_DIVERSITY_LOW = 5.8   # Guiraud's Index at/below this -> reads as maximally "smooth"/repetitive (AI-like)
_VOCAB_DIVERSITY_HIGH = 7.0  # at/above this -> reads as maximally vocabulary-diverse (human-like)
_BURSTINESS_LOW = -0.2    # at/below this -> maximally smooth (AI-like)
_BURSTINESS_HIGH = 0.3    # at/above this -> maximally bursty (human-like)

# Blind spot: unreliable below this many tokens (Section 4 / edge case #2
# in "Anticipated edge cases"). pipeline/aggregate.py uses this flag to
# down-weight this signal for short submissions rather than trusting it
# equally alongside the LLM classifier signal.
MIN_RELIABLE_TOKENS = 50


@dataclass
class StylometricResult:
    score: float                    # P(AI) in [0, 1] — combined signal_1 output
    vocab_diversity: float          # raw Guiraud's Index (unbounded, typically ~3-12)
    burstiness: float               # raw burstiness, in [-1, 1]
    vocab_diversity_score: float    # vocab diversity mapped to P(AI) sub-score, [0, 1]
    burstiness_score: float         # burstiness mapped to P(AI) sub-score, [0, 1]
    token_count: int
    sentence_count: int
    low_reliability: bool           # True if token_count < MIN_RELIABLE_TOKENS

    def as_dict(self) -> dict:
        return asdict(self)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, punctuation stripped."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for length/variance stats,
    not for anything semantic."""
    # Split on ., !, ? followed by whitespace or end of string.
    parts = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return [p for p in parts if p.strip()]


def _compute_vocab_diversity(tokens: list[str]) -> float:
    """
    Guiraud's Index: unique_words / sqrt(total_words).

    Low value -> the writer reuses the same words heavily relative to
    text length (AI-smooth, generic phrasing). High value -> broader,
    more varied vocabulary (human-idiosyncratic). No reference corpus
    needed, unlike the pseudo-perplexity approach this replaced.

    Documented confound (see module docstring): still grows with text
    length, just more slowly than raw TTR -- comparisons across very
    different lengths remain unreliable.
    """
    if not tokens:
        return 0.0
    return len(set(tokens)) / math.sqrt(len(tokens))


def _compute_burstiness(sentences: list[str]) -> float:
    """
    Burstiness of sentence length (in words), using the standard
    Goh-Barabasi formulation:

        B = (sigma - mu) / (sigma + mu)

    where mu and sigma are the mean and standard deviation of sentence
    lengths. Bounded in [-1, 1]: B -> -1 means highly regular/uniform
    sentence lengths (AI-smooth); B -> 1 means highly variable,
    "bursty" sentence lengths (human-idiosyncratic). B = 0 is a
    Poisson-like baseline.

    This is the "variance in sentence length and structure" signal.
    """
    lengths = [len(_tokenize(s)) for s in sentences if _tokenize(s)]
    if len(lengths) < 2:
        # Can't estimate variance from 0-1 sentences; treat as neutral.
        return 0.0

    n = len(lengths)
    mu = sum(lengths) / n
    variance = sum((x - mu) ** 2 for x in lengths) / n
    sigma = math.sqrt(variance)

    if sigma + mu == 0:
        return 0.0
    return (sigma - mu) / (sigma + mu)


def _map_to_unit_interval(value: float, low: float, high: float) -> float:
    """Linearly map value into [0, 1] against (low, high), clamped at
    the ends. low maps to 1.0 (max AI-likelihood), high maps to 0.0
    (max human-likelihood) — i.e. this returns a P(AI)-flavored score,
    not a raw normalization."""
    if high == low:
        return 0.5
    t = (value - low) / (high - low)
    t = max(0.0, min(1.0, t))
    return 1.0 - t


def score_text(text: str) -> StylometricResult:
    """
    Run the stylometric signal over a piece of text and return a full
    breakdown (raw vocab diversity/burstiness plus the combined P(AI)
    score).

    Combination is a plain 50/50 average of the two sub-scores — this is
    the "start 50/50, tune against labeled set" placeholder from
    Section 5/6, not a calibrated model. Real calibration (logistic
    regression / Platt scaling per Section 5) happens later in
    pipeline/aggregate.py, once both signals exist. Note: aggregate.py
    separately down-weights this WHOLE signal (not just this sub-score)
    when low_reliability is True -- that decision lives there, not here.
    """
    tokens = _tokenize(text)
    sentences = _split_sentences(text)

    vocab_diversity = _compute_vocab_diversity(tokens)
    burstiness = _compute_burstiness(sentences)

    vocab_diversity_score = _map_to_unit_interval(
        vocab_diversity, _VOCAB_DIVERSITY_LOW, _VOCAB_DIVERSITY_HIGH
    )
    burstiness_score = _map_to_unit_interval(burstiness, _BURSTINESS_LOW, _BURSTINESS_HIGH)

    combined = 0.5 * vocab_diversity_score + 0.5 * burstiness_score

    return StylometricResult(
        score=combined,
        vocab_diversity=vocab_diversity,
        burstiness=burstiness,
        vocab_diversity_score=vocab_diversity_score,
        burstiness_score=burstiness_score,
        token_count=len(tokens),
        sentence_count=len(sentences),
        low_reliability=len(tokens) < MIN_RELIABLE_TOKENS,
    )


def score(text: str) -> float:
    """
    Convenience entry point matching the "signal_1_score (0-1)" contract
    from the Architecture diagram. Prefer score_text() in
    pipeline/aggregate.py when you need the low_reliability flag too --
    use score() only for quick/manual checks that don't need it.
    """
    return score_text(text).score


if __name__ == "__main__":
    # Quick manual smoke test — the kind of check called for in the
    # M3 "Verify" step: run against a handful of known short/long,
    # human/AI-ish samples and confirm scores move in the expected
    # direction.
    samples = {
        "human_long": (
            "Honestly? I almost didn't submit this one. It's messy, the "
            "middle stanza doesn't quite land, and I rewrote the ending "
            "four times at 2am fueled by cold coffee and stubbornness. "
            "But there's a line in there about my grandmother's kitchen "
            "that I still can't read out loud without my voice cracking, "
            "so I'm sending it anyway, warts and all."
        ),
        "ai_smooth": (
            "In today's fast-paced world, it is important to prioritize "
            "self-care and personal growth. By taking small, consistent "
            "steps, individuals can achieve their goals and improve their "
            "overall well-being. It is essential to remember that progress "
            "takes time, and every step forward is a step in the right "
            "direction."
        ),
        "short": "This is a very short sample text.",
    }
    for name, text in samples.items():
        result = score_text(text)
        print(f"{name}: score={result.score:.3f} "
              f"(vocab_diversity={result.vocab_diversity:.2f}, "
              f"burstiness={result.burstiness:.2f}, "
              f"low_reliability={result.low_reliability})")

