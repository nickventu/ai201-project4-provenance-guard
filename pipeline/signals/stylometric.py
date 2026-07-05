"""
Stylometric signal (Signal 1).

Per planning.md Section 4 ("Why two signals, not one"):

    Measures perplexity (token-level predictability against a reference
    LM) and burstiness (variance in sentence length and structure). AI
    generation optimizes for locally-probable tokens, so it sits in a
    narrow, smooth, low-surprise band; human writing has idiosyncratic,
    less "optimal" word/sentence choices, which shows up as higher
    perplexity and burstiness.

Known blind spots (also Section 4 — worth keeping visible in code, not
just docs, since they explain *why* this signal can be confidently
wrong):
    - unreliable under ~50-100 words
    - naturally formulaic human writing (legal boilerplate, fixed
      poetic forms, ESL sentence patterns) reads as falsely "smooth"
    - heavily human-edited AI drafts can pick up enough irregularity to
      hide
    - no semantic understanding at all
    - sensitive to reference-LM/genre mismatch

No external libraries (per Section 3 tech stack: "Pure Python, no
external libraries needed"). Because there's no pretrained LM available
here, perplexity is approximated against a small built-in table of
common-English-word frequencies (a unigram reference model) rather than
a real trained language model. This is a deliberate simplification —
good enough to separate "smooth/generic" text from "idiosyncratic" text,
not a real perplexity estimate. Revisit if/when a real reference LM
becomes available.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Reference unigram model
#
# Rough relative frequency ranks for a set of very common English words.
# Rank position (not the exact count) is what matters: it's used to
# approximate "how locally-probable is this token" without needing a real
# trained LM. Any token not in this table is treated as low-frequency
# (i.e. surprising), which is the same behavior a real LM would show for
# rare words.
# ---------------------------------------------------------------------------
_COMMON_WORDS_BY_RANK = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "was", "are", "were", "been", "being", "has", "had", "does", "did",
    "having", "am", "should", "may", "might", "must", "shall", "need", "ought",
    "important", "essential", "many", "much", "such", "own", "same", "few", "more", "less",
    "each", "every", "both", "either", "neither", "here", "where", "why", "again", "further",
    "once", "very", "too", "still", "already", "always", "never", "often", "sometimes", "usually",
    "however", "therefore", "although", "though", "while", "since", "until", "unless", "despite", "moreover",
    "furthermore", "additionally", "consequently", "overall", "individuals", "individual", "world", "life", "step", "steps",
    "small", "large", "long", "short", "high", "low", "right", "wrong", "true", "false",
    "progress", "goal", "goals", "growth", "personal", "priority", "prioritize", "achieve", "improve",
    "remember", "consistent", "direction", "forward", "today", "tomorrow", "help", "helps",
    "found", "feel", "feels", "felt", "seem", "seems", "seemed", "become", "became", "becomes",
    "part", "kind", "sort", "point", "case", "fact", "reason", "example", "system", "process",
    "number", "level", "area", "group", "problem", "result", "change", "order", "power", "form",
    "line", "end", "hand", "eye", "eyes", "face", "family", "friend", "friends", "child",
    "children", "man", "men", "woman", "women", "night", "morning", "house", "home", "room",
    "water", "food", "money", "job", "school", "book", "story", "word", "words", "language",
    "table", "door", "window", "car", "road", "city", "country", "state", "government", "company",
    "business", "market", "price", "cost", "value", "data", "information", "study", "research", "science",
    "technology", "computer", "phone", "internet", "email", "message", "call", "talk", "speak", "said",
    "asked", "answered", "told", "wrote", "read", "written", "reading", "writing", "understand", "understood",
    "believe", "hope", "wish", "love", "hate", "fear", "afraid", "happy", "sad", "angry",
]
_WORD_RANK = {w: i + 1 for i, w in enumerate(_COMMON_WORDS_BY_RANK)}
_VOCAB_SIZE = len(_COMMON_WORDS_BY_RANK)

# Tunable thresholds — deliberately not hidden constants, so they can be
# adjusted alongside the aggregation weights in Section 5 without hunting
# through the function body. Calibrate against a labeled set, per
# Section 5 step 1-3, rather than trusting these starting values.
_PERPLEXITY_LOW = 300.0   # at/below this -> reads as maximally "smooth" (AI-like)
_PERPLEXITY_HIGH = 550.0  # at/above this -> reads as maximally "idiosyncratic" (human-like)
_BURSTINESS_LOW = -0.2    # at/below this -> maximally smooth (AI-like)
_BURSTINESS_HIGH = 0.3    # at/above this -> maximally bursty (human-like)

# Blind spot: unreliable below this many tokens (Section 4 / edge case #2
# in "Anticipated edge cases"). Not currently gated on — see planning.md
# Section 6 open question about a minimum-length "uncertain" fallback.
MIN_RELIABLE_TOKENS = 50


@dataclass
class StylometricResult:
    score: float              # P(AI) in [0, 1] — combined signal_1 output
    perplexity: float         # raw pseudo-perplexity
    burstiness: float         # raw burstiness, in [-1, 1]
    perplexity_score: float   # perplexity mapped to P(AI) sub-score, [0, 1]
    burstiness_score: float   # burstiness mapped to P(AI) sub-score, [0, 1]
    token_count: int
    sentence_count: int
    low_reliability: bool     # True if token_count < MIN_RELIABLE_TOKENS

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


def _compute_perplexity(tokens: list[str]) -> float:
    """
    Pseudo-perplexity: exp(average surprisal), where surprisal of a
    token is -log(p) under a Zipfian estimate derived from its rank in
    the reference unigram table above. Unknown tokens are treated as
    rank _VOCAB_SIZE + 1 (rarer than every listed word), matching how a
    real LM would penalize out-of-distribution words.

    Low result -> text stays in a narrow band of highly-predictable,
    common words (AI-smooth). High result -> text draws on less
    predictable vocabulary (human-idiosyncratic). This is the
    "token-level predictability against a reference LM" from Section 4.
    """
    if not tokens:
        return 0.0

    total_surprisal = 0.0
    for tok in tokens:
        rank = _WORD_RANK.get(tok, _VOCAB_SIZE + 1)
        # Zipf's law approximation: p(rank) ~ 1 / (rank * H_N), harmonic
        # normalizer folded into the log for simplicity since we only
        # need relative surprisal, not a calibrated probability.
        p = 1.0 / (rank * math.log(_VOCAB_SIZE + 2))
        p = min(p, 0.999)
        total_surprisal += -math.log(p)

    avg_surprisal = total_surprisal / len(tokens)
    return math.exp(avg_surprisal)


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

    This is the "variance in sentence length and structure" signal from
    Section 4.
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
    breakdown (raw perplexity/burstiness plus the combined P(AI) score).

    Combination is a plain 50/50 average of the two sub-scores — this is
    the "start 50/50, tune against labeled set" placeholder from
    Section 5/6, not a calibrated model. Real calibration (logistic
    regression / Platt scaling per Section 5) happens later in
    pipeline/aggregate.py, once both signals exist.
    """
    tokens = _tokenize(text)
    sentences = _split_sentences(text)

    perplexity = _compute_perplexity(tokens)
    burstiness = _compute_burstiness(sentences)

    perplexity_score = _map_to_unit_interval(perplexity, _PERPLEXITY_LOW, _PERPLEXITY_HIGH)
    burstiness_score = _map_to_unit_interval(burstiness, _BURSTINESS_LOW, _BURSTINESS_HIGH)

    combined = 0.5 * perplexity_score + 0.5 * burstiness_score

    return StylometricResult(
        score=combined,
        perplexity=perplexity,
        burstiness=burstiness,
        perplexity_score=perplexity_score,
        burstiness_score=burstiness_score,
        token_count=len(tokens),
        sentence_count=len(sentences),
        low_reliability=len(tokens) < MIN_RELIABLE_TOKENS,
    )


def score(text: str) -> float:
    """
    Convenience entry point matching the "signal_1_score (0-1)" contract
    from the Architecture diagram — this is what pipeline/aggregate.py
    (Signal Aggregator, M4) will actually call. Use score_text() instead
    when you need the breakdown for debugging, tests, or the audit log.
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
              f"(perplexity={result.perplexity:.1f}, "
              f"burstiness={result.burstiness:.2f}, "
              f"low_reliability={result.low_reliability})")
