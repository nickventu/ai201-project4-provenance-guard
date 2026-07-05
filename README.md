# Content Attribution & Transparency System

A service that analyzes submitted text (poems, story excerpts, blog posts)
and returns an attribution result — likely AI-generated, likely human, or
genuinely uncertain — with a confidence score and a plain-language label
suitable for showing directly to a reader.

## Tech stack

| Component | Tool | Notes |
|---|---|---|
| API framework | Flask | Free, lightweight |
| Detection signal 1 | Groq (`llama-3.3-70b-versatile`) | Free tier |
| Detection signal 2 | Stylometric heuristics | Pure Python, no external libraries needed |
| Rate limiting | Flask-Limiter | Free |
| Audit log | SQLite (built-in) or structured JSON | No additional setup |

## How a submission flows through the system

1. **`POST /submit`** receives the text and immediately passes it through
   a **rate limiter**. Over the limit → rejected with 429, nothing else runs.
2. The **preprocessor** normalizes the text and computes a content hash
   (this becomes the record's ID everywhere downstream).
3. Two independent signals run in parallel:
   - **Stylometric signal**: perplexity + burstiness (statistical/predictability).
   - **LLM classifier signal**: Groq (`llama-3.3-70b-versatile`) judges the text holistically.
4. The **aggregator** combines both into a `raw_score` (P(AI-generated),
   0–1) and a `confidence` (how far that score is from a coin flip).
5. The **label generator** maps `(raw_score, confidence)` to exactly one
   of three fixed label strings (below) — never free text.
6. The **audit logger** writes a structured entry (signals, scores,
   label) for every submission, no exceptions.
7. The response is returned to the caller with the result, score, and label.

Separately, creators can dispute a decision via **`POST /appeal/{content_id}`**,
which logs their reasoning against the original decision and flips the
content's status to `under_review`.

## Why two signals

### Signal 1: Stylometric (perplexity + burstiness)

**What it measures.** Perplexity: how predictable each word is, on
average, given the words before it, scored against a reference language
model. Burstiness: how much sentence length and structure vary across
the piece — a mix of short, punchy sentences and long, meandering ones
vs. a uniform rhythm throughout.

**Why it differs human vs. AI.** Generative models are trained to
produce the *most locally probable* next token, so their output tends to
sit in a narrow, low-surprise band — smooth, evenly-paced, few
structural surprises. Human writers make less "optimal" choices: odd
word order, tangents, a sentence that runs on and then one that doesn't,
inconsistent register. That unevenness shows up directly as higher
perplexity and higher burstiness.

**What it can't capture.**
- Needs enough text to be statistically stable — under ~50-100 words,
  perplexity/burstiness estimates are noisy and unreliable.
- Naturally formulaic *human* writing (legal boilerplate, fixed poetic
  forms like sonnets or villanelles, ESL writers with simpler and more
  regular sentence structures) reads as "smooth" and can false-positive
  as AI, even though the smoothness has nothing to do with generation.
- Heavily human-edited AI drafts (or AI drafts polished by a human) can
  pick up enough irregularity to false-negative as human.
- It has no idea what the text *means* — a fluent but factually or
  semantically strange AI passage can still score as "human-like" if the
  surface statistics happen to be irregular.
- Sensitive to which reference language model computes the perplexity;
  a mismatch between that model's training data and the text's genre
  (e.g. free verse poetry) skews the estimate.

### Signal 2: LLM classifier (Groq `llama-3.3-70b-versatile`)

**What it measures.** A holistic judgment call: the model is prompted to
assess tone, cliché density, thematic coherence, and "voice"
consistency, and to return its own probability estimate of AI origin.

**Why it differs human vs. AI.** The classifier has, in effect, seen a
huge amount of both AI and human text during its own training, and has
picked up on patterns that don't reduce to token-level statistics — AI
text tends toward safe, generic phrasing, over-explaining, hedging, and
a lack of specific concrete or personal detail; human text tends to
carry idiosyncratic voice, concrete anecdote, and emotional messiness
that's hard to fake at the sentence-structure level alone.

**What it can't capture.**
- It's a subjective judgment formatted as a probability, not a
  principled decision boundary — it can be overconfident, and can give
  different estimates on different runs of the same text.
- Biased toward flagging *any* clean, well-edited writing as AI —
  penalizes skilled human writers, ESL writers, and anyone who ran their
  own work through a grammar checker.
- Susceptible to adversarial gaming: a few deliberately "quirky" errors
  inserted into AI text can shift its judgment.
- No actual visibility into provenance — it's estimating typicality
  against patterns it's seen, not detecting generation directly.
- Uneven exposure across genres: it likely has seen far more AI-written
  corporate blog copy than AI-written free verse, so its judgment is
  more reliable in genres it has more reference points for.

A single signal is not acceptable here because each has different blind
spots — and, importantly, some of those blind spots overlap (both
signals key off "polish" and "smoothness" as an AI tell), which means
they aren't always independent votes. Aggregating them raises the bar
for a confident label, but does not eliminate correlated failure —
see the worked example below.

## Worked example: a false positive

**Scenario.** A human poet submits an original villanelle — a strict,
repeating-refrain form — that they also ran through a grammar checker
before submitting.

1. Request clears the rate limiter and preprocessor; gets `content_id = f3a9c1`.
2. **Stylometric signal** sees a fixed refrain structure, even line
   lengths, and grammar-checker-smoothed phrasing → low burstiness, low
   perplexity → estimates `P(AI) = 0.93`. The signal has no concept of
   "villanelle"; strict form *looks* like generation-style uniformity to it.
3. **LLM classifier** (Groq `llama-3.3-70b-versatile`) reads clean, competent, slightly generic imagery
   (typical of the form itself, not of AI) and flags it as "safe" and
   "polished" → estimates `P(AI) = 0.87`.
4. **Aggregator**: `raw_score = 0.90`, `confidence = 2*|0.90-0.5| = 0.80`.
5. **Label generator**: confidence 0.80 clears the 0.7 threshold, raw_score
   > 0.5 → **high-confidence AI** label is shown, even though the poem
   is entirely human-written.

**How the confidence score reflects (and fails to reflect) uncertainty.**
The confidence score measures *how strongly the two signals agree*, not
whether that agreement is correct. Here both signals independently
misfire for the same underlying reason — they both treat polish and
structural regularity as an AI tell — so their agreement is confident
but wrong. This is a correlated-blind-spot failure, not a case of one
signal catching what the other missed. It's exactly what the calibration
testing described above is meant to surface: if "high-confidence AI"
buckets in a labeled test set contain a meaningful fraction of true
human writing in strict forms, that's a sign the thresholds or signal
weights need adjusting, not that the poet did something wrong.

**What the label says.** The reader (and the poet) see, verbatim:
`"This content appears to be AI-generated (confidence: high)."`

**How the creator appeals.** The poet calls
`POST /appeal/f3a9c1` with their reasoning, e.g. *"This is an original
villanelle I wrote myself. I only used a grammar checker for typos; no
generation tool was used."* The appeals handler:
1. Looks up the audit log entry for `f3a9c1`.
2. Writes a new appeal record (reasoning + timestamp) referencing that entry.
3. Sets the content's status to `under_review`.

No automatic re-scoring happens. The audit log entry for `f3a9c1` now
shows the appeal alongside the original decision (see the audit log
example below), so a human reviewer sees both the original signals *and*
the poet's stated context — including the detail the automated pipeline
had no way to know: that the form itself explains the smoothness.

## Confidence scoring

- `raw_score` = P(AI-generated), 0–1 — the *output of a calibration
  model*, not a raw average of the two signals (see below).
- `confidence` = `2 * |raw_score - 0.5|` — distance from a 50/50 guess,
  rescaled to 0–1. A raw_score of 0.51 yields confidence ≈ 0.02 (near
  meaningless); a raw_score of 0.95 yields confidence = 0.90 (strong).
- Labeling thresholds: `confidence >= 0.7` → high-confidence label (AI or
  human, by sign of raw_score vs. 0.5); otherwise → uncertain.

**What a specific score means.** A `raw_score` of 0.6 means: among past
cases where our signals produced this same output pattern, about 60%
turned out to be AI-generated — a statement about frequency in similar
cases, not a claim about the text itself. Its confidence is `2*|0.6-0.5|
= 0.2` — low — so it displays as **uncertain**, not "leaning AI." A 0.6
is barely better than a coin flip and the label says so.

**Raw signals → calibrated score.** Simply averaging the two signals'
outputs is not calibration — it assumes both signals are equally
trustworthy at every point on their scale, which isn't tested and
probably isn't true. Instead: collect a labeled set of known-AI/known-human
text, run both signals over it, and fit a small calibration model
(logistic regression / Platt scaling) mapping `(signal_1, signal_2) →
P(AI)`. That fitted model's output becomes `raw_score`. This corrects
for a signal being systematically over- or under-confident — e.g. if the
LLM classifier's 0.9-scored outputs are only right 70% of the time in
the labeled set, calibration pulls that down toward 0.7 instead of
trusting it at face value.

**How we tested whether the scores mean anything:** we ran both signals
over a held-out labeled set of known-AI and known-human samples and
plotted a calibration curve — predicted raw_score bucket vs. actual
fraction of AI samples in that bucket. A well-calibrated system should
have its "0.9 raw_score" bucket actually be ~90% AI in ground truth, not
just "usually AI." We also explicitly checked that the "uncertain" bucket
has close to 50/50 ground-truth composition — if uncertain cases were
actually 90% AI, our thresholds would be miscalibrated, not just cautious.

## Transparency labels (verbatim)

These are the exact three strings the label generator can return. There
is no fourth option and no freeform text — only these three ever reach a reader.

| Case | Label text shown to reader |
|---|---|
| High-confidence AI | `"This content appears to be AI-generated (confidence: high)."` |
| High-confidence human | `"This content appears to be human-created (confidence: high)."` |
| Uncertain | `"We could not confidently determine whether this content is AI-generated or human-created. Treat this result as inconclusive."` |

## Appeals workflow

`POST /appeal/{content_id}` with a body containing the creator's written
reasoning. This:
1. Looks up the original audit log entry by `content_id`.
2. Writes a new appeal record referencing that entry (reasoning + timestamp).
3. Updates the content's status to `under_review`.

No automated re-classification happens — an appeal is a flag for human
review, not a re-run of the pipeline.

**Reviewer queue.** `GET /appeals?status=under_review` returns open
appeals with the original decision and the appeal reasoning side by
side, so a reviewer doesn't have to cross-reference two views:

```json
[
  {
    "content_id": "f3a9c1",
    "original_decision": {
      "signals": {"stylometric": 0.93, "llm_classifier": 0.87},
      "raw_score": 0.90,
      "confidence": 0.80,
      "label": "This content appears to be AI-generated (confidence: high)."
    },
    "appeal": {
      "reasoning": "This is an original villanelle I wrote myself...",
      "submitted_at": "2026-07-02T09:15:00Z"
    },
    "status": "under_review"
  }
]
```

## Anticipated edge cases

Two specific scenarios the current design is expected to handle badly:

1. **A human poem using heavy repetition and simple vocabulary** — a
   villanelle, a children's-style poem, anything built around a
   repeated refrain. Repetition suppresses the burstiness signal and
   simple vocabulary lowers perplexity, so the stylometric signal reads
   it the same way it would read AI output, even though the repetition
   is a deliberate literary device. Worked through in detail above.
2. **A short submission under ~50-100 words.** Both signals need enough
   text to be statistically stable; perplexity/burstiness get noisy on
   short samples and the LLM classifier has little to judge beyond
   surface tone. There's currently no length-aware fallback, so a short
   piece can still get a confidently-worded label built on a weak read.

## Rate limiting

- **10 requests / minute** and **200 requests / day**, per API key, enforced via **Flask-Limiter**.

**Reasoning:** each submission triggers a live call to an external LLM
(cost per token) plus local compute for the stylometric signal. A
per-minute cap prevents burst abuse or accidental infinite loops from a
client; a per-day cap bounds worst-case cost exposure from a single key
without meaningfully affecting a legitimate individual creator submitting
their own work for review. Both numbers are intentionally conservative
for v1 and expected to move once real usage patterns are known.

## Audit log

Every attribution decision is written as a structured entry (SQLite or
structured JSON — no ORM needed for this scope), viewable via `GET /log`.
Example entries:

```json
[
  {
    "content_id": "a1b2c3d4",
    "timestamp": "2026-07-01T14:02:11Z",
    "signals": {"stylometric": 0.88, "llm_classifier": 0.91},
    "raw_score": 0.895,
    "confidence": 0.79,
    "label": "This content appears to be AI-generated (confidence: high).",
    "appeal": null
  },
  {
    "content_id": "e5f6a7b8",
    "timestamp": "2026-07-01T14:05:47Z",
    "signals": {"stylometric": 0.22, "llm_classifier": 0.15},
    "raw_score": 0.185,
    "confidence": 0.63,
    "label": "We could not confidently determine whether this content is AI-generated or human-created. Treat this result as inconclusive.",
    "appeal": null
  },
  {
    "content_id": "c9d0e1f2",
    "timestamp": "2026-07-01T14:11:03Z",
    "signals": {"stylometric": 0.10, "llm_classifier": 0.08},
    "raw_score": 0.09,
    "confidence": 0.82,
    "label": "This content appears to be human-created (confidence: high).",
    "appeal": {
      "submitted_at": "2026-07-02T09:15:00Z",
      "reasoning": "This is my own original poem; I only used a grammar checker.",
      "status": "under_review"
    }
  }
]
```

## Project structure

```
api/
  routes.py               # Flask: /submit, /appeal/{id}, /log, /appeals
middleware/
  rate_limit.py            # Flask-Limiter
pipeline/
  preprocess.py
  aggregate.py
  label.py
  signals/
    stylometric.py
    llm_classifier.py       # Groq llama-3.3-70b-versatile
storage/
  audit_log.py              # SQLite or structured JSON
  appeals.py
planning.md
README.md
```

## Status

Template/scaffold — see `planning.md` for open questions and TODOs before
implementation.
