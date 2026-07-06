# Content Attribution & Transparency System

A service that analyzes submitted text (poems, story excerpts, blog posts)
and returns an attribution result — likely AI-generated, likely human, or
genuinely uncertain — with a confidence score and a plain-language label
suitable for showing directly to a reader.

## Tech stack

| Component | Tool | Notes |
|---|---|---|
| API framework | Flask | Free, lightweight |
| Detection signal 1 | Stylometric heuristics | Pure Python, no external libraries needed |
| Detection signal 2 | Groq (`llama-3.3-70b-versatile`) | Free tier |
| Rate limiting | Flask-Limiter | Free |
| Audit log | SQLite (stdlib `sqlite3`) | No additional setup |

## Architecture overview

1. **`POST /submit`** receives the text and immediately passes it through
   a **rate limiter**. Over the limit → rejected with `429`, nothing else runs.
2. The text is normalized (whitespace collapsed) and hashed into a
   `content_id` — this becomes the record's ID everywhere downstream.
   (This is currently a small inline step in `api/routes.py`, not a
   separate preprocessing module — real normalization decisions like
   case folding and unicode handling are out of scope for this pass.)
3. Two independent signals run against the cleaned text:
   - **Stylometric signal**: vocabulary diversity (Guiraud's Index) +
     burstiness (sentence-length variance) — pure statistical measures,
     no external calls.
   - **LLM classifier signal**: Groq (`llama-3.3-70b-versatile`) judges
     the text holistically.
4. The **aggregator** combines both into a `raw_score` (P(AI-generated),
   0–1) and a `confidence` (how far that score is from a coin flip). See
   **Confidence scoring** below for exactly how — this is a documented
   heuristic combination, not a fitted/calibrated model.
5. The **label generator** maps `(raw_score, confidence)` to exactly one
   of three fixed label strings (below) — never free text.
6. The **audit logger** writes a structured entry (signals, scores,
   label) for every submission, no exceptions.
7. The response is returned to the caller with the result, score, and label.

Separately, creators can dispute a decision via **`POST /appeal/{content_id}`**,
which logs their reasoning against the original decision and flips the
content's status to `under_review`. Appeals are joined into `GET /log`
output directly.

## Detection signals

### Signal 1: Stylometric (vocabulary diversity + burstiness)

**What it measures.** Vocabulary diversity, via **Guiraud's Index**
(`unique_words / sqrt(total_words)`): how much a piece of writing repeats
itself relative to its length. Burstiness: variance in sentence length
across the piece — a mix of short, punchy sentences and long, meandering
ones vs. a uniform rhythm throughout.

**Why choose it.** Generative models tend toward safe, repetitive
phrasing and even sentence structure — output that sits in a narrow,
low-surprise band. Human writers make less "optimal" choices: varied
vocabulary, a sentence that runs on and then one that doesn't,
inconsistent register. That unevenness shows up as higher vocabulary
diversity and higher burstiness.

An earlier version of this signal used pseudo-perplexity against a small
hardcoded table of common English words instead of vocabulary diversity.
It was replaced after direct testing showed it penalized any text using
vocabulary outside that table — a formulaic AI corporate-speak sample
scored as *more* human-like than a real human narrative, purely because
words like "leveraging" and "stakeholders" weren't in the reference
table. Guiraud's Index needs no reference vocabulary, so it doesn't have
that failure mode.

**What it misses.**
- Needs enough text to be statistically stable — under ~50 words,
  vocabulary-diversity and burstiness estimates are noisy. The signal
  flags this itself (`low_reliability`); see **Confidence scoring**.
- Vocabulary diversity is still length-confounded even after the
  Guiraud correction, just less severely than a raw ratio would be — a
  longer AI sample can out-score a shorter human sample on this
  sub-signal for reasons of length alone, not word choice. Confirmed
  directly in testing (see **Known limitations**).
- The sentence-length-variance measure uses a naive punctuation-based
  splitter, which can misread short, casual, punctuation-light sentences
  as "uniform" (AI-like) when they're actually just casual speech.
- Naturally formulaic *human* writing (legal boilerplate, fixed poetic
  forms, ESL sentence patterns) reads as "smooth" and can false-positive
  as AI, even though the smoothness has nothing to do with generation.
- No semantic understanding at all — a fluent but factually strange AI
  passage can still score as "human-like" if its surface statistics
  happen to be irregular.

### Signal 2: LLM classifier (Groq `llama-3.3-70b-versatile`)

**What it measures.** A holistic judgment call: the model is prompted to
assess tone, cliché density, coherence, and "voice" consistency, and to
return its own probability estimate of AI origin.

**Why choose it.** The classifier has, in effect, seen a large amount
of both AI and human text during its own training and has picked up on
patterns that don't reduce to token-level statistics — AI text tends
toward safe, generic phrasing and a lack of concrete or personal detail;
human text tends to carry idiosyncratic voice and detail that's harder
to fake at the word/sentence-structure level alone. In testing, this
signal consistently ranked samples in the correct direction even when
the stylometric signal didn't.

**What it misses.**
- It's a subjective judgment formatted as a probability, not a
  principled decision boundary — it can be overconfident, and can give
  different estimates on different runs of the same text.
- Biased toward flagging *any* clean, well-edited writing as AI —
  penalizes skilled human writers, ESL writers, and anyone who ran their
  own work through a grammar checker.
- Susceptible to adversarial gaming: a few deliberately "quirky" errors
  inserted into AI text can shift its judgment.
- No actual visibility into provenance, it's just estimating typicality
  against patterns it's seen, not detecting generation directly.

**Why two signals, not one.** Each has different blind spots and
some of those blind spots overlap (both signals can key off
"polish" as an AI tell), so they aren't always independent votes.
Aggregating them raises the bar for a confident label but does not
eliminate correlated failure.

## Confidence scoring

- `raw_score` = P(AI-generated), 0–1.
- `confidence` = `2 * |raw_score - 0.5|` — distance from a 50/50 guess,
  rescaled to 0–1. A raw_score of 0.51 yields confidence ≈ 0.02 (near
  meaningless); a raw_score of 0.95 yields confidence = 0.90 (strong).
- Labeling thresholds: `confidence >= 0.7` → high-confidence label (AI or
  human, by sign of raw_score vs. 0.5); otherwise → uncertain.

**How the two signals are combined.** `raw_score` is produced by a
documented heuristic, not a calibrated model:
- Default: a plain 50/50 average of both signal scores.
- When the stylometric signal flags itself as `low_reliability` (under
  ~50 tokens), it's down-weighted to 20%, with the LLM classifier at
  80% — using a flag the signal already computes, not a new invented
  threshold. Testing surfaced concrete short-text cases where
  stylometric disagreed with ground truth while the LLM classifier
  ranked correctly, so leaning on the LLM signal specifically in the
  regime stylometric already admits it's unreliable in is a direct
  response to that evidence.

A real calibration model — fitting a logistic regression on labeled
`(signal_1, signal_2) → ground_truth` pairs — was part of the original
design and the code path for it was prototyped, but building an actual
labeled dataset of known-AI/known-human text was out of scope for this
assignment. The heuristic above is what actually runs.

**How we tested whether the scores mean anything.** Rather than a
calibration curve (which needs labeled data we don't have), we ran the
combined pipeline against real submissions of known/expected type —
clearly AI-generated text, clearly casual human writing, and several
deliberately extreme/unambiguous constructed examples — and checked
whether confidence actually tracked how unambiguous the input was. The
result: confidence rarely reached the high-confidence threshold except
in the most extreme cases, and high-confidence-human was never reached
in testing at all, even for inputs specifically constructed to be
maximally unambiguous. That's a real, reproducible finding about the
system's current behavior (see **Known limitations**), not a claim that
the scores are statistically well-calibrated.

**Two real example submissions, with noticeably different confidence:**

*High-confidence example* — a short, formulaic AI-generated paragraph
(corporate-jargon style, under 50 tokens, so reliability-weighted 20/80):
```
signals: {"llm_classifier": 0.9, "stylometric": 0.875}
raw_score: 0.895
confidence: 0.790
label: "This content appears to be AI-generated (confidence: high)."
```

*Low-confidence example* — a casual, first-person restaurant review
(over 50 tokens, plain 50/50 average):
```
signals: {"llm_classifier": 0.2, "stylometric": 0.720}
raw_score: 0.460
confidence: 0.080
label: "We could not confidently determine whether this content is
        AI-generated or human-created. Treat this result as inconclusive."
```

## Transparency labels (verbatim)

These are the exact three strings the label generator can return.

| Case | Label text shown to reader |
|---|---|
| High-confidence AI | `"This content appears to be AI-generated (confidence: high)."` |
| High-confidence human | `"This content appears to be human-created (confidence: high)."` |
| Uncertain | `"We could not confidently determine whether this content is AI-generated or human-created. Treat this result as inconclusive."` |

## Appeals workflow

`POST /appeal/{content_id}` with a body containing the creator's written
reasoning. This:
1. Looks up the original audit log entry by `content_id` (404 if unknown).
2. Writes a new appeal record referencing that entry (reasoning + timestamp).
3. Updates the content's status to `under_review`.

No automated re-classification happens — an appeal is a flag for human
review, not a re-run of the pipeline.

**Reviewer queue.** `GET /appeals?status=under_review` returns open
appeals with the original decision and the appeal reasoning side by
side. `GET /log?content_id=...` also shows the appeal directly embedded
in that entry, so appeals are visible both ways — through the reviewer
queue and through the audit log itself.

## Known limitations

**Casual/informal short-form text is consistently misclassified by the
stylometric signal**
Across multiple real test submissions (a casual restaurant review, a
casual personal story, and my own reflective writing), the stylometric 
signal scored *higher* on the AI-likelihood scale than clearly AI-generated samples did — in one case
reaching the maximum possible value (1.0) on real human writing. This
happens because casual, conversational text tends to reuse simple words
and run in short, similarly-sized clauses, which both sub-metrics
(vocabulary diversity and burstiness) read as "smooth"/AI-like, even
though the actual cause is register, not generation. The LLM classifier
signal did not share this failure in the same cases — it's specifically
a stylometric blind spot.

**A related, structural limitation:** because the system currently uses
heuristic averaging rather than a fitted calibration model (see
**Confidence scoring**), reaching a high-confidence label requires both
signals to be simultaneously near-extreme. In testing, high-confidence
AI was reached only once, on a short, maximally unambiguous input, and
high-confidence human was never reached at all — even for inputs
specifically constructed to be unambiguous. The system is structurally
more capable of confidently flagging AI-generated text than confidently
vindicating human-written text, under the current uncalibrated setup.

**A secondary, anticipated (not directly tested) scenario:** a human 
written poem using heavy repetition and simple vocabulary would
likely trigger the same "smooth" misread described above, for the same
underlying reason (repetition and simple vocabulary suppress both
sub-metrics), even though the repetition is a deliberate literary
device rather than a generation artifact.

## Rate limiting

- **10 requests per minute**, per client IP, enforced via
  **Flask-Limiter**, applied only to `POST /submit`.

**Reasoning:** `/submit` is the only endpoint that triggers a real,
paid external API call (Groq) plus local compute on every request — the
read endpoints (`GET /log`, `GET /appeals`) are local SQLite reads with
no comparable cost, so they aren't rate-limited here. 10/minute is
generous enough for a real reader or a manual test session (nobody
legitimately submits more than roughly one piece of content every few
seconds) but low enough to make a scripted abuse attempt slow and
noticeable rather than free and instant. This uses Flask-Limiter's
default in-memory storage, which resets on restart and doesn't share
state across multiple worker processes — acceptable for a single-process
dev/demo deployment

## Audit log

Every attribution decision is written as a structured entry (SQLite,
`storage/audit_log.py`), viewable via `GET /log`. Real entries from
testing:

```json
[
  {
    "content_id": "524f9a",
    "signals": {"llm_classifier": 0.9, "stylometric": 0.5},
    "raw_score": 0.7,
    "confidence": 0.4,
    "label": "We could not confidently determine whether this content is AI-generated or human-created. Treat this result as inconclusive.",
    "appeal": {
      "reasoning": "I wrote this myself, no AI tool involved.",
      "status": "under_review",
      "submitted_at": "2026-07-06T03:29:08.989462+00:00"
    }
  },
  {
    "content_id": "fff4ce",
    "signals": {"llm_classifier": 0.8, "stylometric": 1.0},
    "raw_score": 0.84,
    "confidence": 0.68,
    "label": "We could not confidently determine whether this content is AI-generated or human-created. Treat this result as inconclusive.",
    "appeal": null
  },
  {
    "content_id": "819676",
    "signals": {"llm_classifier": 0.2, "stylometric": 0.7198672168136984},
    "raw_score": 0.45993368840684916,
    "confidence": 0.08013278318630168,
    "label": "We could not confidently determine whether this content is AI-generated or human-created. Treat this result as inconclusive.",
    "appeal": null
  }
]
```

## Project structure

```
api/
  routes.py                 # Flask: /submit, /appeal/{id}, /log, /appeals
middleware/
  rate_limit.py              # Flask-Limiter (10/min on /submit)
pipeline/
  aggregate.py                # combines both signals -> raw_score + confidence
  label.py                    # maps (raw_score, confidence) -> 1 of 3 label strings
  signals/
    stylometric.py             # vocabulary diversity (Guiraud) + burstiness
    llm_classifier.py           # Groq llama-3.3-70b-versatile
storage/
  audit_log.py                # SQLite audit log
  appeals.py                   # SQLite appeal records, linked by content_id
tests/
  test_signals.py              # unit + integration tests
conftest.py
planning.md
README.md
```

## Spec reflection


## AI usage

