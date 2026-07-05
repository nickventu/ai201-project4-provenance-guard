# Planning: Content Attribution & Transparency System

## 1. Problem statement

Given a piece of text-based content (poem, short story excerpt, blog post),
determine whether it was likely AI-generated or human-written, express that
determination with a genuine confidence score (not a coin-flip binary), and
show the reader a plain-language label they can trust — including telling
them honestly when the system doesn't know.

## 2. Architecture narrative

*(Full plain-English walkthrough — also included in README.md)*

Submission → Rate Limiter → Preprocessor (normalize + hash) → Multi-Signal
Detection Pipeline (Stylometric Signal + LLM Classifier Signal, run in
parallel) → Score Aggregator (raw_score + confidence) → Label Generator
(maps to one of 3 fixed label strings) → Audit Logger (writes structured
entry) → Response Builder (returns JSON to caller).

Appeals run as a side path: Appeals Handler looks up the original audit
entry by content_id, writes an appeal record, sets status to
`under_review`.

## 3. Tech stack

| Component | Tool | Notes |
|---|---|---|
| API framework | Flask | Free, lightweight |
| Detection signal 1 | Groq (`llama-3.3-70b-versatile`) | Free tier |
| Detection signal 2 | Stylometric heuristics | Pure Python, no external libraries needed |
| Rate limiting | Flask-Limiter | Free |
| Audit log | SQLite (built-in) or structured JSON | No additional setup |

## 3a. Components

| Component | Responsibility | Notes |
|---|---|---|
| `api/routes.py` | `POST /submit`, `POST /appeal/{id}`, `GET /log`, `GET /appeals` | Flask |
| `middleware/rate_limit.py` | Enforce per-key request limits | Flask-Limiter |
| `pipeline/preprocess.py` | Clean text, compute content hash | Hash = audit log key |
| `pipeline/signals/stylometric.py` | Perplexity + burstiness scoring | Signal 1, pure Python |
| `pipeline/signals/llm_classifier.py` | Groq (`llama-3.3-70b-versatile`) holistic judgment | Signal 2 |
| `pipeline/aggregate.py` | Combine signals → raw_score + confidence | Calibration model (see Section 5), tunable |
| `pipeline/label.py` | Map (raw_score, confidence) → label text | Fixed 3-way thresholding |
| `storage/audit_log.py` | Append-only structured log | SQLite (stdlib `sqlite3`) or structured JSON — no ORM needed |
| `storage/appeals.py` | Appeal records + status updates | Linked by content_id (FK or JSON key) |

## 3b. API surface

Three endpoints. Each one maps directly to a required feature — no
extras, nothing speculative.

### `POST /submit`
**Accepts:**
```json
{ "content": "string, the text to analyze" }
```
**Returns (200):**
```json
{
  "content_id": "f3a9c1",
  "attribution_result": "ai" | "human" | "uncertain",
  "confidence_score": 0.80,
  "label": "This content appears to be AI-generated (confidence: high)."
}
```
**Errors:** `400` (missing/too-short content), `429` (rate-limited).

### `POST /appeal/{content_id}`
**Accepts:**
```json
{ "reasoning": "string, why the creator disputes this" }
```
**Returns (200):**
```json
{
  "content_id": "f3a9c1",
  "status": "under_review",
  "submitted_at": "2026-07-04T10:15:00Z"
}
```
**Errors:** `404` (unknown `content_id`), `400` (missing reasoning).

### `GET /log`
**Accepts:** nothing required (optional `?content_id=` filter).
**Returns (200):** array of audit entries (signals, scores, label, appeal
if any) — see Section 7 example.

### `GET /appeals?status=under_review`
**Accepts:** nothing required (optional `?status=` filter, defaults to
`under_review`).
**Returns (200):** the reviewer queue — for each open appeal, the
original decision *and* the appeal side by side, e.g.:
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
This is a read view over the same audit log data as `GET /log` — not a
separate store — filtered and shaped for a reviewer's workflow rather
than a raw history dump.

Deliberately excluded for now: separate status-check endpoint (covered
by `GET /log?content_id=`), auth endpoints, delete/edit endpoints — none
are required by the feature list, and each one is more surface to build,
test, and explain later.

## Architecture

In the submission flow, text comes in through `POST /submit`, gets cleaned and hashed by the preprocessor, then runs through both detection signals in parallel; their scores are combined into a calibrated confidence score, mapped to one of the three fixed labels, logged, and returned to the client. In the appeal flow, a creator references an existing `content_id` with their reasoning; the handler pulls up that original decision, writes the appeal alongside it in the same audit log, flips the status to `under_review`, and confirms back to the client — no re-scoring happens automatically.

**(1) Submission flow**

```
 Client
   |
   |  raw text  { "content": "..." }
   v
POST /submit  ------------------------------------------------+
   |                                                           |
   | raw text (normalized, hashed -> content_id)               |
   v                                                           |
Preprocessor                                                   |
   |                                                           |
   | cleaned text  --------------------+                       |
   v                                   v                       |
Signal 1: Stylometric          Signal 2: LLM Classifier        |
(perplexity + burstiness)      (Groq llama-3.3-70b-versatile)  |
   |                                   |                       |
   | signal_1_score (0-1)              | signal_2_score (0-1)  |
   +-----------------+-----------------+                       |
                     v                                         |
              Score Aggregator                                 |
                     |                                         |
                     | combined_score: {raw_score, confidence} |
                     v                                         |
              Label Generator                                  |
                     |                                         |
                     | label_text (1 of 3 fixed strings)       |
                     v                                         |
              Audit Logger  <-----------------------------------+
                     |         (content_id, signals, scores, label
                     |          all written together)
                     | log_entry_id (ack)
                     v
              Response Builder
                     |
                     | { content_id, attribution_result,
                     |   confidence_score, label }
                     v
                  Client
```

**(2) Appeal flow**

```
 Client
   |
   |  { "content_id": "f3a9c1", "reasoning": "..." }
   v
POST /appeal/{content_id}
   |
   | content_id  (lookup key)
   v
Appeals Handler  ------------------------------+
   |                                           |
   | fetch original decision                   |
   v                                           |
Audit Log Store                                |
   |                                           |
   | original_entry (signals, scores, label)   |
   v                                           |
Appeals Handler (continued)                    |
   |                                           |
   | status update: "under_review"             |
   v                                           |
Audit Log Store  <------------------------------+
   |         (appeal record: reasoning + timestamp
   |          written alongside original entry)
   | write_ack
   v
Response Builder
   |
   | { content_id, status: "under_review", submitted_at }
   v
Client
```



## 4. Why two signals, not one

**Stylometric signal** — measures perplexity (token-level predictability
against a reference LM) and burstiness (variance in sentence length and
structure). AI generation optimizes for locally-probable tokens, so it
sits in a narrow, smooth, low-surprise band; human writing has
idiosyncratic, less "optimal" word/sentence choices, which shows up as
higher perplexity and burstiness. Blind spots: unreliable under
~50-100 words; naturally formulaic human writing (legal boilerplate,
fixed poetic forms, ESL sentence patterns) reads as falsely "smooth";
heavily human-edited AI drafts can pick up enough irregularity to hide;
no semantic understanding at all; sensitive to reference-LM/genre mismatch.

**LLM classifier signal** — Groq (`llama-3.3-70b-versatile`) judges tone, cliché
density, coherence, and "voice" holistically and returns its own P(AI)
estimate. AI text trends toward safe, generic, hedged phrasing and lacks
concrete/personal detail; human text carries idiosyncratic voice and
messiness. Blind spots: it's a subjective judgment, not a principled
boundary — can be overconfident or inconsistent run-to-run; biased
toward flagging *any* clean/edited writing (penalizes skilled or ESL
writers, or anyone using a grammar checker); gameable by inserting
deliberate "quirky" errors; no real visibility into provenance, only
typicality; uneven reliability across genres it's seen less AI content for.

**Why both, and why that's still not bulletproof.** Neither signal alone
is trustworthy — a strict-form human poem can look statistically
"AI-smooth," and a well-tuned model can read as human to a holistic
judge. But note the two signals share a blind spot: both treat *polish
and structural regularity* as an AI tell. Aggregating raises the bar for
a confident label but does not guarantee independence — see the worked
false-positive example below.

## 4a. Worked false-positive trace

A human poet submits an original villanelle, grammar-checked before
submission.

1. `content_id = f3a9c1` assigned by preprocessor.
2. Stylometric signal: fixed refrain + even lines + grammar-checker
   smoothing → low burstiness/perplexity → `P(AI) = 0.93`.
3. LLM classifier: reads as clean, competent, generic-in-imagery (an
   artifact of the form, not of generation) → `P(AI) = 0.87`.
4. Aggregator: `raw_score = 0.90`, `confidence = 0.80` — clears the 0.7
   threshold.
5. Label generator returns **high-confidence AI**, verbatim:
   `"This content appears to be AI-generated (confidence: high)."`

**Confidence vs. correctness:** the confidence score reflects *agreement
between signals*, not ground truth. Here both signals independently
over-weight polish/structure as an AI tell, so they agree confidently
and wrongly — a correlated failure, not two independent checks catching
different things. This is exactly the failure mode the calibration
testing (Section 5) is meant to catch: if labeled human text in strict
forms clusters into "high-confidence AI," the thresholds/weights need
adjusting.

**Appeal path:** poet calls `POST /appeal/f3a9c1` with reasoning ("I
wrote this myself; only used a grammar checker, no generation tool").
Appeals handler looks up the `f3a9c1` audit entry, writes an appeal
record against it, sets status to `under_review`. No auto re-scoring —
a human reviewer opens `GET /appeals?status=under_review`, sees the
original signals and label side by side with the poet's reasoning, and
decides manually.

## Anticipated edge cases

Two specific scenarios we expect the current design to handle badly —
not "detection can be wrong" in general, but named failure modes:

1. **A human-written poem using heavy repetition and simple, deliberate
   vocabulary** (e.g. a villanelle, a children's-style poem, or a poem
   built around anaphora — repeating an opening phrase each stanza).
   Repetition suppresses burstiness and simple vocabulary lowers
   perplexity, so the stylometric signal reads it as "smooth" the same
   way it would read AI output — even though the repetition here is a
   deliberate literary device, not a generation artifact. This is the
   scenario worked through in detail above.

2. **A short, casual piece of writing under ~50-100 words** (a
   two-stanza poem, a short blog excerpt). Both signals need enough
   text to be stable: perplexity/burstiness estimates get noisy on
   short samples, and the LLM classifier has little to go on beyond
   surface tone. The system will still return a score and a label — it
   has no length-aware fallback yet — so a short submission can get a
   confidently-worded label built on a genuinely weak read. (Open
   question in Section 6: should there be a minimum-length gate that
   forces "uncertain" below some word count, regardless of what the
   signals say?)

## Transparency labels (verbatim)

Exactly these three strings — no fourth option, no freeform text:

| Case | Label text shown to reader |
|---|---|
| High-confidence AI | `"This content appears to be AI-generated (confidence: high)."` |
| High-confidence human | `"This content appears to be human-created (confidence: high)."` |
| Uncertain | `"We could not confidently determine whether this content is AI-generated or human-created. Treat this result as inconclusive."` |

## 5. Confidence score design

**What a score means, concretely.** `raw_score` = P(AI-generated) in
[0, 1]. `confidence = 2 * |raw_score - 0.5|`, distance from a coin flip,
rescaled to [0, 1]. A `raw_score` of 0.6 means: after calibration, in
past cases where our signals produced this same output pattern, about
60% turned out to be AI-generated — it's a claim about *frequency in
similar cases*, not "60% AI by volume" or any property of the text
itself. Its `confidence` is `2*|0.6-0.5| = 0.2` — low, so a 0.6 score
lands as **uncertain**, not "leaning AI." That's intentional: 0.6 is
barely better than a guess and the label should say so, not dress it up
as a lean.

**Raw signal outputs → calibrated score.** Averaging the two signal
outputs is *not* the same as calibrating them, and a fixed 50/50 average
was only a placeholder. The actual mapping:
1. Collect a labeled set of known-AI and known-human text.
2. Run both signals over it to get `(signal_1_score, signal_2_score)`
   pairs with known ground truth.
3. Fit a simple calibration model (logistic regression / Platt scaling)
   on those pairs, so the *output* of that model — not a raw average —
   becomes `raw_score`. This corrects for a signal being systematically
   over- or under-confident (e.g. if the LLM classifier's 0.9 outputs
   are only right 70% of the time, calibration pulls that down toward 0.7).
4. Re-validate the calibration curve periodically as more labeled data
   comes in — calibration drifts if the mix of submitted content changes
   (e.g. more poetry, less blog content).

**Thresholds (tunable, documented in README):**
- `confidence >= 0.7` and `raw_score > 0.5` → high-confidence AI
- `confidence >= 0.7` and `raw_score < 0.5` → high-confidence human
- everything else → uncertain

**How to test whether scores are meaningful:** run both signals over a
held-out labeled set (known-AI and known-human samples), plot a
calibration curve (predicted raw_score vs. actual fraction AI in that
bucket), and confirm the "uncertain" bucket actually has close to 50/50
ground-truth composition — if it doesn't, the thresholds are wrong, not
just conservative.

## 6. Open questions / TODO

- [ ] Pick actual signal weights (start 50/50, tune against labeled set)
- [ ] Decide on minimum submission length (signals are unreliable <~50 words)
- [ ] Persist rate limit state (Redis vs. in-memory for v1)
- [ ] Decide whether appeals trigger a notification to a human reviewer
- [ ] Add per-signal confidence, not just aggregate, to the audit log?
- [ ] Both signals currently over-weight "polish/regularity" as an AI
      tell (see Section 4a false-positive trace) — consider adding a
      genre/form hint field to submissions, or a third signal that isn't
      correlated with structural smoothness

## AI Tool Plan

**M3 (submission endpoint + first signal).** Provide the AI tool the
"Why two signals" stylometric write-up (Section 4) and the submission-flow
ASCII diagram (Architecture). Ask it to generate a Flask app skeleton
(`api/routes.py` with `POST /submit`) and the first signal function
(`pipeline/signals/stylometric.py`, perplexity + burstiness). **Verify**
by running the stylometric function directly against a handful of known
short/long, human/AI text samples and checking the scores move in the
expected direction, before wiring it into the endpoint.

**M4 (second signal + confidence scoring).** Provide the detection
signals section (Section 4), the confidence score design section
(Section 5, including the calibration approach), and the diagram. Ask it
to generate the second signal function (`pipeline/signals/llm_classifier.py`,
calling Groq `llama-3.3-70b-versatile`) plus the aggregation/scoring
logic (`pipeline/aggregate.py`). **Verify** by running both signals
against clearly-AI and clearly-human text and confirming the combined
scores differ meaningfully between the two groups, not just in the same
direction as one signal alone.

**M5 (production layer).** Provide the transparency labels section
(verbatim strings above) and the appeals workflow (Section 3b endpoints
+ the 4a worked trace). Ask it to generate the label-generation logic
(`pipeline/label.py`) and the `/appeal` and `/appeals` endpoints.
**Verify** by constructing inputs that should land in each of the three
label buckets (high-confidence AI, high-confidence human, uncertain) and
confirming all three are actually reachable, then submitting a test
appeal and confirming the audit log entry's status flips to
`under_review` and shows up in `GET /appeals?status=under_review`.

## 7. Out of scope for v1

- Automated re-classification after appeal (explicitly not required)
- Multi-language support
- Image/audio/video content
