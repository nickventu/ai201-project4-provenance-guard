"""
Rate limiting (Section 3a: middleware/rate_limit.py).

Uses Flask-Limiter (Section 3 tech stack: "Free"), keyed by remote
address. Applied to POST /submit specifically, since that's the endpoint
that costs real money/latency (it calls the Groq API on every request)
and is the only one meaningfully abusable at volume.

Limit chosen: 10 requests/minute per IP.

Reasoning (expand on this in the README, per the rubric's
"document the limits you chose and your reasoning" requirement):
  - /submit is the only endpoint that hits an external paid API
    (Groq) per call, so it's the one worth protecting against both
    abuse and runaway cost.
  - 10/min is generous enough for a real reader or a manual test
    session (nobody legitimately submits more than ~1 piece of content
    every few seconds), but low enough to make a scripted abuse attempt
    slow and easy to notice rather than free and instant.
  - Read endpoints (GET /log, GET /appeals) are not rate-limited here:
    they're local SQLite reads with no external API cost, so the
    abuse/cost case that justifies limiting /submit doesn't apply the
    same way. Revisit if this ever sits behind a public, unauthenticated
    deployment.
  - Section 6 open question ("Persist rate limit state (Redis vs.
    in-memory for v1)") is still open: this uses Flask-Limiter's default
    in-memory storage, which resets on restart and doesn't share state
    across multiple app processes/workers. Fine for a single-process
    dev/demo deployment; revisit before running multiple workers.
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

SUBMIT_RATE_LIMIT = "10 per minute"

limiter = Limiter(key_func=get_remote_address)
