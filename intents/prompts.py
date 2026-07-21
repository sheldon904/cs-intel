"""Prompts for the labeler + the deterministic heuristic used in offline mode.

The heuristic (``heuristic_label``) is intentionally *good but imperfect*: it
scores keyword overlap per intent, and because the ticket templates share
vocabulary it misclassifies ambiguous tickets — producing a realistic Cohen's
kappa below 1.0 rather than a suspicious perfect score.
"""

from __future__ import annotations

import json

from datagen.schema import INTENTS, SENTIMENTS, Intent, Sentiment

SYSTEM_PROMPT = f"""You are a customer-success analyst labeling B2B support tickets.

Classify each ticket into exactly one INTENT and one SENTIMENT.

INTENTS ({", ".join(INTENTS)}):
- cancel-risk: the customer signals they may not renew, are evaluating
  competitors, questioning ROI/value, or asking to cancel.
- expansion-signal: the customer wants to grow — add seats, upgrade tiers,
  buy add-ons, roll out to more teams, or discuss larger contracts.
- confusion: the customer does not understand how to use or configure the
  product; how-to questions, unclear docs, setup uncertainty.
- bug-report: something is broken — errors, crashes, data not syncing, wrong
  numbers, failing webhooks.

SENTIMENT ({", ".join(SENTIMENTS)}): positive, neutral, or negative.

Respond with ONLY a JSON object: {{"intent": "<intent>", "sentiment": "<sentiment>"}}."""


def build_user_prompt(text: str) -> str:
    return f'Ticket:\n"""{text}"""\n\nReturn the JSON label.'


# ---- offline heuristic ----------------------------------------------------
_INTENT_KEYWORDS: dict[Intent, list[str]] = {
    Intent.CANCEL_RISK: [
        "cancel", "renew", "churn", "competitor", "roi", "budget", "moving off",
        "chopping block", "value isn't there", "reconsider", "not delivering",
    ],
    Intent.EXPANSION_SIGNAL: [
        "add ", "more seats", "upgrade", "enterprise tier", "expand", "pricing",
        "add-on", "larger contract", "roll it out", "another department",
        "second team", "growing",
    ],
    Intent.CONFUSION: [
        "how do i", "how is", "confused", "not sure", "where do i", "figure out",
        "docs", "guide", "supposed to work", "can't figure",
    ],
    Intent.BUG_REPORT: [
        "error", "500", "crash", "not syncing", "stale", "duplicate", "webhook",
        "wrong totals", "broken", "stopped", "isn't syncing",
    ],
}

_POS_WORDS = ["loving", "great", "enjoy", "big fan", "keep up", "happy", "thanks for the great"]
_NEG_WORDS = [
    "frustrat", "not happy", "broken", "asap", "expect better", "third time",
    "still broken", "raised this before", "affecting our whole",
]


def heuristic_label(text: str) -> dict[str, str]:
    """Keyword-scoring fallback classifier. Deterministic, no network."""
    low = text.lower()
    scores = {
        intent: sum(1 for kw in kws if kw in low) for intent, kws in _INTENT_KEYWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        best = Intent.CONFUSION  # neutral catch-all for keyword-free tickets

    pos = sum(1 for w in _POS_WORDS if w in low)
    neg = sum(1 for w in _NEG_WORDS if w in low)
    if neg > pos:
        sentiment = Sentiment.NEGATIVE
    elif pos > neg:
        sentiment = Sentiment.POSITIVE
    else:
        sentiment = Sentiment.NEUTRAL

    return {"intent": best.value, "sentiment": sentiment.value}


def mock_response(text: str) -> str:
    """Return the heuristic label as the JSON string an LLM would produce."""
    return json.dumps(heuristic_label(text))
