"""Free-text support-ticket synthesis.

Each ticket is generated from a hidden ``(intent, sentiment)`` pair plus a
product feature slot. Templates deliberately overlap in vocabulary (an angry
"confusion" ticket can read like a "cancel-risk" one) so that neither a keyword
heuristic nor an LLM labeler scores a trivial 100% — which makes the
accuracy / Cohen's-kappa evaluation in ``intents/`` meaningful.
"""

from __future__ import annotations

import numpy as np

from datagen.schema import Intent, Sentiment

# Openers keyed by sentiment — set the emotional tone independent of intent.
_OPENERS = {
    Sentiment.NEGATIVE: [
        "This is getting frustrating.",
        "We've raised this before and it's still broken.",
        "Honestly not happy with how this is going.",
        "Third time reaching out about this.",
        "",
    ],
    Sentiment.NEUTRAL: [
        "Quick question for the team.",
        "Following up on our setup.",
        "Hi, hoping you can help.",
        "Wanted to check in about our account.",
        "",
    ],
    Sentiment.POSITIVE: [
        "Loving the product so far!",
        "Thanks for the great support recently.",
        "The team has been really enjoying this.",
        "Big fan of what you've built.",
        "",
    ],
}

# Body templates per intent. ``{feature}`` is slot-filled with a product area.
_BODIES = {
    Intent.CANCEL_RISK: [
        "Leadership is questioning whether we should renew — {feature} just isn't delivering the ROI we expected.",
        "We're evaluating competitors and are close to moving off the platform.",
        "If {feature} keeps failing we'll have to cancel at renewal.",
        "Budget review is coming up and this contract is on the chopping block.",
        "We're seriously considering churning; the value isn't there for us anymore.",
        "Please cancel our upcoming auto-renewal while we reconsider.",
    ],
    Intent.EXPANSION_SIGNAL: [
        "We're growing fast and want to add 15 more seats — can you send pricing?",
        "Interested in upgrading to the Enterprise tier to unlock {feature}.",
        "Another department wants to adopt this; how do we roll it out to them?",
        "Can we discuss a larger contract? {feature} has been a game changer.",
        "We'd like to add the {feature} add-on for our second team.",
        "Looking to expand usage across the org next quarter.",
    ],
    Intent.CONFUSION: [
        "I can't figure out how to configure {feature} — is there a guide?",
        "Not sure why {feature} behaves differently for admins vs members.",
        "Where do I find the setting for {feature}? I've looked everywhere.",
        "The onboarding docs don't match what I see for {feature}.",
        "How is {feature} supposed to work with our SSO setup?",
        "Confused about how billing interacts with {feature}.",
    ],
    Intent.BUG_REPORT: [
        "{feature} is throwing a 500 error every time we export.",
        "Data in {feature} isn't syncing — numbers are stale since yesterday.",
        "{feature} crashes on load in Chrome for our whole team.",
        "We're seeing duplicate records in {feature} after the last release.",
        "The {feature} webhook stopped firing this morning.",
        "{feature} shows the wrong totals compared to our source system.",
    ],
}

# Deliberately cross-cutting templates: the TRUE intent is the key, but the
# surface text leans toward a *confusable neighbor* intent. These model genuine
# annotation ambiguity — a keyword labeler misfires on them, and even a strong
# LLM finds them harder — so Cohen's kappa lands realistically below 1.0.
_AMBIGUOUS_BODIES = {
    Intent.CANCEL_RISK: [  # reads like a bug report
        "Constant 500 errors on {feature} — at this point we're questioning whether to keep paying for it.",
        "{feature} keeps crashing and the team is fed up; leadership is asking hard questions about this tool.",
    ],
    Intent.BUG_REPORT: [  # reads like confusion
        "Not sure if {feature} is broken or if I'm setting it up wrong — the totals look off either way.",
        "Confused whether this {feature} behavior is a defect or just how it's supposed to work.",
    ],
    Intent.CONFUSION: [  # reads like cancel-risk
        "Still can't figure out {feature}; leadership is questioning the ROI if it stays this hard to use.",
        "No idea how {feature} is meant to work and honestly reconsidering whether this platform fits us.",
    ],
}


# Closers add a little more surface area / noise.
_CLOSERS = {
    Sentiment.NEGATIVE: [
        "Need this resolved ASAP.",
        "We expect better.",
        "This is affecting our whole team.",
        "",
    ],
    Sentiment.NEUTRAL: [
        "Let me know, thanks.",
        "Appreciate any pointers.",
        "No rush, just want to understand.",
        "",
    ],
    Sentiment.POSITIVE: [
        "Thanks again for the help!",
        "Keep up the great work.",
        "Happy to hop on a call.",
        "",
    ],
}


def generate_ticket_text(
    rng: np.random.Generator,
    intent: Intent,
    sentiment: Sentiment,
    feature: str,
    ambiguous: bool = False,
) -> str:
    """Compose one ticket body from opener + intent body + closer.

    When ``ambiguous`` is set (and the intent has confusable variants), the body
    is drawn from ``_AMBIGUOUS_BODIES`` so the text lexically resembles a
    different intent than its true label.
    """
    opener = rng.choice(_OPENERS[sentiment])
    if ambiguous and intent in _AMBIGUOUS_BODIES:
        body = rng.choice(_AMBIGUOUS_BODIES[intent]).format(feature=feature)
    else:
        body = rng.choice(_BODIES[intent]).format(feature=feature)
    closer = rng.choice(_CLOSERS[sentiment])
    parts = [p for p in (opener, body, closer) if p]
    return " ".join(parts)
