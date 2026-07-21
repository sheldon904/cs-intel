"""Column names, categorical vocabularies, and label enums for the dataset.

Keeping these in one module means the churn/intents/causal/agent stages all
reference the same string constants instead of re-typing column names.
"""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    CANCEL_RISK = "cancel-risk"
    EXPANSION_SIGNAL = "expansion-signal"
    CONFUSION = "confusion"
    BUG_REPORT = "bug-report"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


INTENTS: list[str] = [i.value for i in Intent]
SENTIMENTS: list[str] = [s.value for s in Sentiment]

SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]

INDUSTRIES = [
    "SaaS",
    "Fintech",
    "Healthcare",
    "E-commerce",
    "Manufacturing",
    "Media",
    "Logistics",
    "Education",
]

# Product feature names referenced by usage events + ticket text.
PRODUCT_FEATURES = [
    "Dashboards",
    "Automations",
    "API access",
    "Team Roles",
    "Reporting",
    "Integrations",
    "Alerts",
    "Audit Log",
]

# Fragments used to synthesize plausible B2B company names.
COMPANY_PREFIXES = [
    "North", "Blue", "Bright", "Vertex", "Quant", "Nimbus", "Iron", "Cedar",
    "Summit", "Delta", "Orbit", "Pine", "Coral", "Atlas", "Echo", "Lumen",
    "Silver", "Onyx", "Harbor", "Meridian",
]
COMPANY_SUFFIXES = [
    "Labs", "Systems", "Works", "Group", "Digital", "Analytics", "Health",
    "Logistics", "Retail", "Networks", "Software", "Partners", "Solutions",
]


# ---- canonical column names ----------------------------------------------
class AccountCols:
    ID = "account_id"
    NAME = "company_name"
    INDUSTRY = "industry"
    SEGMENT = "segment"
    SEATS = "seats_contracted"
    MRR = "mrr"
    RENEWAL_MONTH = "renewal_month"
    TENURE = "tenure_months"
    CHURNED = "churned"
    # latent variables preserved for the causal ground truth (never fed to the model)
    Z_SIZE = "z_size"
    FRICTION_LATENT = "friction_latent"
    INTENT_LATENT = "intent_latent"
    MEAN_ADOPTION = "mean_adoption"
    HIGH_ADOPTION = "high_adoption"


class MonthlyCols:
    ID = "account_id"
    MONTH = "month"
    ACTIVE_SEATS = "active_seats"
    ADOPTION = "adoption_rate"
    LOGINS = "logins"
    FEATURE_EVENTS = "feature_events"
    TICKETS = "tickets_opened"
    AVG_SENTIMENT = "avg_sentiment_score"
    NPS = "nps"
    IS_ACTIVE = "is_active"


class TicketCols:
    ID = "ticket_id"
    ACCOUNT = "account_id"
    MONTH = "month"
    TEXT = "text"
    TRUE_INTENT = "true_intent"
    TRUE_SENTIMENT = "true_sentiment"
    IN_GOLD_EVAL = "in_gold_eval"
