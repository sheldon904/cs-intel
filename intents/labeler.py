"""Run the labeler over all tickets and derive account-level intent features."""

from __future__ import annotations

import pandas as pd

from common.llm import LLMClient, get_client
from common.paths import (
    ACCOUNT_INTENT_FEATURES_PARQUET,
    ACCOUNTS_PARQUET,
    TICKET_LABELS_PARQUET,
    TICKETS_PARQUET,
    ensure_dirs,
)
from datagen.schema import INTENTS, SENTIMENTS, AccountCols, Intent, TicketCols
from intents.prompts import SYSTEM_PROMPT, build_user_prompt, heuristic_label, mock_response

_VALID_INTENTS = set(INTENTS)
_VALID_SENTIMENTS = set(SENTIMENTS)


def label_ticket(client: LLMClient, text: str, account_id: str) -> tuple[str, str]:
    """Label one ticket; fall back to the heuristic on any invalid LLM output."""
    resp = client.complete(
        name="label_ticket",
        system=SYSTEM_PROMPT,
        user=build_user_prompt(text),
        mock=lambda _s, _u, t=text: mock_response(t),  # type: ignore[misc]
        stage="intent_labeling",
        max_tokens=64,
        metadata={"account_id": account_id},
    )
    data = resp.data or {}
    intent = str(data.get("intent", "")).strip()
    sentiment = str(data.get("sentiment", "")).strip()
    if intent not in _VALID_INTENTS or sentiment not in _VALID_SENTIMENTS:
        fallback = heuristic_label(text)
        intent = intent if intent in _VALID_INTENTS else fallback["intent"]
        sentiment = sentiment if sentiment in _VALID_SENTIMENTS else fallback["sentiment"]
    return intent, sentiment


def label_all(limit: int | None = None) -> pd.DataFrame:
    """Label every ticket and persist ``ticket_labels.parquet``."""
    ensure_dirs()
    tickets = pd.read_parquet(TICKETS_PARQUET)
    if limit is not None:
        tickets = tickets.head(limit)
    client = get_client()

    pred_intents: list[str] = []
    pred_sentiments: list[str] = []
    for row in tickets.itertuples(index=False):
        intent, sentiment = label_ticket(
            client, getattr(row, TicketCols.TEXT), getattr(row, TicketCols.ACCOUNT)
        )
        pred_intents.append(intent)
        pred_sentiments.append(sentiment)
    client.flush()

    labels = tickets[
        [TicketCols.ID, TicketCols.ACCOUNT, TicketCols.MONTH, TicketCols.TRUE_INTENT,
         TicketCols.TRUE_SENTIMENT, TicketCols.IN_GOLD_EVAL]
    ].copy()
    labels["pred_intent"] = pred_intents
    labels["pred_sentiment"] = pred_sentiments
    labels.to_parquet(TICKET_LABELS_PARQUET, index=False)
    return labels


def build_account_intent_features() -> pd.DataFrame:
    """Aggregate predicted labels into per-account intent rates.

    Rates are computed over the account's *full pre-renewal history*
    (``month <= renewal_month - 1``): still strictly leakage-free (nothing at or
    after renewal), but pooling more tickets than the churn model's 6-month
    window so the intent rates are estimated with far less noise — which is what
    lets the semantic cancel-risk signal survive and add predictive lift.
    """
    labels = pd.read_parquet(TICKET_LABELS_PARQUET)
    accounts = pd.read_parquet(ACCOUNTS_PARQUET)[[AccountCols.ID, AccountCols.RENEWAL_MONTH]]
    df = labels.merge(accounts, on=AccountCols.ID, how="left")

    hi = df[AccountCols.RENEWAL_MONTH] - 1
    win = df[df[TicketCols.MONTH] <= hi]

    rows: list[dict] = []
    for aid, grp in win.groupby(AccountCols.ID):
        n = len(grp)
        rows.append(
            {
                AccountCols.ID: aid,
                "cancel_risk_rate": float((grp["pred_intent"] == Intent.CANCEL_RISK.value).mean()),
                "expansion_signal_rate": float(
                    (grp["pred_intent"] == Intent.EXPANSION_SIGNAL.value).mean()
                ),
                "confusion_rate": float((grp["pred_intent"] == Intent.CONFUSION.value).mean()),
                "bug_report_rate": float((grp["pred_intent"] == Intent.BUG_REPORT.value).mean()),
                "intent_negative_rate": float((grp["pred_sentiment"] == "negative").mean()),
                "n_tickets_history": n,
            }
        )
    feats = pd.DataFrame(rows)
    # Accounts with no in-window tickets get zero rates.
    all_ids = accounts[[AccountCols.ID]]
    feats = all_ids.merge(feats, on=AccountCols.ID, how="left").fillna(0.0)
    feats.to_parquet(ACCOUNT_INTENT_FEATURES_PARQUET, index=False)
    return feats
