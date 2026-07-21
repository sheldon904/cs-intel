"""Feature engineering for the churn model — with leakage controls baked in.

For each account we aggregate the ``feature_window`` months *strictly before*
its renewal month (``[renewal_month - window, renewal_month - 1]``). Because
churn is decided *at* the renewal month, nothing from the renewal month or later
enters the features. The train/test split is then temporal: accounts are ordered
by renewal date and the later-renewing tail is held out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.paths import (
    ACCOUNT_INTENT_FEATURES_PARQUET,
    ACCOUNTS_PARQUET,
    MONTHLY_PARQUET,
)
from datagen.schema import INDUSTRIES, SEGMENTS, AccountCols, MonthlyCols

FEATURE_WINDOW = 6
TEST_FRACTION = 0.30  # later-renewing tail held out

# Feature groups (kept explicit so the base-vs-enhanced lift study can drop the
# intent features cleanly).
STATIC_NUMERIC = ["log_seats", "seats_contracted", "mrr", "tenure_months"]
USAGE_NUMERIC = [
    "avg_adoption",
    "last_adoption",
    "adoption_trend",
    "active_seat_ratio",
    "avg_logins",
    "avg_feature_events",
]
SUPPORT_NUMERIC = [
    "tickets_total",
    "tickets_per_month",
    "avg_sentiment",
    "pct_months_with_tickets",
    "nps_last",
    "nps_mean",
]
INTENT_FEATURES = [
    "cancel_risk_rate",
    "expansion_signal_rate",
    "confusion_rate",
    "bug_report_rate",
    "intent_negative_rate",
]
CATEG_ONEHOT = [f"segment_{s}" for s in SEGMENTS] + [f"industry_{i}" for i in INDUSTRIES]

TARGET = AccountCols.CHURNED


@dataclass(slots=True)
class FeatureTable:
    df: pd.DataFrame
    base_features: list[str]
    all_features: list[str]
    intent_features: list[str]


def _slope(y: np.ndarray) -> float:
    """OLS slope of ``y`` vs its own index; 0 when fewer than 2 points."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x * (y - y.mean())).sum() / denom) if denom else 0.0


def _window_features(acct_month: pd.DataFrame, renewal_month: int, seats: int) -> dict:
    lo, hi = renewal_month - FEATURE_WINDOW, renewal_month - 1
    win = acct_month[(acct_month[MonthlyCols.MONTH] >= lo) & (acct_month[MonthlyCols.MONTH] <= hi)]
    win = win.sort_values(MonthlyCols.MONTH)
    if win.empty:  # pathological guard; renewal_month is always > window
        win = acct_month.sort_values(MonthlyCols.MONTH).tail(FEATURE_WINDOW)

    adoption = win[MonthlyCols.ADOPTION].to_numpy()
    tickets = win[MonthlyCols.TICKETS].to_numpy()
    sentiment = win[MonthlyCols.AVG_SENTIMENT].to_numpy()
    nps = win[MonthlyCols.NPS].dropna().to_numpy()
    n_months = max(1, len(win))

    return {
        "avg_adoption": float(np.mean(adoption)),
        "last_adoption": float(adoption[-1]),
        "adoption_trend": _slope(adoption),
        "active_seat_ratio": float(win[MonthlyCols.ACTIVE_SEATS].mean() / max(1, seats)),
        "avg_logins": float(win[MonthlyCols.LOGINS].mean()),
        "avg_feature_events": float(win[MonthlyCols.FEATURE_EVENTS].mean()),
        "tickets_total": float(tickets.sum()),
        "tickets_per_month": float(tickets.sum() / n_months),
        "avg_sentiment": float(np.nanmean(sentiment)) if np.isfinite(sentiment).any() else 0.0,
        "pct_months_with_tickets": float((tickets > 0).mean()),
        "nps_last": float(nps[-1]) if len(nps) else np.nan,
        "nps_mean": float(np.mean(nps)) if len(nps) else np.nan,
    }


def build_feature_table(include_intents: bool = True) -> FeatureTable:
    """Assemble the per-account modelling table with a temporal split column."""
    accounts = pd.read_parquet(ACCOUNTS_PARQUET)
    monthly = pd.read_parquet(MONTHLY_PARQUET)

    rows: list[dict] = []
    for acct in accounts.itertuples(index=False):
        aid = getattr(acct, AccountCols.ID)
        am = monthly[monthly[MonthlyCols.ID] == aid]
        feats = _window_features(
            am, int(getattr(acct, AccountCols.RENEWAL_MONTH)), int(getattr(acct, AccountCols.SEATS))
        )
        feats.update(
            {
                AccountCols.ID: aid,
                "log_seats": float(np.log(getattr(acct, AccountCols.SEATS))),
                "seats_contracted": int(getattr(acct, AccountCols.SEATS)),
                "mrr": float(getattr(acct, AccountCols.MRR)),
                "tenure_months": int(getattr(acct, AccountCols.TENURE)),
                "segment": getattr(acct, AccountCols.SEGMENT),
                "industry": getattr(acct, AccountCols.INDUSTRY),
                "renewal_date": acct.renewal_date,
                "renewal_month": int(getattr(acct, AccountCols.RENEWAL_MONTH)),
                TARGET: int(getattr(acct, AccountCols.CHURNED)),
            }
        )
        rows.append(feats)

    df = pd.DataFrame(rows)

    # One-hot encode categoricals against the fixed vocabularies.
    df["segment"] = pd.Categorical(df["segment"], categories=SEGMENTS)
    df["industry"] = pd.Categorical(df["industry"], categories=INDUSTRIES)
    df = pd.concat(
        [
            df,
            pd.get_dummies(df["segment"], prefix="segment").astype(int),
            pd.get_dummies(df["industry"], prefix="industry").astype(int),
        ],
        axis=1,
    )

    # Impute NPS (missing when no survey landed in-window) with the train median.
    base_features = STATIC_NUMERIC + USAGE_NUMERIC + SUPPORT_NUMERIC + CATEG_ONEHOT
    intent_features: list[str] = []

    if include_intents and ACCOUNT_INTENT_FEATURES_PARQUET.exists():
        intent_df = pd.read_parquet(ACCOUNT_INTENT_FEATURES_PARQUET)
        df = df.merge(intent_df, on=AccountCols.ID, how="left")
        intent_features = [c for c in INTENT_FEATURES if c in df.columns]
        df[intent_features] = df[intent_features].fillna(0.0)

    all_features = base_features + intent_features

    # Temporal split: order by renewal date, hold out the latest TEST_FRACTION.
    df = df.sort_values(["renewal_date", AccountCols.ID]).reset_index(drop=True)
    cutoff = df["renewal_date"].quantile(1 - TEST_FRACTION)
    df["split"] = np.where(df["renewal_date"] <= cutoff, "train", "test")

    # Median-impute remaining NaNs using TRAIN statistics only (no leakage).
    train_medians = df.loc[df["split"] == "train", all_features].median(numeric_only=True)
    df[all_features] = df[all_features].fillna(train_medians)

    return FeatureTable(
        df=df, base_features=base_features, all_features=all_features, intent_features=intent_features
    )
