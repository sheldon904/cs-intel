"""Read-side data access over the pipeline artifacts.

Assembles the per-account view (churn risk + SHAP drivers + intent-tagged
tickets + usage series) consumed by both the dashboard API and the NBA agent.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import (
    ACCOUNT_INTENT_FEATURES_PARQUET,
    ACCOUNTS_PARQUET,
    CAUSAL_RESULTS_JSON,
    CHURN_METRICS_JSON,
    CHURN_SCORES_PARQUET,
    GROUND_TRUTH_JSON,
    INTENT_EVAL_JSON,
    LLMOPS_SUMMARY_JSON,
    MONTHLY_PARQUET,
    TICKET_LABELS_PARQUET,
    TICKETS_PARQUET,
)
from datagen.schema import AccountCols, MonthlyCols, TicketCols


def _to_native(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    return v


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): _to_native(v) for k, v in row.items()}


class Repository:
    """Loads the parquet/JSON artifacts once and serves account views."""

    def __init__(self) -> None:
        self.accounts = pd.read_parquet(ACCOUNTS_PARQUET)
        self.scores = pd.read_parquet(CHURN_SCORES_PARQUET)
        self.tickets = pd.read_parquet(TICKETS_PARQUET)
        self.monthly = pd.read_parquet(MONTHLY_PARQUET)
        self.labels = (
            pd.read_parquet(TICKET_LABELS_PARQUET)
            if TICKET_LABELS_PARQUET.exists()
            else pd.DataFrame()
        )
        self.intent_features = (
            pd.read_parquet(ACCOUNT_INTENT_FEATURES_PARQUET)
            if ACCOUNT_INTENT_FEATURES_PARQUET.exists()
            else pd.DataFrame()
        )
        self._merged = self.accounts.merge(self.scores, on=AccountCols.ID, how="left")

    # -- collections --------------------------------------------------------
    def list_accounts(
        self, risk_tier: str | None = None, sort_by_risk: bool = True
    ) -> list[dict[str, Any]]:
        df = self._merged
        cols = [
            AccountCols.ID, AccountCols.NAME, AccountCols.SEGMENT, AccountCols.INDUSTRY,
            AccountCols.SEATS, AccountCols.MRR, "renewal_date", "churn_probability",
            "risk_tier", "actual_churned",
        ]
        out = df[cols].copy()
        if risk_tier:
            out = out[out["risk_tier"] == risk_tier]
        if sort_by_risk:
            out = out.sort_values("churn_probability", ascending=False)
        return [_row_to_dict(r) for _, r in out.iterrows()]

    # -- single account -----------------------------------------------------
    def get_account_detail(self, account_id: str, n_tickets: int = 8) -> dict[str, Any] | None:
        acc = self.accounts[self.accounts[AccountCols.ID] == account_id]
        if acc.empty:
            return None
        acc_row = acc.iloc[0]
        score = self.scores[self.scores[AccountCols.ID] == account_id]
        score_row = score.iloc[0] if not score.empty else None

        top_factors = []
        churn_probability = 0.0
        risk_tier = "Low"
        if score_row is not None:
            churn_probability = float(score_row["churn_probability"])
            risk_tier = str(score_row["risk_tier"])
            top_factors = json.loads(score_row["shap_top"])

        recent_tickets = self._recent_tickets(account_id, n_tickets)
        intent_summary = self._intent_summary(account_id)
        series = self._usage_series(account_id)

        return {
            "account_id": account_id,
            "company_name": str(acc_row[AccountCols.NAME]),
            "segment": str(acc_row[AccountCols.SEGMENT]),
            "industry": str(acc_row[AccountCols.INDUSTRY]),
            "seats_contracted": int(acc_row[AccountCols.SEATS]),
            "mrr": float(acc_row[AccountCols.MRR]),
            "renewal_date": _to_native(acc_row["renewal_date"]),
            "tenure_months": int(acc_row[AccountCols.TENURE]),
            "actual_churned": int(acc_row[AccountCols.CHURNED]),
            "churn_probability": churn_probability,
            "risk_tier": risk_tier,
            "top_factors": top_factors,
            "recent_tickets": recent_tickets,
            "intent_summary": intent_summary,
            "usage_series": series,
        }

    def _recent_tickets(self, account_id: str, n: int) -> list[dict[str, Any]]:
        tk = self.tickets[self.tickets[TicketCols.ACCOUNT] == account_id]
        if tk.empty:
            return []
        if not self.labels.empty:
            tk = tk.merge(
                self.labels[[TicketCols.ID, "pred_intent", "pred_sentiment"]],
                on=TicketCols.ID,
                how="left",
            )
        tk = tk.sort_values(TicketCols.MONTH, ascending=False).head(n)
        out = []
        for _, r in tk.iterrows():
            out.append(
                {
                    "ticket_id": r[TicketCols.ID],
                    "month": int(r[TicketCols.MONTH]),
                    "text": str(r[TicketCols.TEXT]),
                    "intent": str(r.get("pred_intent", r[TicketCols.TRUE_INTENT])),
                    "sentiment": str(r.get("pred_sentiment", r[TicketCols.TRUE_SENTIMENT])),
                }
            )
        return out

    def _intent_summary(self, account_id: str) -> dict[str, float]:
        if self.intent_features.empty:
            return {}
        row = self.intent_features[self.intent_features[AccountCols.ID] == account_id]
        if row.empty:
            return {}
        return {str(k): _to_native(v) for k, v in row.iloc[0].items() if k != AccountCols.ID}

    def _usage_series(self, account_id: str) -> list[dict[str, Any]]:
        m = self.monthly[self.monthly[MonthlyCols.ID] == account_id].sort_values(MonthlyCols.MONTH)
        cols = [
            MonthlyCols.MONTH, MonthlyCols.ADOPTION, MonthlyCols.TICKETS,
            MonthlyCols.NPS, MonthlyCols.AVG_SENTIMENT, MonthlyCols.ACTIVE_SEATS,
        ]
        return [_row_to_dict(r) for _, r in m[cols].iterrows()]

    # -- pipeline artifacts -------------------------------------------------
    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def churn_metrics(self) -> dict[str, Any]:
        return self._load_json(CHURN_METRICS_JSON)

    def causal_results(self) -> dict[str, Any]:
        return self._load_json(CAUSAL_RESULTS_JSON)

    def intent_eval(self) -> dict[str, Any]:
        return self._load_json(INTENT_EVAL_JSON)

    def ground_truth(self) -> dict[str, Any]:
        return self._load_json(GROUND_TRUTH_JSON)

    def llmops_summary(self) -> dict[str, Any]:
        return self._load_json(LLMOPS_SUMMARY_JSON)

    def portfolio_summary(self) -> dict[str, Any]:
        df = self._merged
        return {
            "n_accounts": int(len(df)),
            "n_high_risk": int((df["risk_tier"] == "High").sum()),
            "n_medium_risk": int((df["risk_tier"] == "Medium").sum()),
            "n_low_risk": int((df["risk_tier"] == "Low").sum()),
            "mrr_at_risk": float(df.loc[df["risk_tier"] == "High", AccountCols.MRR].sum()),
            "avg_churn_probability": float(df["churn_probability"].mean()),
        }


@lru_cache(maxsize=1)
def get_repository() -> Repository:
    return Repository()
