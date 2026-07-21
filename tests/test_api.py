"""Tests for the read-side Repository over the pre-built artifacts."""

from __future__ import annotations

DETAIL_KEYS = {
    "account_id",
    "company_name",
    "segment",
    "industry",
    "seats_contracted",
    "mrr",
    "renewal_date",
    "churn_probability",
    "risk_tier",
    "top_factors",
    "recent_tickets",
    "usage_series",
}


def test_list_accounts_non_empty(repo):
    accounts = repo.list_accounts()
    assert len(accounts) > 0
    first = accounts[0]
    assert "account_id" in first
    assert "churn_probability" in first
    assert "risk_tier" in first


def test_list_accounts_sorted_by_risk_descending(repo):
    probs = [a["churn_probability"] for a in repo.list_accounts(sort_by_risk=True)]
    assert probs == sorted(probs, reverse=True)


def test_get_account_detail_unknown_returns_none(repo):
    assert repo.get_account_detail("DOES-NOT-EXIST-9999") is None


def test_get_account_detail_real_id_returns_expected_dict(repo):
    account_id = repo.list_accounts()[0]["account_id"]
    detail = repo.get_account_detail(account_id)
    assert isinstance(detail, dict)
    assert DETAIL_KEYS.issubset(detail.keys())
    assert detail["account_id"] == account_id
    assert isinstance(detail["top_factors"], list)
    assert isinstance(detail["recent_tickets"], list)
    assert 0.0 <= detail["churn_probability"] <= 1.0


def test_portfolio_summary_counts_sum_to_total(repo):
    summary = repo.portfolio_summary()
    tier_sum = summary["n_high_risk"] + summary["n_medium_risk"] + summary["n_low_risk"]
    assert tier_sum == summary["n_accounts"]
    assert summary["n_accounts"] == len(repo.list_accounts())
    assert 0.0 <= summary["avg_churn_probability"] <= 1.0


def test_churn_metrics_available(repo):
    metrics = repo.churn_metrics()
    assert "xgboost" in metrics
    assert "temporal_split" in metrics


def test_causal_results_available(repo):
    results = repo.causal_results()
    assert "true_ate_risk_diff" in results
    assert "naive_estimate" in results
    assert "backdoor_gformula_logistic" in results
