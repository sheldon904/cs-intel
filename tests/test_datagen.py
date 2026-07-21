"""Tests for the synthetic data generator + schema constants."""

from __future__ import annotations

import numpy as np
import pandas as pd

from datagen.generate import GenParams, generate
from datagen.schema import (
    INTENTS,
    SENTIMENTS,
    AccountCols,
    Intent,
    MonthlyCols,
    Sentiment,
    TicketCols,
)

GT_KEYS = {
    "true_ate_risk_diff",
    "naive_diff_risk_diff",
    "churn_base_rate",
    "params",
    "confounder",
    "treatment",
    "outcome",
    "n_accounts",
}


def test_generate_is_deterministic_for_same_seed():
    """Same seed + params -> byte-identical accounts and identical churn rate."""
    a = generate(GenParams(seed=7, n_accounts=60), write=False)
    b = generate(GenParams(seed=7, n_accounts=60), write=False)

    assert a["ground_truth"]["churn_base_rate"] == b["ground_truth"]["churn_base_rate"]
    pd.testing.assert_frame_equal(a["accounts"], b["accounts"])


def test_generate_differs_for_different_seed():
    """Different seeds must produce different realized churn outcomes."""
    a = generate(GenParams(seed=7, n_accounts=60), write=False)
    b = generate(GenParams(seed=99, n_accounts=60), write=False)

    churned_a = a["accounts"][AccountCols.CHURNED].to_numpy()
    churned_b = b["accounts"][AccountCols.CHURNED].to_numpy()
    assert not np.array_equal(churned_a, churned_b)


def test_generate_shapes_and_columns(tiny_gen):
    """generate() returns the expected frames/keys with the expected columns."""
    assert set(tiny_gen.keys()) == {"accounts", "monthly", "tickets", "ground_truth"}

    accounts = tiny_gen["accounts"]
    assert len(accounts) == 60
    for col in (
        AccountCols.ID,
        AccountCols.SEATS,
        AccountCols.MRR,
        AccountCols.CHURNED,
        AccountCols.HIGH_ADOPTION,
        "renewal_date",
    ):
        assert col in accounts.columns

    monthly = tiny_gen["monthly"]
    for col in (MonthlyCols.ID, MonthlyCols.MONTH, MonthlyCols.ADOPTION, MonthlyCols.TICKETS):
        assert col in monthly.columns

    tickets = tiny_gen["tickets"]
    for col in (TicketCols.ID, TicketCols.TEXT, TicketCols.TRUE_INTENT, TicketCols.TRUE_SENTIMENT):
        assert col in tickets.columns


def test_ground_truth_has_expected_keys(tiny_gen):
    gt = tiny_gen["ground_truth"]
    assert GT_KEYS.issubset(gt.keys())
    assert gt["n_accounts"] == 60
    assert gt["outcome"] == AccountCols.CHURNED


def test_churn_rate_in_plausible_range(tiny_gen):
    """The realized churn base rate should be genuinely stochastic, not degenerate."""
    rate = tiny_gen["ground_truth"]["churn_base_rate"]
    assert 0.1 < rate < 0.6
    # sanity: it equals the mean of the churned column
    assert rate == float(tiny_gen["accounts"][AccountCols.CHURNED].mean())


def test_true_intents_and_sentiments_are_valid(tiny_gen):
    tickets = tiny_gen["tickets"]
    assert set(tickets[TicketCols.TRUE_INTENT].unique()).issubset(set(INTENTS))
    assert set(tickets[TicketCols.TRUE_SENTIMENT].unique()).issubset(set(SENTIMENTS))


def test_schema_enum_lists_match_enums():
    assert [i.value for i in Intent] == INTENTS
    assert [s.value for s in Sentiment] == SENTIMENTS
    assert len(INTENTS) == 4
    assert len(SENTIMENTS) == 3
    # a couple of canonical column-name anchors
    assert AccountCols.ID == "account_id"
    assert AccountCols.CHURNED == "churned"
    assert TicketCols.TRUE_INTENT == "true_intent"
