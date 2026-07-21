"""Tests for the causal (backdoor adjustment) layer.

Key invariant: adjusting for account size shrinks the (negative) adoption effect
toward zero, moving the estimate closer to the known structural ground truth.
"""

from __future__ import annotations

from causal.analysis import gformula_logistic, naive_effect


def test_build_causal_frame_columns(causal_frame):
    assert list(causal_frame.columns) == ["high_adoption", "churned", "log_seats"]
    assert set(causal_frame["high_adoption"].unique()).issubset({0, 1})
    assert set(causal_frame["churned"].unique()).issubset({0, 1})


def test_adjustment_shrinks_effect_toward_zero(causal_frame):
    """Naive effect is more negative than the size-adjusted g-formula estimate."""
    naive = naive_effect(causal_frame)
    adjusted = gformula_logistic(causal_frame, ["log_seats"])
    # both should indicate adoption protects against churn (negative)
    assert naive < 0
    assert adjusted < 0
    # adjusting for the size confounder pulls the estimate toward zero
    assert naive < adjusted


def test_adjusted_estimate_is_closer_to_truth(causal_frame, causal_run):
    """The backdoor-adjusted estimate beats the naive one against ground truth."""
    true_ate = causal_run["true_ate_risk_diff"]
    naive = naive_effect(causal_frame)
    adjusted = gformula_logistic(causal_frame, ["log_seats"])

    assert abs(adjusted - true_ate) < abs(naive - true_ate)


def test_run_causal_analysis_result_shape(causal_run):
    for key in (
        "true_ate_risk_diff",
        "naive_estimate",
        "backdoor_gformula_logistic",
        "refutations",
    ):
        assert key in causal_run
    # the standalone helpers agree with the full-pipeline numbers
    assert causal_run["naive_estimate"] < causal_run["backdoor_gformula_logistic"]
    assert causal_run["true_ate_risk_diff"] < 0
