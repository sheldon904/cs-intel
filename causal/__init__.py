"""Causal layer: why the naive 'adoption reduces churn' number is a lie.

Uses DoWhy to show the naive association between feature adoption and churn is
confounded by account size, then recovers the true effect via backdoor
adjustment — comparing both against the *known* structural effect baked into the
synthetic data.
"""
