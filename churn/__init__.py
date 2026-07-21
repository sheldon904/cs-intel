"""Supervised churn model: logistic baseline + gradient-boosted trees.

Emphasis on visible ML hygiene: a strict *temporal* train/test split (features
are built only from months strictly before each account's renewal, and training
renewals all precede test renewals in calendar time), plus AUC / PR-AUC /
calibration / SHAP reporting.
"""
