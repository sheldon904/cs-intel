"""Tests for churn feature engineering: temporal split + leakage controls."""

from __future__ import annotations

from churn.features import FEATURE_WINDOW, TEST_FRACTION, build_feature_table
from churn.train import load_bundle


def test_feature_window_constants():
    assert FEATURE_WINDOW == 6
    assert TEST_FRACTION == 0.30


def test_split_column_values(feature_table):
    df = feature_table.df
    assert "split" in df.columns
    assert set(df["split"].unique()) == {"train", "test"}


def test_both_splits_non_empty(feature_table):
    df = feature_table.df
    assert (df["split"] == "train").sum() > 0
    assert (df["split"] == "test").sum() > 0


def test_temporal_split_has_no_leakage(feature_table):
    """Train renewals must all precede (or equal at the boundary) test renewals."""
    df = feature_table.df
    train_max = df.loc[df["split"] == "train", "renewal_date"].max()
    test_min = df.loc[df["split"] == "test", "renewal_date"].min()
    assert train_max <= test_min


def test_base_features_strict_subset_of_all(feature_table):
    base = set(feature_table.base_features)
    allf = set(feature_table.all_features)
    assert base < allf  # strict subset (intents add the extra columns)


def test_intent_features_are_the_extras(feature_table):
    ft = feature_table
    assert len(ft.intent_features) > 0
    # all_features is exactly base + intents, in order, with no overlap
    assert ft.all_features == ft.base_features + ft.intent_features
    assert set(ft.intent_features).isdisjoint(set(ft.base_features))
    assert set(ft.all_features) - set(ft.base_features) == set(ft.intent_features)


def test_required_columns_present(feature_table):
    df = feature_table.df
    for col in ("churned", "renewal_date", "renewal_month", "split"):
        assert col in df.columns
    # every declared feature actually exists as a column
    for col in feature_table.all_features:
        assert col in df.columns


def test_no_nans_in_model_features(feature_table):
    """Train-median imputation should leave no NaNs in the modelling columns."""
    df = feature_table.df
    assert not df[feature_table.all_features].isna().any().any()


def test_load_bundle_has_expected_keys():
    bundle = load_bundle()
    for key in ("xgb", "logistic", "features", "base_features", "intent_features", "shap_base_value"):
        assert key in bundle
    # the saved feature list is base + intents, matching the feature table contract
    assert bundle["features"] == bundle["base_features"] + bundle["intent_features"]


def test_include_intents_false_drops_intent_features():
    ft = build_feature_table(include_intents=False)
    assert ft.intent_features == []
    assert ft.all_features == ft.base_features
