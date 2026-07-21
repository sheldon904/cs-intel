"""Train + persist the churn model and score every account.

Produces:
  * ``artifacts/churn_model.joblib``   – fitted XGBoost + logistic + metadata
  * ``artifacts/churn_metrics.json``   – AUC/PR-AUC/Brier for both models + lift
  * ``artifacts/{calibration,pr,shap_summary}.png``
  * ``data/churn_scores.parquet``      – per-account risk + top SHAP contributions
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churn.evaluate import (
    ShapExplainer,
    calibration_points,
    classification_metrics,
    plot_calibration,
    plot_pr_curves,
    plot_shap_summary,
    pr_curve_points,
)
from churn.features import FeatureTable, build_feature_table
from common.paths import (
    CHURN_METRICS_JSON,
    CHURN_MODEL_PKL,
    CHURN_SCORES_PARQUET,
    ensure_dirs,
)
from datagen.schema import AccountCols

RANDOM_STATE = 42


def _fit_logistic(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _make_xgb() -> XGBClassifier:
    # Tuned for a ~500-account dataset: shallow trees + strong regularization to
    # exploit the size x friction interaction without overfitting the small n.
    return XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=4,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )


def _fit_xgb(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    model = _make_xgb()
    model.fit(X, y)
    return model


def _split_xy(ft: FeatureTable, features: list[str]) -> dict[str, np.ndarray]:
    df = ft.df
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    return {
        "X_train": tr[features].to_numpy(dtype=float),
        "y_train": tr[AccountCols.CHURNED].to_numpy(dtype=int),
        "X_test": te[features].to_numpy(dtype=float),
        "y_test": te[AccountCols.CHURNED].to_numpy(dtype=int),
    }


def _risk_tier(p: float) -> str:
    if p >= 0.50:
        return "High"
    if p >= 0.25:
        return "Medium"
    return "Low"


def _cv_auc(ft: FeatureTable, features: list[str], seeds: tuple[int, ...]) -> float:
    """Mean out-of-fold AUC across several 5-fold shuffles (stable lift estimate)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = ft.df[features].to_numpy(dtype=float)
    y = ft.df[AccountCols.CHURNED].to_numpy(dtype=int)
    scores = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        oof = cross_val_predict(_make_xgb(), X, y, cv=skf, method="predict_proba")[:, 1]
        scores.append(float(roc_auc_score(y, oof)))
    return float(np.mean(scores))


def compute_intent_lift(ft: FeatureTable, seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any] | None:
    """Measure the AUC lift from intent features via cross-validated OOF AUC.

    A single temporal holdout (~150 accounts) is too noisy to read a few-point
    feature lift, so lift is estimated with repeated 5-fold cross-validation
    (headline metrics still use the temporal split). Same XGBoost config in both.
    """
    if not ft.intent_features:
        return None
    auc_base = _cv_auc(ft, ft.base_features, seeds)
    auc_full = _cv_auc(ft, ft.all_features, seeds)
    return {
        "method": "mean OOF AUC over repeated 5-fold CV (seeds " + str(list(seeds)) + ")",
        "auc_base": auc_base,
        "auc_with_intents": auc_full,
        "auc_lift": auc_full - auc_base,
        "intent_features_added": ft.intent_features,
    }


def train_and_save() -> dict[str, Any]:
    ensure_dirs()
    ft = build_feature_table(include_intents=True)
    features = ft.all_features
    data = _split_xy(ft, features)

    logistic = _fit_logistic(data["X_train"], data["y_train"])
    xgb = _fit_xgb(data["X_train"], data["y_train"])

    p_log = logistic.predict_proba(data["X_test"])[:, 1]
    p_xgb = xgb.predict_proba(data["X_test"])[:, 1]
    met_log = classification_metrics(data["y_test"], p_log)
    met_xgb = classification_metrics(data["y_test"], p_xgb)

    # Calibration + PR plots (both models).
    plot_calibration(
        {
            "XGBoost": calibration_points(data["y_test"], p_xgb),
            "Logistic": calibration_points(data["y_test"], p_log),
        }
    )
    plot_pr_curves(
        {
            "XGBoost": pr_curve_points(data["y_test"], p_xgb),
            "Logistic": pr_curve_points(data["y_test"], p_log),
        }
    )

    # SHAP (global summary over the full population, computed on the XGB model).
    explainer = ShapExplainer(xgb, features)
    X_all = ft.df[features].to_numpy(dtype=float)
    plot_shap_summary(explainer, X_all)

    shap_all = explainer.shap_matrix(X_all)
    global_importance = {
        f: float(v) for f, v in zip(features, np.abs(shap_all).mean(axis=0), strict=True)
    }
    global_importance = dict(
        sorted(global_importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    # Per-account scores + top SHAP contributions for the dashboard.
    probs = xgb.predict_proba(X_all)[:, 1]
    contributions = []
    for i in range(len(ft.df)):
        sv = shap_all[i]
        order = np.argsort(np.abs(sv))[::-1][:6]
        contributions.append(
            json.dumps(
                [
                    {
                        "feature": features[j],
                        "shap_value": round(float(sv[j]), 4),
                        "value": round(float(X_all[i, j]), 4),
                        "direction": "increases risk" if sv[j] > 0 else "decreases risk",
                    }
                    for j in order
                ]
            )
        )
    scores = pd.DataFrame(
        {
            AccountCols.ID: ft.df[AccountCols.ID].to_numpy(),
            "churn_probability": probs,
            "risk_tier": [_risk_tier(p) for p in probs],
            "actual_churned": ft.df[AccountCols.CHURNED].to_numpy(),
            "split": ft.df["split"].to_numpy(),
            "shap_top": contributions,
            "shap_base_value": explainer.base_value,
        }
    )
    scores.to_parquet(CHURN_SCORES_PARQUET, index=False)

    lift = compute_intent_lift(ft)

    metrics = {
        "temporal_split": {
            "train_n": int((ft.df["split"] == "train").sum()),
            "test_n": int((ft.df["split"] == "test").sum()),
            "test_fraction": 0.30,
            "method": "accounts ordered by renewal date; latest 30% held out",
        },
        "xgboost": met_xgb,
        "logistic": met_log,
        "intent_lift": lift,
        "shap_global_importance": dict(list(global_importance.items())[:15]),
        "n_features": len(features),
        "features_used": features,
    }
    CHURN_METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    bundle = {
        "xgb": xgb,
        "logistic": logistic,
        "features": features,
        "base_features": ft.base_features,
        "intent_features": ft.intent_features,
        "shap_base_value": explainer.base_value,
    }
    joblib.dump(bundle, CHURN_MODEL_PKL)

    return metrics


def load_bundle() -> dict[str, Any]:
    return joblib.load(CHURN_MODEL_PKL)


if __name__ == "__main__":
    m = train_and_save()
    xgb, log = m["xgboost"], m["logistic"]
    print(f"XGBoost  AUC={xgb['auc']:.3f}  PR-AUC={xgb['pr_auc']:.3f}  Brier={xgb['brier']:.3f}")
    print(f"Logistic AUC={log['auc']:.3f}  PR-AUC={log['pr_auc']:.3f}  Brier={log['brier']:.3f}")
    if m["intent_lift"]:
        lift = m["intent_lift"]
        print(
            f"Intent lift: AUC {lift['auc_base']:.3f} -> {lift['auc_with_intents']:.3f} "
            f"({lift['auc_lift']:+.3f})"
        )
