"""Evaluation + explainability for the churn model.

Reports the metrics the model card promises — ROC-AUC, PR-AUC, Brier score,
a calibration curve — plus SHAP feature attributions (global summary and
per-account contributions used by the dashboard).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display needed inside containers/CI
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from common.paths import CALIBRATION_PNG, PR_CURVE_PNG, SHAP_SUMMARY_PNG


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def plot_calibration(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path = CALIBRATION_PNG
) -> None:
    """``curves`` maps model name -> (prob_true, prob_pred)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfectly calibrated")
    for name, (prob_true, prob_pred) in curves.items():
        ax.plot(prob_pred, prob_true, "o-", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn frequency")
    ax.set_title("Calibration (temporal test set)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def calibration_points(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return prob_true, prob_pred


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path = PR_CURVE_PNG
) -> None:
    """``curves`` maps model name -> (precision, recall, pr_auc)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for name, (precision, recall, ap) in curves.items():
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (temporal test set)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    return precision, recall, ap


class ShapExplainer:
    """Thin wrapper around ``shap.TreeExplainer`` for a fitted XGBoost model."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        base = self.explainer.expected_value
        self.base_value = float(np.ravel([base])[0])

    def shap_matrix(self, X: np.ndarray) -> np.ndarray:
        vals = self.explainer.shap_values(X)
        if isinstance(vals, list):  # some versions return per-class lists
            vals = vals[-1]
        return np.asarray(vals)

    def top_contributions(self, x_row: np.ndarray, k: int = 6) -> list[dict[str, Any]]:
        """Ranked per-account SHAP contributions for the dashboard explanation."""
        sv = self.shap_matrix(x_row.reshape(1, -1))[0]
        order = np.argsort(np.abs(sv))[::-1][:k]
        return [
            {
                "feature": self.feature_names[i],
                "shap_value": float(sv[i]),
                "value": float(x_row[i]),
                "direction": "increases risk" if sv[i] > 0 else "decreases risk",
            }
            for i in order
        ]


def plot_shap_summary(
    explainer: ShapExplainer, X: np.ndarray, path: Path = SHAP_SUMMARY_PNG, max_display: int = 14
) -> None:
    sv = explainer.shap_matrix(X)
    fig = plt.figure()
    shap.summary_plot(
        sv,
        features=X,
        feature_names=explainer.feature_names,
        max_display=max_display,
        show=False,
        plot_size=(7.5, 6.0),
    )
    plt.title("SHAP feature importance (churn model)")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
