"""Evaluate labeler quality on the frozen 100-ticket gold set.

Reports accuracy and Cohen's kappa (chance-corrected agreement) for both intent
and sentiment, plus a per-class breakdown and a confusion matrix — the same
agreement-metric vocabulary used to judge human annotators.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from common.paths import INTENT_EVAL_JSON, TICKET_LABELS_PARQUET, ensure_dirs
from datagen.schema import INTENTS, SENTIMENTS, TicketCols


def _agreement(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    acc = float((pd.Series(y_true).to_numpy() == pd.Series(y_pred).to_numpy()).mean())
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    per_class_f1 = {
        lbl: float(
            f1_score(
                [t == lbl for t in y_true],
                [p == lbl for p in y_pred],
                zero_division=0,
            )
        )
        for lbl in labels
    }
    return {
        "accuracy": acc,
        "cohen_kappa": kappa,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "labels": labels,
        "confusion_matrix": cm,
    }


def evaluate_labeler() -> dict:
    """Compare predicted labels to gold on the frozen eval set."""
    ensure_dirs()
    labels = pd.read_parquet(TICKET_LABELS_PARQUET)
    gold = labels[labels[TicketCols.IN_GOLD_EVAL]].copy()

    intent_eval = _agreement(
        gold[TicketCols.TRUE_INTENT].tolist(), gold["pred_intent"].tolist(), INTENTS
    )
    sentiment_eval = _agreement(
        gold[TicketCols.TRUE_SENTIMENT].tolist(), gold["pred_sentiment"].tolist(), SENTIMENTS
    )

    result = {
        "gold_set_size": int(len(gold)),
        "note": (
            "Gold labels come from the generative process (synthetic tickets), "
            "i.e. a perfect annotator. Kappa/accuracy measure how well the "
            "LLM/heuristic labeler recovers those labels."
        ),
        "intent": intent_eval,
        "sentiment": sentiment_eval,
    }
    INTENT_EVAL_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
