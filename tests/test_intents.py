"""Tests for the offline heuristic labeler + labeler evaluation."""

from __future__ import annotations

import json

import pytest

from datagen.schema import INTENTS, SENTIMENTS
from intents.prompts import heuristic_label, mock_response

SAMPLE_TEXTS = [
    "We are seriously considering whether to cancel and not renew next quarter.",
    "Can we add more seats and upgrade to the enterprise tier for another team?",
    "How do I configure the dashboard? I'm confused about the setup docs.",
    "I keep getting a 500 error and the webhook is broken, data not syncing.",
    "Just checking in, everything is fine.",  # keyword-free -> catch-all
    "",  # empty text must still return a valid label
]


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_heuristic_label_returns_valid_labels(text):
    label = heuristic_label(text)
    assert set(label.keys()) == {"intent", "sentiment"}
    assert label["intent"] in INTENTS
    assert label["sentiment"] in SENTIMENTS


def test_heuristic_label_is_deterministic():
    text = "The value isn't there anymore, we may move off to a competitor."
    assert heuristic_label(text) == heuristic_label(text)


def test_heuristic_label_detects_obvious_intents():
    assert heuristic_label("Please cancel our subscription.")["intent"] == "cancel-risk"
    assert heuristic_label("We want to add more seats and upgrade.")["intent"] == "expansion-signal"


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_mock_response_is_parseable_json(text):
    payload = json.loads(mock_response(text))
    assert payload["intent"] in INTENTS
    assert payload["sentiment"] in SENTIMENTS
    # mock_response is just the JSON serialization of heuristic_label
    assert payload == heuristic_label(text)


def test_evaluate_labeler_structure(intent_eval):
    for section in ("intent", "sentiment"):
        assert section in intent_eval
        for metric in ("accuracy", "cohen_kappa", "macro_f1"):
            assert metric in intent_eval[section]


def test_evaluate_labeler_intent_quality(intent_eval):
    """Loose, realistic bounds: good-but-imperfect agreement with gold labels."""
    intent = intent_eval["intent"]
    assert 0.6 <= intent["cohen_kappa"] <= 1.0
    assert 0.7 <= intent["accuracy"] <= 1.0
    assert 0.0 <= intent["macro_f1"] <= 1.0


def test_evaluate_labeler_sentiment_quality(intent_eval):
    sentiment = intent_eval["sentiment"]
    assert 0.0 <= sentiment["cohen_kappa"] <= 1.0
    assert 0.5 <= sentiment["accuracy"] <= 1.0
