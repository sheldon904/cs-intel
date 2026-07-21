"""Canonical filesystem locations for datasets and model artifacts.

Every stage of the pipeline reads and writes through these constants so the
data contract between ``datagen`` -> ``churn`` -> ``intents`` -> ``causal`` ->
``agent`` -> ``api`` stays in exactly one place.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = parent of the ``common`` package directory.
ROOT = Path(__file__).resolve().parent.parent

# Allow containers / CI to relocate the writable data + artifact roots.
DATA_DIR = Path(os.environ.get("CS_INTEL_DATA_DIR", ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("CS_INTEL_ARTIFACTS_DIR", ROOT / "artifacts"))
MEMORY_DIR = Path(os.environ.get("CS_INTEL_MEMORY_DIR", ROOT / "agent" / "memory_store"))

# ---- datagen outputs ------------------------------------------------------
ACCOUNTS_PARQUET = DATA_DIR / "accounts.parquet"
MONTHLY_PARQUET = DATA_DIR / "monthly.parquet"          # account x month panel
TICKETS_PARQUET = DATA_DIR / "tickets.parquet"          # free-text support tickets
GROUND_TRUTH_JSON = DATA_DIR / "ground_truth.json"      # known causal effect + params

# ---- churn outputs --------------------------------------------------------
CHURN_MODEL_PKL = ARTIFACTS_DIR / "churn_model.joblib"
CHURN_FEATURES_PARQUET = DATA_DIR / "churn_features.parquet"
CHURN_METRICS_JSON = ARTIFACTS_DIR / "churn_metrics.json"
CHURN_SCORES_PARQUET = DATA_DIR / "churn_scores.parquet"       # per-account risk + SHAP
CALIBRATION_PNG = ARTIFACTS_DIR / "calibration_curve.png"
SHAP_SUMMARY_PNG = ARTIFACTS_DIR / "shap_summary.png"
PR_CURVE_PNG = ARTIFACTS_DIR / "pr_curve.png"

# ---- intents outputs ------------------------------------------------------
TICKET_LABELS_PARQUET = DATA_DIR / "ticket_labels.parquet"     # LLM/heuristic labels
INTENT_EVAL_JSON = ARTIFACTS_DIR / "intent_eval.json"          # kappa/accuracy vs gold
ACCOUNT_INTENT_FEATURES_PARQUET = DATA_DIR / "account_intent_features.parquet"

# ---- causal outputs -------------------------------------------------------
CAUSAL_RESULTS_JSON = ARTIFACTS_DIR / "causal_results.json"

# ---- agent memory + proposals ---------------------------------------------
MEMORY_DECISIONS_JSONL = MEMORY_DIR / "decisions.jsonl"       # learned operator decisions
PROPOSALS_JSONL = MEMORY_DIR / "proposals.jsonl"              # every agent proposal (audit)

# ---- LLMOps outputs -------------------------------------------------------
LLM_USAGE_JSONL = ARTIFACTS_DIR / "llm_usage.jsonl"           # every traced call
LLMOPS_SUMMARY_JSON = ARTIFACTS_DIR / "llmops_summary.json"   # cost/account, eval over time
EVAL_HISTORY_JSONL = ARTIFACTS_DIR / "eval_history.jsonl"     # eval scores per pipeline run


def ensure_dirs() -> None:
    """Create every writable directory the pipeline needs (idempotent)."""
    for d in (DATA_DIR, ARTIFACTS_DIR, MEMORY_DIR):
        d.mkdir(parents=True, exist_ok=True)
