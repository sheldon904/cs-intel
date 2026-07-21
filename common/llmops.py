"""LLMOps rollup: turn the raw per-call trace log into a dashboard summary.

Reads ``artifacts/llm_usage.jsonl`` (written by ``common.llm`` on every call —
labeling, agent proposals, evals) and produces cost-per-account, per-stage
token/latency/cost aggregates, and an eval-scores-over-time series. This is the
local mirror of what Langfuse shows; both are populated by the same traced calls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.paths import (
    ACCOUNTS_PARQUET,
    CHURN_METRICS_JSON,
    EVAL_HISTORY_JSONL,
    INTENT_EVAL_JSON,
    LLM_USAGE_JSONL,
    LLMOPS_SUMMARY_JSON,
    ensure_dirs,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_eval_history(now: str | None = None) -> dict[str, Any]:
    """Snapshot the current eval scores into an append-only history file."""
    ensure_dirs()
    intent_eval = json.loads(INTENT_EVAL_JSON.read_text(encoding="utf-8")) if INTENT_EVAL_JSON.exists() else {}
    churn = json.loads(CHURN_METRICS_JSON.read_text(encoding="utf-8")) if CHURN_METRICS_JSON.exists() else {}
    row = {
        "ts": now or datetime.now(UTC).isoformat(),
        "intent_kappa": intent_eval.get("intent", {}).get("cohen_kappa"),
        "intent_accuracy": intent_eval.get("intent", {}).get("accuracy"),
        "sentiment_kappa": intent_eval.get("sentiment", {}).get("cohen_kappa"),
        "churn_auc": churn.get("xgboost", {}).get("auc"),
        "churn_pr_auc": churn.get("xgboost", {}).get("pr_auc"),
    }
    with EVAL_HISTORY_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def build_llmops_summary() -> dict[str, Any]:
    """Aggregate the trace log into ``artifacts/llmops_summary.json``."""
    ensure_dirs()
    usage = _read_jsonl(LLM_USAGE_JSONL)

    import pandas as pd

    n_accounts = int(len(pd.read_parquet(ACCOUNTS_PARQUET))) if ACCOUNTS_PARQUET.exists() else 0

    by_stage: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                 "projected_cost_usd": 0.0, "latency_ms_sum": 0.0}
    )
    provider = "mock"
    for row in usage:
        s = by_stage[row.get("stage", "unknown")]
        s["calls"] += 1
        s["input_tokens"] += row.get("input_tokens", 0)
        s["output_tokens"] += row.get("output_tokens", 0)
        s["cost_usd"] += row.get("cost_usd", 0.0)
        s["projected_cost_usd"] += row.get("projected_cost_usd", 0.0)
        s["latency_ms_sum"] += row.get("latency_ms", 0.0)
        provider = row.get("provider", provider)

    stage_summary = {}
    total_cost = total_proj = total_calls = 0.0
    for stage, s in by_stage.items():
        calls = s["calls"]
        stage_summary[stage] = {
            "calls": int(calls),
            "input_tokens": int(s["input_tokens"]),
            "output_tokens": int(s["output_tokens"]),
            "cost_usd": round(s["cost_usd"], 6),
            "projected_cost_usd": round(s["projected_cost_usd"], 6),
            "avg_latency_ms": round(s["latency_ms_sum"] / calls, 2) if calls else 0.0,
        }
        total_cost += s["cost_usd"]
        total_proj += s["projected_cost_usd"]
        total_calls += calls

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "note": (
            "In mock (offline) mode actual cost is $0; projected_cost prices the "
            "same token counts at the configured model's rate so the dashboard "
            "shows a realistic cost-per-account."
        ),
        "total_calls": int(total_calls),
        "total_cost_usd": round(total_cost, 6),
        "total_projected_cost_usd": round(total_proj, 6),
        "n_accounts": n_accounts,
        "projected_cost_per_account": round(total_proj / n_accounts, 6) if n_accounts else None,
        "actual_cost_per_account": round(total_cost / n_accounts, 6) if n_accounts else None,
        "by_stage": stage_summary,
        "eval_history": _read_jsonl(EVAL_HISTORY_JSONL),
    }
    LLMOPS_SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    append_eval_history()
    s = build_llmops_summary()
    print(f"provider={s['provider']}  calls={s['total_calls']}")
    print(f"projected cost/account = ${s['projected_cost_per_account']}")
    for stage, st in s["by_stage"].items():
        print(f"  {stage:18s} calls={st['calls']:>5} proj_cost=${st['projected_cost_usd']:.5f}")
